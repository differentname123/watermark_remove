import collections
import math
import re
import statistics
import time
from itertools import groupby
import difflib

import unicodedata
from difflib import SequenceMatcher

from common_utils.ASR.funasr_utils import run_funasr
from common_utils.ASR.speech_brain_utils import perform_speaker_diarization
from common_utils.ASR.whisper_utils import transcribe_words_to_json


import collections
import json
import math
import statistics
import unicodedata
from functools import lru_cache

# 在运行前，请确保已安装 pypinyin 库:
# pip install pypinyin
from pypinyin import Style, pinyin
import collections
import json
import math
import statistics
import unicodedata
from difflib import SequenceMatcher
import unicodedata


def read_json(filepath):
    """从文件路径读取JSON数据。"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def save_json(filepath, data):
    """将数据保存为格式化的JSON文件。"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving to {filepath}: {e}")


def calculate_pinyin_similarity(pinyin1, pinyin2):
    """
    使用 difflib.SequenceMatcher 计算两个拼音字符串的相似度。
    返回值在 0.0 到 1.0 之间。
    """
    if not pinyin1 or not pinyin2 or len(pinyin1) != len(pinyin2):
        return 0.0
    return difflib.SequenceMatcher(None, pinyin1, pinyin2).ratio()

def fuse_asr_results_final(all_asr_lists):
    """
    基于拼音共识与时间聚类的精简型ASR融合算法 (循环基准版)。

    该方案严格遵循三步策略，并引入了“循环基准”机制以提高稳定性：
    1.  线性化单字拆分：
        - 将所有ASR结果强制拆分为单个字符，时间戳按比例插值，置信度直接继承。

    2.  基于时间与发音的幻觉抑制 (循环基准)：
        - **轮流**将每一个ASR源作为基准进行聚类，并将所有轮次的结果合并。
        - 对合并后的聚类进行去重，确保每个字只参与一次投票。
        - 如果一个聚类的成员总数小于预设阈值，则判定为“幻觉”并整体删除。

    3.  最终选择与聚合：
        - 对通过抑制测试的“可信聚类”进行投票，选出最终结果。
    """

    # ----------------------- 可调参数 -----------------------
    TIME_WINDOW_MS = 150.0
    # 注意：这里的阈值含义稍有变化。原代码是 >=, 这里改为 >
    # len(cluster) > THRESHOLD, 意味着至少要有 THRESHOLD + 1 个成员
    # 如果您希望至少3个成员，就设为2。
    HALLUCINATION_THRESHOLD = 2

    # ----------------------- 工具函数 -----------------------
    def to_float(x, default=0.0):
        try:
            val = float(x)
            return default if math.isnan(val) else val
        except (ValueError, TypeError):
            return default

    def norm_text(s: str) -> str:
        if s is None: return ""
        return unicodedata.normalize("NFKC", str(s)).strip()

    def get_prob(item):
        p = item.get("probability", item.get("confidence", 0.5))
        return p if 0.0 <= p <= 1.0 else 0.5

    @lru_cache(maxsize=1024)
    def get_pinyin(char):
        # 如果是英文字母，直接返回小写
        if char.isalpha() and len(char) == 1:
            return char.lower()

        # 否则尝试转拼音
        p = pinyin(char, style=Style.NORMAL)
        if p:
            return p[0][0]
        else:
            return char.lower()

    # ----------------------- 1. 线性化单字拆分 -----------------------
    all_char_tokens = []
    for si, asr_list in enumerate(all_asr_lists or []):
        char_sequence = []
        for item_idx, item in enumerate(asr_list or []):
            word = norm_text(item.get("word", ""))
            if not word:
                continue
            start, end = to_float(item.get("start")), to_float(item.get("end"))
            prob = get_prob(item)
            duration = end - start
            if duration < 0:
                continue

            # 拆分为单字或数字
            if re.fullmatch(r'[A-Za-z]+', word):
                units = list(word)
            else:
                units = [word]

            if not units:
                continue

            for i, char in enumerate(units):
                token = {
                    "uid": (si, item_idx, i),
                    "text": char,
                    "start": start,  # 不再线性分割，直接用原始 start
                    "end": end,  # 不再线性分割，直接用原始 end
                    "prob": prob,
                    "pinyin": get_pinyin(char),
                }
                char_sequence.append(token)

        all_char_tokens.append(char_sequence)

    if not any(all_char_tokens): return []

    # ----------------------- 2. 基于“循环基准”的幻觉抑制 -----------------------

    all_clusters_from_all_rounds = []

    # --- 核心修改：外层循环，轮流选择基准 ---
    for base_idx in range(len(all_char_tokens)):
        base_sequence = all_char_tokens[base_idx]
        other_tokens = [token for i, seq in enumerate(all_char_tokens) if i != base_idx for token in seq]

        processed_uids_in_this_round = set()

        for base_token in base_sequence:
            if base_token['uid'] in processed_uids_in_this_round:
                continue

            # 1. 定义以 base_token 为中心的时间窗口
            t_center = (base_token['start'] + base_token['end']) / 2
            t_min = t_center - TIME_WINDOW_MS
            t_max = t_center + TIME_WINDOW_MS

            current_cluster = [base_token]
            processed_uids_in_this_round.add(base_token['uid'])

            # 寻找伙伴
            for other_token in other_tokens:
                if other_token['uid'] in processed_uids_in_this_round:
                    continue

                # --- 核心修改部分 ---

                # 2. 【新】稳健的时间重叠判断
                # base_token 的窗口是 [t_min, t_max]
                # other_token 的时间段是 [other_token['start'], other_token['end']]
                # 判断两个时间段是否有交集的条件是: start1 <= end2 AND start2 <= end1
                intervals_overlap = (t_min <= other_token['end']) and (other_token['start'] <= t_max)

                if not intervals_overlap:
                    continue  # 如果时间上不重叠，直接跳过，没必要再算拼音相似度

                # 3. 【新】拼音相似度判断
                pinyin_similarity = calculate_pinyin_similarity(base_token['pinyin'], other_token['pinyin'])

                # 同时满足时间重叠和拼音相似度阈值
                if pinyin_similarity > 0.8:
                    current_cluster.append(other_token)
                    processed_uids_in_this_round.add(other_token['uid'])

            all_clusters_from_all_rounds.append(current_cluster)

    if not all_clusters_from_all_rounds: return []

    # --- 合并与去重，确保每个 token 只属于一个最终聚类 ---
    # 按聚类大小降序排序，优先保留成员更多的聚类
    all_clusters_from_all_rounds.sort(key=len, reverse=True)

    final_clusters = []
    processed_token_uids = set()

    for cluster in all_clusters_from_all_rounds:
        # 检查当前聚类是否包含已被处理过的 token
        if any(token['uid'] in processed_token_uids for token in cluster):
            continue

        # 如果是全新的聚类，则采纳它，并将其所有 token 标记为已处理
        final_clusters.append(cluster)
        for token in cluster:
            processed_token_uids.add(token['uid'])

    # --- 应用幻觉抑制规则 ---
    valid_clusters = [c for c in final_clusters if len(c) >= HALLUCINATION_THRESHOLD]

    if not valid_clusters: return []

    # 对可信聚类按时间排序
    valid_clusters.sort(key=lambda c: statistics.median(t['start'] for t in c))

    # ----------------------- 3. 最终选择与聚合 -----------------------
    fused_result = []
    for cluster in valid_clusters:
        counts = collections.Counter(t['text'] for t in cluster)
        max_freq = max(counts.values())
        top_chars = [char for char, freq in counts.items() if freq == max_freq]

        winner_text = ""
        if len(top_chars) == 1:
            winner_text = top_chars[0]
        else:
            max_prob = -1
            for char in top_chars:
                avg_prob = statistics.mean(t['prob'] for t in cluster if t['text'] == char)
                if avg_prob > max_prob:
                    max_prob = avg_prob
                    winner_text = char

        final_start = statistics.median(t['start'] for t in cluster)
        final_end = statistics.median(t['end'] for t in cluster)
        if final_end <= final_start:
            final_end = final_start + 50

        final_prob = statistics.mean(t['prob'] for t in cluster)
        fused_result.append({
            "word": winner_text, "start": final_start, "end": final_end, "probability": final_prob
        })

    return fused_result


def foolproof_merge(speech_file, transcript_file, output_file,
                    include_word_timestamps=False, sentence_split_threshold=0.5):
    """
    Merges speaker segments and ASR transcripts, ensuring every single ASR word
    is assigned to a speaker. It also splits segments into sub-sentences based on
    time gaps and can optionally include word-level timestamps.

    The logic is word-centric:
    1. For each word, find the speaker segment with the maximum time overlap.
    2. If a word has no overlap (is in a gap), assign it to the chronologically
       closest speaker segment.
    3. Group consecutive words from the same speaker.
    4. Within each speaker group, create sub-sentences by splitting where the
       time gap between words exceeds `sentence_split_threshold`.
    5. Optionally include detailed timestamps for every single word.

    Args:
        speech_file (str): Path to the JSON file with speaker segments.
        transcript_file (str): Path to the JSON file with word-level transcripts.
        output_file (str): Path to save the final merged output.
        include_word_timestamps (bool): If True, adds a 'word_timestamps' list
                                        to each segment with per-word timing.
                                        Defaults to False.
        sentence_split_threshold (float): The time gap in seconds between words
                                          to trigger a new sub-sentence.
                                          Defaults to 0.5.
    """
    # 1. 加载数据
    print("Loading data...")
    try:
        with open(speech_file, 'r', encoding='utf-8') as f:
            segments = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Speaker file '{speech_file}' not found or is invalid. Assuming a single unknown speaker.")
        segments = []

    with open(transcript_file, 'r', encoding='utf-8') as f:
        words = json.load(f)

    if not words:
        print("Transcript file is empty. Nothing to process.")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
        return

    # 预处理：将说话人日志的时间单位从毫秒转换为秒
    for seg in segments:
        seg['start_s'] = seg['start'] / 1000.0
        seg['end_s'] = seg['end'] / 1000.0

    # 2. 为每一个 ASR 词语分配一个说话人
    print("Assigning a speaker to every word...")
    words_with_speaker = []
    for word in words:
        word_start = word['start']
        word_end = word['end']

        best_speaker = "SPEAKER_UNKNOWN"  # 默认值
        max_overlap_duration = -1

        # --- 首选规则：寻找最大重叠 ---
        if segments:
            for seg in segments:
                overlap = max(0, min(word_end, seg['end_s']) - max(word_start, seg['start_s']))
                if overlap > max_overlap_duration:
                    max_overlap_duration = overlap
                    best_speaker = seg['speaker']

        # --- 备用规则：寻找最近邻 ---
        if max_overlap_duration <= 0 and segments:
            min_distance = float('inf')
            # 找到时间上最近的说话人分段
            for seg in segments:
                # 计算词语和分段之间的时间间隙
                if word_end <= seg['start_s']:
                    distance = seg['start_s'] - word_end
                else:  # word_start >= seg['end_s']
                    distance = word_start - seg['end_s']

                if distance < min_distance:
                    min_distance = distance
                    best_speaker = seg['speaker']

        word['speaker'] = best_speaker
        words_with_speaker.append(word)

    # 3. 合并连续属于同一说话人的词语，并生成子句
    print("Grouping words and creating sub-sentences...")
    final_data = []
    if not words_with_speaker:
        print("No words to process after speaker assignment.")
    else:
        # 使用 groupby 按说话人对连续的词语进行分组
        for speaker, group in groupby(words_with_speaker, key=lambda x: x['speaker']):
            words_in_group = list(group)

            # --- 计算整个句段的宏观信息 ---
            segment_start = words_in_group[0]['start']
            segment_end = words_in_group[-1]['end']
            full_text = "".join(w['word'] for w in words_in_group)

            # --- 生成 sub_text_list ---
            sub_text_list = []
            if words_in_group:
                current_sub = {
                    'words': [words_in_group[0]['word']],
                    'start': words_in_group[0]['start'],
                    'end': words_in_group[0]['end']
                }
                for i in range(1, len(words_in_group)):
                    prev_word = words_in_group[i - 1]
                    curr_word = words_in_group[i]
                    gap = curr_word['start'] - prev_word['end']

                    if gap > sentence_split_threshold:
                        # 间隔过大，结束当前子句，开始新子句
                        current_sub['text'] = "".join(current_sub.pop('words'))
                        sub_text_list.append(current_sub)
                        current_sub = {
                            'words': [curr_word['word']],
                            'start': curr_word['start'],
                            'end': curr_word['end']
                        }
                    else:
                        # 间隔不大，继续向当前子句添加词语
                        current_sub['words'].append(curr_word['word'])
                        current_sub['end'] = curr_word['end']

                # 不要忘记添加最后一个子句
                current_sub['text'] = "".join(current_sub.pop('words'))
                sub_text_list.append(current_sub)

            # --- 组装最终数据 ---
            segment_data = {
                "speaker": speaker,
                "start": segment_start,
                "end": segment_end,
                "text": full_text,
                "sub_text_list": sub_text_list
            }

            # --- 根据参数添加可选的字级别时间戳 ---
            if include_word_timestamps:
                segment_data['word_timestamps'] = [
                    {'word': w['word'], 'start': w['start'], 'end': w['end']}
                    for w in words_in_group
                ]

            final_data.append(segment_data)

    # 4. 保存结果
    print(f"Saving final merged data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print("Done! All ASR words have been processed and included.")


def gen_precise_asr(audio_file, output_file):
    """
    生成融合后准确的asr文件
    """
    funasr_file = run_funasr(audio_file)
    whisper_v2_file = transcribe_words_to_json(audio_file, MODEL_SIZE="large-v2")
    # time.sleep(10)
    whisper_v3_file = transcribe_words_to_json(audio_file)
    ASR_FILES = [
        funasr_file,
        whisper_v2_file,
        whisper_v3_file,
    ]
    fuse_asr_file = 'output/fused_transcript_final.json'
    all_asr_lists = [read_json(f)[-10000:] for f in ASR_FILES if read_json(f) is not None]

    final_result = fuse_asr_results_final(all_asr_lists)


    # 将结果时间戳转换为秒，并格式化
    for item in final_result:
        item['start'] = round(item['start'] / 1000.0, 3)
        item['end'] = round(item['end'] / 1000.0, 3)
        item['probability'] = round(item['probability'], 4)

    save_json(fuse_asr_file, final_result)
    print(f"融合成功！最终结果已保存到 {fuse_asr_file}")

    speaker_file = perform_speaker_diarization(audio_file)
    foolproof_merge(speaker_file, fuse_asr_file, output_file)
    return output_file




# --- 示例用法 ---
if __name__ == '__main__':
    audio_file = r"mix.mp3"
    # audio_file = r"test.wav"


    OUTPUT_FILE = 'output/final_asr.json'

    gen_precise_asr(audio_file, OUTPUT_FILE)


    fuse_asr_file = 'output/fused_transcript_final.json'
    speaker_file = "output/segments_speech.json"
    foolproof_merge(speaker_file, fuse_asr_file, OUTPUT_FILE)