import collections
import math
import statistics
import unicodedata
from difflib import SequenceMatcher

from common_utils.ASR.funasr_utils import run_funasr
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
        p = pinyin(char, style=Style.NORMAL)
        return p[0][0] if p else char

    # ----------------------- 1. 线性化单字拆分 -----------------------
    all_char_tokens = []
    for si, asr_list in enumerate(all_asr_lists or []):
        char_sequence = []
        for item_idx, item in enumerate(asr_list or []):
            word = norm_text(item.get("word", ""))
            if not word: continue
            start, end = to_float(item.get("start")), to_float(item.get("end"))
            prob = get_prob(item)
            duration = end - start
            if duration < 0: continue
            units = list(word)
            unit_count = len(units)
            if unit_count == 0: continue
            for i, char in enumerate(units):
                u_start = start + duration * i / unit_count
                u_end = start + duration * (i + 1) / unit_count
                token = {
                    "uid": (si, item_idx, i), "text": char, "start": u_start,
                    "end": u_end, "prob": prob, "pinyin": get_pinyin(char)
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

            t_center = (base_token['start'] + base_token['end']) / 2
            t_min = t_center - TIME_WINDOW_MS
            t_max = t_center + TIME_WINDOW_MS

            current_cluster = [base_token]
            processed_uids_in_this_round.add(base_token['uid'])

            # 寻找伙伴
            for other_token in other_tokens:
                if other_token['uid'] in processed_uids_in_this_round:
                    continue

                other_t_center = (other_token['start'] + other_token['end']) / 2
                if t_min <= other_t_center <= t_max and other_token['pinyin'] == base_token['pinyin']:
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
def foolproof_merge(speech_file, transcript_file, output_file):
    """
    Merges speaker segments and ASR transcripts, ensuring every single ASR word
    is assigned to a speaker.

    The logic is word-centric:
    1. For each word, find the speaker segment with the maximum time overlap.
    2. If a word has no overlap (is in a gap), assign it to the chronologically
       closest speaker segment.
    3. Group consecutive words from the same speaker into sentences.

    Args:
        speech_file (str): Path to the JSON file with speaker segments.
        transcript_file (str): Path to the JSON file with word-level transcripts.
        output_file (str): Path to save the final merged output.
    """
    # 1. 加载数据
    print("Loading data...")
    with open(speech_file, 'r', encoding='utf-8') as f:
        segments = json.load(f)

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

        best_speaker = None
        max_overlap_duration = -1

        # --- 首选规则：寻找最大重叠 ---
        if segments:  # 仅当有说话人分段时才进行此操作
            for seg in segments:
                overlap = max(0, min(word_end, seg['end_s']) - max(word_start, seg['start_s']))
                if overlap > max_overlap_duration:
                    max_overlap_duration = overlap
                    best_speaker = seg['speaker']

        # --- 备用规则：寻找最近邻 ---
        if max_overlap_duration == 0:
            min_distance = float('inf')
            # 找到时间上最近的说话人分段
            if segments:
                for seg in segments:
                    # 计算词语和分段之间的时间间隙
                    if word_end <= seg['start_s']:
                        distance = seg['start_s'] - word_end
                    else:  # word_start >= seg['end_s']
                        distance = word_start - seg['end_s']

                    if distance < min_distance:
                        min_distance = distance
                        best_speaker = seg['speaker']
            else:
                # 如果没有说话人日志，则分配一个默认标签
                best_speaker = "SPEAKER_UNKNOWN"

        word['speaker'] = best_speaker
        words_with_speaker.append(word)

    # 3. 合并连续属于同一说话人的词语
    print("Grouping consecutive words into sentences...")
    final_data = []
    if not words_with_speaker:
        print("No words to process after speaker assignment.")
    else:
        current_group = {
            "speaker": words_with_speaker[0]['speaker'],
            "text_list": [words_with_speaker[0]['word']],
            "start": words_with_speaker[0]['start'],
            "end": words_with_speaker[0]['end']
        }

        for i in range(1, len(words_with_speaker)):
            word_data = words_with_speaker[i]
            if word_data['speaker'] == current_group['speaker']:
                # 如果说话人相同，则继续添加到当前组
                current_group['text_list'].append(word_data['word'])
                current_group['end'] = word_data['end']  # 更新结束时间
            else:
                # 如果说话人不同，则完成当前组并开始一个新组
                # 完成当前组
                current_group['text'] = "".join(current_group.pop('text_list'))
                final_data.append(current_group)

                # 开始新组
                current_group = {
                    "speaker": word_data['speaker'],
                    "text_list": [word_data['word']],
                    "start": word_data['start'],
                    "end": word_data['end']
                }

        # 不要忘记添加最后一个组
        current_group['text'] = "".join(current_group.pop('text_list'))
        final_data.append(current_group)

    # 4. 保存结果
    print(f"Saving final merged data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print("Done! All ASR words have been processed and included.")

# --- 示例用法 ---
if __name__ == '__main__':
    audio_file = r"mix.mp3"
    # audio_file = r"test.wav"




    # 假设你的ASR文件都在同一个目录下
    ASR_FILES = [
        run_funasr(audio_file),
        transcribe_words_to_json(audio_file),
        transcribe_words_to_json(audio_file, MODEL_SIZE="large-v2"),
    ]
    OUTPUT_FILE = 'output/fused_transcript_final.json'

    all_asr_lists = [read_json(f) for f in ASR_FILES if read_json(f) is not None]

    if len(all_asr_lists) > 1:
        # 执行最终版融合
        final_result = fuse_asr_results_final(all_asr_lists)

        # 将结果时间戳转换为秒，并格式化
        for item in final_result:
            item['start'] = round(item['start'] / 1000.0, 3)
            item['end'] = round(item['end'] / 1000.0, 3)
            item['probability'] = round(item['probability'], 4)

        save_json(OUTPUT_FILE, final_result)
        print(f"融合成功！最终结果已保存到 {OUTPUT_FILE}")
    else:
        print("未能加载足够的ASR文件进行融合。")