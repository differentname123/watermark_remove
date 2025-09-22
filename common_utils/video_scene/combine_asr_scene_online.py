# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/8/21 18:35
:last_date:
    2025/8/21 18:35
:description:

"""
import collections
import copy
import os
import time
import traceback
from typing import List, Dict, Any, Optional
from collections import Counter
from pypinyin import lazy_pinyin, Style

from typing import List, Dict, Any, Tuple

from LLM.gemini import get_llm_content, get_llm_content_gemini_flash_video
from common_utils.ASR.asr_fusion import gen_precise_asr
from common_utils.ASR.speech_brain_utils import perform_speaker_diarization
from common_utils.common_utils import read_json, time_to_ms, save_json, ms_to_time, read_file_to_str, string_to_object, \
    timeit_print, is_valid_target_file_simple
from common_utils.image_utils import save_frames_around_timestamp
from common_utils.split_audio import separate_with_cli
from common_utils.split_scenes import find_and_split_scenes, split_scenes_json
from common_utils.tts.edge_tts_utils import generate_audio_and_get_duration_sync
from common_utils.video_utils import extract_audio_from_video, clip_video_ms, merge_videos_ffmpeg, probe_duration, \
    add_subtitles_to_video
from common_utils.video_utils1 import redub_video_with_ffmpeg
from common_utils.video_utils2 import add_bgm_to_video

import string
import re
from copy import deepcopy

from common_utils.video_utils_cut import gen_video

# ==============================================================================
# 1. 辅助常量与函数 (Helpers & Constants)
# ==============================================================================

# --- 单一数据源：定义用于分割句子的正则表达式 ---
# 这是我们唯一的“真相来源”。所有关于标点的逻辑都将从此派生。
# 括号 () 用于捕获分隔符，方括号 [] 内是所有作为分隔符的字符。
SPLIT_SENTENCE_REGEX = r'([,，.。!?！、？])'

# --- 派生逻辑：根据正则表达式自动生成标点符号集 ---
# 我们从正则表达式中提取出所有标点，用于 is_not_punctuation 函数。
# 这样，只要修改上面的正则表达式，下面的集合就会自动更新。
PUNCTUATION_SET = set(re.findall(r'\[(.*?)\]', SPLIT_SENTENCE_REGEX)[0])


def is_not_punctuation(char: str) -> bool:
    """
    判断一个字符是否为标点符号。
    此函数的逻辑完全由 SPLIT_SENTENCE_REGEX 派生，确保了定义的一致性。
    """
    # 我们认为，任何不参与分割的字符，如果它也不是空格，就是有效字符。
    return char and char.strip() and char not in PUNCTUATION_SET


def split_text_into_sentences(text: str) -> list[str]:
    """
    将段落文本分割成子句列表。
    此函数是“真相来源”，其行为定义了哪些字符是标点。
    """
    if not text:
        return []

    sentences = re.split(SPLIT_SENTENCE_REGEX, text)
    if not sentences:
        return [text]

    # 将分隔符合并到它们前面的句子中
    result = []
    for i in range(0, len(sentences), 2):
        # 确保不添加由连续分隔符产生的空字符串
        if sentences[i]:
            current_sentence = sentences[i]
            if i + 1 < len(sentences) and sentences[i + 1]:
                current_sentence += sentences[i + 1]
            result.append(current_sentence)

    # 如果最后没有分割，返回原始文本列表
    if not result:
        return [text]

    return result


# --- 模拟函数，请替换为您自己的实现 ---
def get_pinyin_for_char(s: str) -> str:
    """
    - 若输入不含中文，原样返回 s（例如 "ABC" -> "ABC"）。
    - 若输入含中文，则对每个字符返回无声调拼音并以空格分隔（例如 "你好" -> "ni hao"，"你A" -> "ni A"）。
    """
    if not s:
        return s
    if not any('\u4e00' <= ch <= '\u9fff' for ch in s):
        return s
    return " ".join(lazy_pinyin(s, style=Style.NORMAL, errors="keep"))


# ------------------------------------------

def _find_char_in_asr(
        char_pinyin: str,
        search_center_index: int,
        asr_data: list[dict],
        window_size: int = 5
) -> tuple[dict | None, int]:
    if not asr_data:
        return None, -1

    center_idx = int(round(search_center_index))

    n = len(asr_data)
    if center_idx < 0:
        center_idx = 0
    elif center_idx >= n:
        center_idx = n - 1

    if 0 <= center_idx < len(asr_data):
        asr_word_info = asr_data[center_idx]
        asr_word = asr_word_info.get("word", "")
        if asr_word and char_pinyin == get_pinyin_for_char(asr_word[0]):
            return asr_word_info, center_idx

    for offset in range(1, window_size + 1):
        right_idx = center_idx + offset
        if right_idx < len(asr_data):
            asr_word_info = asr_data[right_idx]
            asr_word = asr_word_info.get("word", "")
            if asr_word and char_pinyin == get_pinyin_for_char(asr_word[0]):
                return asr_word_info, right_idx

        left_idx = center_idx - offset
        if left_idx >= 0:
            asr_word_info = asr_data[left_idx]
            asr_word = asr_word_info.get("word", "")
            if asr_word and char_pinyin == get_pinyin_for_char(asr_word[0]):
                return asr_word_info, left_idx

    return None, -1


# ==============================================================================
# 2. 主逻辑函数 (Main Logic)
# ==============================================================================

def compute_sentence_time(starts: List[Dict], key: str = "start") -> Optional[int]:
    """
    用 items 中每项的 probability^2 作为权重，计算 key 对应值的加权平均并返回 int。
    - items: 列表，每项为 dict，期望包含 key 和 "probability"（若无 probability 则视为 1）。
    - 若没有有效的数值则返回 None。
    """
    total_w = 0.0
    weighted_sum = 0.0

    for item in starts:
        # 提取并验证 value
        val = item.get(key)
        if val is None:
            continue
        try:
            v = float(val)
        except Exception:
            continue

        # 提取并验证 probability（缺失则默认 1）
        p = item.get("probability", 1)
        try:
            p = float(p)
        except Exception:
            p = 1.0

        w = p * p
        if w <= 0:
            continue

        weighted_sum += v * w
        total_w += w

    if total_w == 0:
        return None

    avg = weighted_sum / total_w
    return int(round(avg))


def split_tokens_linear(tokens):
    out = []
    for it in tokens:
        w = it.get("word", "") or ""
        try:
            s = int(it.get("start", 0));
            e = int(it.get("end", 0))
        except Exception:
            out.append(it.copy());
            continue
        if not w or len(w) <= 1 or s >= e:
            out.append(it.copy());
            continue
        parts = list(w);
        total = e - s;
        n = len(parts)
        base, rem = divmod(total, n)
        cur = s
        for i, p in enumerate(parts):
            dur = base + (1 if i < rem else 0)
            new = it.copy()
            new.update({"word": p, "start": cur, "end": cur + dur})
            out.append(new)
            cur += dur
        if out and out[-1]["end"] != e:
            out[-1]["end"] = e
    return out


def generate_sub_sentence_timestamps(
        asr_data_list: List[List[Dict]],
        corrected_text_data: Dict,
        search_window: int = 20,
        time_margin: int = 1000
) -> Dict:
    """
    asr_data_list: 一个包含多个 asr_data 的列表，每个 asr_data 本身是 list[dict]
    corrected_text_data: 原来的 corrected_text_data（包含 'fixed_asr_list'）
    返回值结构保持不变，但每个 sub_text 的 start/end 为各 ASR 结果的平均（若存在）
    """
    processed_data = deepcopy(corrected_text_data)

    for segment in processed_data:
        final_text = segment.get("final_text", "")
        segment['speaker'] = 'owner_speaker'
        if not final_text:
            segment['sub_text'] = []
            continue

        # 针对每个 asr_data 预先筛选与当前 segment 相关的 words 列表
        relevant_asr_per_source = []
        for asr_data in asr_data_list:
            asr_data = split_tokens_linear(asr_data)
            relevant_asr = [
                word for word in asr_data
                if word.get('end', 0) > segment.get('start', 0) - time_margin and
                   word.get('start', 0) < segment.get('end', 0) + time_margin
            ]
            relevant_asr_per_source.append(relevant_asr)

        # 如果所有来源都没有相关 asr，则直接置空
        if all(len(r) == 0 for r in relevant_asr_per_source):
            segment['sub_text'] = []
            continue

        sub_sentence_list = split_text_into_sentences(final_text)
        pass1_results = []

        segment_char_cursor = 0
        # 每个 asr source 单独维护 offset（用于 search center index）
        offsets = [0.0 for _ in asr_data_list]

        for sentence_text in sub_sentence_list:
            effective_chars = [c for c in sentence_text if is_not_punctuation(c)]

            if not effective_chars:
                pass1_results.append({"text": sentence_text, "start": None, "end": None})
                continue

            # 为每个 asr source 尝试找到 first/last char 的信息（可能为 None）
            sentence_start_candidates = []  # 每个 source 的 start（或 None）
            sentence_end_candidates = []  # 每个 source 的 end（或 None）

            # 先处理首字符
            first_char_pinyin = get_pinyin_for_char(effective_chars[0])
            text_expected_pos = segment_char_cursor

            for idx, relevant_asr in enumerate(relevant_asr_per_source):
                # 如果该 source 没有相关 asr，跳过
                if not relevant_asr:
                    sentence_start_candidates.append(None)
                    continue

                search_center_index = text_expected_pos + offsets[idx]
                first_info, first_match_idx = _find_char_in_asr(
                    first_char_pinyin, search_center_index, relevant_asr, window_size=search_window
                )
                if first_info:
                    sentence_start_candidates.append(first_info)
                    # 更新对应 source 的 offset
                    offsets[idx] = first_match_idx - text_expected_pos
                else:
                    sentence_start_candidates.append(None)

            # 取所有非 None start 的平均
            starts = [s for s in sentence_start_candidates if s is not None]
            sentence_start_time = compute_sentence_time(starts)
            # 处理 end（单字与多字不同处理）
            if len(effective_chars) == 1:
                # end 来自首字符的 end（每个 source）
                for idx, relevant_asr in enumerate(relevant_asr_per_source):
                    if not relevant_asr:
                        sentence_end_candidates.append(None)
                        continue
                    # 如果我们之前在该 source 找到 first_info，我们 should get its 'end' value.
                    # 重新 run 找一次首字（为了得到 end）——可以优化复用，但保持接口一致
                    search_center_index = text_expected_pos + offsets[idx]
                    first_info, _ = _find_char_in_asr(
                        first_char_pinyin, search_center_index, relevant_asr, window_size=search_window
                    )
                    if first_info:
                        sentence_end_candidates.append(first_info)
                    else:
                        sentence_end_candidates.append(None)
            else:
                # 多字符，找最后一个字符
                last_char_pinyin = get_pinyin_for_char(effective_chars[-1])
                text_expected_pos_last = segment_char_cursor + len(effective_chars) - 1

                for idx, relevant_asr in enumerate(relevant_asr_per_source):
                    if not relevant_asr:
                        sentence_end_candidates.append(None)
                        continue

                    search_center_index_last = text_expected_pos_last + offsets[idx]
                    last_info, last_match_idx = _find_char_in_asr(
                        last_char_pinyin, search_center_index_last, relevant_asr, window_size=search_window
                    )
                    if last_info:
                        sentence_end_candidates.append(last_info)
                        # 更新对应 source 的 offset（基于最后字符的位置）
                        offsets[idx] = last_match_idx - text_expected_pos_last
                    else:
                        sentence_end_candidates.append(None)

            ends = [e for e in sentence_end_candidates if e is not None]
            sentence_end_time = compute_sentence_time(ends, key="end")

            # 如果长度小且没有时间信息，则跳过（保留你原有的规则）
            if len(effective_chars) <= 5 and sentence_start_time is None and sentence_end_time is None:
                # 不添加这一短句
                segment_char_cursor += len(effective_chars)
                continue

            pass1_results.append({
                "text": sentence_text,
                "start": sentence_start_time,
                "end": sentence_end_time
            })

            segment_char_cursor += len(effective_chars)

        # 第二遍：填补缺失 start/end（沿用原逻辑）
        final_sub_text = []
        for i, sub in enumerate(pass1_results):
            if sub['start'] is None:
                prev_end = segment.get('start', 0)
                for j in range(i - 1, -1, -1):
                    if pass1_results[j]['end'] is not None:
                        prev_end = pass1_results[j]['end']
                        break
                sub['start'] = prev_end

            if sub['end'] is None:
                next_start = segment.get('end', 0)
                for j in range(i + 1, len(pass1_results)):
                    if pass1_results[j]['start'] is not None:
                        next_start = pass1_results[j]['start']
                        break
                sub['end'] = next_start

            if sub['start'] > sub['end']:
                sub['end'] = sub['start']

            final_sub_text.append(sub)

        segment['sub_text'] = final_sub_text

    return processed_data


def fill_speaker_texts(
        words: List[Dict[str, Any]],
        speaker_segments: List[Dict[str, Any]],
        keep_word_list: bool = False,
        joiner: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    将 ASR 的 words 填充进说话人 segment 列表。

    Args:
        words: 每个元素类似 {"word": "盘", "start": 130, "end": 290, ...}
        speaker_segments: 每个元素类似 {"start": 31, "end": 14138, "speaker": "SPEAKER_00", ...}
        keep_word_list: 是否在返回的每个 segment 中保留 "words" 列表（默认 False）
        joiner: 用于拼接 words 的分隔符；
                if None -> 自动选择：如果大多数 word 都是 ASCII（包含字母/数字），用 ' '（空格），否则用 ''（不加空格）
                可以显式传入 '' 或 ' '。

    Returns:
        返回一个新的 speaker_segments 列表（每个为原 dict 的浅拷贝），
        每个 dict 会新增:
            - 'text': 按时间排序拼接得到的字符串
            - 'words': 若 keep_word_list True，则为包含的 word 列表（按时间排序）
    """
    # defensive copies
    segs = [dict(s) for s in speaker_segments]
    words_sorted = sorted(words, key=lambda w: w.get('start', 0))

    # choose joiner if None
    if joiner is None:
        # heuristic: if any word contains ascii letters/digits, prefer space; else no space (for Chinese)
        ascii_like = sum(1 for w in words_sorted if
                         any((c.isascii() and (c.isalpha() or c.isdigit())) for c in str(w.get('word', ''))))
        joiner = ' ' if ascii_like >= len(words_sorted) / 3 else ''

    # prepare container for assigned words
    for s in segs:
        s['_assigned_words'] = []

    # helper: overlap length
    def overlap_len(a_start, a_end, b_start, b_end):
        return max(0, min(a_end, b_end) - max(a_start, b_start))

    # assign each word to the segment with max overlap
    for w in words_sorted:
        w_s = w.get('start', 0)
        w_e = w.get('end', 0)
        best_idx = None
        best_ol = 0
        # iterate all segments and find best overlap
        for i, s in enumerate(segs):
            s_s = s.get('start', 0)
            s_e = s.get('end', 0)
            ol = overlap_len(w_s, w_e, s_s, s_e)
            if ol > best_ol:
                best_ol = ol
                best_idx = i
        # assign if any positive overlap
        if best_idx is not None and best_ol > 0:
            segs[best_idx]['_assigned_words'].append(dict(w))  # append a shallow copy

    # build text and optionally keep words
    for s in segs:
        s['_assigned_words'].sort(key=lambda x: x.get('start', 0))
        s['text'] = joiner.join([str(w.get('word', '')) for w in s['_assigned_words']])
        if keep_word_list:
            s['words'] = s['_assigned_words']
        # cleanup internal key
        s.pop('_assigned_words', None)

    return segs


def merge_by_key(temp_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    简单按 (start,end,speaker) 分组合并 temp_list。
    返回每个分组字典，包含原始 start,end,speaker 和若干 text_{basename} 字段。
    """
    if not temp_list:
        return []

    # 推断每个子列表对应的 basename
    basenames = []
    for i, segs in enumerate(temp_list):
        name = None
        if isinstance(segs, list) and len(segs) > 0:
            name = segs[0].get('ASR_FILE') or segs[0].get('asr_file')
        if name:
            name = os.path.basename(name)
        else:
            name = f"asr_{i}"
        basenames.append(name)

    base_map: Dict[Tuple[int, int, str], Dict[str, Any]] = {}

    # 聚合：把每个 segment 的 text 塞到对应 key 下
    for asr_idx, segs in enumerate(temp_list):
        name = basenames[asr_idx]
        tf = f"text_{name}"
        for seg in (segs or []):
            # 仅按 start,end,speaker 分组；缺少这些字段的 segment 会被跳过
            if not all(k in seg for k in ("start", "end", "speaker")):
                # 若你希望保留无 loc 的 seg，可改为把它们放到特殊 key
                continue
            key = (seg["start"], seg["end"], seg["speaker"])
            entry = base_map.setdefault(key, {"start": seg["start"], "end": seg["end"], "speaker": seg["speaker"]})
            # 直接写入（若同一 (start,end,speaker) 在同一 ASR 中出现多次，后者会覆盖）
            entry[tf] = seg.get("text", "") or ""

    # 确保每个 entry 都有所有 text_{basename} 字段（缺失则填 ""）
    for entry in base_map.values():
        for name in basenames:
            entry.setdefault(f"text_{name}", "")

    # 按 start 排序并返回列表
    merged = sorted(base_map.values(), key=lambda e: (e.get("start", 0), e.get("end", 0)))
    return merged


def check_fix_speech_asr(fixed_speech_asr_info, speech_asr_info):
    """
    检查修复后的说话人文本是否合理
    """
    if not fixed_speech_asr_info or 'fixed_asr_list' not in fixed_speech_asr_info:
        print("[ERROR] 修复后的说话人文本信息无效或缺失 'fixed_asr_list' 字段")
        return False

    fixed_list = fixed_speech_asr_info['fixed_asr_list']
    if len(fixed_list) != len(speech_asr_info):
        print(f"[ERROR] 修复后的说话人文本长度与原始不匹配: {len(fixed_list)} != {len(speech_asr_info)}")
        return False

    # 检查'owner_speaker'是否为空或者是否是已存在的说话人
    speaker_list = {entry.get('speaker') for entry in speech_asr_info if 'speaker' in entry}
    owner_speaker = fixed_speech_asr_info.get('owner_speaker', '')
    if owner_speaker and owner_speaker not in speaker_list:
        print(f"[ERROR] 修复后的说话人文本中 'owner_speaker' 无效: '{owner_speaker}'")
        return False

    # 将fixed_list中所有speaker为owner_speaker的值设置为 'owner_speaker'
    for entry in fixed_list:
        if entry.get('speaker') == owner_speaker:
            entry['speaker'] = 'owner_speaker'

    print("[INFO] 修复后的说话人文本信息通过基本检查")
    return True

def check_owner_asr(owner_asr_info, speaker_info):
    """
    检查生成的asr文本是否正确，第一是验证每个时间是否合理（1.最长跨度不能够超过20s 2.时长的合理性（也就是最快和最慢的语速就能够知道文本对应的时长是否合理） 3.owner语音和本地speaker说话人日志的差异不能够太大）
    """
    return True

def gen_owner_asr_by_llm(video_path, speaker_info):
    """
    通过大模型生成asr文本，附带主人识
    """
    retry_delay = 10
    max_retries = 3
    prompt_file_path = '../../content_community/app/视频分解素材_直接进行asr转录与owner识别.txt'
    prompt = read_file_to_str(prompt_file_path)
    full_prompt = f'{prompt}'
    raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            model_name = "gemini-2.5-pro"
            raw = get_llm_content_gemini_flash_video(prompt=full_prompt, video_path=video_path, model_name=model_name)

            owner_asr_info = string_to_object(raw)
            # 检测fix_speech_asr_info和speech_asr_info长度是否一致
            if check_owner_asr(owner_asr_info, speaker_info) is False:
                raise ValueError(f"[ERROR] 生成的视频信息与原始不匹配，尝试重新生成 (尝试 {attempt}/{max_retries})")
            return owner_asr_info
        except Exception as e:
            print(f"[ERROR] 生成视频信息失败 (尝试 {attempt}/{max_retries}): {e} {raw}")
            if attempt < max_retries:
                print(f"[INFO] 正在重试... (等待 {retry_delay} 秒)")
                time.sleep(retry_delay)  # 等待一段时间后再重试
            else:
                print("[ERROR] 达到最大重试次数，失败.")
                return None  # 达到最大重试次数后返回 None
            traceback.print_exc()

def gen_audio_path(input_path: str, split_vocal=True) -> str:
    """
    生成处理好的音频文件路径。
    """
    processed_audio_path = input_path
    input_audio_path = input_path
    if input_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
        new_audio_file = input_path.replace('.mp4', '.wav')
        extract_audio_from_video(input_path, new_audio_file)
        input_audio_path = new_audio_file
        processed_audio_path = new_audio_file
    # 获取绝对路径
    abs_input_path = os.path.abspath(input_path)
    base_dir = os.path.dirname(abs_input_path)


    if split_vocal:
        split_audio_path = os.path.join(base_dir, "htdemucs", os.path.basename(input_audio_path).rsplit('.', 1)[0], "vocals.wav")
        if not is_valid_target_file_simple(split_audio_path):
            separate_with_cli(input_audio_path, base_dir)
        if is_valid_target_file_simple(split_audio_path):
            processed_audio_path = split_audio_path
    print(f"处理后的音频文件路径: {processed_audio_path}")
    return processed_audio_path

def fix_owner_speaker(video_path, fixed_speech_asr_info):
    """
    进一步的修复主人语音，避免漏识别和误识别
    """
    owner_asr_list = []
    fixed_list = fixed_speech_asr_info['fixed_asr_list']
    for entry in fixed_list:
        if entry.get('speaker') == 'owner_speaker' and entry.get('final_text', '').strip():
            # 删除speaker字段
            entry.pop('speaker', None)
            owner_asr_list.append(entry)

    retry_delay = 10
    max_retries = 3
    prompt_file_path = '../../content_community/app/视频分解素材_进一步进行准确的主人音频纠正.txt'
    prompt = read_file_to_str(prompt_file_path)
    full_prompt = f'{prompt}\n{owner_asr_list}'
    raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            model_name = "gemini-2.5-pro"
            raw = get_llm_content_gemini_flash_video(prompt=full_prompt, video_path=video_path, model_name=model_name)

            fix_speech_asr_info = string_to_object(raw)
            return fix_speech_asr_info
        except Exception as e:
            print(f"[ERROR] 生成视频信息失败 (尝试 {attempt}/{max_retries}): {e} {raw}")
            if attempt < max_retries:
                print(f"[INFO] 正在重试... (等待 {retry_delay} 秒)")
                time.sleep(retry_delay)  # 等待一段时间后再重试
            else:
                print("[ERROR] 达到最大重试次数，失败.")
                return None  # 达到最大重试次数后返回 None
            traceback.print_exc()

@timeit_print
def gen_asr(video_path):
    """
    生成修复后的asr以及句子时间段
    """
    start_time = time.time()
    base_name = os.path.basename(video_path).split('.')[0]
    speech_asr_output_file = f'output/{base_name}/{base_name}_speech_asr_with_owner.json'

    processed_audio_path = gen_audio_path(video_path)
    base_name = os.path.basename(processed_audio_path).split('.')[0]
    output_dir = os.path.dirname(speech_asr_output_file)

    speaker_file_path = os.path.join(output_dir, f"{base_name}_speaker.json")
    # return ASR_FILES, ASR_FILES
    speaker_file_path = perform_speaker_diarization(processed_audio_path, output_path=speaker_file_path)
    speaker_info = read_json(speaker_file_path)
    if not is_valid_target_file_simple(speech_asr_output_file):
        owner_asr_info = gen_owner_asr_by_llm(video_path, speaker_info)
        save_json(speech_asr_output_file, owner_asr_info)
    print(f"生成精准asr与说话人信息文件耗时: {time.time() - start_time} 秒")
    owner_asr_info = read_json(speech_asr_output_file)
    return owner_asr_info


def merge_scene_timestamps(scene_dict, min_count=3, count_by_threshold=True):
    """
    合并不同阈值下的场景时间点。

    行为说明：
      - kept_sorted: 返回**未过滤**的时间戳及其出现次数，类型为列表 [(timestamp_str, count), ...]，
                    并按真实时间升序排序。
      - pairs: 仍然基于满足 min_count 的时间戳构建相邻配对区间，格式为
               {'场景1': {'start': s, 'end': e, 'duration': ms}, ...}

    参数:
      scene_dict: 嵌套字典，外层 key 为阈值（例如 40,50,60），内层为场景名 -> [start, end]
      min_count: 只把出现次数 >= min_count 的时间戳用于构建 pairs（默认 3）
      count_by_threshold: True 时在每个阈值内先去重再计数（推荐），
                          False 时把所有出现次数都计入（同阈值重复会被计多次）

    返回:
      (kept_sorted, pairs)
        - kept_sorted: [(timestamp_str, count), ...] （未过滤，按时间升序）
        - pairs: dict，键为 '场景1','场景2',...，值为 {'start','end','duration'}
    """
    from collections import Counter

    counts = Counter()

    for thr, scenes in scene_dict.items():
        ts_list = []
        for scene_name, bounds in scenes.items():
            if not bounds:
                continue
            # 期望 bounds = [start, end]
            ts_list.extend(time_to_ms(t) for t in bounds if isinstance(t, str) and t.strip())
        if count_by_threshold:
            for ts in set(ts_list):
                counts[ts] += 1
        else:
            for ts in ts_list:
                counts[ts] += 1

    # kept_sorted: 未过滤，包含每个时间戳的出现次数，按时间升序
    kept_sorted = sorted(counts.items(), key=lambda kv: time_to_ms(kv[0]))  # [(ts, count), ...]

    # 下面为构建 pairs：仍然只使用出现次数 >= min_count 的时间戳（按时间排序）
    filtered_ts = [ts for ts, c in counts.items() if c >= min_count]
    filtered_sorted = sorted(filtered_ts, key=time_to_ms)

    pairs = {}
    n = len(filtered_sorted)
    if n == 0:
        return kept_sorted, pairs
    if n == 1:
        start = filtered_sorted[0]
        end = filtered_sorted[0]
        td = time_to_ms(end) - time_to_ms(start)
        pairs['场景1'] = {
            'start': start,
            'end': end,
            'duration': td
        }
        return kept_sorted, pairs

    for i in range(n - 1):
        key = f"场景{i + 1}"
        start = filtered_sorted[i]
        end = filtered_sorted[i + 1]
        td = time_to_ms(end) - time_to_ms(start)
        pairs[key] = {
            'start': start,
            'end': end,
            'duration': td
        }

    return kept_sorted, pairs


@timeit_print
def get_scene(video_path):
    basename = os.path.basename(video_path).split('.')[0]

    all_scene_info_dict = {}
    for high_threshold in [30, 40, 50, 60, 70]:
        start_time = time.time()
        scene_info_file = f'output/{basename}/scenes_{basename}_{high_threshold}/scene_info.json'
        if is_valid_target_file_simple(scene_info_file):
            # print(f"场景信息文件已存在，跳过处理: {scene_info_file}")
            all_scene_info_dict[high_threshold] = read_json(scene_info_file)
            continue

        # 运行带有精炼功能的场景分割
        scene_info_dict = split_scenes_json(
            video_path,
            high_threshold=high_threshold,  # 初始高阈值
            min_scene_len=25,  # 最小场景长度（帧）
        )
        print(
            f"阈值为 {high_threshold}场景信息字典已生成并打印。共 {len(scene_info_dict)} 个场景。 耗时: {time.time() - start_time} 秒\n")
        # for key, value in scene_info_dict.items():
        #     timestamp = value[1]
        #     save_frames_around_timestamp(video_path, timestamp, 3, str(os.path.join(f'output/{basename}/scenes_{basename}_{high_threshold}', key)))

        save_json(scene_info_file, scene_info_dict)
        all_scene_info_dict[high_threshold] = scene_info_dict
    kept_sorted, pairs = merge_scene_timestamps(all_scene_info_dict, min_count=3)

    print(f"\n合并后的场景数量为: {len(kept_sorted)}")
    # 将kept_sorted保存到文件
    save_json(f'output/{basename}/scenes_fused_{basename}/merged_timestamps.json', kept_sorted)

    # for key, value in pairs.items():
    #     timestamp = value[1]
    #     save_frames_around_timestamp(my_video_path, timestamp, 3,
    #                                  str(os.path.join(f'scenes_fused_{basename}', key)))

    return kept_sorted


def _contains_owner_speaker(scene, owner_speaker):
    """辅助函数：检查场景的 content_list 是否包含 owner_speaker 的文本。"""
    if not scene['content_list']:
        return False
    for text in scene['content_list']:
        if text.get('speaker') == owner_speaker:
            return True
    return False


def process_scenes_complete_fix(
        text_results,
        scene_splits,
        owner_speaker='owner',
        min_overlap_ms=500,
        keep_empty_scenes=True,
        min_scene_duration_ms=10000
):
    """
    一个完整的、结合了“主动合并”与“短场景清理”的最终修复版函数 (V9)。

    - 核心修复 1 (您的方案): 在文本分配阶段，如果 owner_speaker 的文本跨越多个场景，
      则立即将这些场景合并。
    - 核心修复 2 (功能恢复): 恢复了对短场景的清理合并逻辑，正确使用 min_scene_duration_ms。
    """
    # --- 步骤 0: 预处理 (无变化) ---
    if not scene_splits: return []
    scenes = []
    if len(scene_splits) > 1:
        for i in range(len(scene_splits) - 1):
            scenes.append({'start': scene_splits[i][0], 'end': scene_splits[i + 1][0], 'content_list': []})
        last_scene_start = scene_splits[-1][0]
        max_text_end = max(item['end'] for item in text_results) if text_results else last_scene_start
        scenes.append({'start': last_scene_start, 'end': max(last_scene_start, max_text_end), 'content_list': []})
    else:
        start_time = scene_splits[0][0] if scene_splits else 0
        max_text_end = max(item['end'] for item in text_results) if text_results else start_time
        scenes.append({'start': start_time, 'end': max_text_end, 'content_list': []})
    if not scenes: return []

    # --- [核心修复] 步骤 1: 智能分配与主动合并 ---
    text_queue = sorted(text_results, key=lambda x: x['start'])
    for text in text_queue:
        overlapping_indices = []
        for i, scene in enumerate(scenes):
            overlap_duration = max(0, min(text['end'], scene['end']) - max(text['start'], scene['start']))
            is_owner = text.get('speaker') == owner_speaker
            min_req_overlap = min_overlap_ms if is_owner else 1
            if overlap_duration >= min_req_overlap:
                overlapping_indices.append(i)

        if not overlapping_indices:
            min_distance, best_scene_idx = float('inf'), -1
            for i, scene in enumerate(scenes):
                distance = min(abs(scene['start'] - text['end']), abs(text['start'] - scene['end']))
                if distance < min_distance:
                    min_distance, best_scene_idx = distance, i
            if best_scene_idx != -1:
                scenes[best_scene_idx]['content_list'].append(text)
            continue

        if len(overlapping_indices) == 1:
            scenes[overlapping_indices[0]]['content_list'].append(text)
            continue

        if text.get('speaker') == owner_speaker and len(overlapping_indices) > 1:
            first_idx = overlapping_indices[0]
            for idx in sorted(overlapping_indices[1:], reverse=True):
                scenes[first_idx]['start'] = min(scenes[first_idx]['start'], scenes[idx]['start'])
                scenes[first_idx]['end'] = max(scenes[first_idx]['end'], scenes[idx]['end'])
                scenes[first_idx]['content_list'].extend(scenes[idx]['content_list'])
                scenes.pop(idx)
            scenes[first_idx]['content_list'].append(text)
        else:
            best_scene_idx = -1
            max_overlap = 0
            for i in overlapping_indices:
                scene = scenes[i]
                overlap_duration = max(0, min(text['end'], scene['end']) - max(text['start'], scene['start']))
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_scene_idx = i
            if best_scene_idx != -1:
                scenes[best_scene_idx]['content_list'].append(text)

    # --- 步骤 2: 中间处理 ---
    # 排序 & 过滤空场景
    for scene in scenes:
        scene['content_list'].sort(key=lambda x: x['start'])

    first_pass_merged = []
    if not keep_empty_scenes:
        for scene in scenes:
            if scene['content_list']:
                first_pass_merged.append(scene)
    else:
        first_pass_merged = scenes

    # --- [功能恢复] 步骤 3: 第二阶段合并 (Short Scene Cleanup) ---
    # 这里的代码逻辑与你最初版本中的步骤4完全相同，现在它作用于主动合并之后的结果
    if len(first_pass_merged) <= 1:
        final_scenes_unformatted = first_pass_merged
    else:
        final_scenes_unformatted = []
        i = 0
        while i < len(first_pass_merged):
            current_scene = first_pass_merged[i]
            is_short = (current_scene['end'] - current_scene['start']) < min_scene_duration_ms
            has_owner = _contains_owner_speaker(current_scene, owner_speaker)

            if not is_short or has_owner:
                final_scenes_unformatted.append(current_scene)
                i += 1
                continue

            # 尝试向上合并
            can_merge_up = False
            if final_scenes_unformatted:
                prev_scene = final_scenes_unformatted[-1]
                if not prev_scene['content_list'] or prev_scene['content_list'][-1].get('speaker') != owner_speaker:
                    can_merge_up = True
            if can_merge_up:
                prev_scene['end'] = max(prev_scene['end'], current_scene['end'])
                prev_scene['content_list'].extend(current_scene['content_list'])
                prev_scene['content_list'].sort(key=lambda x: x['start'])
                i += 1
                continue

            # 尝试向下合并
            can_merge_down = False
            if (i + 1) < len(first_pass_merged):
                next_scene = first_pass_merged[i + 1]
                if not next_scene['content_list'] or next_scene['content_list'][0].get('speaker') != owner_speaker:
                    can_merge_down = True
            if can_merge_down:
                next_scene = first_pass_merged[i + 1]
                merged_down_scene = current_scene.copy()
                merged_down_scene['end'] = max(current_scene['end'], next_scene['end'])
                merged_down_scene['content_list'].extend(next_scene['content_list'])
                merged_down_scene['content_list'].sort(key=lambda x: x['start'])
                final_scenes_unformatted.append(merged_down_scene)
                i += 2
                continue

            # 无法合并，保持独立
            final_scenes_unformatted.append(current_scene)
            i += 1

    # --- 步骤 4: 格式化输出 ---
    final_scenes = []
    for scene in final_scenes_unformatted:
        formatted_scene = {
            'scene_start': scene['start'],
            'scene_end': scene['end'],
            'content_list': scene['content_list']
        }
        final_scenes.append(formatted_scene)

    return final_scenes


@timeit_print
def get_scene_sub_text(video_path, sorted_scene_timestamp, owner_asr):
    base_name = os.path.basename(video_path).split('.')[0]
    merged_timestamps = sorted_scene_timestamp
    # 将all_sub_text_list按照end升序排序
    owner_asr.sort(key=lambda x: x.get('end', 0))
    for asr in owner_asr:
        if 'text' not in asr or not asr['text'].strip():
            asr['text'] = asr['final_text']
        asr['start'] = time_to_ms(asr.get('start', 0))
        asr['end'] = time_to_ms(asr.get('end', 0))
    scene_sub_text = process_scenes_complete_fix(owner_asr, merged_timestamps)
    output_file_scene_sub_text = f'output/{base_name}/{base_name}_scene_sub_text.json'
    save_json(output_file_scene_sub_text, scene_sub_text)
    return scene_sub_text


def extract_and_merge_owner_other(scenes, target_speaker):
    """
    scenes: list of scene dicts, each contains 'scene_start','scene_end','content_list'
    target_speaker: 要作为 owner 的说话人字符串

    返回: list，格式示例：
    [
      {
        'scene_name': 'scene_1',
        'scene_start': 0,
        'scene_end': 5433,
        'owner_text': '拼接后的目标说话人文本',
        'owner_text_start': 130 or None,
        'owner_text_end': 5490 or None,
        'other_text': '拼接后的其它说话人文本',
        'other_text_start': 20011 or None,
        'other_text_end': 25917 or None,
      },
      ...
    ]
    """
    result = []
    idx = 1

    for scene in scenes:
        contents = scene.get('content_list', [])
        # 如果场景没有内容就跳过（可根据需要改成保留空条目）
        if not contents:
            continue

        # 分出 owner 和 other
        owner_segs = [c for c in contents if c.get('speaker') == target_speaker]
        other_segs = [c for c in contents if c.get('speaker') != target_speaker]

        # 按时间排序，保证拼接顺序和时间一致
        owner_segs.sort(key=lambda x: x.get('start', 0))
        other_segs.sort(key=lambda x: x.get('start', 0))

        # 合并文本与时间
        if owner_segs:
            owner_text = ''.join(seg.get('text', '') for seg in owner_segs)
            owner_start = min(seg.get('start', 0) for seg in owner_segs)
            owner_end = max(seg.get('end', 0) for seg in owner_segs)
        else:
            owner_text = ''
            owner_start = None
            owner_end = None

        if other_segs:
            other_text = ''.join(seg.get('text', '') for seg in other_segs)
            other_start = min(seg.get('start', 0) for seg in other_segs)
            other_end = max(seg.get('end', 0) for seg in other_segs)
        else:
            other_text = ''
            other_start = None
            other_end = None

        result.append({
            'scene_number': f'{idx}',
            'scene_start': scene.get('scene_start'),
            'scene_end': scene.get('scene_end'),
            'owner_text': owner_text,
            'owner_text_start': owner_start,
            'owner_text_end': owner_end,
            'other_text': other_text,
            'other_text_start': other_start,
            'other_text_end': other_end
        })

        idx += 1

    return result


def check_new_video_script(new_video_script, scene_info):
    """
    生成的original_scene_number是否都在scene_info中
    """
    scene_numbers = {str(scene['scene_number']) for scene in scene_info}
    for detail_new_video_script in new_video_script:
        for scene in detail_new_video_script.get('场景顺序与新文案', []):
            if str(scene['original_scene_number']) not in scene_numbers:
                print(f"[ERROR] original_scene_number {scene['original_scene_number']} 不在 scene_info 中")
                return False
    return True


def gen_new_video_script_llm(scene_info, video_path):
    """
    生成新的视频方案
    """
    retry_delay = 10
    max_retries = 3
    prompt_file_path = '../../content_community/app/视频生成详细的客观场景描述.txt'
    prompt = read_file_to_str(prompt_file_path)
    full_prompt = f'{prompt}\n{scene_info}'
    raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            model_name = "gemini-2.5-flash"
            raw = get_llm_content_gemini_flash_video(prompt=full_prompt, video_path=video_path, model_name=model_name)
            new_video_script = string_to_object(raw)
            check_result = check_new_video_script(new_video_script, scene_info)
            if not check_result:
                raise ValueError("生成的视频脚本检查未通过")
            return new_video_script
        except Exception as e:
            print(f"[ERROR] 生成视频信息失败 (尝试 {attempt}/{max_retries}): {e} {raw}")
            if attempt < max_retries:
                print(f"[INFO] 正在重试... (等待 {retry_delay} 秒)")
                time.sleep(retry_delay)  # 等待一段时间后再重试
            else:
                print("[ERROR] 达到最大重试次数，失败.")
                return None  # 达到最大重试次数后返回 None
            traceback.print_exc()


@timeit_print
def gen_new_video_script(video_path, scene_sub_text, target_speaker='owner'):
    """
    生成新视频的文本脚本
    """
    base_name = os.path.basename(video_path).split('.')[0]
    scene_sub_text_list = scene_sub_text
    output_file_final = f'output/{base_name}/{base_name}_scene_format_new_script.json'
    output_file_scene_info = f'output/{base_name}/{base_name}_merge_speaker_scene_info.json'

    # if is_valid_target_file_simple(output_file_final):
    #     new_video_script = read_json(output_file_final)
    #     scene_info = read_json(output_file_scene_info)
    #     return new_video_script, scene_info

    scene_info = extract_and_merge_owner_other(scene_sub_text_list, target_speaker)
    save_json(output_file_scene_info, scene_info)

    new_video_script = gen_new_video_script_llm(scene_info, video_path=video_path)
    save_json(output_file_final, new_video_script)

    return new_video_script, scene_info


def gen_new_video_by_scene_and_script(video_path, new_video_script, scene_info):
    """
    生成新视频的文本脚本
    """
    max_diff = 500
    base_name = os.path.basename(video_path).split('.')[0]
    # 将new_video_script安装 方案整体评分 降序排序
    new_video_script.sort(key=lambda x: x.get('方案整体评分', 0), reverse=True)
    new_video_script_result = new_video_script
    final_video_script = new_video_script_result[0]

    need_merge_video_file = []
    for new_scene in final_video_script['场景顺序与新文案']:
        original_scene_number = new_scene['original_scene_number']
        # 找到scene_info中scene_number等于original_scene_number的场景
        for scene in scene_info:
            if scene['scene_number'] == original_scene_number:
                new_scene.update(scene)
    print(f'完成场景信息合并')

    new_scene_list = final_video_script['场景顺序与新文案']
    # new_scene_list = new_scene_list[:2]
    for fused_new_scene in new_scene_list:
        print(
            f'处理新场景: new_{fused_new_scene["new_scene_number"]}_origin_{fused_new_scene["scene_number"]} 进度: {new_scene_list.index(fused_new_scene) + 1}/{len(new_scene_list)}')
        scene_start = fused_new_scene.get('scene_start')
        scene_end = fused_new_scene.get('scene_end')

        new_owner_text = fused_new_scene.get('new_owner_text', '').strip()
        owner_text = fused_new_scene.get('owner_text', '').strip()
        if not new_owner_text:
            new_owner_text = owner_text

        if new_owner_text:
            def _to_int(v):
                try:
                    return int(float(v))
                except:
                    return None

            s = _to_int(fused_new_scene.get('owner_text_start')) or int(scene_start)
            e = _to_int(fused_new_scene.get('owner_text_end'))

            if e is None:
                MS_PER_CHAR = 200
                MIN_MS = 500
                est = max(MIN_MS, len(new_owner_text) * MS_PER_CHAR)
                scene_end = _to_int(fused_new_scene.get('scene_end'))
                e = min(s + est, scene_end) if scene_end is not None else s + est

            if e <= s:
                e = s + max(500, len(new_owner_text) * MS_PER_CHAR)

            owner_text_start, owner_text_end = s, e

            # 规范化时间，确保在场景时间范围内
            format_start_time = max(scene_start, owner_text_start)
            format_end_time = min(scene_end, owner_text_end)
            if abs(format_start_time - scene_start) < max_diff:
                format_start_time = scene_start
            if abs(format_end_time - scene_end) < max_diff:
                format_end_time = scene_end
            # 获取三个时间段，分别是scene_start到format_start_time，format_start_time到format_end_time，format_end_time到scene_end
            video_time_segments = []
            video_time_segments.append((scene_start, format_start_time))
            video_time_segments.append((format_start_time, format_end_time))
            video_time_segments.append((format_end_time, scene_end))
            sub_count = 0
            for video_time_segment in video_time_segments:
                sub_count += 1
                seg_start, seg_end = video_time_segment
                if seg_end > seg_start:
                    segment_output_scene_file = f'output/{base_name}/split_scene/new_{fused_new_scene["new_scene_number"]}_origin_{fused_new_scene["scene_number"]}_part{sub_count}.mp4'
                    output_path = segment_output_scene_file
                    if not is_valid_target_file_simple(segment_output_scene_file):
                        print(f'正在裁剪视频片段: {segment_output_scene_file} 时间段: {seg_start}-{seg_end}')
                        clip_video_ms(video_path, seg_start, seg_end, segment_output_scene_file)

                    if sub_count == 2:
                        output_path = segment_output_scene_file.replace('.mp4', '_with_text.mp4')
                        gen_video(new_owner_text, output_path, segment_output_scene_file)

                    need_merge_video_file.append(output_path)
        else:
            output_scene_file = f'output/{base_name}/split_scene/new_{fused_new_scene["new_scene_number"]}_origin_{fused_new_scene["scene_number"]}_part{0}.mp4'
            if not is_valid_target_file_simple(output_scene_file):
                clip_video_ms(video_path, scene_start, scene_end, output_scene_file)
            need_merge_video_file.append(output_scene_file)
    final_output_path = f'output/{base_name}/{base_name}_remake.mp4'
    merge_videos_ffmpeg(need_merge_video_file, output_path=final_output_path)
    bgm_path = r"W:\project\python_project\watermark_remove\content_community\app\bgm_audio" + os.sep + '4f7ed367245a6ba525d07f21d4790a25.wav'
    if bgm_path and os.path.exists(bgm_path):
        # print(f"正在为视频添加背景音乐: {bgm_path}")
        final_with_bgm_path = final_output_path.replace('.mp4', '_with_bgm.mp4')
        add_bgm_to_video(final_output_path, bgm_path, str(final_with_bgm_path), volume_percentage=50)
        return final_with_bgm_path
    return final_output_path


@timeit_print
def video_remake(video_path):
    fixed_speech_asr_with_sub_text = gen_asr(video_path)

    sorted_scene_timestamp = get_scene(video_path)

    scene_sub_text = get_scene_sub_text(video_path, sorted_scene_timestamp, fixed_speech_asr_with_sub_text)

    new_video_script, scene_info = gen_new_video_script(video_path, scene_sub_text)

    final_video_path = gen_new_video_by_scene_and_script(video_path, new_video_script, scene_info)
    return final_video_path


if __name__ == '__main__':
    video_remake('test2.mp4')
