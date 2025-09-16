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

from typing import List, Dict, Any, Tuple

from LLM.gemini import get_llm_content, get_llm_content_gemini_flash_video
from common_utils.ASR.asr_fusion import gen_precise_asr
from common_utils.common_utils import read_json, time_to_ms, save_json, ms_to_time, read_file_to_str, string_to_object
from common_utils.image_utils import save_frames_around_timestamp
from common_utils.split_scenes import find_and_split_scenes, split_scenes_json
from common_utils.tts.edge_tts_utils import generate_audio_and_get_duration_sync
from common_utils.video_utils import extract_audio_from_video, clip_video_ms, merge_videos_ffmpeg, probe_duration, \
    add_subtitles_to_video
from common_utils.video_utils1 import redub_video_with_ffmpeg
from common_utils.video_utils2 import add_bgm_to_video

import string
import re
from copy import deepcopy

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
def get_pinyin_for_char(char: str) -> str:
    # 示例实现
    if char == "你": return "ni"
    if char == "好": return "hao"
    if char == "世": return "shi"
    if char == "界": return "jie"
    return char  # 默认返回原字符


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

def generate_sub_sentence_timestamps(
        asr_data: list[dict],
        corrected_text_data: list[dict],
        search_window: int = 5,
        time_margin: int = 500
) -> list[dict]:
    processed_data = deepcopy(corrected_text_data)

    for segment in processed_data:
        final_text = segment.get("final_text", "")
        if not final_text:
            segment['sub_text'] = []
            continue

        relevant_asr = [
            word for word in asr_data
            if word.get('end', 0) > segment.get('start', 0) - time_margin and
               word.get('start', 0) < segment.get('end', 0) + time_margin
        ]

        if not relevant_asr:
            segment['sub_text'] = []
            continue

        sub_sentence_list = split_text_into_sentences(final_text)
        pass1_results = []

        segment_char_cursor = 0
        offset = 0.0

        for sentence_text in sub_sentence_list:
            # 现在 is_not_punctuation 完全依赖于分割逻辑
            effective_chars = [c for c in sentence_text if is_not_punctuation(c)]

            if not effective_chars:
                pass1_results.append({"text": sentence_text, "start": None, "end": None})
                continue

            sentence_start_time, sentence_end_time = None, None

            first_char_pinyin = get_pinyin_for_char(effective_chars[0])
            text_expected_pos = segment_char_cursor
            search_center_index = text_expected_pos + offset

            first_char_info, first_char_match_idx = _find_char_in_asr(
                first_char_pinyin, search_center_index, relevant_asr, window_size=search_window
            )

            if first_char_info:
                sentence_start_time = first_char_info['start']
                offset = first_char_match_idx - text_expected_pos

            if len(effective_chars) == 1:
                if first_char_info:
                    sentence_end_time = first_char_info['end']
            else:
                last_char_pinyin = get_pinyin_for_char(effective_chars[-1])
                text_expected_pos_last = segment_char_cursor + len(effective_chars) - 1
                search_center_index_last = text_expected_pos_last + offset

                last_char_info, last_char_match_idx = _find_char_in_asr(
                    last_char_pinyin, search_center_index_last, relevant_asr, window_size=search_window
                )
                if last_char_info:
                    sentence_end_time = last_char_info['end']
                    offset = last_char_match_idx - text_expected_pos_last

            pass1_results.append({
                "text": sentence_text,
                "start": sentence_start_time,
                "end": sentence_end_time
            })

            segment_char_cursor += len(effective_chars)

        final_sub_text = []
        for i, sub in enumerate(pass1_results):
            if sub['start'] is None:
                prev_end = segment.get('start', 0)
                for j in range(i - 1, -1, -1):
                    if pass1_results[j]['end'] is not None:
                        prev_end = pass1_results[j]['end'];
                        break
                sub['start'] = prev_end

            if sub['end'] is None:
                next_start = segment.get('end', 0)
                for j in range(i + 1, len(pass1_results)):
                    if pass1_results[j]['start'] is not None:
                        next_start = pass1_results[j]['start'];
                        break
                sub['end'] = next_start

            if sub['start'] > sub['end']:
                sub['end'] = sub['start']

            final_sub_text.append(sub)

        segment['sub_text'] = final_sub_text

    return processed_data


def find_silent_scene_timestamps(scenes: dict,
                                 speakers: list,
                                 margin_ms: int = 50) -> list:
    """
    找到场景时间戳中，那些在该时间点“没有人说话”的剪切点。

    参数:
      scenes: dict，场景字典，例如:
          {
            "场景1": ["00:00:00.000", "00:00:09.867"],
            "场景2": ["00:00:09.867", "00:00:23.933"],
            ...
          }
      speakers: list，说话人事件列表，每项包含 'start' (秒，float) 和 'end' (秒，float)。
          例如 [{'speaker':'SPEAKER_03','start':0.0,'end':1.15,'text':...}, ...]
      time_to_ms: 函数，接收场景中的时间字符串并返回对应的毫秒整数（用户已提供）。
      margin_ms: int，可选。安全边界（毫秒）。如果某个说话段在 timestamp 的 margin_ms 范围内，也认为“在说话”，默认 50ms。

    返回:
      list，按时间升序的安全剪切点，每项为 dict:
        {
          "time_str": 原始时间字符串,
          "time_ms": 毫秒整数
        }
      如果没有安全点，返回空列表。
    """
    # 收集所有场景时间戳（可能有重复）
    candidate_strs = []
    for scene_key, times in scenes.items():
        if not isinstance(times, (list, tuple)) or len(times) < 2:
            # 兼容只有一个时间的情况（仍然收集）
            for t in times:
                candidate_strs.append(t)
        else:
            candidate_strs.extend(times[:2])  # 只取前两个（start,end）
    # 去重并保持可比较顺序
    candidate_strs = sorted(set(candidate_strs), key=lambda s: time_to_ms(s))

    # 把说话人时间转换为 ms 列表
    speaker_intervals_ms = []
    for sp in speakers:
        try:
            s_ms = int(round(float(sp.get('start', 0.0)) * 1000))
            e_ms = int(round(float(sp.get('end', 0.0)) * 1000))
        except Exception:
            # 如果 start/end 不是数值，跳过该条
            continue
        # 保证 start <= end
        if e_ms < s_ms:
            s_ms, e_ms = e_ms, s_ms
        speaker_intervals_ms.append((s_ms, e_ms))

    safe_points = []
    for t_str in candidate_strs:
        try:
            t_ms = int(time_to_ms(t_str))
        except Exception:
            # time_to_ms 出错则跳过该候选
            continue

        # 检查是否与任一说话段冲突（考虑 margin_ms）
        conflict = False
        left = t_ms - margin_ms
        right = t_ms + margin_ms
        for s_ms, e_ms in speaker_intervals_ms:
            # 如果时间点（连同 margin）与说话段有重叠，即认为冲突
            # 也即： not (right < s_ms or left > e_ms)
            if not (right < s_ms or left > e_ms):
                conflict = True
                break

        if not conflict:
            safe_points.append({"time_str": t_str, "time_ms": t_ms})

    # 按时间升序返回
    safe_points.sort(key=lambda x: x["time_ms"])
    return safe_points


def create_speech_segments(scenes: dict,
                           speakers: list,
                           margin_ms: int = 50) -> list:
    """
    基于场景边界找到静音切点，并以此为边界生成时间段。
    完全采纳用户指定的精确归属逻辑：
    1. 生成一个从场景结束时间戳字符串(times[1])到 scene_key 的映射表。
    2. 在生成时间段时，使用其 end_time_str 作为键，直接从映射表中查找归属的 scene_key。
    (最终修正版 + 用户指定映射逻辑)
    """

    # 假设 time_to_ms 函数已定义
    def time_to_ms(time_str):
        # 这是一个示例实现，您可能需要根据您的时间格式进行调整
        parts = time_str.split(':')
        h, m, s_ms_str = parts[0], parts[1], parts[2].replace(',', '.')
        s_parts = s_ms_str.split('.')
        s = int(s_parts[0])
        ms = int(s_parts[1]) if len(s_parts) > 1 else 0
        return int((int(h) * 3600 + int(m) * 60 + s) * 1000 + ms)

    # =========================================================================
    # 步骤 1: 创建从 end_time_str 到 scene_key 的映射表 (您的逻辑)
    # =========================================================================
    end_time_to_scene_key_map = {}
    candidate_strs = set()
    for scene_key, times in scenes.items():
        if isinstance(times, (list, tuple)) and len(times) >= 2:
            start_str, end_str = times[0], times[1]
            candidate_strs.add(start_str)
            candidate_strs.add(end_str)

            # 核心映射逻辑：使用场景的结束时间字符串作为键
            # 这意味着任何以这个时间点结束的时间段，都归属于这个场景
            end_time_to_scene_key_map[end_str] = scene_key

    if not candidate_strs:
        return []

    # =========================================================================
    # (此部分保持不变) 步骤 2: 找到所有“安全切点”并构建最终边界点
    # =========================================================================
    sorted_candidate_strs = sorted(list(candidate_strs), key=lambda s: time_to_ms(s))

    speaker_intervals_for_conflict_check = []
    for sp in speakers:
        try:
            s_ms = int(round(float(sp.get('start', 0.0)) * 1000))
            e_ms = int(round(float(sp.get('end', 0.0)) * 1000))
            speaker_intervals_for_conflict_check.append({'start': s_ms, 'end': e_ms})
        except (ValueError, TypeError):
            continue

    safe_points = []
    for t_str in sorted_candidate_strs:
        try:
            t_ms = time_to_ms(t_str)
        except Exception:
            continue
        conflict = False
        left = t_ms - margin_ms
        right = t_ms + margin_ms
        for interval in speaker_intervals_for_conflict_check:
            if not (right < interval['start'] or left > interval['end']):
                conflict = True
                break
        if not conflict:
            safe_points.append({"time_str": t_str, "time_ms": t_ms})

    if not sorted_candidate_strs:
        return []
    timeline_start_point = {"time_str": sorted_candidate_strs[0], "time_ms": time_to_ms(sorted_candidate_strs[0])}
    timeline_end_point = {"time_str": sorted_candidate_strs[-1], "time_ms": time_to_ms(sorted_candidate_strs[-1])}

    boundary_points = [timeline_start_point]
    for sp in safe_points:
        if timeline_start_point['time_ms'] < sp['time_ms'] < timeline_end_point['time_ms']:
            boundary_points.append(sp)
    boundary_points.append(timeline_end_point)

    unique_points = {p['time_ms']: p for p in boundary_points}
    boundary_points = sorted(unique_points.values(), key=lambda x: x['time_ms'])

    if len(boundary_points) < 2:
        return []

    # =========================================================================
    # 步骤 3: 生成时间段，并使用映射表直接查找 scene_key (您的逻辑)
    # =========================================================================
    segments = []
    for i in range(len(boundary_points) - 1):
        start_point = boundary_points[i]
        end_point = boundary_points[i + 1]

        segment_start_ms = start_point["time_ms"]
        segment_end_ms = end_point["time_ms"]

        if segment_start_ms >= segment_end_ms:
            continue

        # --- MODIFICATION START: 实施您指定的直接查找逻辑 ---
        # 使用 segment 的 end_time_str 作为 key 来查找 scene_key
        end_time_key = end_point["time_str"]
        current_scene_key = end_time_to_scene_key_map.get(end_time_key, None)
        # --- MODIFICATION END ---

        speakers_in_segment = []
        for speaker_element in speakers:
            try:
                s_ms = int(round(float(speaker_element.get('start', 0.0)) * 1000))
                e_ms = int(round(float(speaker_element.get('end', 0.0)) * 1000))
            except (ValueError, TypeError):
                continue
            if not (e_ms <= segment_start_ms or s_ms >= segment_end_ms):
                speakers_in_segment.append(speaker_element)

        speakers_in_segment.sort(key=lambda x: x.get('start', 0.0))

        segments.append({
            "scene_key": current_scene_key,
            "start_time_str": start_point["time_str"],
            "end_time_str": end_point["time_str"],
            "start_time_ms": segment_start_ms,
            "end_time_ms": segment_end_ms,
            "speakers": speakers_in_segment
        })

    return segments


def find_asr_indices_at_boundaries_old(scenes: dict, asr_results: list, window_ms: int = 50) -> dict:
    """
    为场景边界时间戳查找在指定时间窗口内的ASR结果索引。
    ASR结果是一个列表的列表 (list[list[dict]])。

    Args:
        scenes (dict): 场景信息字典，键为场景名，值为[开始时间戳, 结束时间戳]。
        asr_results (list[list[dict]]): ASR识别结果，这是一个嵌套列表。
        window_ms (int): 在时间戳周围搜索的时间窗口半径（毫秒）。

    Returns:
        dict: 一个字典，键是场景边界的时间戳字符串，
              值是对应的ASR结果索引元组 (子列表索引, 词语索引) 的列表。
    """
    boundary_asr_indices = collections.defaultdict(list)

    # 1. 提取所有唯一的边界时间戳
    unique_timestamps = set()
    for start_time, end_time in scenes.values():
        unique_timestamps.add(start_time)
        unique_timestamps.add(end_time)
    print(f'unique_timestamps{len(unique_timestamps)}')
    # 2. 遍历每一个唯一的时间戳
    for ts_str in unique_timestamps:
        boundary_asr_indices[ts_str] = []
        target_ms = time_to_ms(ts_str)
        target_ms = target_ms - 1000 / 60
        window_start_ms = target_ms - window_ms
        window_end_ms = target_ms + window_ms

        # 3. 遍历ASR结果的嵌套列表
        # asr_segment_index 是外层列表的索引
        # asr_segment 是内层列表（即一个完整的ASR识别结果）
        for asr_segment_index, asr_segment in enumerate(asr_results):
            # word_index 是内层列表的索引
            # asr_item 是单个词的字典
            for word_index, asr_item in enumerate(asr_segment):
                asr_start_ms = asr_item['start']
                asr_end_ms = asr_item['end']

                # 4. 检查时间区间是否重叠
                if asr_start_ms <= window_end_ms and window_start_ms <= asr_end_ms:
                    # 记录复合索引 (外层列表索引, 内层词语索引)
                    compound_index = (asr_segment_index, word_index)
                    asr_item['compound_index'] = compound_index
                    boundary_asr_indices[ts_str].append(asr_item)

    return dict(boundary_asr_indices)


def find_asr_at_boundaries_sorted_by_overlap(scenes: dict, asr_results: list, window_ms: int = 10) -> list:
    """
    为场景边界时间戳查找重叠的ASR结果，并按重叠时间总和升序排序。

    新增功能:
    1. 计算每个时间戳下，所有匹配词的重叠时间总和 (total_overlap_ms)。
    2. 最终返回一个列表，该列表根据 'total_overlap_ms' 升序排序。

    Args:
        scenes (dict): 场景信息字典。
        asr_results (list[list[dict]]): ASR识别结果。
        window_ms (int): 搜索窗口半径（毫秒）。

    Returns:
        list: 一个已排序的列表，每个元素是一个元组 `(timestamp, result_info)`。
              `result_info` 是一个字典，包含:
              - 'found_words': 找到的ASR词语列表 (每个词都包含 'overlap_ms')。
              - 'total_overlap_ms': 重叠时间的总和。
    """
    # 1. 提取所有唯一的边界时间戳
    unique_timestamps = set()
    for start_time, end_time in scenes.values():
        unique_timestamps.add(start_time)
        unique_timestamps.add(end_time)

    # 临时存储结果，键是时间戳，值是包含词列表和总和的字典
    temp_results = {}

    # 2. 遍历时间戳，计算每个时间戳的匹配结果和重叠时间
    for ts_str in unique_timestamps:
        target_ms_original = time_to_ms(ts_str)
        target_ms = target_ms_original - 1000 / 60
        window_start_ms = target_ms - window_ms
        window_end_ms = target_ms

        found_items = []
        total_overlap = 0

        for asr_segment_index, asr_segment in enumerate(asr_results):
            for word_index, asr_item in enumerate(asr_segment):
                asr_start_ms = asr_item['start']
                asr_end_ms = asr_item['end']

                if asr_start_ms <= window_end_ms and window_start_ms <= asr_end_ms:
                    overlap_start = max(asr_start_ms, window_start_ms)
                    overlap_end = min(asr_end_ms, window_end_ms)
                    overlap_duration = overlap_end - overlap_start

                    # 只有当重叠时间 > 0 时才计算在内
                    if overlap_duration > 0:
                        result_item = asr_item.copy()
                        result_item['compound_index'] = (asr_segment_index, word_index)
                        result_item['overlap_ms'] = round(overlap_duration)

                        found_items.append(result_item)
                        total_overlap += result_item['overlap_ms']

        # 如果找到了匹配的词，才记录结果
        if found_items:
            temp_results[ts_str] = {
                'found_words': found_items,
                'total_overlap_ms': total_overlap
            }
        else:
            # 如果没有找到匹配的词，也记录一个空的结果
            temp_results[ts_str] = {
                'found_words': [],
                'total_overlap_ms': 0
            }

    # 3. 新增功能：将字典转换为列表，并根据 'total_overlap_ms' 排序
    # dict.items() 会得到 [(key1, value1), (key2, value2), ...]
    sorted_result_list = sorted(
        temp_results.items(),
        key=lambda item: item[1]['total_overlap_ms']  # item[1]是值(dict)，我们根据这个dict里的'total_overlap_ms'排序
    )

    return sorted_result_list


def get_detail_seg(video_path):
    """
    获取最详细的分割点
    """
    new_audio_file = video_path.replace('.mp4', '.wav')
    extract_audio_from_video(video_path, new_audio_file)
    OUTPUT_FILE = f'output/{new_audio_file.split('.')[0]}_final_asr.json'

    output_file, ASR_FILES = gen_precise_asr(new_audio_file, OUTPUT_FILE)
    scene_info = new_audio_file.replace('.mp4', '.json').replace('.wav', '.json')

    scenes = read_json(scene_info)
    asr_list = []
    for ASR_FILE in ASR_FILES:
        asr_list.append(read_json(ASR_FILE))

    result = find_asr_at_boundaries_sorted_by_overlap(scenes, asr_list)
    print(result)
    return result


def get_scene_word(scene_info, asr_list, need_detail=False):
    # 按场景开始时间排序并准备容器
    scenes = sorted(
        [(name, time_to_ms(t[0]), time_to_ms(t[1])) for name, t in scene_info.items()],
        key=lambda x: x[1]
    )
    if not scenes:
        return {}

    res_map = {name: [] for name, _, _ in scenes}

    for w in asr_list:
        ws, we = int(w['start']), int(w['end'])
        placed = False

        for i, (name, s_start, s_end) in enumerate(scenes):
            # 完全在场景内
            if ws >= s_start and we <= s_end:
                res_map[name].append(w);
                placed = True;
                break
            # 跨过当前场景结束 -> 放到下一个场景（无下一个则放当前）
            if ws < s_end and we > s_end:
                next_name = scenes[i + 1][0] if i + 1 < len(scenes) else name
                res_map[next_name].append(w);
                placed = True;
                break
            # 从前一场景延伸进来，end 落在当前场景内
            if ws < s_start and we > s_start and we <= s_end:
                res_map[name].append(w);
                placed = True;
                break

        # 回退策略：放到首或尾场景
        if not placed:
            if we <= scenes[0][1]:
                res_map[scenes[0][0]].append(w)
            else:
                res_map[scenes[-1][0]].append(w)

    # 构造输出，增加 last_end_ms / last_end_time
    out = {}
    for name, s_start, s_end in scenes:
        words = res_map[name]
        text = "".join(x['word'] for x in words)
        last_end_ms = max((int(x['end']) for x in words), default=None)
        out[name] = {
            "times": (s_start if callable(globals().get('ms_to_time')) else s_start,
                      s_end if callable(globals().get('ms_to_time')) else s_end),
            "text": text,
            "last_end_ms": last_end_ms}
        if need_detail:
            out[name]["words"] = words

    return out


def reorganize_scene_asr(scene_map):
    """
    将类似
      {'场景1': [ { 'times': (0,4683), 'text': '...' }, ... ], ...}
    的数据重组织为
      {'场景1': {'start_time': 0, 'end_time': 4683, 'end_t': 4683, 'asr_list': [...]}, ...}
    """
    out = {}
    for scene, entries in scene_map.items():
        starts = []
        ends = []
        texts = []

        for e in entries or []:
            # times 可能是 tuple/list，也可能是字符串或缺失，尽量转换为 int
            times = e.get('times') if isinstance(e, dict) else None
            if times and len(times) >= 2:
                try:
                    starts.append(int(times[0]))
                    ends.append(int(times[1]))
                except Exception:
                    pass

            # 文本字段可能叫 'text'，也可能别的名字，优先取 text
            txt = None
            if isinstance(e, dict):
                txt = e.get('text') or e.get('asr') or e.get('sentence') or e.get('word')
            if txt is None:
                txt = ""
            texts.append(txt)

        start_time = min(starts) if starts else None
        end_time = max(ends) if ends else None

        out[scene] = {
            "start_time": start_time,
            "end_time": end_time,
            "asr_list": texts
        }

    return out


def split_video():
    video_path = 'test1.mp4'
    scene_info = video_path.replace('.mp4', '.json')
    scenes = read_json(scene_info)
    count = 0
    for key, value in scenes.items():
        count += 1
        start_time = value[0]
        end_time = value[1]
        clip_video_ms(video_path, start_time, end_time, f'scenes/{count}.mp4')


def gen_new_video():
    # 读取test1_scene_format_new_script.json
    script_info = read_json('test1_scene_format_new_script.json')
    acts = script_info['new_video_script']['acts']
    audio_video_path_list = []
    for act in acts:
        scenes = act['scenes']
        for scene in scenes:
            original_scene_ref = scene['original_scene_ref']
            original_video_path = f'scenes/{original_scene_ref}.mp4'
            text = scene['new_script_text']
            output_path = f'scenes/{original_scene_ref}_new.mp4'
            audio_video_path = output_path.replace('.mp4', '_av.mp4')
            # if os.path.exists(audio_video_path):
            #     audio_video_path_list.append(audio_video_path)
            #     continue
            video_duration = probe_duration(original_video_path)
            audio_path = output_path.replace('.mp4', '.wav')
            duration = generate_audio_and_get_duration_sync(
                text=text,
                output_filename=str(audio_path),
                voice_name="zh-CN-YunjianNeural",
                trim_silence=False,
                # rate="+15%",
                # pitch='+10Hz',
            )

            segments_info = [{
                'startTime': "00:00:00.000",
                'endTime': ms_to_time(video_duration * 1000),
                'outputPath': str(audio_path),
                'trimmedDuration': duration,
            }]
            redub_video_with_ffmpeg(original_video_path, segments_info, output_path=str(audio_video_path))

            subtitle_data = [{
                'startTime': "00:00:00.000",
                'endTime': ms_to_time(duration * 1000),
                'optimizedText': text
            }]
            subtitle_video_path = output_path.replace('.mp4', '_sub.mp4')
            add_subtitles_to_video(
                video_path=str(audio_video_path),
                subtitles_info=subtitle_data,
                output_path=str(subtitle_video_path),
                font_size=70,
                bottom_margin=30
            )

            audio_video_path_list.append(subtitle_video_path)

    merge_videos_ffmpeg(audio_video_path_list, output_path=f'scenes/final_video.mp4')
    bgm_path = r"W:\project\python_project\watermark_remove\content_community\app\bgm_audio" + os.sep + '4f7ed367245a6ba525d07f21d4790a25.wav'
    if bgm_path and os.path.exists(bgm_path):
        # print(f"正在为视频添加背景音乐: {bgm_path}")
        final_with_bgm_path = 'scenes/final_video_with_bgm.mp4'
        add_bgm_to_video(f'scenes/final_video.mp4', bgm_path, str(final_with_bgm_path))


def reorganize_scene_asr_fun():
    video_path = 'test1.mp4'
    scene_info = video_path.replace('.mp4', '.json')
    scenes = read_json(scene_info)
    ASR_FILES, ASR_FILES = gen_precise_asr(video_path, '')
    result_dict = {}
    for ASR_FILE in ASR_FILES:
        asr_list = read_json(ASR_FILE)
        temp = get_scene_word(scenes, asr_list)
        for key, value in temp.items():
            value['ASR_FILE'] = ASR_FILE
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(value)
    print(result_dict)

    sentence_info = reorganize_scene_asr(result_dict)
    print(sentence_info)


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


def fix_speech_asr(speech_asr_info):
    """
    纠正每个说话人的文本
    """
    retry_delay = 10
    max_retries = 3
    prompt_file_path = '../../content_community/app/视频分解素材_纠正说话人文本.txt'
    prompt = read_file_to_str(prompt_file_path)
    full_prompt = f'{prompt}\n{speech_asr_info}'
    raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            raw = get_llm_content(prompt=full_prompt, model_name="gemini-2.5-flash")
            fix_speech_asr_info = string_to_object(raw)
            # 检测fix_speech_asr_info和speech_asr_info长度是否一致
            if len(fix_speech_asr_info) != len(speech_asr_info):
                raise ValueError(
                    f"返回的数据长度与输入数据长度不一致: {len(fix_speech_asr_info)} != {len(speech_asr_info)}")
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


def reorganize_speech_asr_fun():
    video_path = 'test2.mp4'
    speech_info_path = "output/segments_speech.json"
    speech_info = read_json(speech_info_path)
    ASR_FILES, ASR_FILES = gen_precise_asr(video_path, '')
    temp_list = []
    for ASR_FILE in ASR_FILES:
        asr_list = read_json(ASR_FILE)
        temp = fill_speaker_texts(asr_list, speech_info)
        for value in temp:
            value['ASR_FILE'] = os.path.basename(ASR_FILE)
        temp_list.append(temp)
    sentence_info = merge_by_key(temp_list)
    origin_sentence_info = sentence_info.copy()
    # 遍历sentence_info，删除start，end,speaker这三个字段，替换为一个自增的id
    for i, entry in enumerate(sentence_info):
        entry['id'] = i + 1
        entry.pop('start', None)
        entry.pop('end', None)
        entry.pop('speaker', None)
    fix_speech_asr_info = fix_speech_asr(origin_sentence_info)
    save_json('output/final_speech_asr.json', fix_speech_asr_info)
    print(fix_speech_asr_info)


def fun():
    video_path = 'test2.mp4'
    base_name = os.path.basename(video_path).split('.')[0]
    output_file = f'output/{base_name}/{base_name}_fixed_speech_asr.json'
    result_file_info = gen_precise_asr(video_path, output_file)

    asr_file_list = result_file_info['asr_file']
    speaker_file = result_file_info['speaker_file']

    speech_info = read_json(speaker_file)
    temp_list = []
    for asr_file in asr_file_list:
        asr_info = read_json(asr_file)
        break
        temp = fill_speaker_texts(asr_info, speech_info)
        for value in temp:
            value['asr_file'] = os.path.basename(asr_file)

        temp_list.append(temp)
    sentence_info = merge_by_key(temp_list)
    origin_sentence_info = copy.deepcopy(sentence_info)
    # 遍历sentence_info，删除start，end,speaker这三个字段，替换为一个自增的id
    for i, entry in enumerate(sentence_info):
        entry['id'] = i + 1
        entry.pop('start', None)
        entry.pop('end', None)
        entry.pop('speaker', None)

    # fixed_speech_asr_info = fix_speech_asr(origin_sentence_info)
    # save_json(output_file, fixed_speech_asr_info)
    fixed_speech_asr_info = read_json(output_file)
    # fixed_speech_asr_info = fixed_speech_asr_info[:1]
    result = generate_sub_sentence_timestamps(asr_info, fixed_speech_asr_info)
    save_json(output_file, result)


    print()


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
        key = f"场景{i+1}"
        start = filtered_sorted[i]
        end = filtered_sorted[i+1]
        td = time_to_ms(end) - time_to_ms(start)
        pairs[key] = {
            'start': start,
            'end': end,
            'duration': td
        }

    return kept_sorted, pairs



def get_scene():
    my_video_path = 'test2.mp4'
    basename = os.path.basename(my_video_path).split('.')[0]

    all_scene_info_dict = {}
    for high_threshold in [30, 40, 50, 60, 70]:
        scene_info_file = f'scenes_{basename}_{high_threshold}/scene_info.json'
        if os.path.exists(scene_info_file):
            print(f"场景信息文件已存在，跳过处理: {scene_info_file}")
            all_scene_info_dict[high_threshold] = read_json(scene_info_file)
            continue

        # 运行带有精炼功能的场景分割
        scene_info_dict = split_scenes_json(
            my_video_path,
            high_threshold=high_threshold,  # 初始高阈值
            min_scene_len=25,  # 最小场景长度（帧）
        )
        print("\n场景信息字典已生成并打印。")
        for key, value in scene_info_dict.items():
            timestamp = value[1]
            save_frames_around_timestamp(my_video_path, timestamp, 3, str(os.path.join(f'scenes_{basename}_{high_threshold}', key)))

        save_json(scene_info_file, scene_info_dict)
        all_scene_info_dict[high_threshold] = scene_info_dict
    kept_sorted, pairs = merge_scene_timestamps(all_scene_info_dict, min_count=3)

    print(f"\n合并后的场景数量为: {len(pairs)}")
    # 将kept_sorted保存到文件
    save_json(f'scenes_fused_{basename}/merged_timestamps.json', kept_sorted)

    for key, value in pairs.items():
        timestamp = value[1]
        save_frames_around_timestamp(my_video_path, timestamp, 3,
                                     str(os.path.join(f'scenes_fused_{basename}', key)))

    print(f"所有场景信息: {all_scene_info_dict}")


def process_scenes(text_results, scene_splits, inclusion_threshold=0.9, merge_threshold=0.2):
    """
    将文本识别结果根据场景切分进行组织，并根据重叠率合并场景。(最终版)

    Args:
        text_results (list): 文本识别结果列表。
        scene_splits (list): 最小场景切分列表。
        inclusion_threshold (float): 判断句子是否属于场景的重叠率阈值。
        merge_threshold (float): 触发场景合并的最低重叠率阈值。

    Returns:
        list: 整理后的场景列表。
    """
    if not text_results or not scene_splits:
        return []

    # 1. 预处理场景切分列表
    scenes = []
    for i in range(len(scene_splits) - 1):
        start_time = scene_splits[i][0]
        end_time = scene_splits[i + 1][0]
        scenes.append({'start': start_time, 'end': end_time})

    last_scene_start = scene_splits[-1][0]
    max_text_end = max(item['end'] for item in text_results) if text_results else last_scene_start
    scenes.append({'start': last_scene_start, 'end': max(last_scene_start, max_text_end)})

    # 2. 初始化指针和最终结果列表
    final_scenes = []
    text_cursor = 0
    scene_cursor = 0

    # 3. 遍历场景
    while scene_cursor < len(scenes):
        current_merged_scene = {
            'scene_start': scenes[scene_cursor]['start'],
            'scene_end': scenes[scene_cursor]['end'],
            'content_list': []
        }

        merged_scene_count = 1

        # 4. 为当前场景分配文本
        while text_cursor < len(text_results):
            sentence = text_results[text_cursor]

            # 优化：如果句子完全在新场景的后面，当前场景肯定处理结束
            if sentence['start'] >= current_merged_scene['scene_end']:
                # 检查句子和当前场景是否有任何重叠
                # 如果完全没有重叠，且场景已有内容，则结束
                if current_merged_scene['content_list']:
                    break

            sentence_duration = sentence['end'] - sentence['start']

            overlap_start = max(sentence['start'], current_merged_scene['scene_start'])
            overlap_end = min(sentence['end'], current_merged_scene['scene_end'])
            overlap_duration = max(0, overlap_end - overlap_start)

            overlap_ratio = 0.0
            if sentence_duration > 0:
                overlap_ratio = overlap_duration / sentence_duration
            elif current_merged_scene['scene_start'] <= sentence['start'] < current_merged_scene['scene_end']:
                overlap_ratio = 1.0

            # 5. 根据重叠率进行三段式判断
            if overlap_ratio >= inclusion_threshold:
                # 场景1: 高度重叠，直接包含
                current_merged_scene['content_list'].append(sentence)
                text_cursor += 1
            elif overlap_ratio > merge_threshold:
                # 场景2: 中度重叠，不满足包含条件但满足合并条件
                # 尝试合并下一个场景来容纳它
                next_scene_index = scene_cursor + merged_scene_count
                if next_scene_index < len(scenes):
                    current_merged_scene['scene_end'] = scenes[next_scene_index]['end']
                    merged_scene_count += 1
                    # 保持 text_cursor 不动，用新场景重新判断当前句子
                else:
                    # 没有更多场景可合并，强制加入
                    current_merged_scene['content_list'].append(sentence)
                    text_cursor += 1
            else:  # overlap_ratio <= merge_threshold
                # 场景3: 低度或无重叠，认为句子与当前场景组无关
                # 结束当前场景的构建，让这个句子在下一个新场景中被处理
                # 但如果当前场景是空的，我们必须处理这个句子，所以需要合并
                if not current_merged_scene['content_list']:
                    # 强制合并以处理第一个句子
                    next_scene_index = scene_cursor + merged_scene_count
                    if next_scene_index < len(scenes):
                        current_merged_scene['scene_end'] = scenes[next_scene_index]['end']
                        merged_scene_count += 1
                    else:
                        current_merged_scene['content_list'].append(sentence)
                        text_cursor += 1
                else:
                    # 场景已有内容，则结束
                    break

        if current_merged_scene['content_list']:
            final_scenes.append(current_merged_scene)

        scene_cursor += merged_scene_count

    return final_scenes


def get_scene_sub_text():
    my_video_path = 'test2.mp4'
    base_name = os.path.basename(my_video_path).split('.')[0]
    merged_timestamps_file = f'scenes_fused_{base_name}/merged_timestamps.json'
    merged_timestamps = read_json(merged_timestamps_file)
    output_file = f'output/{base_name}/{base_name}_fixed_speech_asr.json'
    fixed_speech_asr = read_json(output_file)
    all_sub_text_list = []
    for speech_info in fixed_speech_asr:
        sub_text_list = speech_info.get('sub_text', '')
        speaker = speech_info.get('speaker', '')
        for sub_text in sub_text_list:
            sub_text['speaker'] = speaker
            all_sub_text_list.append(sub_text)
    mapped = process_scenes(all_sub_text_list, merged_timestamps)
    output_file_scene_sub_text = f'output/{base_name}/{base_name}_scene_sub_text.json'
    save_json(output_file_scene_sub_text, mapped)
    print(mapped)


def extract_and_merge_by_speaker(scenes, target_speaker):
    """
    scenes: list of scenes, each scene 是 dict，包含 'scene_start','scene_end','content_list'
    target_speaker: 要筛选的 speaker 字符串

    返回: list，格式示例：
    [
      {
        'scene_name': 'scene_1',
        'scene_start': 0,
        'scene_end': 5433,
        'text': '拼接后的文本...',
        'text_start': 130,
        'text_end': 5490
      },
      ...
    ]
    """
    result = []
    out_idx = 1

    for scene in scenes:
        contents = scene.get('content_list', [])
        # 筛选出目标说话人的片段
        target_segments = [c for c in contents if c.get('speaker') == target_speaker]

        if not target_segments:
            continue

        # 按 start 排序（以保证文本顺序与时间顺序一致）
        target_segments.sort(key=lambda x: x.get('start', 0))

        # 拼接文本（中文常见直接拼接，保留原始标点）
        merged_text = ''.join([seg.get('text', '') for seg in target_segments])

        text_start = min(seg.get('start', 0) for seg in target_segments)
        text_end = max(seg.get('end', 0) for seg in target_segments)

        result.append({
            'scene_name': f'scene_{out_idx}',
            'scene_start': scene.get('scene_start'),
            'scene_end': scene.get('scene_end'),
            'text': merged_text,
            'text_start': text_start,
            'text_end': text_end
        })

        out_idx += 1

    return result

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

def gen_new_video_script_llm(scene_info, video_path):
    """
    生成新的视频方案
    """
    retry_delay = 10
    max_retries = 3
    prompt_file_path = '../../content_community/app/视频场景生成新视频.txt'
    prompt = read_file_to_str(prompt_file_path)
    full_prompt = f'{prompt}\n{scene_info}'
    raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            model_name = "gemini-2.5-flash"
            raw = get_llm_content_gemini_flash_video(prompt=full_prompt,video_path=video_path,model_name=model_name)
            new_video_script = string_to_object(raw)
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

def gen_new_video_script():
    """
    生成新视频的文本脚本
    """
    my_video_path = 'test2.mp4'
    base_name = os.path.basename(my_video_path).split('.')[0]
    output_file = f'output/{base_name}/{base_name}_scene_sub_text.json'
    scene_sub_text_list = read_json(output_file)
    target_speaker = 'SPEAKER_00'
    scene_info = extract_and_merge_owner_other(scene_sub_text_list, target_speaker)
    output_file_scene_info = f'output/{base_name}/{base_name}_new_video_script.json'
    save_json(output_file_scene_info, scene_info)

    result = gen_new_video_script_llm(scene_info, video_path=my_video_path)
    print(result)

if __name__ == '__main__':
    # fun()

    # get_scene()

    # get_scene_sub_text()
    #

    gen_new_video_script()
    #
    # reorganize_speech_asr_fun()
    #
    # # gen_new_video()
    # # split_video()
    # #
    # # fun()
    # #
    # # video_path = 'test1.mp4'
    # # get_detail_seg(video_path)
    # # asr_and_scene('test2.mp4')
    #
