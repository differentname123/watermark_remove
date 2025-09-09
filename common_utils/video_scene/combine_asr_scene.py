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

from LLM.gemini import get_llm_content
from common_utils.ASR.asr_fusion import gen_precise_asr
from common_utils.common_utils import read_json, time_to_ms, save_json, ms_to_time, read_file_to_str, string_to_object
from common_utils.image_utils import save_frames_around_timestamp
from common_utils.split_scenes import find_and_split_scenes, split_scenes_json
from common_utils.tts.edge_tts_utils import generate_audio_and_get_duration_sync
from common_utils.video_utils import extract_audio_from_video, clip_video_ms, merge_videos_ffmpeg, probe_duration, \
    add_subtitles_to_video
from common_utils.video_utils1 import redub_video_with_ffmpeg
from common_utils.video_utils2 import add_bgm_to_video


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

    fixed_speech_asr_info = fix_speech_asr(origin_sentence_info)
    save_json(output_file, fixed_speech_asr_info)
    print()


def merge_scene_timestamps(scene_dict, min_count=3, count_by_threshold=True, return_pairs=True):
    """
    合并不同阈值下的场景时间点，保留出现次数 >= min_count 的时间戳并按时间升序返回。

    参数:
      scene_dict: 嵌套字典，外层 key 为阈值（例如 40,50,60），内层为场景名 -> [start, end]
      min_count: 只保留出现次数 >= min_count 的时间戳（默认 2）
      count_by_threshold: True 时在每个阈值内先去重再计数（推荐），
                          False 时把所有出现次数都计入（同阈值重复会被计多次）
      return_pairs: True 则同时把相邻时间点两两配对成区间返回

    返回:
      如果 return_pairs=False，返回按升序的时间戳字符串列表。
      如果 return_pairs=True，返回 (time_points_list, [(start,end), ...])。
    """
    counts = Counter()

    for thr, scenes in scene_dict.items():
        # 收集该阈值下所有时间戳
        ts_list = []
        for scene_name, bounds in scenes.items():
            if not bounds:
                continue
            # 期望 bounds = [start, end]
            ts_list.extend(bounds)
        if count_by_threshold:
            for ts in set(ts_list):
                counts[ts] += 1
        else:
            for ts in ts_list:
                counts[ts] += 1

    # 过滤出出现次数 >= min_count 的时间点
    kept = [ts for ts, c in counts.items() if c >= min_count]

    # 按真实时间排序
    kept_sorted = sorted(kept, key=time_to_ms)

    # 构建 pairs_dict，键名为 '场景1','场景2',...
    pairs = {}
    n = len(kept_sorted)
    if n == 0:
        return kept_sorted, pairs
    if n == 1:
        start = kept_sorted[0]
        end = kept_sorted[0]
        td = time_to_ms(end) - time_to_ms(start)
        pairs['场景1'] = {
            'start': start,
            'end': end,
            'duration': td
        }
        return kept_sorted, pairs

    for i in range(n - 1):
        key = f"场景{i+1}"
        start = kept_sorted[i]
        end = kept_sorted[i+1]
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
    for high_threshold in [30, 40, 50, 70]:
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
    kept_sorted, pairs = merge_scene_timestamps(all_scene_info_dict, min_count=2)

    print(f"\n合并后的场景数量为: {len(pairs)}")
    for key, value in pairs.items():
        timestamp = value[1]
        save_frames_around_timestamp(my_video_path, timestamp, 3,
                                     str(os.path.join(f'scenes_fused_{basename}', key)))

    print(f"所有场景信息: {all_scene_info_dict}")


if __name__ == '__main__':
    # fun()

    get_scene()
    #
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
