# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/8/21 18:35
:last_date:
    2025/8/21 18:35
:description:
    
"""
from common_utils.common_utils import read_json, time_to_ms


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


if __name__ == '__main__':
    scenes = read_json('test.json')
    speakers = read_json('../ASR/output/final_asr.json')
    safe = find_silent_scene_timestamps(scenes, speakers, margin_ms=50)
    safe1 = create_speech_segments(scenes, speakers, margin_ms=50)
    print(safe)