# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/8/21 18:35
:last_date:
    2025/8/21 18:35
:description:

"""

import os
import random
import time
import traceback
import logging  # 1. 引入 logging 模块
from LLM.gemini import get_llm_content, get_llm_content_gemini_flash_video
from common_utils.common_utils import read_json, time_to_ms, save_json, ms_to_time, read_file_to_str, string_to_object, \
    timeit_print, is_valid_target_file_simple
from common_utils.ocr.paddle_ocr_utils import find_overall_subtitle_box_target_number
from common_utils.split_audio import separate_with_cli
from common_utils.split_scenes import split_scenes_json
from common_utils.video_utils import extract_audio_from_video, clip_video_ms, merge_videos_ffmpeg, probe_duration
from common_utils.video_utils2 import add_bgm_to_video

import re

from common_utils.video_utils_cut import gen_video
from content_community.app.remake_video import adjust_subtitle_box, cover_subtitle

base_output_dir = "W:/project/python_project/watermark_remove/content_community/bilibili/output"


# 2. 新增 setup_logger 函数，这是日志系统的核心
def setup_logger(log_file):
    """
    配置并返回一个日志记录器，支持并行调用。
    """
    # 使用日志文件的绝对路径作为记录器的唯一名称，防止冲突
    logger_name = os.path.abspath(log_file)
    logger = logging.getLogger(logger_name)

    # 如果这个logger已经配置过了，直接返回，防止重复添加handler
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    # 创建一个handler，用于写入日志文件
    # 确保日志文件目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 创建一个handler，用于输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def check_owner_asr(owner_asr_info, video_duration, logger):
    """
        检查生成的asr文本是否正确，第一是验证每个时间是否合理（1.最长跨度不能够超过20s 2.时长的合理性（也就是最快和最慢的语速就能够知道文本对应的时长是否合理） 3.owner语音和本地speaker说话人日志的差异不能够太大）

    :param owner_asr_info: 包含 ASR 信息的字典列表
    :return: 错误信息列表，若没有错误则返回空列表
    """
    max_time = 0
    for i in range(len(owner_asr_info)):
        start_time = time_to_ms(owner_asr_info[i]["start"])
        end_time = time_to_ms(owner_asr_info[i]["end"])
        max_time = max(max_time, end_time)
        duration = end_time - start_time

        # 1. 最大跨度不能超过 20s
        if len(owner_asr_info[i]['final_text']) > 200 and owner_asr_info[i]['speaker'] == 'owner':
            logger.error(
                f"[ERROR] 片段 {i} 跨度过长: {duration} ms 文案长度: {len(owner_asr_info[i]['final_text'])} 文案为:{owner_asr_info[i]['final_text']}")
            return False

        # # 2. 检查时长合理性：使用最快和最慢语速来估算时长范围
        # word_count = len(owner_asr_info[i]["final_text"].strip())
        # min_duration = (word_count / 1000) * 60 * 1000  # 最快语速 (150词/分钟)
        # max_duration = (word_count / 50) * 60 * 1000  # 最慢语速 (50词/分钟)
        #
        # if not (min_duration <= duration <= max_duration):
        #     logger.error(f"[ERROR] 片段 {i} 时长不合理: {duration} ms, 预计范围: [{min_duration} ms, {max_duration} ms] 文案为:{owner_asr_info[i]["final_text"]}")
        #     return False
    if max_time > video_duration + 1000:
        logger.error(f"[ERROR] 最大结束时间 {max_time} ms 超过视频总时长 {video_duration} ms")
        return False

    return True


def gen_owner_asr_by_llm(video_path, has_author_voice):
    """
    通过大模型生成带说话人识别的ASR文本。
    （已重构，提升可读性和健壮性）
    """
    basename = os.path.basename(video_path).split('.mp4')[0]
    output_dir = os.path.join(base_output_dir, basename)
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

    # --- 1. 配置常量 ---
    MAX_RETRIES = 3
    RETRY_DELAY = 10  # 秒
    PROMPT_FILE_PATH = '../../content_community/app/视频分解素材_直接进行asr转录与owner识别严格.txt'
    MODEL_NAME = "gemini-2.5-pro"

    # --- 2. 初始化和预处理 ---
    try:
        video_duration = probe_duration(video_path)
        video_duration_ms = int(video_duration * 1000)
    except Exception as e:
        logger.error(f"获取视频时长失败: {e}")
        return None

    # --- 3. 前置条件判断 (Guard Clause) ---
    # 如果视频中没有作者声音，直接返回一个覆盖全时长的默认结构
    if not has_author_voice:
        logger.info("视频无作者声音，返回默认ASR结构。")
        return [
            {
                "start": "00:00.000",
                "end": ms_to_time(video_duration_ms),
                "speaker": "other",
                "final_text": ""
            }
        ]

    # --- 4. 准备Prompt ---
    try:
        prompt = read_file_to_str(PROMPT_FILE_PATH)
    except Exception as e:
        logger.error(f"读取Prompt文件失败: {PROMPT_FILE_PATH}, 错误: {e}")
        return None

    # --- 5. 带重试机制的核心逻辑 ---
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"尝试生成ASR信息... (第 {attempt}/{MAX_RETRIES} 次)")
        raw_response = ""
        try:
            # 调用大模型API
            raw_response = get_llm_content_gemini_flash_video(
                prompt=prompt,
                video_path=video_path,
                model_name=MODEL_NAME
            )

            # 解析和校验
            owner_asr_info = string_to_object(raw_response)

            if not check_owner_asr(owner_asr_info, video_duration_ms, logger):
                logger.warning(f"生成的ASR文本校验失败 (尝试 {attempt}/{MAX_RETRIES})")
                # 校验失败，继续下一次重试
                continue

            # 处理LLM返回空列表的情况
            if not owner_asr_info:
                logger.warning("大模型返回为空列表，使用默认值。")
                return [
                    {
                        "start": "00:00.000",
                        "end": ms_to_time(video_duration_ms),
                        "speaker": "other",
                        "final_text": ""
                    }
                ]

            # 成功获取并校验通过，直接返回结果
            logger.info("成功生成ASR信息。")
            return owner_asr_info

        except Exception as e:
            logger.error(f"生成或处理ASR时发生异常 (尝试 {attempt}/{MAX_RETRIES}): {e}")
            logger.error(f"       原始响应内容 (raw_response): {raw_response}")
            logger.exception("详细堆栈信息：")  # logger.exception 会自动记录堆栈信息

        # 如果当前尝试失败且不是最后一次，则等待后重试
        if attempt < MAX_RETRIES:
            logger.info(f"将在 {RETRY_DELAY} 秒后重试...")
            time.sleep(RETRY_DELAY)

    # --- 6. 所有重试均告失败 ---
    logger.error("已达到最大重试次数，无法生成ASR信息。")
    return None


def gen_audio_path(input_path: str, split_vocal=True) -> str:
    """
    生成处理好的音频文件路径。
    """
    basename = os.path.basename(input_path).split('.')[0]
    output_dir = os.path.join(base_output_dir, basename)
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

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
        split_audio_path = os.path.join(base_dir, "htdemucs", os.path.basename(input_audio_path).rsplit('.', 1)[0],
                                        "vocals.wav")
        if not is_valid_target_file_simple(split_audio_path):
            separate_with_cli(input_audio_path, base_dir)
        if is_valid_target_file_simple(split_audio_path):
            processed_audio_path = split_audio_path
    logger.info(f"处理后的音频文件路径: {processed_audio_path}")
    return processed_audio_path


@timeit_print
def gen_asr(video_path, output_dir, has_author_voice):
    """
    生成修复后的asr以及句子时间段
    """
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

    start_time = time.time()
    speech_asr_output_file = os.path.join(output_dir, 'speech_asr_with_owner.json')

    if not is_valid_target_file_simple(speech_asr_output_file, min_size_bytes=10):
        owner_asr_info = gen_owner_asr_by_llm(video_path, has_author_voice)
        # 判断owner_asr_info是否为dict
        if owner_asr_info is None:
            logger.error("生成asr文本失败，返回空结果")
            raise ValueError("生成asr文本失败，返回空结果")
        # 为owner_asr_info增加一个字段叫做source_clip_id，从1开始编号
        for idx, item in enumerate(owner_asr_info):
            item['source_clip_id'] = idx + 1
        save_json(speech_asr_output_file, owner_asr_info)
    logger.info(f"生成精准asr与说话人信息文件耗时: {time.time() - start_time} 秒")
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
def get_scene(video_path, output_dir, high_thresholds=[30, 50, 70]):
    """
    生成视频不同阈值下的场景划分
    """
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

    merged_timestamps_file = os.path.join(output_dir, 'merged_timestamps.json')
    all_scene_info_dict = {}
    for high_threshold in high_thresholds:
        start_time = time.time()
        file_name = f'scenes_{high_threshold}_info.json'
        scene_info_file = os.path.join(output_dir, file_name)
        if is_valid_target_file_simple(scene_info_file):
            all_scene_info_dict[high_threshold] = read_json(scene_info_file)
            continue
        # 运行带有精炼功能的场景分割
        scene_info_dict = split_scenes_json(video_path, high_threshold=high_threshold, min_scene_len=25)
        logger.info(f"阈值为 {high_threshold}场景信息字典已生成并打印。共 {len(scene_info_dict)} 个场景。 耗时: {time.time() - start_time} 秒")
        save_json(scene_info_file, scene_info_dict)
        all_scene_info_dict[high_threshold] = scene_info_dict

    kept_sorted, pairs = merge_scene_timestamps(all_scene_info_dict, min_count=3)
    logger.info(f"场景识别合并完成:场景数量为: {len(kept_sorted)}")
    save_json(merged_timestamps_file, kept_sorted)
    # 进行kept_sorted合法性判断，如果小于3直接抛出异常
    if len(kept_sorted) < 3:
        error_msg = f"场景识别合并后时间点过少: {len(kept_sorted)} < 3，无法进行有效分割，请检查输入视频内容或调整阈值。"
        logger.error(error_msg)
        raise ValueError(error_msg)
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
        if last_scene_start < max_text_end:
            scenes.append({'start': last_scene_start, 'end': max(last_scene_start, max_text_end), 'content_list': []})
    else:
        start_time = scene_splits[0][0] if scene_splits else 0
        max_text_end = max(item['end'] for item in text_results) if text_results else start_time
        if start_time < max_text_end:
            scenes.append({'start': start_time, 'end': max_text_end, 'content_list': []})
    if not scenes: return []

    # --- [核心修复] 步骤 1: 智能分配与主动合并 ---
    text_queue = sorted(text_results, key=lambda x: x['start'])
    for text in text_queue:
        overlapping_indices = []
        is_owner = text.get('speaker') == owner_speaker
        for i, scene in enumerate(scenes):
            overlap_duration = max(0, min(text['end'], scene['end']) - max(text['start'], scene['start']))
            scene_duration = scene['end'] - scene['start']

            # 非 owner 仍用宽松策略（只要沾边）
            if not is_owner:
                if overlap_duration >= 1:
                    overlapping_indices.append(i)
                continue

            # === owner 的重叠判断：双条件任一满足即算重叠 ===
            cond1 = overlap_duration >= min_overlap_ms
            cond2 = scene_duration > 0 and (overlap_duration / scene_duration) >= 0.2
            if cond1 or cond2:
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


def merge_scenes_advanced(
        scene_timestamps,
        min_scene_duration_ms,
        high_occurrence_threshold=2,
        max_merge_duration_ms=10000,
        final_cleanup_threshold_ms=1500  # 新增参数：最终清理的阈值
):
    """
    最终版场景合并函数，采用“分块处理”的健壮逻辑。

    1.  用高频时间戳定义不可逾越的“大块”。
    2.  在每个“大块”内部，利用低频点进行精细分割和智能合并。
    3.  最后全局清理极短的碎片。
    """
    if not scene_timestamps:
        return []
    if len(scene_timestamps) < 10:
        # 直接使用所有时间戳（包括0）作为切分点
        # 使用 set 确保时间戳唯一，然后排序
        boundaries = sorted(list(set([0] + [t for t, c in scene_timestamps])))

        simple_scenes = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            if start < end:  # 确保时间戳有效
                simple_scenes.append([start, end])

        # --- 直接套用步骤 4 的格式化逻辑进行输出 ---
        formatted_output = []
        for start, end in simple_scenes:
            formatted_output.append({
                "scene_start": start,
                "scene_end": end,
                "content_list": [{"start": start, "end": end, "speaker": "other", "final_text": "", "text": ""}]
            })
        return formatted_output
    # --- 步骤 1: 定义“大块”边界 ---
    video_duration = scene_timestamps[-1][0]
    hard_boundaries = sorted(list(set(
        [0] + [t for t, c in scene_timestamps if c >= high_occurrence_threshold] + [video_duration]
    )))

    all_processed_scenes = []

    # --- 步骤 2: 逐个“大块”进行内部分割与合并 ---
    for i in range(len(hard_boundaries) - 1):
        block_start, block_end = hard_boundaries[i], hard_boundaries[i + 1]
        if block_start >= block_end:
            continue

        # 收集落在这个大块内部的所有时间戳
        internal_points = [block_start]
        for timestamp, _ in scene_timestamps:
            if block_start < timestamp < block_end:
                internal_points.append(timestamp)
        internal_points.append(block_end)

        # 利用内部所有点进行最精细的分割
        internal_segments = []
        unique_points = sorted(list(set(internal_points)))
        for j in range(len(unique_points) - 1):
            start, end = unique_points[j], unique_points[j + 1]
            if start < end:
                internal_segments.append([start, end])

        if not internal_segments:
            continue

        # 在这个“大块”内部执行合并逻辑
        block_merged_scenes = []
        for segment in internal_segments:
            if not block_merged_scenes:
                block_merged_scenes.append(segment)
                continue

            last_scene = block_merged_scenes[-1]
            current_duration = segment[1] - segment[0]
            last_duration = last_scene[1] - last_scene[0]

            # 强制合并：如果上一个场景不合格
            if last_duration < min_scene_duration_ms:
                last_scene[1] = segment[1]
            # 选择性合并：如果当前场景不合格，且上一个场景还有空间
            elif current_duration < min_scene_duration_ms and last_duration < max_merge_duration_ms:
                last_scene[1] = segment[1]
            # 不合并：当前场景成为新的独立场景
            else:
                block_merged_scenes.append(segment)

        all_processed_scenes.extend(block_merged_scenes)

    # --- 步骤 3: 全局清理 ---
    if not all_processed_scenes:
        return []

    final_scenes = []
    for scene in all_processed_scenes:
        duration = scene[1] - scene[0]
        if final_scenes and duration < final_cleanup_threshold_ms:
            final_scenes[-1][1] = scene[1]
        else:
            final_scenes.append(scene)

    # --- 步骤 4: 格式化输出 ---
    formatted_output = []
    for start, end in final_scenes:
        formatted_output.append({
            "scene_start": start,
            "scene_end": end,
            "content_list": [{"start": start, "end": end, "speaker": "other", "final_text": "", "text": ""}]
        })
    return formatted_output


def get_scene_sub_text(sorted_scene_timestamp, owner_asr, output_dir):
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

    output_file_scene_sub_text_file = os.path.join(output_dir, 'scene_sub_text.json')

    merged_timestamps = sorted_scene_timestamp
    # 将all_sub_text_list按照end升序排序
    owner_asr.sort(key=lambda x: x.get('end', 0))
    is_have_owner = False
    for asr in owner_asr:
        speaker = asr.get('speaker', '')
        if speaker != 'owner':
            asr['speaker'] = 'other'
        else:
            is_have_owner = True
        if 'text' not in asr or not asr['text'].strip():
            asr['text'] = asr['final_text']
        asr['start'] = time_to_ms(asr.get('start', 0))
        asr['end'] = time_to_ms(asr.get('end', 0))

    if len(owner_asr) > 2:
        owner_speaker = 'owner' if is_have_owner else 'other'
        min_scene_duration_ms = 10000 if is_have_owner else 1000
        scene_sub_text = process_scenes_complete_fix(owner_asr, merged_timestamps, owner_speaker=owner_speaker,
                                                     min_scene_duration_ms=min_scene_duration_ms)
    else:
        logger.warning("说话人文本过少，直接使用场景分割点进行划分")
        scene_sub_text = merge_scenes_advanced(merged_timestamps, min_scene_duration_ms=5000)
    save_json(output_file_scene_sub_text_file, scene_sub_text)
    logger.info(f"初步场景划分完成:数量{len(scene_sub_text)}")

    # 检测scene_sub_text的长度是否小于3，如果是则抛出异常
    if len(scene_sub_text) < 3:
        error_msg = f"最终场景划分结果过少: {len(scene_sub_text)} < 3，无法进行有效分割，请检查输入视频内容或调整阈值。"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return scene_sub_text


def merge_short_blank_segments(processed_scenes, min_duration=1000):
    """
    合并连续的、总时长小于 min_duration 的空白段落。
    实际上，这里的逻辑是：将所有连续的空白段落合并成一个，
    然后检查这个合并后的总时长，如果小于 min_duration，则丢弃它。

    processed_scenes: 经过 extract_and_merge_owner_other 处理后的列表。
    min_duration: 空白段落需要保留的最小毫秒时长。
    """
    if not processed_scenes:
        return []

    merged_result = []
    # 只用于累积连续的【短】空白场景
    short_blanks_accumulator = []

    def flush_short_blanks():
        """处理累积的短空白段落的辅助函数"""
        nonlocal short_blanks_accumulator, merged_result
        if not short_blanks_accumulator:
            return

        # 将所有累积的短空白段合并成一个
        start_time = short_blanks_accumulator[0]['scene_start']
        end_time = short_blanks_accumulator[-1]['scene_end']

        # 检查这个合并后的总时长是否达标
        if (end_time - start_time) > 0:
            merged_scene = {
                'scene_number': 'temp',  # 稍后统一重新编号
                'scene_start': start_time,
                'scene_end': end_time,
                'narration_script': '',
                'narration_script_start': None,
                'narration_script_end': None,
                'original_script': '',
                'original_script_start': None,
                'original_script_end': None
            }
            merged_result.append(merged_scene)

        # 清空累积器，为下一批做准备
        short_blanks_accumulator = []

    # --- 主循环 ---
    for scene in processed_scenes:
        is_blank = not scene.get('narration_script') and not scene.get('original_script')
        duration = scene['scene_end'] - scene['scene_start']

        if is_blank and duration < min_duration:
            # 1. 如果是【短】空白段，则加入累积器
            short_blanks_accumulator.append(scene)
        else:
            # 2. 如果是【内容段】或【长空白段】，则它是一个边界
            #    首先，处理掉之前累积的所有短空白段
            flush_short_blanks()
            #    然后，将这个边界段落自身加入结果
            merged_result.append(scene)

    # 循环结束后，别忘了处理可能遗留在累积器中的最后一批短空白段
    flush_short_blanks()

    # 最后，重新编号，确保 scene_number 是连续的
    for i, scene in enumerate(merged_result, 1):
        scene['scene_number'] = str(i)

    return merged_result


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
            result.append({
                'scene_number': f'{idx}',
                'scene_start': scene.get('scene_start'),
                'scene_end': scene.get('scene_end'),
                'narration_script': '',
                'narration_script_start': None,
                'narration_script_end': None,
                'original_script': '',
                'original_script_start': None,
                'original_script_end': None
            })
            idx += 1
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
            'narration_script': owner_text,
            'narration_script_start': owner_start,
            'narration_script_end': owner_end,
            'original_script': other_text,
            'original_script_start': other_start,
            'original_script_end': other_end,
            'original_content_list': contents
        })

        idx += 1
    new_result = merge_short_blank_segments(result, min_duration=2000)
    return new_result


def check_new_video_script(new_video_script, scene_info, logger):
    """
    检测 new_video_script 的有效性，确保：
    1. 每个场景的 original_scene_number 都能在 scene_info 中找到。
    2. 找到的原始场景中 'narration_script_list' 的 source_clip_id 集合
       必须与 new_script 场景中 'new_narration_script_list' 的 source_clip_id 集合完全一致。
    """
    # 创建一个从 scene_number 到整个场景对象的映射，便于快速查找
    scene_info_map = {scene['scene_number']: scene for scene in scene_info}

    # 遍历 new_video_script 中的每个方案和场景
    for solution_index, detail_new_video_script in enumerate(new_video_script):
        solution_title = detail_new_video_script.get('title', f"方案 {solution_index + 1}")

        scenes_list = detail_new_video_script.get('场景顺序与新文案')
        if not isinstance(scenes_list, list) or len(scenes_list) == 0:
            logger.error(
                f"方案 '{solution_title}' 的字段 '场景顺序与新文案' 不存在或为空（必须提供且至少包含 1 个场景）。")
            return False

        for scene_index, scene in enumerate(detail_new_video_script.get('场景顺序与新文案', [])):
            original_scene_num = scene.get('original_scene_number')

            # 检测点 1: original_scene_number 是否在 scene_info 中存在
            if original_scene_num not in scene_info_map:
                logger.error(f"在方案 '{solution_title}' 的第 {scene_index + 1} 个场景中：")
                logger.error(f"  - original_scene_number '{original_scene_num}' 不在 scene_info 中。")
                return False

            original_scene = scene_info_map[original_scene_num]

            # --- 检测点 2: 比较 source_clip_id 是否完全一致 ---

            # 安全地获取原始场景和新场景的 narration list，如果键不存在则视为空列表[]
            original_narration_list = original_scene.get('narration_script_list', [])
            new_narration_list = scene.get('new_narration_script_list', [])

            # 分别提取两边的 source_clip_id 到集合中
            original_clip_ids = {item['source_clip_id'] for item in original_narration_list}
            new_clip_ids = {item['source_clip_id'] for item in new_narration_list}

            # 并且要求new_narration_list中每个元素包含new_narration_script字段
            for item in new_narration_list:
                if 'new_narration_script' not in item:
                    logger.error(f"在方案 '{solution_title}' 的第 {scene_index + 1} 个场景中：")
                    logger.error(f"  - new_narration_script_list 中的元素缺少 'new_narration_script' 字段！")
                    return False

            # 比较两个集合是否相等。集合比较能确保元素和数量都一致，且忽略顺序。
            if original_clip_ids != new_clip_ids:
                logger.error(
                    f"在方案 '{solution_title}' 的第 {scene_index + 1} 个场景 (对应 original_scene_number: {original_scene_num}) 中：")
                logger.error(f"  - source_clip_id 不匹配！")
                logger.error(f"  - 期望的 ID 集合 (来自 scene_info): {original_clip_ids or '空'}")
                logger.error(f"  - 实际的 ID 集合 (来自 new_script): {new_clip_ids or '空'}")
                return False

    # 如果所有循环都正常完成，说明没有发现错误
    logger.info("检测通过！所有场景引用均有效，且 source_clip_id 完全匹配。")
    return True


def gen_new_video_script_llm(scene_info, output_dir, has_author_voice=True):
    """
    生成新的视频方案
    """
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

    for temp in scene_info:
        # 去掉scene_start和scene_end字段
        temp.pop('scene_start', None)
        temp.pop('scene_end', None)

    retry_delay = 10
    max_retries = 3
    prompt_file_path = '../../content_community/app/视频场景生成新视频无原始视频输入增强版本.txt'

    if not has_author_voice:
        logger.info("使用无主人说话人版本的提示词")
        prompt_file_path = '../../content_community/app/视频场景生成新视频无原始视频输入增强版本纯重排场景.txt'

    prompt = read_file_to_str(prompt_file_path)
    full_prompt = f'{prompt}\n{scene_info}'
    raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"正在生成新的视频脚本 (尝试 {attempt}/{max_retries})")
            model_name = "gemini-2.5-pro"
            # raw = get_llm_content_gemini_flash_video(prompt=full_prompt, video_path=video_path, model_name=model_name)
            raw = get_llm_content(prompt=full_prompt, model_name=model_name)

            new_video_script = string_to_object(raw)
            check_result = check_new_video_script(new_video_script, scene_info, logger)
            if not check_result:
                raise ValueError("生成的视频脚本检查未通过")
            return new_video_script
        except Exception as e:
            logger.error(f"生成视频信息失败 (尝试 {attempt}/{max_retries}): {e} {raw}")
            if attempt < max_retries:
                logger.info(f"正在重试... (等待 {retry_delay} 秒)")
                time.sleep(retry_delay)  # 等待一段时间后再重试
            else:
                logger.error("达到最大重试次数，失败.")
                return None  # 达到最大重试次数后返回 None
            logger.exception("详细堆栈信息：")


def check_logical_scene(logical_scene_info, scene_info):
    """
    生成的original_scene_number是否都在scene_info中
    """

    return True


def gen_logical_scene_llm(scene_info, video_path):
    """
    生成新的视频方案
    """
    basename = os.path.basename(video_path).split('.mp4')[0]
    output_dir = os.path.join(base_output_dir, basename)
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

    retry_delay = 10
    max_retries = 3
    prompt_file_path = '../../content_community/app/视频场景根据逻辑性的合并增强版本.txt'
    format_scene_info = []
    for scene in scene_info:
        format_scene = {
            'scene_number': scene.get('scene_number'),
            'scene_start': scene.get('scene_start'),
            'scene_end': scene.get('scene_end'),
            'narration_script': scene.get('narration_script', ''),
            'original_script': scene.get('original_script', '')
        }
        format_scene_info.append(format_scene)

    prompt = read_file_to_str(prompt_file_path)
    full_prompt = f'{prompt}\n{format_scene_info}'
    raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            model_name = "gemini-2.5-pro"
            logger.info(f"正在生成逻辑性场景划分 (尝试 {attempt}/{max_retries})")
            raw = get_llm_content_gemini_flash_video(prompt=full_prompt, video_path=video_path, model_name=model_name)
            logical_scene_info = string_to_object(raw)
            check_result = check_logical_scene(logical_scene_info, format_scene_info)
            if not check_result:
                raise ValueError("生成的视频脚本检查未通过")
            return logical_scene_info
        except Exception as e:
            logger.error(f"生成视频信息失败 (尝试 {attempt}/{max_retries}): {e} {raw}")
            if attempt < max_retries:
                logger.info(f"正在重试... (等待 {retry_delay} 秒)")
                time.sleep(retry_delay)  # 等待一段时间后再重试
            else:
                logger.error("达到最大重试次数，失败.")
                return None  # 达到最大重试次数后返回 None
            logger.exception("详细堆栈信息：")


def gen_final_scene_info(logical_scene_info, origin_scene_info, output_dir):
    """
    通过逻辑性的场景划分和原始的场景信息，生成最终的场景信息
    """
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

    final_scene_info = []
    new_scene_list = logical_scene_info.get('new_scene_info', [])
    for logical_scene in new_scene_list:
        temp_dict = {}
        narration_script_list = []
        original_script_list = []
        scene_start = 10000000
        scene_end = 0

        origin_scene_number_list = logical_scene.get('origin_scene_number_list', [])
        # 将origin_scene_number_list元素变成int类型并且升序
        origin_scene_number_list = sorted([int(num) for num in origin_scene_number_list if str(num).isdigit()])
        for origin_scene_number in origin_scene_number_list:
            target_origin_scene = None
            for origin_scene in origin_scene_info:
                if str(origin_scene['scene_number']) == str(origin_scene_number):
                    target_origin_scene = origin_scene
                    break
            if not target_origin_scene:
                logger.error(f"逻辑场景中的原始场景编号 {origin_scene_number} 不存在于原始场景信息中")
                continue
            scene_start = min(scene_start, int(target_origin_scene.get('scene_start', scene_start)))
            scene_end = max(scene_end, int(target_origin_scene.get('scene_end', scene_end)))
            original_content_list = target_origin_scene.get('original_content_list', [])
            # 筛选出speaker为owner的文本
            owner_content_list = [item for item in original_content_list if item.get('speaker') == 'owner']
            if owner_content_list:
                for owner_content in owner_content_list:
                    narration_script = owner_content.get('final_text', '').strip()
                    source_clip_id = owner_content.get('source_clip_id', '')
                    if narration_script:
                        narration_script_list.append({
                            'source_clip_id': source_clip_id,
                            'narration_script': narration_script
                        })

            origin_content_list = [item for item in original_content_list if item.get('speaker') != 'owner']

            if origin_content_list:
                for origin_content in origin_content_list:
                    original_script = origin_content.get('final_text', '').strip()
                    source_clip_id = origin_content.get('source_clip_id', '')
                    if original_script:
                        original_script_list.append({
                            'source_clip_id': source_clip_id,
                            # 'original_script_start': target_origin_scene.get('original_script_start'),
                            # 'original_script_end': target_origin_scene.get('original_script_end'),
                            'original_script': original_script
                        })

        temp_dict['scene_number'] = logical_scene.get('new_scene_number', '')
        temp_dict['scene_start'] = scene_start if scene_start != 10000000 else 0
        temp_dict['scene_end'] = scene_end
        temp_dict['narration_script_list'] = narration_script_list
        temp_dict['original_script_list'] = original_script_list
        temp_dict['visual_description'] = logical_scene.get('visual_description', '')
        # temp_dict['reason'] = logical_scene.get('reason', '')
        temp_dict['scene_potential'] = logical_scene.get('scene_potential', '')
        final_scene_info.append(temp_dict)

    return final_scene_info


@timeit_print
def gen_new_video_script(video_path, scene_sub_text, output_dir, target_speaker='owner', has_author_voice=False):
    """
    生成新视频的文本脚本
    """
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

    scene_sub_text_list = scene_sub_text
    output_file_final_path = os.path.join(output_dir, 'new_script.json')
    output_file_scene_info_path = os.path.join(output_dir, 'merge_speaker_scene_info.json')
    output_file_logical_scene_info_path = os.path.join(output_dir, 'logical_scene_info.json')
    final_scene_info_path = os.path.join(output_dir, 'final_scene_info.json')

    # if is_valid_target_file_simple(output_file_final_path, 10):
    #     new_video_script = read_json(output_file_final_path)
    #     scene_info = read_json(output_file_scene_info_path)
    #     return new_video_script, scene_info

    scene_info = extract_and_merge_owner_other(scene_sub_text_list, target_speaker)
    save_json(output_file_scene_info_path, scene_info)

    if is_valid_target_file_simple(output_file_logical_scene_info_path):
        logical_scene_info = read_json(output_file_logical_scene_info_path)
    else:
        logical_scene_info = gen_logical_scene_llm(scene_info, video_path=video_path)
        save_json(output_file_logical_scene_info_path, logical_scene_info)
    logger.info(
        f"场景逻辑合并完成:数量{len(logical_scene_info.get('new_scene_info'))} 删除的子场景数量:{len(logical_scene_info.get('deleted_scene'))}")

    final_scene_info = gen_final_scene_info(logical_scene_info, scene_info, output_dir)
    save_json(final_scene_info_path, final_scene_info)

    new_video_script = gen_new_video_script_llm(final_scene_info, output_dir, has_author_voice=has_author_voice)
    save_json(output_file_final_path, new_video_script)

    return new_video_script, scene_info


def generate_scene_segments(scene_start, scene_end, narration_script_list):
    # 排序
    narration_script_list.sort(key=lambda x: x['scene_start'])

    # 收集所有时间点（包括边界和片段的开始/结束）
    time_points = {scene_start, scene_end}
    for item in narration_script_list:
        time_points.add(item['scene_start'])
        time_points.add(item['scene_end'])

    # 排序时间点
    time_points = sorted(time_points)

    result = []

    # 遍历每两个相邻时间点
    for i in range(len(time_points) - 1):
        start = time_points[i]
        end = time_points[i + 1]

        # 如果当前段为空（start == end），跳过
        if start >= end:
            continue

        # 查找是否有片段覆盖这个区间
        matched_item = None
        for item in narration_script_list:
            if item['scene_start'] <= start and item['scene_end'] >= end:
                # 完全覆盖，使用该片段裁剪
                matched_item = item
                break

        # 如果有匹配的片段，裁剪它
        if matched_item:
            # 裁剪片段：只保留与当前子段重叠的部分
            clip_start = matched_item['scene_start']
            clip_end = matched_item['scene_end']
            overlap_start = max(start, clip_start)
            overlap_end = min(end, clip_end)

            if overlap_start < overlap_end:
                new_item = {
                    'source_clip_id': matched_item['source_clip_id'],
                    'new_narration_script': matched_item['new_narration_script'],
                    'scene_number': matched_item['scene_number'],
                    'scene_start': overlap_start,
                    'scene_end': overlap_end,
                    'narration_script': matched_item['narration_script'],
                    'narration_script_start': matched_item['narration_script_start'],
                    'narration_script_end': matched_item['narration_script_end'],
                    'original_script': matched_item['original_script'],
                    'original_script_start': matched_item['original_script_start'],
                    'original_script_end': matched_item['original_script_end']
                }
                result.append(new_item)
        else:
            # 没有匹配的片段，添加空段
            result.append({
                'new_narration_script': '',
                'scene_start': start,
                'scene_end': end,
            })

    return result


def process_video_with_owner_text(video_path, new_owner_text, fused_new_scene, scene_start, scene_end, base_name,
                                  max_diff, need_merge_video_file, name_key, subtitle_box):
    output_dir = os.path.join(base_output_dir, base_name)
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

    def _to_int(v):
        try:
            return int(float(v))
        except:
            return None

    if new_owner_text:
        original_script = fused_new_scene.get('original_script', '')
        offset = 100
        if not original_script or fused_new_scene.get('original_script_start', 10) > fused_new_scene.get(
                'narration_script_start'):
            offset = 500
        else:
            pass  # old: print()
        s = _to_int(fused_new_scene.get('narration_script_start')) or int(scene_start)
        s = s - offset
        e = _to_int(fused_new_scene.get('narration_script_end')) + 500

        if e is None:
            MS_PER_CHAR = 200
            MIN_MS = 500
            est = max(MIN_MS, len(new_owner_text) * MS_PER_CHAR)
            scene_end_time = _to_int(fused_new_scene.get('scene_end'))
            e = min(s + est, scene_end_time) if scene_end_time is not None else s + est

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
        video_time_segments = [
            (scene_start, format_start_time),
            (format_start_time, format_end_time),
            (format_end_time, scene_end)
        ]

        sub_count = 0
        for video_time_segment in video_time_segments:
            sub_count += 1
            seg_start, seg_end = video_time_segment
            if seg_end > seg_start + 100:
                segment_output_scene_file = r"W:\project\python_project\watermark_remove\content_community\bilibili" + f'/output/{base_name}/split_scene/{name_key}_part{sub_count}.mp4'
                logger.info(f'处理: {segment_output_scene_file} 时间段: {seg_start}-{seg_end}')
                output_path = segment_output_scene_file
                if not is_valid_target_file_simple(segment_output_scene_file):
                    clip_video_ms(video_path, seg_start, seg_end, segment_output_scene_file)

                if sub_count == 2:
                    output_path = segment_output_scene_file.replace('.mp4', '_with_text.mp4')
                    origin_video_path = segment_output_scene_file
                    keep_original_audio = False
                    if not is_valid_target_file_simple(output_path):
                        # audio_path = gen_audio_path(video_path).replace("vocals.wav", "no_vocals.wav")
                        # pure_audio_path = gen_audio_path(video_path).replace(".wav", "_pure.wav")
                        # if not is_valid_target_file_simple(pure_audio_path):
                        #     process_media_by_volume(audio_path, pure_audio_path)
                        # segment_output_scene_background_file = segment_output_scene_file.replace('.mp4', '_with_background.mp4')
                        # replace_video_audio(segment_output_scene_file,seg_start, seg_end, pure_audio_path, segment_output_scene_background_file)
                        # origin_video_path = segment_output_scene_background_file
                        # keep_original_audio = True
                        gen_video(new_owner_text, output_path, origin_video_path,
                                  keep_original_audio=keep_original_audio, fixed_rect=subtitle_box)

                need_merge_video_file.append(output_path)
    else:
        if scene_start >= scene_end - 100:
            logger.warning(f"跳过无效时间段: {scene_start}-{scene_end}")
            return need_merge_video_file
        output_scene_file = r"W:\project\python_project\watermark_remove\content_community\bilibili" + f'/output/{base_name}/split_scene/{name_key}_part{0}.mp4'
        if not is_valid_target_file_simple(output_scene_file):
            clip_video_ms(video_path, scene_start, scene_end, output_scene_file)
        need_merge_video_file.append(output_scene_file)

    return need_merge_video_file


def get_bgm_path(tags, logger):
    """
    根据标签匹配数量对BGM进行排序，并选择一个合适的BGM路径。

    Args:
        tags (dict): 输入的标签字典，例如 {'style': ['清新'], 'mood': ['愉快']}
        logger: 日志记录器实例

    Returns:
        str: 选中的BGM文件路径。
    """
    all_tags = []
    for key, value in tags.items():
        all_tags.extend(value)
    # 使用集合以便快速计算交集
    all_tags_set = set(all_tags)

    bgm_dir = r"W:\project\python_project\watermark_remove\content_community\app\bgm_audio"
    bgm_info_list = read_json(r"W:\project\python_project\watermark_remove\content_community\app\bgm_info.json")

    bgm_info_map = {}
    for bgm_info in bgm_info_list:
        bgm_name = bgm_info.get('bgm_name', '未知').split('.')[0]
        bgm_tags_dict = bgm_info.get('selected_tags', {})
        bgm_all_tags = []
        for key, bgm_tag_list in bgm_tags_dict.items():
            bgm_all_tags.extend(bgm_tag_list)
        bgm_info_map[bgm_name] = bgm_all_tags

    # 获取所有有效的BGM文件
    bgm_files = [f for f in os.listdir(bgm_dir) if f.lower().endswith('.wav')]
    bgm_file_names = [os.path.splitext(f)[0] for f in bgm_files]

    bgm_with_match_count = []
    for bgm_file_name in bgm_file_names:
        bgm_tags = bgm_info_map.get(bgm_file_name, [])
        if not bgm_tags:
            continue

        # 计算交集，获取匹配的标签数量
        match_count = len(all_tags_set.intersection(set(bgm_tags)))

        if match_count > 0:
            bgm_path = os.path.join(bgm_dir, f"{bgm_file_name}.wav")
            bgm_with_match_count.append({'path': bgm_path, 'match_count': match_count})

    if not bgm_with_match_count:
        # 如果没有任何匹配的BGM，可以采取备用策略，例如随机选择一个BGM
        logger.warning(f"在 {bgm_dir} 目录下未找到任何与给定标签匹配的音频文件，将随机选择一个文件。")
        if not bgm_files:
            raise FileNotFoundError(f"在 {bgm_dir} 目录下找不到任何音频文件！")
        return os.path.join(bgm_dir, random.choice(bgm_files))

    # 根据匹配数量进行降序排序
    bgm_with_match_count.sort(key=lambda x: x['match_count'], reverse=True)

    # --- 选择策略 ---
    # 策略2：在匹配度最高的几个BGM中随机选择一个（例如前3个）
    top_n = 3
    top_choices = bgm_with_match_count[:top_n]
    if not top_choices:
        # 理论上，如果bgm_with_match_count不为空，这里就不会为空
        raise ValueError("未能确定顶部的BGM选项。")

    selected_bgm = random.choice(top_choices)

    logger.info(f"最终选择的BGM: {selected_bgm['path']} (匹配数: {selected_bgm['match_count']})")
    return selected_bgm['path']


@timeit_print
def gen_new_video_by_scene_and_script(video_path, new_video_script, scene_info, subtitle_box, base_name):
    """
    生成新视频的文本脚本
    """
    output_dir = os.path.join(base_output_dir, base_name)
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

    max_diff = 500
    new_video_script.sort(key=lambda x: x.get('方案整体评分', 0), reverse=True)
    new_video_script_result = new_video_script
    final_video_script = new_video_script_result[0]
    final_scene_info_path = r"W:\project\python_project\watermark_remove\content_community\bilibili" + f'/output/{base_name}/final_scene_info.json'
    final_scene_info = read_json(final_scene_info_path)

    need_merge_video_file = []
    for new_scene in final_video_script['场景顺序与新文案']:
        original_scene_number = new_scene.get('original_scene_number')
        for final_scene in final_scene_info:
            final_scene_number = final_scene.get('scene_number')
            if str(final_scene_number) == str(original_scene_number):
                new_scene.update(final_scene)
                break

        new_narration_script_list = new_scene.get('new_narration_script_list', [])
        for new_narration_script in new_narration_script_list:
            original_scene_number = new_narration_script['source_clip_id']
            for scene in scene_info:
                if str(scene['scene_number']) == str(original_scene_number):
                    new_narration_script.update(scene)
    logger.info('完成场景信息合并')

    new_scene_list = final_video_script['场景顺序与新文案']
    # new_scene_list = new_scene_list[7:]
    for fused_new_scene in new_scene_list:
        scene_start = fused_new_scene.get('scene_start')
        scene_end = fused_new_scene.get('scene_end')
        name_key = f"new_scene_{fused_new_scene.get('new_scene_number')}_original_scene_{fused_new_scene.get('original_scene_number')}"

        new_narration_script_list = fused_new_scene.get('new_narration_script_list', [])
        split_scene_list = generate_scene_segments(scene_start, scene_end, new_narration_script_list)
        # split_scene_list = [{'new_narration_script': '', 'scene_end': scene_end, 'scene_start': scene_start}]
        start_time = time.time()
        count = 0
        for split_scene in split_scene_list:
            count += 1
            name_key_full = f"{name_key}_part{count}"
            new_narration_script = split_scene.get('new_narration_script', '').strip()
            process_video_with_owner_text(video_path, new_narration_script, split_scene, split_scene['scene_start'],
                                          split_scene['scene_end'], base_name, max_diff, need_merge_video_file,
                                          name_key_full, subtitle_box)
        logger.info(
            f'处理新场景:{name_key} 分割后的场景数量{len(split_scene_list)} 进度: {new_scene_list.index(fused_new_scene) + 1}/{len(new_scene_list)} 耗时: {time.time() - start_time:.2f} 秒')

    final_output_path = r"W:\project\python_project\watermark_remove\content_community\bilibili" + f'/output/{base_name}/remake.mp4'
    merge_videos_ffmpeg(need_merge_video_file, output_path=final_output_path)
    tags = final_video_script.get('tags', [])
    bgm_path = get_bgm_path(tags, logger)
    if bgm_path and os.path.exists(bgm_path):
        # logger.info(f"正在为视频添加背景音乐: {bgm_path}")
        final_with_bgm_path = final_output_path.replace('.mp4', '_with_bgm.mp4')
        add_bgm_to_video(final_output_path, bgm_path, str(final_with_bgm_path), auto_compute=True)
        return final_with_bgm_path, final_video_script
    return final_output_path, final_video_script


def merge_intervals(intervals):
    """
    合并相邻或重叠的时间段

    Args:
        intervals: 时间段列表，每个元素为 (start, end) 的元组

    Returns:
        合并后的时间段列表
    """
    # 处理空列表的情况
    if not intervals:
        return []

    # 按开始时间排序
    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    # 初始化结果列表，第一个时间段作为起点
    merged = [sorted_intervals[0]]

    # 遍历剩余的时间段
    for current in sorted_intervals[1:]:
        # 获取当前合并结果中的最后一个时间段
        last = merged[-1]

        # 如果当前时间段与最后一个时间段相邻或重叠
        # 相邻的条件是：current[0] <= last[1]
        # （因为如果 current[0] == last[1]，它们是连续的，应该合并）
        if current[0] <= last[1]:
            # 合并时间段，结束时间取两者中的最大值
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            # 不相邻，直接添加到结果中
            merged.append(current)

    return merged


@timeit_print
def gen_subtitle_box_and_cover_subtitle(video_path, merged_scene_info_list, output_dir):
    """
    找到字幕区域并且遮挡字幕
    """
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

    time_ranges = []

    duration_list = []
    for scene_info in merged_scene_info_list:
        narration_script = scene_info.get('narration_script', '').strip()
        if not narration_script:
            continue
        scene_start = scene_info.get('scene_start')
        scene_end = scene_info.get('scene_end')
        narration_script_start = scene_info.get('narration_script_start')
        narration_script_end = scene_info.get('narration_script_end')
        narration_script_start -= 500
        narration_script_end += 500

        final_narration_script_start = max(scene_start, narration_script_start)
        final_narration_script_end = min(scene_end, narration_script_end)
        duration_list.append((final_narration_script_start, final_narration_script_end))
    merge_intervals_list = merge_intervals(duration_list)
    merged_timerange_list = []
    if not merge_intervals_list:
        return video_path, None

    for start, end in merge_intervals_list:
        merged_timerange_list.append(
            {
                "startTime": ms_to_time(start),
                "endTime": ms_to_time(end)
            }
        )
        time_ranges.append((start / 1000, end / 1000))
    final_box_path = os.path.join(output_dir, 'final_subtitle_box.json')

    if not is_valid_target_file_simple(final_box_path):
        final_box = find_overall_subtitle_box_target_number(video_path, merged_timerange_list, output_dir=output_dir)
        save_json(final_box_path, final_box)
    final_box = read_json(final_box_path)
    top_left, bottom_right, vid_w, vid_h = adjust_subtitle_box(video_path, final_box)

    cover_video_path = os.path.join(output_dir, 'subtitle_covered.mp4')
    if is_valid_target_file_simple(cover_video_path, 10):
        logger.info(f"已存在遮挡字幕的视频: {cover_video_path}")
        return cover_video_path, [top_left, bottom_right]
    cover_subtitle(video_path, cover_video_path, top_left, bottom_right, time_ranges=time_ranges)
    if not is_valid_target_file_simple(cover_video_path, 10):
        raise ValueError(f"生成遮挡字幕视频失败: {cover_video_path}")

    return cover_video_path, [top_left, bottom_right]


@timeit_print
def gen_sub_scene(video_path, output_dir, sorted_scene_timestamp, has_author_voice=True):
    """
    根据场景分割点和是否包含原始作者语音进行子场景划分
    """
    fixed_speech_asr_with_sub_text = gen_asr(video_path, output_dir, has_author_voice)
    for idx, item in enumerate(fixed_speech_asr_with_sub_text):
        item['source_clip_id'] = idx + 1
    scene_sub_text = get_scene_sub_text(sorted_scene_timestamp, fixed_speech_asr_with_sub_text, output_dir)
    return scene_sub_text


def gen_video_script(video_path, params={}):
    """
    根据原始视频以及一些参数生成新的视频方案
    参数包括 是否包含原始作者语音（默认为True表示包含作者语音） 创作指导 评论参考
    """
    # 信息准备
    basename = os.path.basename(video_path).split('.mp4')[0]
    output_dir = os.path.join(base_output_dir, basename)
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

    has_author_voice = params.get('has_author_voice', True)

    # 获取场景分割点信息
    logger.info("开始获取场景分割点信息...")
    start_time = time.time()
    sorted_scene_timestamp = get_scene(video_path, output_dir)
    logger.info(f"场景分割点信息获取完成。耗时: {time.time() - start_time:.2f} 秒")

    # 进行子场景的划分
    logger.info("开始进行子场景划分...")
    start_time = time.time()
    scene_sub_text = gen_sub_scene(video_path, output_dir, sorted_scene_timestamp, has_author_voice=has_author_voice)
    logger.info(f"子场景划分完成。耗时: {time.time() - start_time:.2f} 秒")

    # 进行新文案的生成
    logger.info("开始生成新视频脚本...")
    start_time = time.time()
    new_video_script, scene_info = gen_new_video_script(video_path, scene_sub_text, output_dir, has_author_voice=has_author_voice)
    logger.info(f"新视频脚本生成完成。耗时: {time.time() - start_time:.2f} 秒")


def gen_new_video(video_path):
    """
    根据新的方案生成新的视频
    """
    basename = os.path.basename(video_path).split('.mp4')[0]
    output_dir = os.path.join(base_output_dir, basename)
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

    output_file_final_path = os.path.join(output_dir, 'new_script.json')
    output_file_scene_info_path = os.path.join(output_dir, 'merge_speaker_scene_info.json')
    all_files = [output_file_final_path, output_file_scene_info_path]

    all_valid = all(is_valid_target_file_simple(f, 10) for f in all_files)
    if not all_valid:
        logger.warning(f"检测到部分输出文件缺失或无效，将重新处理视频: {video_path}")
        return None, None, []
    scene_info = read_json(output_file_scene_info_path)
    new_video_script = read_json(output_file_final_path)

    logger.info("开始处理字幕区域并生成遮罩...")
    subtitle_video_path, subtitle_box = gen_subtitle_box_and_cover_subtitle(video_path, scene_info, output_dir)
    logger.info("字幕处理完成。")

    logger.info("开始根据新脚本生成最终视频...")
    final_video_path, final_video_script = gen_new_video_by_scene_and_script(
        subtitle_video_path, new_video_script, scene_info, subtitle_box, basename
    )
    logger.info("最终视频生成完成。")
    return final_video_path, final_video_script


@timeit_print
def video_remake(video_path, no_owner=False, video_info={}, is_half=False):
    """
    重制视频，并在发生异常时将日志保存到指定文件。
    """
    basename = os.path.basename(video_path).split('.mp4')[0]
    output_dir = os.path.join(base_output_dir, basename)
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

    try:
        logger.info(f"========== 开始处理视频: {video_path} ==========")
        gen_video_script(video_path)
        gen_new_video(video_path)
        logger.info(f"========== 视频处理成功: {video_path} ==========")
    except Exception as e:
        logger.error(f"处理视频 {video_path} 时发生未捕获的严重错误: {e}")
        logger.exception("详细错误堆栈信息:")  # 记录完整的堆栈信息到日志
        # 这里可以选择重新抛出异常，或者根据业务需要决定是否继续
        # raise


if __name__ == '__main__':
    gen_video_script('7535033987593440527.mp4')