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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from LLM.gemini import get_llm_content, get_llm_content_gemini_flash_video
from common_utils.common_utils import read_json, time_to_ms, save_json, ms_to_time, read_file_to_str, string_to_object, \
    timeit_print, is_valid_target_file_simple
from common_utils.ocr.paddle_ocr_utils import find_overall_subtitle_box_target_number
from common_utils.split_audio import separate_with_cli
from common_utils.split_scenes import split_scenes_json
from common_utils.video_utils import extract_audio_from_video, clip_video_ms, merge_videos_ffmpeg, probe_duration, \
    cover_subtitle
from common_utils.video_utils2 import add_bgm_to_video

import re

from common_utils.video_utils_cut import gen_video
from content_community.app.remake_video import adjust_subtitle_box

base_output_dir = "W:/project/python_project/watermark_remove/douyin_video"


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
    max_end_time_ms = 0
    # 使用 enumerate 获取索引和元素，便于日志记录
    for i, segment in enumerate(owner_asr_info):
        try:
            start_str = segment.get("start")
            end_str = segment.get("end")

            # 检查 start 和 end 是否为字符串，如果不是，则格式错误
            if not isinstance(start_str, str) or not isinstance(end_str, str):
                logger.error(f"[ERROR] 片段 {i} 的时间格式不正确，应为字符串。数据: {segment}")
                return False

            start_time_ms = time_to_ms(start_str)
            end_time_ms = time_to_ms(end_str)

            # --- 核心修改步骤：原地更新字典 ---
            segment["start"] = start_time_ms
            segment["end"] = end_time_ms
            # ------------------------------------

            # 更新整个 ASR 列表的最大结束时间
            max_end_time_ms = max(max_end_time_ms, end_time_ms)

            duration_ms = end_time_ms - start_time_ms

            # 1. 最大文案长度不能超过 20s
            if len(owner_asr_info[i]['final_text']) > 200 and owner_asr_info[i]['speaker'] == 'owner':
                logger.error(f"[ERROR] 片段 {i} 文案长度：{len(owner_asr_info[i]['final_text'])} 跨度过长: {duration_ms} ms 文案为:{owner_asr_info[i]['final_text']}")
                return False

        except (ValueError, TypeError) as e:
            logger.error(f"[ERROR] 处理片段 {i} 时发生时间转换错误: {e}. 数据: {segment}")
            return False

    # 循环结束后，检查 ASR 的最大时间是否超过视频总时长（允许1秒的误差）
    if max_end_time_ms > video_duration + 1000:
        logger.error(f"[ERROR] ASR 最大结束时间 {max_end_time_ms} ms 超过视频总时长 {video_duration} ms")
        return False

    # 为owner_asr_info增加source_clip_id字段，从1开始
    source_clip_id = 0
    for segment in owner_asr_info:
        source_clip_id += 1
        segment['source_clip_id'] = source_clip_id


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
                "source_clip_id": 1,
                "start": 0,
                "end": video_duration_ms,
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
                        "start": 0,
                        "end": video_duration_ms,
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


def check_logical_scene(logical_scene_info: dict, video_duration_ms: int) -> tuple[bool, str]:
    """
     检查 logical_scene_info 的有效性，并在检查过程中将时间字符串转换为毫秒整数（in-place-modification）。

     Args:
         logical_scene_info (dict): 包含 'new_scene_info' 和 'deleted_scene' 的字典。
                                    此字典中的时间格式将被直接修改。
         video_duration_ms (int): 视频总时长（毫秒）。

     Returns:
         tuple[bool, str]: 一个元组，第一个元素是检查结果 (True/False)，
                            第二个元素是具体的检查信息。
     """
    # 临时列表，用于存储转换后的时间信息以进行排序和连续性检查
    all_scenes_for_sorting = []

    # 待处理的场景列表（new_scene_info 和 deleted_scene）
    scene_lists_to_process = [
        logical_scene_info.get('new_scene_info', []),
        logical_scene_info.get('deleted_scene', [])
    ]

    # 1. 遍历并转换所有场景，同时进行初步检查
    for scene_list in scene_lists_to_process:
        for i, scene in enumerate(scene_list):
            try:
                start_str, end_str = scene['start'], scene['end']

                # 确保 start 和 end 都是字符串，如果已经是数字则跳过转换
                if not isinstance(start_str, str) or not isinstance(end_str, str):
                    return False, f"检查失败：场景 {i + 1} 的时间格式不正确，期望是字符串但不是。场景: {scene}"

                start_ms = time_to_ms(start_str)
                end_ms = time_to_ms(end_str)

                # --- 核心修改步骤 ---
                # 直接在原始字典上更新值为毫秒整数
                scene['start'] = start_ms
                scene['end'] = end_ms
                # --------------------

                # 要求1：start < end
                if start_ms >= end_ms:
                    return False, f"检查失败：场景 {i + 1} 的开始时间 {start_str} ({start_ms}ms) 必须小于结束时间 {end_str} ({end_ms}ms)。"

                # 要求3：在视频时长范围内
                if not (0 <= start_ms <= video_duration_ms and 0 <= end_ms <= video_duration_ms + 2000):
                    return False, f"检查失败：场景 {i + 1} 的时间范围 [{start_str}, {end_str}] 超出视频时长 [0, {video_duration_ms}ms]。"

                # 将信息存入临时列表，用于后续排序和检查
                all_scenes_for_sorting.append({
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'original_start': start_str,  # 保留原始字符串用于错误报告
                    'original_end': end_str,
                })

            except (ValueError, TypeError) as e:
                return False, f"检查失败：场景 {i + 1} 的时间格式无效。原始场景: {scene}, 错误: {e}"

    # 如果视频时长为0，且没有场景，这是有效情况
    if not all_scenes_for_sorting and video_duration_ms == 0:
        return True, "OK. 视频时长为0，且没有场景。"

    if not all_scenes_for_sorting:
        return False, "检查失败：未提供任何场景信息，但视频时长大于0。"

    # 2. 按开始时间排序，为连续性检查做准备
    all_scenes_for_sorting.sort(key=lambda x: x['start_ms'])

    # 3. 检查时间轴的完整性
    if all_scenes_for_sorting[0]['start_ms'] != 0:
        return False, f"检查失败：时间轴不连续。第一个场景从 {all_scenes_for_sorting[0]['original_start']} 开始，而不是从 00:00.000 开始。"

    if abs(all_scenes_for_sorting[-1]['end_ms'] - video_duration_ms) > 2000:
        return False, f"检查失败：时间轴不完整。最后一个场景在 {all_scenes_for_sorting[-1]['original_end']} ({all_scenes_for_sorting[-1]['end_ms']}ms) 结束，与视频总时长 {video_duration_ms}ms 不匹配。"

    # 4. 遍历排序后的场景，检查重叠和间隔
    for i in range(len(all_scenes_for_sorting) - 1):
        current = all_scenes_for_sorting[i]
        next_s = all_scenes_for_sorting[i + 1]

        # 要求2：不能重叠
        if current['end_ms'] > next_s['start_ms']:
            return False, (f"检查失败：场景之间存在重叠。场景 "
                           f"[{current['original_start']} - {current['original_end']}] 与 "
                           f"[{next_s['original_start']} - {next_s['original_end']}] 重叠。")

        # 要求4：不能有间隔
        if current['end_ms'] < next_s['start_ms']:
            return False, (f"检查失败：场景之间存在间隔。场景 "
                           f"[{current['original_start']} - {current['original_end']}] 之后与 "
                           f"[{next_s['original_start']} - {next_s['original_end']}] 之前有时间空缺。")

    # 为logical_scene_info增加一个字段，表示scene_number
    scene_number = 1
    for scene_list in scene_lists_to_process:
        for scene in scene_list:
            scene['scene_number'] = scene_number
            scene_number += 1

    return True, "检查并转换成功：所有场景的时间有效、连续且无重叠，格式已更新为毫秒。"


def gen_logical_scene_llm(video_path):
    """
    生成新的视频方案
    """
    basename = os.path.basename(video_path).split('.mp4')[0]
    output_dir = os.path.join(base_output_dir, basename)
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

    # --- 2. 初始化和预处理 ---
    try:
        video_duration = probe_duration(video_path)
        video_duration_ms = int(video_duration * 1000)
    except Exception as e:
        logger.error(f"获取视频时长失败: {e}")
        return None

    retry_delay = 10
    max_retries = 3
    prompt_file_path = '../../content_community/app/视频场景逻辑切分只根据视频内容.txt'
    prompt = read_file_to_str(prompt_file_path)
    full_prompt = f'{prompt}'
    raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            model_name = "gemini-2.5-pro"
            logger.info(f"正在生成逻辑性场景划分 (尝试 {attempt}/{max_retries})")
            raw = get_llm_content_gemini_flash_video(prompt=full_prompt, video_path=video_path, model_name=model_name)
            logical_scene_info = string_to_object(raw)
            check_result, check_info = check_logical_scene(logical_scene_info, video_duration_ms)
            if not check_result:
                logger.error(f"逻辑性场景划分检查未通过: {check_info} {raw}")
                raise ValueError(f"逻辑性场景划分检查未通过: {check_info} {raw}")
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


def process_scenes_improved(logical_scene_info, owner_asr_info):
    """
    将ASR信息整合到场景信息中，并根据说话人区分为旁白和原始脚本。
    此函数通过计算时间上的最大重叠，确保跨场景的ASR条目能被分配到
    最合适的场景中，避免任何ASR信息被丢弃。

    Args:
        logical_scene_info (dict): 包含'new_scene_info'键的字典，其值为场景信息列表。
        owner_asr_info (list): 包含语音识别结果的列表，每个元素是一个包含时间、
                               说话人、文本等信息的字典。

    Returns:
        list: 一个新的场景信息列表，每个场景字典都包含了
              'narration_script_list' 和 'original_script_list'。
    """
    # 1. 初始化最终结果列表，并预先填充好场景的基本信息
    #    这样做可以方便后续直接向场景中添加脚本
    processed_scene_list = []
    scenes_map = {}  # 使用字典方便快速查找
    for scene_data in logical_scene_info.get("new_scene_info", []):
        new_scene = {
            "scene_number": scene_data.get("scene_number"),
            "scene_start": scene_data.get("start"),
            "scene_end": scene_data.get("end"),
            "narration_script_list": [],
            "original_script_list": [],
            "visual_description": scene_data.get("visual_description"),
            "scene_potential": scene_data.get("scene_potential")
        }
        processed_scene_list.append(new_scene)
        # 键为场景编号，值为列表中的场景对象引用
        scenes_map[new_scene["scene_number"]] = new_scene

    # 2. 遍历每一条ASR记录
    for asr_item in owner_asr_info:
        asr_start = asr_item.get("start")
        asr_end = asr_item.get("end")

        best_scene_number = None
        max_overlap = -1

        # 3. 为当前ASR记录计算与所有场景的重叠时间，找到重叠最长的场景
        for scene in processed_scene_list:
            scene_start = scene.get("scene_start")
            scene_end = scene.get("scene_end")

            # 计算重叠区间的长度
            # overlap = max(0, min(end1, end2) - max(start1, start2))
            overlap_duration = max(0, min(asr_end, scene_end) - max(asr_start, scene_start))

            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_scene_number = scene.get("scene_number")

        # 4. 如果找到了最匹配的场景，则将ASR记录添加到该场景中
        if best_scene_number is not None:
            script_item = {
                "source_clip_id": asr_item.get("source_clip_id"),
                "original_script": asr_item.get("final_text")
            }

            target_scene = scenes_map[best_scene_number]

            if asr_item.get("speaker") == "owner":
                target_scene["narration_script_list"].append(script_item)
            else:
                target_scene["original_script_list"].append(script_item)

    return processed_scene_list


def adjust_clips_to_range(data, start, end):
    """
    (新增步骤) 预处理函数：过滤和裁剪片段，确保它们严格在[start, end]范围内。

    参数:
        data (list): 原始数据列表。
        start (int): 期望的开始时间。
        end (int): 期望的结束时间。

    返回:
        list: 经过调整后的新列表。
    """
    adjusted_data = []
    for clip in data:
        clip_start = clip["narration_script_start"]
        clip_end = clip["narration_script_end"]

        # 检查片段与[start, end]范围是否有重叠
        # 条件：片段的结束时间必须在范围开始之后，且片段的开始时间必须在范围结束之前
        if clip_end > start and clip_start < end:
            # 计算裁剪后的新起始和结束时间
            new_start = max(clip_start, start)
            new_end = min(clip_end, end)

            # 只有当裁剪后仍然是有效的时间段时才添加
            if new_start < new_end:
                new_clip = clip.copy()
                new_clip["narration_script_start"] = new_start
                new_clip["narration_script_end"] = new_end
                adjusted_data.append(new_clip)

    return adjusted_data


def fill_time_gaps(start, end, data):
    """
    填充给定时间段内的空白部分。
    """
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        raise ValueError("start和end必须是有效的非负整数，且end不应小于start。")

    sorted_data = sorted(data, key=lambda x: x["narration_script_start"])

    result = []
    current_time = start

    for item in sorted_data:
        item_start = item["narration_script_start"]
        item_end = item["narration_script_end"]

        if current_time < item_start:
            result.append({
                "new_narration_script": "",
                "narration_script_start": current_time,
                "narration_script_end": item_start
            })

        result.append({
            "new_narration_script": item.get("new_narration_script", item.get("narration_script", "")),
            "narration_script_start": item_start,
            "narration_script_end": item_end
        })

        current_time = item_end

    if current_time < end:
        result.append({
            "new_narration_script": "",
            "narration_script_start": current_time,
            "narration_script_end": end
        })

    return result


def merge_short_clips(clips, min_duration=500):
    """
    合并列表中时长过短的片段。
    """
    if not clips:
        return []

    merged_list = [clips[0]]

    for i in range(1, len(clips)):
        last_clip = merged_list[-1]
        current_clip = clips[i]

        last_duration = last_clip["narration_script_end"] - last_clip["narration_script_start"]
        current_duration = current_clip["narration_script_end"] - current_clip["narration_script_start"]

        if last_duration < min_duration:
            current_clip["narration_script_start"] = last_clip["narration_script_start"]
            if last_clip["new_narration_script"] and not current_clip["new_narration_script"]:
                current_clip["new_narration_script"] = last_clip["new_narration_script"]
            merged_list[-1] = current_clip
        elif current_duration < min_duration:
            last_clip["narration_script_end"] = current_clip["narration_script_end"]
            if current_clip["new_narration_script"] and not last_clip["new_narration_script"]:
                last_clip["new_narration_script"] = current_clip["new_narration_script"]
        else:
            merged_list.append(current_clip)

    return merged_list


def process_narration_clips(start, end, data, min_duration=500):
    """
    (最终主函数) 完整流程：裁剪、填充、合并。

    参数:
        start (int): 最终结果的开始时间。
        end (int): 最终结果的结束时间。
        data (list): 原始数据列表。
        min_duration (int): 片段的最小允许时长。

    返回:
        list: 经过所有处理后的最终列表。
    """
    # 步骤 1: 过滤和裁剪所有片段，确保它们在[start, end]范围内
    adjusted_data = adjust_clips_to_range(data, start, end)

    # 步骤 2: 使用处理过的数据来填充[start, end]范围内的空白
    filled_clips = fill_time_gaps(start, end, adjusted_data)

    # 步骤 3: 合并所有时长过短的片段
    final_clips = merge_short_clips(filled_clips, min_duration)

    return final_clips

@timeit_print
def gen_video_script(logical_scene_info, owner_asr_info, output_dir, has_author_voice=False):
    """
    生成新视频的文本脚本
    """
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)
    final_scene_info_path = os.path.join(output_dir, 'final_scene_info.json')
    new_video_script_path = os.path.join(output_dir, 'new_video_script.json')

    if is_valid_target_file_simple(new_video_script_path) and is_valid_target_file_simple(final_scene_info_path):
        logger.info("检测到已存在的输出文件，直接加载返回")
        return read_json(new_video_script_path)

    final_scene_info = process_scenes_improved(logical_scene_info, owner_asr_info)
    save_json(final_scene_info_path, final_scene_info)

    new_video_script = gen_new_video_script_llm(final_scene_info, output_dir, has_author_voice=has_author_voice)
    save_json(new_video_script_path, new_video_script)

    return new_video_script


def process_video_with_owner_text(video_path, split_scene, output_dir, name_key, subtitle_box):
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)
    new_narration_script = split_scene.get('new_narration_script', '')
    narration_script_start = split_scene.get('narration_script_start', 0)
    narration_script_end = split_scene.get('narration_script_end', 0)
    segment_output_scene_file = os.path.join(output_dir,'split_scene/' f'{name_key}.mp4')
    start_time = time.time()

    if narration_script_start >= narration_script_end - 100:
        logger.warning(f"跳过无效时间段: {narration_script_start}-{narration_script_end}")
        return None

    if not is_valid_target_file_simple(segment_output_scene_file):
        clip_video_ms(video_path, narration_script_start, narration_script_end, segment_output_scene_file)

    if new_narration_script.strip() != '':
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
            gen_video(new_narration_script, output_path, origin_video_path, keep_original_audio=keep_original_audio, fixed_rect=subtitle_box)
        need_merge_video_file = output_path
    else:
        need_merge_video_file = segment_output_scene_file

    logger.info(f"处理片段 {name_key} 完成，耗时 {time.time() - start_time:.2f} 秒\n")
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


def choose_script(new_video_script, need_different=False):
    """
    选择最优的视频脚本方案。

    当 need_different 为 True 时，会优先选择“第一个场景是否改变”为True的方案，
    在此基础上再按“方案整体评分”从高到低排序。
    否则，仅按“方案整体评分”排序。

    Args:
        new_video_script (list or dict): 包含一个或多个脚本方案的列表或单个方案的字典。
        need_different (bool): 是否优先选择第一个场景已改变的方案。默认为 False。

    Returns:
        dict or None: 返回最优的脚本方案。如果没有输入脚本，则返回 None。
    """
    if not new_video_script:
        return None

    if isinstance(new_video_script, list):
        if not new_video_script:
            return None

        if need_different:
            # 使用一个元组作为排序的key。
            # Python会先按元组的第一个元素排序，如果相同，再按第二个元素排序。
            # 布尔值True在排序时被视为1，False被视为0。
            # reverse=True使得True（1）排在False（0）之前。
            new_video_script.sort(
                key=lambda x: (x.get('第一个场景是否改变', False), x.get('方案整体评分', 0)),
                reverse=True
            )
        else:
            # 原始的排序逻辑，仅按评分排序
            new_video_script.sort(key=lambda x: x.get('方案整体评分', 0), reverse=True)

        return new_video_script[0]

    return new_video_script


@timeit_print
def gen_new_video_by_script(video_path, fused_new_video_script_info, subtitle_box, output_dir):
    """
    生成新视频的文本脚本
    """
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

    final_output_path = os.path.join(output_dir, 'remake.mp4')
    final_with_bgm_path = final_output_path.replace('.mp4', '_with_bgm.mp4')
    final_video_script = choose_script(fused_new_video_script_info, need_different=True)
    need_merge_video_file_list = []

    new_scene_list = final_video_script['场景顺序与新文案']
    for fused_new_scene in new_scene_list:
        scene_start = fused_new_scene.get('scene_start')
        scene_end = fused_new_scene.get('scene_end')
        name_key = f"new_scene_{fused_new_scene.get('new_scene_number')}_original_scene_{fused_new_scene.get('original_scene_number')}"

        new_narration_script_list = fused_new_scene.get('new_narration_script_list', [])
        split_scene_list = process_narration_clips(scene_start, scene_end, new_narration_script_list)
        start_time = time.time()
        count = 0
        for split_scene in split_scene_list:
            count += 1
            name_key_full = f"{name_key}_part{count}"
            need_merge_video_file = process_video_with_owner_text(video_path, split_scene, output_dir, name_key_full, subtitle_box)
            if need_merge_video_file:
                need_merge_video_file_list.append(need_merge_video_file)
        logger.info(f'处理新场景:{name_key} 分割后的场景数量{len(split_scene_list)} 进度: {new_scene_list.index(fused_new_scene) + 1}/{len(new_scene_list)} 耗时: {time.time() - start_time:.2f} 秒')

    merge_videos_ffmpeg(need_merge_video_file_list, output_path=final_output_path)
    tags = final_video_script.get('tags', [])
    bgm_path = get_bgm_path(tags, logger)
    if bgm_path and os.path.exists(bgm_path):
        # logger.info(f"正在为视频添加背景音乐: {bgm_path}")
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
def gen_subtitle_box_and_cover_subtitle(video_path, owner_asr_info, output_dir):
    """
    找到字幕区域并且遮挡字幕
    """
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

    try:
        video_duration = probe_duration(video_path)
        video_duration_ms = int(video_duration * 1000)
    except Exception as e:
        logger.error(f"获取视频时长失败: {e}")
        return None

    time_ranges = []

    duration_list = []
    for asr_info in owner_asr_info:
        final_text = asr_info.get('final_text', '').strip()
        speaker = asr_info.get('speaker', 'unknown')
        if speaker != 'owner':
            continue
        if not final_text:
            continue
        asr_start = asr_info.get('start')
        asr_start = max(0, asr_start-500)
        asr_end = asr_info.get('end')
        asr_end = min(video_duration_ms, asr_end+500)
        duration_list.append((asr_start, asr_end))
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
        logger.info(f"已保存最终字幕框: {final_box_path}")
    final_box = read_json(final_box_path)
    top_left, bottom_right, vid_w, vid_h = adjust_subtitle_box(video_path, final_box)

    cover_video_path = os.path.join(output_dir, 'subtitle_covered.mp4')
    # 获取原始文件的大小，单位是字节
    video_size = os.path.getsize(video_path)

    if is_valid_target_file_simple(cover_video_path, video_size * 0.1):
        logger.info(f"已存在遮挡字幕的视频: {cover_video_path}")
        return cover_video_path, [top_left, bottom_right]

    start_time = time.time()
    logger.info(f"开始生成遮挡字幕视频: {cover_video_path} final_box: {final_box}")
    cover_subtitle(video_path, cover_video_path, top_left, bottom_right, time_ranges=time_ranges)
    if is_valid_target_file_simple(cover_video_path, video_size * 0.1):
        raise ValueError(f"生成遮挡字幕视频失败: {cover_video_path} 文件大小Mb为 {os.path.getsize(cover_video_path) / (1024 * 1024):.2f}，小于原始文件的10%")
    logger.info(f"完成生成遮挡字幕视频: {cover_video_path} 耗时: {time.time() - start_time:.2f} 秒")

    return cover_video_path, [top_left, bottom_right]


@timeit_print
def gen_logical_scene(video_path, output_dir):
    """
    直接根据视频生成逻辑性场景划分
    """
    # sorted_scene_timestamp = get_scene(video_path, output_dir)

    output_file_logical_scene_info_path = os.path.join(output_dir, 'logical_scene_info.json')
    if is_valid_target_file_simple(output_file_logical_scene_info_path, 10):
        logical_scene_info = read_json(output_file_logical_scene_info_path)
    else:
        logical_scene_info = gen_logical_scene_llm(video_path=video_path)
        save_json(output_file_logical_scene_info_path, logical_scene_info)
    return logical_scene_info

def gen_new_video_script_robus(video_path, params={}):
    """
    最多尝试3次生成新的视频方案
    """
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            return gen_new_video_script(video_path, params)
        except Exception as e:
            print(f"尝试 {attempt} 失败: {e} traceback: {traceback.format_exc()}")
            if attempt == max_attempts:
                raise
            else:
                print("正在重试...")
                time.sleep(2)  # 等待一段时间后再重试

def is_contain_owner_speaker(owner_asr_info):
    """
    检查是否包含owner的文本
    """
    for asr_info in owner_asr_info:
        speaker = asr_info.get('speaker', 'unknown')
        final_text = asr_info.get('final_text', '').strip()
        if speaker == 'owner' and final_text:
            return True
    return False

def gen_new_video_script(video_path, params={}):
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

    # 获取场景分割信息
    logical_scene_info = gen_logical_scene(video_path, output_dir)
    logger.info(f"场景逻辑合并完成:数量{len(logical_scene_info.get('new_scene_info'))} 删除的子场景数量:{len(logical_scene_info.get('deleted_scene'))}")


    # 生成asr信息
    owner_asr_info = gen_asr(video_path, output_dir, has_author_voice)

    # # --- 多进程并行执行 ---
    #
    # # 将要执行的函数和它们的参数打包
    # tasks = [
    #     (gen_logical_scene, (video_path, output_dir)),
    #     (gen_asr, (video_path, output_dir, has_author_voice))
    # ]
    #
    # # 核心改动：将 ThreadPoolExecutor 换成 ProcessPoolExecutor
    # with ProcessPoolExecutor(max_workers=2) as executor:
    #     logger.info("开始并行处理场景分割和语音识别 (多进程)...")
    #
    #     # 提交和获取结果的逻辑完全不用变！
    #     futures = [executor.submit(func, *args) for func, args in tasks]
    #     results = [f.result() for f in futures]
    #
    # # 按提交顺序解包结果
    # logical_scene_info, owner_asr_info = results
    # logger.info("场景分割和语音识别均已完成。")
    #
    # # --- 并行执行结束 ---

    # 后续处理
    logger.info(
        f"场景逻辑合并完成:数量{len(logical_scene_info.get('new_scene_info', []))} 删除的子场景数量:{len(logical_scene_info.get('deleted_scene', []))}")


    # 进行新文案的生成
    logger.info("开始生成新视频脚本...")
    # 检查has_author_voice是否包含owner的文本
    has_author_voice = is_contain_owner_speaker(owner_asr_info)
    start_time = time.time()
    new_video_script = gen_video_script(logical_scene_info, owner_asr_info, output_dir, has_author_voice=has_author_voice)
    logger.info(f"新视频脚本生成完成。耗时: {time.time() - start_time:.2f} 秒")

def fuse_all_info(owner_asr_info, final_scene_info, new_video_script_list):
    """
    将所有信息融合到new_video_script中
    """
    for new_video_script in new_video_script_list:
        first_original_scene_number = new_video_script['场景顺序与新文案'][0]['original_scene_number']
        new_video_script['第一个场景是否改变'] = False
        if first_original_scene_number != 1:
            new_video_script['第一个场景是否改变'] = True

        for new_scene in new_video_script['场景顺序与新文案']:
            original_scene_number = new_scene.get('original_scene_number')
            for final_scene in final_scene_info:
                final_scene_number = final_scene.get('scene_number')
                if str(final_scene_number) == str(original_scene_number):
                    new_scene['scene_start'] = final_scene.get('scene_start')
                    new_scene['scene_end'] = final_scene.get('scene_end')
                    break

            new_narration_script_list = new_scene.get('new_narration_script_list', [])
            for new_narration_script in new_narration_script_list:
                source_clip_id = new_narration_script['source_clip_id']
                for asr_info in owner_asr_info:
                    if str(asr_info['source_clip_id']) == str(source_clip_id):
                        new_narration_script['narration_script'] = asr_info.get('final_text', '')
                        new_narration_script['narration_script_start'] = asr_info.get('fix_start')
                        new_narration_script['narration_script_end'] = asr_info.get('fix_end')
                        break
    return new_video_script_list


def correct_owner_timestamps(asr_result: list) -> list:
    """
    对ASR结果列表中speaker为owner的文本时间进行纠正。

    Args:
        asr_result: ASR结果列表。

    Returns:
        带有 'fix_start' 和 'fix_end' 字段的ASR结果列表。
    """
    # 1. 初始化 fix_start 和 fix_end 字段
    for segment in asr_result:
        segment['fix_start'] = segment['start']
        segment['fix_end'] = segment['end']

    # 2. 遍历列表，应用修正逻辑
    for i in range(len(asr_result)):
        current_segment = asr_result[i]

        # 只处理 speaker 为 'owner' 的情况
        if current_segment['speaker'] == 'owner':

            # --- 向前修正逻辑 (修正 start) ---
            # 查看上一个文本
            if i > 0:
                prev_segment = asr_result[i - 1]
                # 如果上一个不是 owner，则尝试移动 start
                if prev_segment['speaker'] != 'owner':
                    gap = current_segment['start'] - prev_segment['end']
                    if gap > 0:
                        # 最多移动500ms
                        movement = min(500, gap)
                        current_segment['fix_start'] = current_segment['start'] - movement

            # --- 向后修正逻辑 (修正 end) ---
            # 查看下一个文本
            if i < len(asr_result) - 1:
                next_segment = asr_result[i + 1]

                # 如果下一个也是 owner
                if next_segment['speaker'] == 'owner':
                    gap = next_segment['start'] - current_segment['end']
                    if gap > 0:
                        if gap < 1000:
                            # 间隔小于1000ms，取中点
                            midpoint = round(current_segment['end'] + gap / 2)
                            current_segment['fix_end'] = midpoint
                            # 注意：这里直接修正了下一个owner的fix_start
                            next_segment['fix_start'] = midpoint
                        else:
                            # 间隔大于等于1000ms，各自移动，但最多500ms
                            # 同时要保证移动后两者间隔至少500ms
                            movement = min(500, (gap - 500) / 2)
                            if movement > 0:
                                current_segment['fix_end'] = round(current_segment['end'] + movement)
                                # 注意：这里直接修正了下一个owner的fix_start
                                next_segment['fix_start'] = round(next_segment['start'] - movement)

                # 如果下一个不是 owner
                else:
                    gap = next_segment['start'] - current_segment['end']
                    if gap > 0:
                        # 最多移动500ms
                        movement = min(500, gap)
                        current_segment['fix_end'] = current_segment['end'] + movement

    return asr_result

def gen_new_video_robus(video_path):
    """
    最多尝试3次生成新的视频
    """
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            return gen_new_video(video_path)
        except Exception as e:
            print(f"尝试 {attempt} 失败: {e} traceback: {traceback.format_exc()}")
            if attempt == max_attempts:
                raise
            else:
                print("正在重试...")
                time.sleep(2)  # 等待一段时间后再重试




def gen_new_video(video_path):
    """
    根据新的方案生成新的视频
    """
    basename = os.path.basename(video_path).split('.mp4')[0]
    output_dir = os.path.join(base_output_dir, basename)
    log_file_path = os.path.join(output_dir, 'log.txt')
    logger = setup_logger(log_file_path)

    output_file_final_path = os.path.join(output_dir, 'new_video_script.json')
    final_scene_info_path = os.path.join(output_dir, 'final_scene_info.json')
    owner_asr_path = os.path.join(output_dir, 'speech_asr_with_owner.json')
    all_files = [output_file_final_path, final_scene_info_path, owner_asr_path]
    fuse_all_info_path = os.path.join(output_dir, 'fuse_all_info.json')
    fixed_owner_asr_path = os.path.join(output_dir, 'fixed_owner_asr.json')

    all_valid = all(is_valid_target_file_simple(f, 10) for f in all_files)
    if not all_valid:
        logger.warning(f"检测到部分输出文件缺失或无效，将重新处理视频: {video_path}")
        return None, None, []

    final_scene_info = read_json(final_scene_info_path)
    new_video_script = read_json(output_file_final_path)
    owner_asr_info = read_json(owner_asr_path)
    fixed_owner_asr_info = correct_owner_timestamps(owner_asr_info)
    save_json(fixed_owner_asr_path, fixed_owner_asr_info)


    logger.info("开始处理字幕区域并生成遮罩...")
    subtitle_video_path, subtitle_box = gen_subtitle_box_and_cover_subtitle(video_path, owner_asr_info, output_dir)
    logger.info("字幕处理完成。")

    logger.info("开始根据新脚本生成最终视频...")
    # 综合三个信息
    fused_new_video_script_info = fuse_all_info(fixed_owner_asr_info, final_scene_info, new_video_script)
    save_json(fuse_all_info_path, fused_new_video_script_info)
    final_video_path, final_video_script = gen_new_video_by_script(subtitle_video_path, fused_new_video_script_info, subtitle_box, output_dir)
    logger.info("最终视频生成完成。")

    return final_video_path, final_video_script



if __name__ == '__main__':
    video_path = '7558521660698119481.mp4'
    # print(check_video_integrity(video_path))

    # gen_new_video_script_robus(video_path)
    gen_new_video_robus(video_path)
