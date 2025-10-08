#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动上传脚本（B 站版）

功能说明
1. 读取权威元数据文件 metadata_cache.json，获取需要处理的全部视频任务。
2. 读取 / 创建 上传日志文件 metadata_cache_with_uploads.json，判断哪些视频已成功投稿。
3. 仅对尚未记录成功投稿的视频执行完整上传流程。
4. 上传成功后，把「权威元数据 + upload_info」写入 / 更新到 metadata_cache_with_uploads.json。
5. 任何情况下都不修改 metadata_cache.json。
"""
import datetime
import hashlib
import json
import os
import copy
import random
import time
import traceback
import datetime as dt

from common_utils.common_utils import get_config, format_seconds_to_mmss, read_json, is_valid_target_file_simple, \
    scan_generated_files
from common_utils.video_scene.combine_asr_scene import gen_new_video_robus
from common_utils.video_scene.combine_asr_scene_online import video_remake
from common_utils.video_utils import add_image_to_video_end, get_video_duration_seconds, create_enhanced_cover, \
    merge_videos_ffmpeg, apply_all_subtle_tweaks, _get_video_resolution, process_video_with_template, probe_duration, \
    add_transparent_watermark
from common_utils.video_utils_cut import text_image_to_video_with_subtitles, gen_ending_video
from content_community.app.remake_video import remake_video_robust

config_map = {}

base_SESSDATA = get_config("bilibili_sessdata_cookie")  # 必需。你的B站登录会话 SESSDATA cookie 值。
base_BILI_JCT = get_config("bilibili_csrf_token")
base_total_cookie = get_config("bilibili_total_cookie")

config_map['base'] = (base_SESSDATA, base_BILI_JCT, base_total_cookie)

# mama_SESSDATA = get_config("mama_bilibili_sessdata_cookie")  # 可选。妈妈账号的 SESSDATA cookie 值。
# mama_BILI_JCT = get_config("mama_bilibili_csrf_token")
# mama_total_cookie = get_config("mama_bilibili_total_cookie")
# config_map['mama'] = (mama_SESSDATA, mama_BILI_JCT, mama_total_cookie)
group_info = {
    'fun': ['ruru', 'jj', 'xiaosu', 'chabian', 'dan', 'yiyi', 'qiqixiao', 'yang',
            'xiaodan', 'qiqixiao', 'dahao', 'lin', 'xiaohao', 'xue', 'jj', 'ruru'
            ],
    'sport': ['nana', 'jun'],
    'game': ['cai', 'tao', 'taoxiao', 'ning', 'xiaoxue', 'yan', 'hong', 'junxiao', 'mama', 'jie', 'qiqi', 'junda', 'ruruxiao']
}


video_recommend_user_list = ["cai","yang","dahao","ruru","yiyi","lin","mama","hong","yan","jie","qiqi","xiaosu","jun","jj","qiqixiao","xiaoxue"]
# 定义需要处理的账号名及其对应的config_map键名（区分大小写）
accounts = {
    'tao': 'tao',
    'taoxiao': 'taoxiao',
    'junxiao': 'junxiao',
    'junda': 'junda',
    'ruru': 'ruru',
    'nana': 'nana',
    'jie': 'jie',
    'qiqi': 'qiqi',
    'mama': 'mama',
    'hong': 'hong',
    # 'su': 'su',
    'yan': 'yan',
    'xue': 'xue',
    'cai': 'cai',
    'jun': 'jun',
    'xiaosu': 'xiaosu',
    'chabian': 'chabian',
    'lin': 'lin',
    'jj': 'jj',
    'hao': 'hao',
    # 'xiaohao': 'xiaohao',
    'dan': 'dan',
    'ning': 'ning',
    'dahao': 'dahao',
    'yang': 'yang',
    'ruruxiao': 'ruruxiao',
    'qiqixiao': 'qiqixiao',
    'yiyi': 'yiyi',
    'xiaodan': 'xiaodan',
    'xiaoxue': 'xiaoxue',

}

for name, map_key in accounts.items():
    sessdata = get_config(f"{name}_bilibili_sessdata_cookie")
    bili_jct = get_config(f"{name}_bilibili_csrf_token")
    total_cookie = get_config(f"{name}_bilibili_total_cookie")
    config_map[map_key] = (sessdata, bili_jct, total_cookie)

error_user_map = {}
from content_community.bilibili.bilibili_uploader import upload_to_bilibili, fetch_bili_topics

# ---------- 文件路径常量 ----------
METADATA_FILE = '../../LLM/TikTokDownloader/back_up/metadata_cache.json'           # 权威源
UPLOAD_LOG_FILE = '../../LLM/TikTokDownloader/back_up/metadata_cache_with_uploads.json'  # 上传日志
persistent_tasks_file = "../../LLM/TikTokDownloader/back_up/persistent_tasks.json"
bvid_file_path = '../../LLM/TikTokDownloader/back_up/bvid_file.json'

# ---------- 工具函数 ----------
def load_json(path: str, default):
    """安全地加载 JSON 文件；不存在或格式错误时返回 default。"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as e:
        print(f"⚠️  警告：文件 {path} JSON 解析失败，原因：{e}。将使用默认值。")
        return default


def get_watermark_path(user_type, user_name):
    """
    生成合适的水印图片路径
    """
    # 获取asset下面所有的图片，并且包含user_type
    asset_dir = 'asset'
    all_files = os.listdir(asset_dir)
    filtered_files = [f for f in all_files if user_type in f and f.endswith('.png')]

    if filtered_files:
        # 对文件列表进行排序，以确保每次执行的顺序一致
        filtered_files.sort()

        # 使用 user_name 的哈希值来计算一个固定的索引
        # 这样同一个 user_name 总是会得到相同的索引
        user_hash_hex = hashlib.sha256(user_name.encode('utf-8')).hexdigest()
        user_hash_int = int(user_hash_hex, 16)
        selected_index = user_hash_int % len(filtered_files)

        selected_file = filtered_files[selected_index]
        watermark_path = os.path.join(asset_dir, selected_file)
        print(f"{user_name} ✅ 使用水印图片 {watermark_path} 筛选池大小 {len(filtered_files)}")
        return watermark_path
    else:
        print(f"⚠️ 未找到符合条件的水印图片，使用默认水印。")
        return 'asset/default_watermark.png'


def _deep_update(orig: dict, new: dict):
    """
    将 new 合并到 orig：
      - 如果某个 key 在 orig 和 new 中对应的 value 都是 dict，则递归合并；
      - 否则直接用 new[key] 覆盖 orig[key]（或新增）。
    """
    for k, v in new.items():
        if k in orig and isinstance(orig[k], dict) and isinstance(v, dict):
            _deep_update(orig[k], v)
        else:
            orig[k] = v


def add_template(video_path, output_video_path, user_name):
    """
    增加框信息，大大提高通过率
    """
    try:
        width, height = _get_video_resolution(video_path)  # 确保视频路径有效

        if width > height:
            image_key = 'height'
            left_up_point = (0, 416)
            box_info = (768, 576)
        else:
            image_key = 'width'
            left_up_point = (416, 0)
            box_info = (576, 768)

        # 随机生成1-3的一个数字
        random_number = str(random.randint(1, 12))
        template_image = f"template_images/{image_key}{random_number}_transparent.png"
        process_video_with_template(input_video=video_path, template_image=template_image, output_video=output_video_path, left_up_point=left_up_point, box_info=box_info)
    except Exception as e:
        print(f"⚠️ 添加模板失败：{e}")
        return False

def save_json(path: str, data):
    """
    1. 确保目录存在
    2. 如果 data 不是 dict，直接写入覆盖
    3. 如果 data 是 dict，则先读已有内容（若不是 dict 则丢弃），深度合并，然后写回
    """
    # 1. 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # 2. 非 dict 直接写入
    if not isinstance(data, dict):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return

    # 3. 尝试加载已有内容
    try:
        with open(path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
            if not isinstance(existing, dict):
                existing = {}
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}

    # 深度合并
    _deep_update(existing, data)

    # 写回
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=4, ensure_ascii=False)


def get_best_plan_by_potential(data: dict) -> dict:
    """根据“爆款潜力指数”选出分值最高的方案。"""
    best_plan, highest_score = None, float('-inf')
    for plan_info in data.values():
        if not isinstance(plan_info, dict):
            continue
        score = plan_info.get("增长潜力", {}).get("爆款潜力指数", 0)
        score = float(score)
        if score > highest_score:
            highest_score, best_plan = score, plan_info
    return best_plan

def time_str_to_seconds(time_str: str) -> int | None:
    """
    将 "HH:MM:SS" 或 "MM:SS" 格式的时间字符串转换为总秒数。

    Args:
        time_str: "时:分:秒" 或 "分:秒" 格式的字符串。

    Returns:
        一个整数，表示总秒数。
        如果输入格式无效，则返回 None。
    """
    try:
        # 确保输入是字符串类型
        if not isinstance(time_str, str):
            raise TypeError("输入必须是字符串")

        parts = time_str.split(':')
        num_parts = len(parts)

        if num_parts == 3:  # 格式为 HH:MM:SS
            h = int(parts[0])
            m = int(parts[1])
            s = int(parts[2])
            # 检查分钟和秒是否在有效范围内（可选，但推荐）
            if m >= 60 or s >= 60:
                raise ValueError("分钟或秒的值不能大于等于60")
            return h * 3600 + m * 60 + s

        elif num_parts == 2:  # 格式为 MM:SS
            m = int(parts[0])
            s = int(parts[1])
            # 检查秒是否在有效范围内（可选）
            if s >= 60:
                raise ValueError("秒的值不能大于等于60")
            return m * 60 + s

        else:
            # 如果部分数量不是2或3，则格式错误
            raise ValueError("时间格式应为 'HH:MM:SS' 或 'MM:SS'")

    except (ValueError, TypeError) as e:
        # 捕获所有可能的错误，例如 int() 转换失败或我们主动抛出的错误
        print(f"错误: 无法解析时间字符串 '{time_str}'。详情: {e}")
        return None

# ---------- 主逻辑 ----------
# ---------- 新增的 imports（放到文件顶部或确保已导入） ----------
import concurrent.futures
import threading
from collections import defaultdict

# ---------- 全局变量（放到模块层） ----------
# 每个账号使用一个单独的 ThreadPoolExecutor(max_workers=1) —— 保证同账号串行上传
account_executors = defaultdict(lambda: concurrent.futures.ThreadPoolExecutor(max_workers=1))

# 保护 upload_log 的并发写入
upload_lock = threading.Lock()

# 全局引用的 upload_log（在 auto_upload 开头会被赋值）
upload_log_global = {}

# ---------- upload_worker：在 per-account executor 中执行的完整上传与后处理逻辑 ----------
def upload_worker(upload_params, key, updated_entry, files_to_cleanup, stage_times, userName):
    """
    后台上传任务（在各自账号的单线程 executor 中运行，保证同账号串行）；
    完整地执行上传重试、结果处理、metadata 更新、临时文件清理与日志持久化。
    参数：
      - upload_params: dict, 传给 upload_to_bilibili 的参数
      - key: str, metadata_cache 的 key
      - updated_entry: dict, 深拷贝的 metadata entry（用于写回 upload_log）
      - files_to_cleanup: list[str|None], 上传成功后要删除的临时文件路径
      - stage_times: dict, 各阶段耗时（worker 会写 '上传'）
      - userName: str, 账号名（用于记录错误 map 等）
    """
    global upload_log_global, error_user_map

    max_retries = 3
    result = None
    t_upload = time.time()

    # 上传重试
    for attempt in range(1, max_retries + 1):
        try:
            result = upload_to_bilibili(**upload_params)
            break
        except Exception as e:
            print(f"❌ 上传接口异常 (第 {attempt} 次重试) user={userName} key={key}：{e} {upload_params}")
            if attempt < max_retries:
                # 等候一会再试（与原逻辑一致）
                time.sleep(60)
            else:
                print("已达最大重试次数，放弃本次上传（后台）。")

    stage_times['上传'] = time.time() - t_upload

    # 上传成功分支
    if result and isinstance(result, dict) and result.get("aid") and result.get("bvid"):
        try:
            print(f"🎉 后台投稿成功！AID={result['aid']}  BVID={result['bvid']} key={key} user={userName} 上传耗时 {stage_times.get('上传', 0):.2f} 秒。")
            # 尝试获取最终视频时长并更新 metadata（和原逻辑保持一致）
            try:
                final_duration_sec = get_video_duration_seconds(upload_params.get("video_path"))
                if final_duration_sec is not None:
                    formatted_duration = format_seconds_to_mmss(final_duration_sec)
                    if 'metadata' in updated_entry and isinstance(updated_entry['metadata'], list) and updated_entry['metadata']:
                        updated_entry['metadata'][0]['duration'] = formatted_duration
                else:
                    print("⚠️ 未能获取最终视频时长，metadata 中的 duration 字段将不被更新。")
            except Exception as e:
                print(f"⚠️ 获取最终视频时长失败：{e}")

            # 删除临时文件（上传成功后清理）
            for p in files_to_cleanup or []:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception as e:
                    print(f"⚠️ 清理文件 {p} 失败：{e}")

        except Exception as e:
            print(f"⚠️ 后台上传后处理异常：{e}")

        # 把 upload_info 写入 updated_entry
        updated_entry["upload_info"] = {
            "upload_params": upload_params,
            "upload_result": result,
        }

        # 安全地写入全局 upload_log 并持久化（加锁）
        with upload_lock:
            upload_log_global[key] = updated_entry
            try:
                save_json(UPLOAD_LOG_FILE, upload_log_global)
                # 打印阶段耗时汇总
                if stage_times:
                    stage_lines = [f"{k}: {v:.2f} 秒" for k, v in stage_times.items()]
                    print(f"✅ 后台上传日志已更新 -> {UPLOAD_LOG_FILE}。阶段耗时：{' | '.join(stage_lines)} {userName} {datetime.datetime.now().isoformat()}")
                else:
                    print(f"✅ 后台上传日志已更新 -> {UPLOAD_LOG_FILE} {userName}.")
            except Exception as e:
                print(f"🔥 后台写入日志文件失败：{e}")

    else:
        # 上传失败：记录 error_user_map，并把错误信息写到 upload_log（加锁）
        err = None
        try:
            err = result.get("message", str(result)) if isinstance(result, dict) else str(result)
        except Exception:
            err = str(result)
        error_user_map[userName] = err or "未知错误"
        print(f"❌ 后台投稿失败 user={userName} key={key}：{err}")
        with upload_lock:
            upload_log_global[key] = upload_log_global.get(key, {})
            upload_log_global[key]['status'] = 'error'
            upload_log_global[key]['error_message'] = err
            try:
                save_json(UPLOAD_LOG_FILE, upload_log_global)
            except Exception as e:
                print(f"🔥 后台写入失败（失败记录）：{e}")


# ---------- auto_upload：完整实现（主线程做预处理，上传按账号串行提交） ----------
import os
import time
import copy
import traceback
import concurrent.futures
from typing import Dict, Any, Tuple, List, Set

# -------------- 注意 --------------
# 下面的代码依赖你原有模块里的函数与全局变量（不要删改）：
# load_json, save_json, remake_video_robust, add_image_to_video_end, apply_all_subtle_tweaks,
# merge_videos_ffmpeg, _get_video_resolution, text_image_to_video_with_subtitles,
# add_template, create_enhanced_cover, fetch_bili_topics, time_str_to_seconds,
# account_executors, upload_worker, config_map, error_user_map,
# METADATA_FILE, UPLOAD_LOG_FILE, persistent_tasks_file
# ---------------------------------

# 我们在模块级保留 upload_log_global 的声明，和原来行为一致（worker 使用时仍受 upload_lock 保护）
upload_log_global = {}

def _load_metadata_and_log() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """加载 metadata 与 upload_log，并设置 upload_log_global（与原脚本行为一致）"""
    global upload_log_global
    metadata_cache: dict = load_json(METADATA_FILE, default={})
    upload_log: dict = load_json(UPLOAD_LOG_FILE, default={})
    upload_log_global = upload_log
    return metadata_cache, upload_log

def _basic_task_checks(key: str, value: Dict[str, Any], video_id_key) -> Tuple[bool, str]:
    """
    基本合法性检查。返回 (should_skip(bool), reason(str))。
    不做任何副作用，只用于上游决定是否跳过。
    """
    status = value.get('status', '未处理')
    if status != 'complete':
        return True, f"⚠️ 跳过 {key}：当前状态为{status} 不为 complete"
    if video_id_key in upload_log_global and upload_log_global[video_id_key].get("upload_info"):
        return True, f"⏭️ 跳过 {key}：已记录上传成功"
    metadata = value.get('metadata')
    if not (isinstance(metadata, list) and metadata):
        return True, f"⏭️ 跳过 {key}：metadata 字段缺失或格式错误。{metadata}"
    video_id = metadata[0].get('id')
    if not video_id:
        return True, f"⏭️ 跳过 {key}：metadata 中缺少 id。"
    video_path = value.get('video_path')
    if not video_path or not os.path.exists(video_path):
        return True, f"⏭️ 跳过 {key} (ID: {video_id})：视频文件缺失 -> {video_path}"

    best_scheme = value.get('best_scheme') or get_best_plan_by_potential(value.get('title_schemes', {}))
    if not best_scheme:
        return True, f"⏭️ 跳过 {key}：无法选取投稿方案。"
    return False, ""

def _select_config_for_user(userName: str) -> Tuple[str, Any]:
    """
    选择 config（保留原逻辑：若 userName 不在 config_map，回退到 'base' 并打印提示）
    返回实际使用的 userName 与 config
    """
    if userName in error_user_map:
        return "error_user", None
    if userName not in config_map.keys():
        print(f"⚠️ 跳过 {userName} 用户上传 请检查配置数据。")
        userName = 'base'
    config = config_map.get(userName, config_map['base'])
    return userName, config

def _preprocess_media_steps(
    key: str,
    value: Dict[str, Any],
    best_scheme: Dict[str, Any],
    userName: str
):
    """
    执行视频 / 封面等一系列预处理步骤（复刻原逻辑的顺序与异常处理）。
    返回：
      - video_path: 最终用于上传的视频路径
      - cover_path: 最终用于上传的封面路径
      - stage_times: 每一步耗时字典
      - files_to_cleanup: 可能需要清理的临时文件列表（与原脚本相同内容）
    """
    stage_times: Dict[str, float] = {}
    metadata = value.get('metadata', [])
    generation_options = value.get('generation_options', {}) or {}

    # 初始设定（和原脚本保持一致）
    video_path = value.get('video_path')
    current_video_path = video_path
    duration = metadata[0].get('duration', "00:10")
    duration = time_str_to_seconds(duration)
    full_title = metadata[0].get('full_title', "00:10")

    # 初始化可能产生的临时路径变量（与原脚本一致名称）
    new_video_path = None
    temp_video_path = None
    tweak_video_path = None
    addPrologue_video_path = None
    template_video_path = None
    # --------- 重制视频分支（原脚本里始终 False） ---------
    # 保留原来判断（即：永远不会执行），以确保逻辑一致
    if generation_options.get('is_original', False) and duration < 600:
        t0 = time.time()
        has_author_voice = generation_options.get('has_author_voice', True)
        # 反转has_author_voice
        no_owner = not has_author_voice
        creative_guidance = generation_options.get('creative_guidance', '')
        print(f"🔄 重制视频 {video_path}... userName: {userName} 是否不包含作者语音{no_owner} 创作指导：{creative_guidance} 视频名称：{full_title} duration{duration}")
        try:

            final_video_path, final_video_script = gen_new_video_robus(video_path)
            if is_valid_target_file_simple(final_video_path):
                print(f"✅ 重制视频成功，保存为 {final_video_path}")
                video_path = final_video_path
                if has_author_voice:
                    title = final_video_script.get('title')
                    best_scheme['标题'] = title if title else best_scheme.get('标题', '欢迎来看我的视频！')
                    cover_text = final_video_script.get('cover_text')
                    best_scheme['封面']['配文'] = cover_text if cover_text else best_scheme.get('封面', {}).get('配文', '欢迎来看我的视频！')
            else:
                upload_log_global[key] = upload_log_global.get(key, {})
                upload_log_global[key]['status'] = 'error'
                save_json(UPLOAD_LOG_FILE, upload_log_global)
                print(f"❌ 重制视频失败")
            stage_times['重制视频'] = time.time() - t0
        except Exception as e:
            stage_times['重制视频'] = time.time() - t0
            upload_log_global[key] = upload_log_global.get(key, {})
            upload_log_global[key]['status'] = 'error'
            save_json(UPLOAD_LOG_FILE, upload_log_global)
            print(f"❌ 重制视频失败：{e}")

    # # ---------- 预处理：在尾部插入引导图片 ----------
    # new_video_path = current_video_path.replace('.mp4', '_new.mp4')
    # if duration < 600:
    #     try:
    #         t0 = time.time()
    #         image_duration = int(duration / 100)
    #         image_duration = max(1, image_duration)
    #         print(f"🔄 尾部插图处理：视频时长 {duration} 秒，插图持续 {image_duration} 秒。 文件路径：{current_video_path} -> {new_video_path}")
    #         final_jpg_path = f'{userName}_final.jpg'
    #         if not os.path.exists(final_jpg_path):
    #             final_jpg_path = 'final.jpg'
    #             print(f"⚠️ 尾部插图文件 {final_jpg_path} 不存在，使用默认图片。")
    #         add_image_to_video_end(current_video_path, final_jpg_path, new_video_path, image_duration)
    #         video_path = new_video_path
    #         stage_times['尾部插图'] = time.time() - t0
    #     except Exception as e:
    #         stage_times['尾部插图'] = time.time() - t0
    #         print(f"⚠️ 尾部插图失败，继续使用原视频：{e}")

    # # 视频细节调整（当 duration < 600）
    # if duration < 6000:
    #     tweak_video_path = video_path.replace('.mp4', '_tweaked.mp4')
    #     try:
    #         t0 = time.time()
    #         result = apply_all_subtle_tweaks(video_path, output_path=tweak_video_path)
    #         if os.path.exists(tweak_video_path) and result and os.path.getsize(tweak_video_path) > 0:
    #             video_path = tweak_video_path
    #             print(f"✅ 视频细节调整成功，保存为 {tweak_video_path}")
    #         else:
    #             print(f"❌ 视频细节调整失败，继续使用原视频。")
    #         stage_times['视频细节调整'] = time.time() - t0
    #     except Exception as e:
    #         stage_times['视频细节调整'] = time.time() - t0
    #         print(f"⚠️ 视频细节调整失败：{e}")

    # # 添加结尾片段
    # temp_video_path = video_path.replace('.mp4', '_temp.mp4')
    # try:
    #     if generation_options.get('add_epilogue', False):
    #         t0 = time.time()
    #         print(f"🔄 添加结尾视频片段到 {video_path}... userName: {userName}")
    #         copyright_video_path = f'{userName}_final.mp4'
    #         if not os.path.exists(copyright_video_path):
    #             copyright_video_path = 'final.mp4'
    #             print(f"⚠️ 版权视频文件 {copyright_video_path} 不存在，使用默认视频。")
    #         video_path_list = [video_path, copyright_video_path]
    #         merge_videos_ffmpeg(video_path_list, output_path=temp_video_path)
    #         if os.path.exists(temp_video_path) and os.path.getsize(temp_video_path) > 0:
    #             video_path = temp_video_path
    #             print(f"✅ 合并视频成功，保存为 {temp_video_path}")
    #         stage_times['添加结尾片段'] = time.time() - t0
    # except Exception as e:
    #     stage_times['添加结尾片段'] = time.time() - t0
    #     print(f"⚠️ 合并视频失败：{e}")

    # 封面路径选择（遵循原脚本判断优先级）
    cover_path = (
        metadata[0].get('abs_cover_path') if os.path.exists(metadata[0].get('abs_cover_path', ''))
        else best_scheme.get('封面', {}).get('图片路径', 'default_cover.jpg')
    )
    is_duplicate = value.get('is_duplicate', False)
    if is_duplicate:
        cover_path = (
            best_scheme.get('封面', {}).get('图片路径', 'default_cover.jpg') if os.path.exists(
                best_scheme.get('封面', {}).get('图片路径', 'default_cover.jpg'))
            else metadata[0].get('abs_cover_path')
        )
        print(f"⚠️ 重复视频，使用方案封面 {cover_path}。")

    # 添加开场白
    addPrologue_video_path = video_path.replace('.mp4', '_prologue.mp4')
    if generation_options.get('add_prologue', False):
        print(f"🔄 添加开场白到 {video_path}... userName: {userName}")
        try:
            t0 = time.time()
            width, height = _get_video_resolution(video_path)
            resolution = (width, height)
            addPrologueStr = best_scheme.get('开场白', {}).get('脚本', '')
            if not addPrologueStr:
                print(f"⚠️ 开场白脚本为空，跳过添加开场白。")

            temp_video_path = text_image_to_video_with_subtitles(text=addPrologueStr, image_path=cover_path, output_path=addPrologue_video_path, resolution=resolution)
            if os.path.exists(temp_video_path):
                print(f"✅ 添加开场白成功，保存为 {temp_video_path}")
                merge_videos_ffmpeg([temp_video_path, video_path], output_path=addPrologue_video_path)
                if os.path.exists(addPrologue_video_path) and os.path.getsize(addPrologue_video_path) > 0:
                    video_path = addPrologue_video_path
                    print(f"✅ 合并开场白视频成功，保存为 {addPrologue_video_path}")
            stage_times['添加开场白'] = time.time() - t0
        except Exception as e:
            stage_times['添加开场白'] = time.time() - t0
            print(f"⚠️ 添加开场白失败：{e}")

    # 添加模板
    template_video_path = video_path.replace('.mp4', '_template.mp4')
    if generation_options.get('need_template', False):
        try:
            t0 = time.time()
            add_template(video_path, template_video_path, userName)
            if os.path.exists(template_video_path) and os.path.getsize(template_video_path) > 0:
                video_path = template_video_path
                print(f"✅ 添加模板成功，保存为 {template_video_path}")
            stage_times['添加模板'] = time.time() - t0
        except Exception as e:
            stage_times['添加模板'] = time.time() - t0
            print(f"⚠️ 添加模板失败：{e}")

    # 封面增强处理
    try:
        t0 = time.time()
        output_image_path = cover_path.replace('.jpg', '_enhanced.jpg')
        create_enhanced_cover(
            input_image_path=cover_path,
            output_image_path=output_image_path,
            text_lines=[best_scheme.get('封面', {}).get('配文', '')],
        )
        cover_path = output_image_path if os.path.exists(output_image_path) else cover_path
        stage_times['封面处理'] = time.time() - t0
    except Exception as e:
        stage_times['封面处理'] = time.time() - t0
        traceback.print_exc()
        print(f"⚠️  封面处理失败：{e}")

    return video_path, cover_path, stage_times

def _build_upload_params(
    metadata_entry: Dict[str, Any],
    best_scheme: Dict[str, Any],
    cover_path: str,
    video_path: str,
    config: Any,
    userName: str
) -> Dict[str, Any]:
    """基于 best_scheme 与 metadata 生成 upload_params（保留原逻辑）"""
    metadata = metadata_entry.get('metadata', [])
    origin_tag = metadata[0].get('tag', [])
    if userName in video_recommend_user_list:
        origin_tag.insert(0, 'B站好片有奖种草')  # 放最前面
    origin_tag.extend(metadata[0].get('text_extra', []))

    title = best_scheme.get('标题', '欢迎来看我的视频！')
    if len(title) > 80:
        title = title[:70]
        print(f"⚠️ 标题过长，已截断为：{title}")

    human_type2 = best_scheme.get('分区编号', 21)
    topic_json = fetch_bili_topics(config[2], type_pid=human_type2)
    topic_name = '骑行去追夏天的风'  # 默认话题名称
    topic_id = 1313687  # 默认话题ID
    topic_detail = {
        "from_topic_id": 1313687,
        "from_source": "arc.web.recommend",
        'topic_name': '骑行去追夏天的风'
    }
    # 尝试获取话题
    if isinstance(topic_json, dict) and 'data' in topic_json:
        topics = topic_json.get('data', {}).get('topics', [])
        if topics:
            topic_id = topics[0].get('topic_id', human_type2)
            topic_name = topics[0].get('topic_name', '骑行去追夏天的风')
            topic_detail['from_topic_id'] = topic_id
            topic_detail['topic_name'] = topic_name
    else:
        print(f"⚠️ 获取分区 {human_type2} 的话题失败，使用默认值。{topic_json}")

    description_json = best_scheme.get('简介', {})
    target_keys = ["核心看点", "价值承诺", "互动引导", "补充信息"]
    description = "\n".join(
        str(description_json[k]) for k in target_keys if k in description_json
    )
    tags = best_scheme.get('标签', ['AI修复', '视频剪辑'])
    origin_tag.extend(tags)
    tags = list(set(origin_tag))
    tags = [tag for tag in tags if len(tag) <= 18]
    tags = tags[:12]
    tags_str = ",".join(tags) if isinstance(tags, list) else str(tags)
    dynamic = best_scheme.get('简介', {}).get('互动引导', '希望大家喜欢')

    upload_params = {
        "title": title,
        "description": description,
        "tags": tags_str,
        "dynamic": dynamic,
        "cover_path": cover_path,
        "video_path": video_path,
        "sessdata": config[0],
        "bili_jct": config[1],
        "human_type2": human_type2,
        "topic_detail": topic_detail,
        "topic_id": topic_id,
    }
    return upload_params

def full_video_info(video_info):
    """
    将视频信息补充完整，主要是互动信息和标题信息
    """
    hudong_path = video_info.get('hudong_path')
    if is_valid_target_file_simple(hudong_path):
        hudong_info = read_json(hudong_path)
        video_info['hudong'] = hudong_info
    titles_path = video_info.get('titles_path')
    if is_valid_target_file_simple(titles_path):
        titles_info = read_json(titles_path)
        video_info['title_schemes'] = titles_info
    return video_info

def gen_clean_files(video_path_list):
    """
    根据 video_path_list 生成需要清理的文件列表
    """
    # 遍历视频
    cleaner_file_list = []
    file_names = ['hudong.json', 'title_schemes.json', 'new_video_script.json', 'final_scene_info.json', 'speech_asr_with_owner.json', 'log.txt', 'logical_scene_info.json']

    all_files = []
    for video_path in video_path_list:
        # 获取目录
        dir_name = os.path.dirname(video_path)
        file_name = os.path.basename(video_path)
        file_names.append(file_name)
        all_sub_files = scan_generated_files(dir_name)
        all_files.extend(all_sub_files)
    # 对all_files进行去重
    all_files = list(set(all_files))

    # 剔除文件名是file_names中的文件
    for f in all_files:
        if os.path.basename(f) not in file_names:
            cleaner_file_list.append(f)
    print(f"🧹 生成清理文件列表，共 {len(cleaner_file_list)} 个文件。")
    return cleaner_file_list





def auto_upload():
    """
    非阻塞版 auto_upload（主线程负责预处理，投稿提交到每个账号的单线程 executor）：
    - 保留并执行原脚本的全部预处理逻辑
    - 在生成 upload_params 后，使用 account_executors[userName].submit(...) 提交 upload_worker，
      以确保同一用户同一时刻只会有一个上传任务在运行。
    """
    global upload_log_global



    temp_set: Set[str] = set()  # 临时集合，记录被跳过/需持久化的任务
    metadata_cache, upload_log = _load_metadata_and_log()
    upload_log_global = upload_log  # 赋值全局，worker 会在后台修改（受 upload_lock 保护）

    if not metadata_cache:
        print(f"❌ 无可用任务：{METADATA_FILE} 为空或不存在。")
        return

    if not upload_log and (len(metadata_cache) - len(upload_log)) > 20:
        print(f"❌ 无可用任务：{UPLOAD_LOG_FILE} 为空或不存在。")
        return

    futures = []  # 保存提交到各账号 executor 的 future（如需等待可用）
    new_uploads_made = False
    error_count = 0
    processed_video_id = []
    # 遍历所有权威元数据任务
    for key, value in metadata_cache.items():
        if key in processed_video_id:
            continue
        userName = value.get('userName', 'other')
        today_start = dt.datetime.combine(dt.date.today(), dt.time.min).timestamp()



        start_time = time.time()
        best_score_max = float('-inf')
        should_skip = False
        # 先把一些变量初始化（与原脚本一致）
        updated_entry = copy.deepcopy(value)
        updated_entry = full_video_info(updated_entry)

        video_id_list = value.get('video_id_list', [key])
        # 排序video_id_list
        video_id_list = sorted(video_id_list)
        video_id_key = '_'.join(video_id_list)
        processed_video_id.extend(video_id_list)

        for video_id in video_id_list:
            value_info = metadata_cache.get(video_id, {})
            value_info = full_video_info(value_info)
            # 基本检查（同原逻辑）
            should_skip, reason = _basic_task_checks(video_id, value_info, video_id_key)
            if should_skip:
                if '已记录上传成功' in reason:
                    continue
                # 某些检查需要增加 error_count 或打印额外信息（与原脚本行为一致）
                if '之前处理失败' in reason:
                    print(f"{reason} {userName}")
                    error_count += 1
                    break
                else:
                    print(f"{reason} {userName}")
                    # 如果是 metadata 格式错误或缺少 id 或视频不存在等情况，直接 continue
                    break

        if should_skip:
            continue


        metadata = value.get('metadata')
        if userName in error_user_map:
            print(f"⚠️ 跳过 {userName} 用户上传：之前上传失败，错误信息：{error_user_map[userName]}")
            error_count += 1
            continue

        # 选择 config（保留原逻辑）
        if userName not in config_map.keys():
            print(f"⚠️ 跳过 {userName} 用户上传 请检查配置数据。")
            userName = 'base'
            continue
        config = config_map.get(userName, config_map['base'])

        try:
            bvid_file_data = read_json(bvid_file_path)
        except Exception as e:
            print(f"❌ 读取 {bvid_file_path} 失败：{e}")
            bvid_file_data = {}
        user_videos = bvid_file_data.get(userName, [])
        recent_videos = [v for v in user_videos if v.get('created') and v['created'] >= today_start]
        print(f"🔍 处理 {key} (用户: {userName}) 今日已上传 {len(recent_videos)} 个视频，时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
        if len(recent_videos) >= 20:
            print(f"⚠️ 跳过 {userName} 用户上传：今日已上传 {len(recent_videos)} 个视频，达到上限。")
            continue
        video_path_list = []
        origin_video_path_list = []
        best_scheme_final = None
        best_cover_path = None
        comment_list_all = []
        for video_id in video_id_list:
            video_info = metadata_cache.get(video_id, {})
            video_info = full_video_info(video_info)
            origin_video_path_list.append(video_info.get('video_path'))
            comment_list = video_info.get('hudong', {}).get('comment_list', [])
            comment_list_all.extend(comment_list)
            # 选择最佳投稿方案
            best_scheme = value.get('best_scheme') or get_best_plan_by_potential(video_info.get('title_schemes', {}))
            if not best_scheme:
                print(f"⏭️ 跳过 {key}：无法选取投稿方案。")
                temp_set.add(key)
                continue

            score = best_scheme.get("增长潜力", {}).get("爆款潜力指数", 0)
            score = float(score)


            print(f"\n⏳ {userName} 开始处理任务 {key}，视频ID列表：{video_id_list}，时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
            # 执行一系列的媒体预处理与封面处理
            try:
                video_path, cover_path, stage_times = _preprocess_media_steps(video_id, video_info, best_scheme, userName)
                if score > best_score_max:
                    best_score_max = score
                    best_scheme_final = best_scheme
                    # 将video_path插入video_path_list第一个
                    best_cover_path = cover_path
                    video_path_list.insert(0, video_path)
                else:
                    video_path_list.append(video_path)
            except Exception as e:
                # 严格保持原脚本在异常处的行为：记录错误并继续
                print(f"⚠️ 处理媒体过程中出现异常：{e} {video_id} {userName}")
                traceback.print_exc()
                error_count += 1
                break
        final_output_path = video_path.replace('.mp4', '_final.mp4')
        # 将comment_list_all 按照第二个元素降序排序
        comment_list_all = sorted(comment_list_all, key=lambda x: x[1], reverse=True)
        comment_list_all = comment_list_all[:30]
        updated_entry['hudong']['comment_list'] = comment_list_all

        merge_videos_ffmpeg(video_path_list, output_path=final_output_path)
        if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
            duration = probe_duration(final_output_path)

            # ---------- 预处理：在尾部插入引导视频 ----------
            new_video_path = final_output_path.replace('.mp4', '_new.mp4')
            temp_ending_video_path = final_output_path.replace('.mp4', '_ending.mp4')
            if duration < 6000:
                try:
                    origin_ending_video_path = "origin_ending_video.mp4"
                    ending_text = best_scheme_final.get('简介', {}).get('结尾语', '感谢观看本视频，欢迎点赞、评论、关注、投币、分享！')
                    gen_ending_video(ending_text, temp_ending_video_path, origin_ending_video_path)
                    merge_videos_ffmpeg([final_output_path, temp_ending_video_path], output_path=new_video_path)

                    final_output_path = new_video_path
                except Exception as e:
                    print(f"⚠️ 尾部引导视频失败，继续使用原视频：{e}")


        # 进行水印的增加

        try:
            output_watermark_path = final_output_path.replace('.mp4', '_watermark.mp4')
            user_type = 'fun'
            for group, users in group_info.items():
                if userName in users:
                    user_type = group
                    break
            start_time = time.time()
            watermark_path = get_watermark_path(user_type, userName)
            add_transparent_watermark(final_output_path, watermark_path, output_watermark_path)
            if os.path.exists(output_watermark_path) and os.path.getsize(output_watermark_path) > 0:
                print(f"✅ 水印增加成功，保存为 {output_watermark_path} 耗时 {time.time() - start_time:.2f} 秒")
                final_output_path = output_watermark_path
        except Exception as e:
            print(f"⚠️ 水印增加失败，继续使用原视频：{e}")

        # 构建上传参数（title/description/tags/topic 等）
        upload_params = _build_upload_params(value, best_scheme_final, best_cover_path, final_output_path, config, userName)

        # ---------- 非阻塞提交上传（按账号串行） ----------
        print(f"🚀 准备为用户 {userName} 后台投稿 {key} (ID: {video_id_key}) - 《{upload_params.get('title')}》（按账号串行）")

        task_stage_times = dict(stage_times)

        # 获取该账号的 executor（默认每账号单线程）
        account_executor = account_executors[userName]
        # 尝试移除all_files_to_cleanup中的video_path

        all_files_to_cleanup = gen_clean_files(origin_video_path_list)

        future = account_executor.submit(upload_worker, upload_params, video_id_key, updated_entry, all_files_to_cleanup, task_stage_times, userName)
        futures.append(future)
        new_uploads_made = True

        # 主循环继续处理下一条任务（不等待上传完成）

    # 如果需要在一次运行结束前等待所有后台上传完成，可取消下面注释：
    print("等待所有后台上传完成...")
    concurrent.futures.wait(futures, timeout=None)

    # 处理被跳过的 persistent tasks
    if len(temp_set) > 0:
        print(f"⚠️ 跳过了 {len(temp_set)} 个任务：{', '.join(temp_set)}")
        persistent_tasks = load_json(persistent_tasks_file, default={})
        persistent_tasks = set(persistent_tasks)
        persistent_tasks.update(temp_set)
        save_json(persistent_tasks_file, list(persistent_tasks))

    print(f"错误数量为{error_count}  全部任务处理完毕。时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")





# ---------- CLI ----------
if __name__ == "__main__":
    while True:
        auto_upload()
        time.sleep(60)  # 每小时运行一次