#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动上传脚本（B 站版）- 重构版

功能说明（保持原逻辑不变）
1. 读取权威元数据文件 metadata_cache.json，获取需要处理的全部视频任务。
2. 读取 / 创建 上传日志文件 metadata_cache_with_uploads.json，判断哪些视频已成功投稿。
3. 仅对尚未记录成功投稿的视频执行完整上传流程。
4. 上传成功后，把「权威元数据 + upload_info」写入 / 更新到 metadata_cache_with_uploads.json。
5. 任何情况下都不修改 metadata_cache.json。

重构要点
- 将“视频处理流水线（预处理/合并/结尾/水印）”封装为 process_video_batch(...)，并实现断点续跑：
  - 若存在 _final.mp4 则跳过合并；
  - 若存在 _new.mp4 则跳过末尾拼接；
  - 若存在 _watermark.mp4 则跳过水印；
  - 若封面增强 _enhanced.jpg 已存在，则跳过再次生成。
- 保持上传前参数构建、上传重试、日志更新、清理与节流策略不变。
"""
import concurrent.futures
import datetime
import hashlib
import json
import os
import threading
import time
import traceback
import shutil
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from common_utils.common_utils import (
    get_config,
    format_seconds_to_mmss,
    read_json,
    is_valid_target_file_simple,
    scan_generated_files,
    ms_to_time, init_config,
)
from common_utils.video_scene.combine_asr_scene import gen_new_video_robus
from common_utils.video_utils import (
    get_video_duration_seconds,
    create_enhanced_cover,
    merge_videos_ffmpeg,
    probe_duration,
    add_transparent_watermark,
)
from common_utils.video_utils_cut import gen_ending_video
from content_community.bilibili.bilibili_uploader import upload_to_bilibili, fetch_bili_topics

# ---------- 全局配置 ----------
config_map: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]] = {}

# 基础账号（base）
base_SESSDATA = get_config("bilibili_sessdata_cookie")
base_BILI_JCT = get_config("bilibili_csrf_token")
base_total_cookie = get_config("bilibili_total_cookie")
config_map["base"] = (base_SESSDATA, base_BILI_JCT, base_total_cookie)


# 题材分组
group_info = {
    "fun": [
        "ruru",
        "chabian",
        "dan",
        "yiyi",
        "xiaodan",
        "xiaohao",
        "xue",
        "ruru",
        "xiaosu",
        "shun",
        "ping"
    ],
    "sport": ["nana", "jun", "xiaomu"],
    "game": [
        "cai",
        "tao",
        "taoxiao",
        "ning",
        "xiaoxue",
        "yan",
        "hong",
        "junxiao",
        "mama",
        "jie",
        "qiqi",
        "junda",
        "ruruxiao",
        "qiqixiao",
        "mu",
        "xiaocai",
        "qizhu",
        "dahao",
        "zhong",
        "jj",
        "lin",
        "yang"
    ],
}
fun_user_list = []

right_now_user_list = ["qiqixiao", "jie"]
video_recommend_user_list = ['nana', 'taoxiao', 'hong', 'jie', 'cai', 'xiaosu', 'jun', 'dan', 'ning', 'yiyi', 'qiqixiao', 'xiaoxue', 'shun', 'ping']

# 错误记录
error_user_map: Dict[str, str] = {}

# ---------- 文件路径常量 ----------
METADATA_FILE = "../../LLM/TikTokDownloader/back_up/metadata_cache.json"  # 权威源
UPLOAD_LOG_FILE = "../../LLM/TikTokDownloader/back_up/metadata_cache_with_uploads.json"  # 上传日志
USER_UPLOADS_INFO_FILE = "../../LLM/TikTokDownloader/back_up/user_uploads_info.json"  # 用户上传统计
persistent_tasks_file = "../../LLM/TikTokDownloader/back_up/persistent_tasks.json"
bvid_file_path = "../../LLM/TikTokDownloader/back_up/bvid_file.json"
SKIPPED_CANDIDATES_FILE = "../../LLM/TikTokDownloader/back_up/skipped_candidates.json"

# ---------- 并发与日志 ----------
# 每个账号使用一个单独的 ThreadPoolExecutor(max_workers=1) —— 保证同账号串行上传
account_executors: Dict[str, concurrent.futures.ThreadPoolExecutor] = defaultdict(
    lambda: concurrent.futures.ThreadPoolExecutor(max_workers=1)
)
# 保护 upload_log 的并发写入
upload_lock = threading.Lock()
# 全局引用的 upload_log（在 auto_upload 开头会被赋值）
upload_log_global: Dict[str, Any] = {}


# ---------- 工具函数 ----------
def load_json(path: str, default: Any) -> Any:
    """安全地加载 JSON 文件；不存在或格式错误时返回 default。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as e:
        print(f"⚠️  警告：文件 {path} JSON 解析失败，原因：{e}。将使用默认值。")
        return default


def save_json(path: str, data: Any) -> None:
    """
    保存 JSON：
    1. 确保目录存在
    2. 如果 data 不是 dict，直接写入覆盖
    3. 如果 data 是 dict，则先读已有内容（若不是 dict 则丢弃），深度合并，然后写回
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if not isinstance(data, dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return

    # 读取已有内容
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
            if not isinstance(existing, dict):
                existing = {}
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}

    # 深度合并
    _deep_update(existing, data)

    # 写回
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4, ensure_ascii=False)


def _deep_update(orig: Dict[str, Any], new: Dict[str, Any]) -> None:
    """字典深度合并：dict 值递归合并，其他覆盖。"""
    for k, v in new.items():
        if k in orig and isinstance(orig[k], dict) and isinstance(v, dict):
            _deep_update(orig[k], v)
        else:
            orig[k] = v


# ---------- 新增的安全保存逻辑 ----------

def local_safe_save_log():
    """
    本地实现的原子保存函数。
    直接将内存中的 upload_log_global 写入磁盘，不读取旧文件，防止读到空文件。
    使用写临时文件 + 替换的方式，防止断电/崩溃导致文件损坏。
    """
    global upload_log_global

    # 1. 写入临时文件
    dir_name = os.path.dirname(os.path.abspath(UPLOAD_LOG_FILE))
    os.makedirs(dir_name, exist_ok=True)

    try:
        # delete=False 保证文件关闭后不会自动删除，以便我们手动移动
        with tempfile.NamedTemporaryFile(mode='w', dir=dir_name, delete=False, encoding='utf-8') as tmp:
            json.dump(upload_log_global, tmp, indent=4, ensure_ascii=False)
            tmp_path = tmp.name

        # 2. 原子替换（这一步极快，不容易出错）
        shutil.move(tmp_path, UPLOAD_LOG_FILE)

    except Exception as e:
        print(f"🔥 [严重错误] 日志写入失败: {e}")
        # 如果写入失败，尝试清理临时文件
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass


def update_log_status(key: str, updates: Dict[str, Any]):
    """
    线程安全的更新并保存日志。
    替代代码中零散的 upload_log_global[key] = ... 和 save_json 调用。
    """
    with upload_lock:
        # 1. 更新内存
        if key not in upload_log_global:
            upload_log_global[key] = {}

        # 简单的深度合并或直接更新
        _deep_update(upload_log_global[key], updates)

        # 2. 调用上面的安全保存
        local_safe_save_log()


def analyze_user_uploads_by_day(
        metadata_cache_with_uploads: Any,
        metadata_cache: Any  # 原始投稿信息
) -> Dict[str, Dict[str, Any]]:
    """
    汇总每个用户【当天】的投稿数量与状态信息。
    所有统计数据都仅限于当天（从今日0点到现在）。

    新增统计 (来自 metadata_cache):
    - total_count_today: 用户今日总投稿数
    - error_count_today: 今日失败的投稿数 (status='error')
    - unprocessed_count_today: 今日等待处理的投稿数 (status!='error' 且 key 未在 metadata_cache_with_uploads 中)

    原有统计 (来自 metadata_cache_with_uploads):
    - uploads_today: 当天已处理的投稿数
    - uploads_last_hour: 近1小时已处理的投稿数
    - latest_upload_time: 最近已处理的投稿时间
    - latest_timestamp: 最近已处理投稿的Unix时间戳
    """
    stats_result: Dict[str, Dict[str, Any]] = {}

    # --- 1. 设置时间范围并标准化输入 ---
    now = datetime.datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    one_hour_ago = now.timestamp() - 3600

    if isinstance(metadata_cache, list) and len(metadata_cache) > 0:
        metadata_cache = metadata_cache[0]
    if not isinstance(metadata_cache, dict):
        metadata_cache = {}

    if isinstance(metadata_cache_with_uploads, list) and len(metadata_cache_with_uploads) > 0:
        metadata_cache_with_uploads = metadata_cache_with_uploads[0]
    if not isinstance(metadata_cache_with_uploads, dict):
        metadata_cache_with_uploads = {}

    # --- 辅助函数：为用户初始化统计字典，包含所有新旧字段 ---
    def get_default_stats():
        return {
            # 新增字段
            "total_count_today": 0,
            "error_count_today": 0,
            "unprocessed_count_today": 0,
            # 原始字段
            "uploads_today": 0,
            "uploads_last_hour": 0,
            "latest_upload_time": "无记录",
            "latest_timestamp": 0.0,
        }

    upload_log_keys = set(metadata_cache_with_uploads.keys())
    # --- 2. 从原始投稿(metadata_cache)计算【当天】的总数、失败数、未处理数 ---
    for key, data in metadata_cache.items():
        try:
            ts = float(data.get("timestamp", 0))
            if ts < today_start:
                continue
        except (ValueError, TypeError):
            continue

        user_name = data.get("userName")
        if not user_name:
            continue

        user_stats = stats_result.setdefault(user_name, get_default_stats())

        user_stats["total_count_today"] += 1

        is_contain = False
        # 判断key是否被包含在upload_log_keys的元素中
        for log_key in upload_log_keys:
            if key in log_key:
                is_contain = True
                break
        if is_contain:
            continue

        status = data.get("status")
        if status == "error":
            user_stats["error_count_today"] += 1
        elif is_contain is False:
            user_stats["unprocessed_count_today"] += 1

    # --- 3. 从已处理投稿(metadata_cache_with_uploads)计算【当天/小时】已处理数 (保留原始逻辑和字段) ---
    user_timestamps: Dict[str, List[float]] = {}
    for _, data in metadata_cache_with_uploads.items():
        user_name = data.get("userName")
        if not user_name:
            continue
        try:
            ts = data["upload_info"]["timestamp"]
            # 同样确保只处理今天的数据
            if ts >= today_start:
                user_timestamps.setdefault(user_name, []).append(ts)
        except (KeyError, TypeError):
            continue

    for user_name, timestamps in user_timestamps.items():
        if not timestamps:
            continue

        user_stats = stats_result.setdefault(user_name, get_default_stats())

        # 这里的计算逻辑和字段名完全遵照您的原始代码
        uploads_today = len(timestamps)  # 因为上面已经筛选过，列表长度就是当天数量
        uploads_last_hour = sum(1 for ts in timestamps if ts > one_hour_ago)
        latest_ts = max(timestamps)
        latest_time_str = datetime.datetime.fromtimestamp(latest_ts).strftime("%Y-%m-%d %H:%M:%S")

        user_stats.update({
            "uploads_today": uploads_today,
            "uploads_last_hour": uploads_last_hour,
            "latest_upload_time": latest_time_str,
            "latest_timestamp": latest_ts,
        })

    return stats_result


def get_user_type(user_name: str) -> str:
    """根据用户映射表得到用户类型。"""
    for group, users in group_info.items():
        if user_name in users:
            return group
    return "fun"


def get_watermark_path(user_type: str, user_name: str) -> str:
    """
    生成合适的水印图片路径。
    从 asset/ 目录中筛选包含 user_type 的 .png，按 user_name 的哈希稳定选择。
    """
    asset_dir = "asset"
    try:
        all_files = os.listdir(asset_dir)
    except FileNotFoundError:
        print("⚠️ 未找到 asset 目录，使用默认水印。")
        return "asset/default_watermark.png"

    filtered_files = [f for f in all_files if user_type in f and f.endswith(".png")]
    if not filtered_files:
        print("⚠️ 未找到符合条件的水印图片，使用默认水印。")
        return "asset/default_watermark.png"

    filtered_files.sort()
    user_hash_hex = hashlib.sha256(user_name.encode("utf-8")).hexdigest()
    user_hash_int = int(user_hash_hex, 16)
    selected_index = user_hash_int % len(filtered_files)
    selected_file = filtered_files[selected_index]
    watermark_path = os.path.join(asset_dir, selected_file)
    print(f"{user_name} ✅ 使用水印图片 {watermark_path} 筛选池大小 {len(filtered_files)} {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return watermark_path


def get_best_plan_by_potential(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """根据“爆款潜力指数”选出分值最高的方案。"""
    best_plan, highest_score = None, float("-inf")
    for plan_info in data.values():
        if not isinstance(plan_info, dict):
            continue
        score = float(plan_info.get("增长潜力", {}).get("爆款潜力指数", 0))
        if score > highest_score:
            highest_score, best_plan = score, plan_info
    return best_plan


def time_str_to_seconds(time_str: str) -> Optional[int]:
    """将 'HH:MM:SS' 或 'MM:SS' 格式的时间字符串转换为总秒数。"""
    try:
        if not isinstance(time_str, str):
            raise TypeError("输入必须是字符串")
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = map(int, parts)
            if m >= 60 or s >= 60:
                raise ValueError("分钟或秒的值不能大于等于60")
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(int, parts)
            if s >= 60:
                raise ValueError("秒的值不能大于等于60")
            return m * 60 + s
        else:
            raise ValueError("时间格式应为 'HH:MM:SS' 或 'MM:SS'")
    except (ValueError, TypeError) as e:
        print(f"错误: 无法解析时间字符串 '{time_str}'。详情: {e}")
        return None


def file_valid(path: Optional[str]) -> bool:
    """判断文件存在且非空。"""
    try:
        return bool(path) and os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:
        return False


# ---------- 上传后台任务 ----------
def upload_worker(
        upload_params: Dict[str, Any],
        key: str,
        updated_entry: Dict[str, Any],
        files_to_cleanup: List[Optional[str]],
        stage_times: Dict[str, float],
        userName: str,
        video_duration_str: str,
) -> None:
    """
    后台上传任务（在各自账号的单线程 executor 中运行，保证同账号串行）；
    完整地执行上传重试、结果处理、metadata 更新、临时文件清理与日志持久化。
    """
    global upload_log_global, error_user_map

    max_retries = 3
    result: Optional[Dict[str, Any]] = None
    t_upload = time.time()

    # 上传重试
    for attempt in range(1, max_retries + 1):
        try:
            result = upload_to_bilibili(**upload_params)
            break
        except Exception as e:
            print(
                f"❌ 上传接口异常 (第 {attempt} 次重试) user={userName} key={key}：{e} {upload_params}"
            )
            if attempt < max_retries:
                time.sleep(60)
            else:
                print("已达最大重试次数，放弃本次上传（后台）。")

    stage_times["上传"] = time.time() - t_upload

    # 上传成功
    if result and isinstance(result, dict) and result.get("aid") and result.get("bvid"):
        try:
            print(
                f"🎉 后台投稿成功！AID={result['aid']}  BVID={result['bvid']} key={key} "
                f"user={userName} 上传耗时 {stage_times.get('上传', 0):.2f} 秒。"
            )
            # 尝试获取最终视频时长并更新 metadata
            try:
                final_duration_sec = get_video_duration_seconds(upload_params.get("video_path"))
                if final_duration_sec is not None:
                    formatted_duration = format_seconds_to_mmss(final_duration_sec)
                    if (
                            "metadata" in updated_entry
                            and isinstance(updated_entry["metadata"], list)
                            and updated_entry["metadata"]
                    ):
                        updated_entry["metadata"][0]["duration"] = formatted_duration
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

        # 写入 upload_info
        updated_entry["upload_info"] = {
            "upload_params": upload_params,
            "upload_result": result,
            "timestamp": time.time(),
        }

        # 再次更新 duration（与原逻辑一致）
        if "metadata" in updated_entry and isinstance(updated_entry["metadata"], list) and updated_entry["metadata"]:
            updated_entry["metadata"][0]["duration"] = video_duration_str

        # 更新全局 upload_log 并持久化 - 替换为安全更新
        try:
            update_log_status(key, updated_entry)

            if stage_times:
                stage_lines = [f"{k}: {v:.2f} 秒" for k, v in stage_times.items()]
                print(
                    f"✅ 后台上传日志已更新 -> {UPLOAD_LOG_FILE}。阶段耗时：{' | '.join(stage_lines)} "
                    f"{userName} {key} {datetime.datetime.now().isoformat()}"
                )
            else:
                print(f"✅ 后台上传日志已更新 -> {UPLOAD_LOG_FILE} {userName}.")
        except Exception as e:
            print(f"🔥 后台写入日志文件失败：{e}")

    else:
        # 上传失败：记录 error_user_map，并把错误信息写到 upload_log
        try:
            err = result.get("message", str(result)) if isinstance(result, dict) else str(result)
        except Exception:
            err = str(result)
        error_user_map[userName] = err or "未知错误"
        print(f"❌ 后台投稿失败 user={userName} key={key}：{err}")

        # 写入失败日志 - 替换为安全更新
        try:
            update_log_status(key, {"status": "error", "error_message": err})
        except Exception as e:
            print(f"🔥 后台写入失败（失败记录）：{e}")


# ---------- 加载与检查 ----------
def _load_metadata_and_log() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """加载 metadata 与 upload_log，并设置 upload_log_global。"""
    global upload_log_global
    metadata_cache: Dict[str, Any] = load_json(METADATA_FILE, default={})
    upload_log: Dict[str, Any] = load_json(UPLOAD_LOG_FILE, default={})
    upload_log_global = upload_log
    return metadata_cache, upload_log


def _basic_task_checks(key: str, value: Dict[str, Any], video_id_key: str) -> Tuple[bool, str]:
    """
    基本合法性检查。返回 (should_skip(bool), reason(str))。
    """
    status = value.get("status", "未处理")
    if status != "complete":
        return True, f"⚠️ 跳过 {key}：当前状态为{status} 不为 complete"

    if video_id_key in upload_log_global and upload_log_global[video_id_key].get("upload_info"):
        return True, f"⏭️ 跳过 {key}：已记录上传成功"

    metadata = value.get("metadata")
    if not (isinstance(metadata, list) and metadata):
        return True, f"⏭️ 跳过 {key}：metadata 字段缺失或格式错误。{metadata}"

    video_id = metadata[0].get("id")
    if not video_id:
        return True, f"⏭️ 跳过 {key}：metadata 中缺少 id。"

    video_path = value.get("video_process_path")
    if not is_valid_target_file_simple(video_path):
        return True, f"⏭️ 跳过 {key} (ID: {video_id})：视频文件缺失 -> {video_path}"

    best_scheme = value.get("best_scheme") or get_best_plan_by_potential(value.get("title_schemes", {}))
    if not best_scheme:
        return True, f"⏭️ 跳过 {key}：无法选取投稿方案。"

    need_process_users = ['qizhu', 'mama', 'dahao', 'shun', 'xiaoxue', 'nana', 'hong', 'zhong']

    userName = value.get("userName", "other")
    if userName in need_process_users:
        if not check_type(value):
            return True, f"⏭️ 跳过 {key}：userName 题材不匹配 {value.get('userName')}"

    return False, ""


# ---------- 媒体预处理 ----------
def _preprocess_media_steps(
        key: str,
        value: Dict[str, Any],
        best_scheme: Dict[str, Any],
        userName: str,
) -> Tuple[str, str, Dict[str, float]]:
    """
    执行视频 / 封面的一系列预处理步骤（顺序与原逻辑一致）。
    返回：
      - video_path: 最终用于后续处理的视频路径（可能被重制覆盖）
      - cover_path: 最终用于上传的封面路径（可能被增强覆盖）
      - stage_times: 本视频的阶段耗时（仅统计“重制视频”“封面处理”，保持原逻辑）
    """
    stage_times: Dict[str, float] = {}
    metadata = value.get("metadata", [])
    generation_options = value.get("generation_options", {}) or {}

    # 初始视频路径
    video_path = value.get("video_process_path")
    duration = metadata[0].get("duration", "00:10")
    duration_sec = time_str_to_seconds(duration)
    full_title = metadata[0].get("full_title", "00:10")

    # 重制视频（保持原条件）
    if generation_options.get("is_original", False) and (duration_sec is None or duration_sec < 600):
        t0 = time.time()
        try:
            has_author_voice = generation_options.get("has_author_voice", True)
            no_owner = not has_author_voice
            creative_guidance = generation_options.get("creative_guidance", "")
            print(
                f"🔄 重制视频 {video_path}... userName: {userName} 是否不包含作者语音{no_owner} "
                f"创作指导：{creative_guidance} 视频名称：{full_title} duration{duration_sec}"
            )
            final_video_path, final_video_script = gen_new_video_robus(video_path)
            cut_type = final_video_script.get("cut_type", "all")
            if is_valid_target_file_simple(final_video_path):
                print(f"✅ 重制视频成功，保存为 {final_video_path}")
                video_path = final_video_path
                if cut_type != "no_owner_voice":
                    title = final_video_script.get("title")
                    if title:
                        best_scheme["标题"] = title
                    cover_text = final_video_script.get("cover_text")
                    if cover_text:
                        best_scheme.setdefault("封面", {}).setdefault("配文", cover_text)
            else:
                print("❌ 重制失败（结果文件不存在），记录错误状态。")
                # 替换为安全更新
                update_log_status(key, {"status": "error"})
            stage_times["重制视频"] = time.time() - t0
        except Exception as e:
            stage_times["重制视频"] = time.time() - t0
            # 替换为安全更新
            update_log_status(key, {"status": "error"})
            print(f"❌ 重制失败：{e}")
            return

    # 封面路径
    meta_cover = metadata[0].get("abs_cover_path", "")
    scheme_cover = best_scheme.get("封面", {}).get("图片路径", "default_cover.jpg")
    cover_path = meta_cover if os.path.exists(meta_cover) else scheme_cover

    # 重复视频使用方案封面
    if value.get("is_duplicate", False):
        cover_path = scheme_cover if os.path.exists(scheme_cover) else meta_cover
        print(f"⚠️ 重复视频，使用方案封面 {cover_path}。")

    # 封面增强处理（支持断点续跑：若增强文件已存在则直接使用）
    try:
        t0 = time.time()
        output_image_path = cover_path.replace(".jpg", "_enhanced.jpg")
        if file_valid(output_image_path):
            cover_path = output_image_path
            print(f"✅ 发现已增强封面，复用 {output_image_path}")
        else:
            create_enhanced_cover(
                input_image_path=cover_path,
                output_image_path=output_image_path,
                text_lines=[best_scheme.get("封面", {}).get("配文", "")],
            )
            if os.path.exists(output_image_path):
                cover_path = output_image_path
        stage_times["封面处理"] = time.time() - t0
    except Exception as e:
        stage_times["封面处理"] = time.time() - t0
        traceback.print_exc()
        print(f"⚠️  封面处理失败：{e}")

    return video_path, cover_path, stage_times


def _build_upload_params(
        metadata_entry: Dict[str, Any],
        best_scheme: Dict[str, Any],
        cover_path: str,
        video_path: str,
        config: Tuple[Optional[str], Optional[str], Optional[str]],
        userName: str,
) -> Dict[str, Any]:
    """基于 best_scheme 与 metadata 生成 upload_params（保留原逻辑）"""
    metadata = metadata_entry.get("metadata", [])
    origin_tag = metadata[0].get("tag", [])
    if userName in video_recommend_user_list:
        origin_tag.insert(0, "B站好片有奖种草")
    if userName in fun_user_list:
        origin_tag.insert(0, "娱乐盘点")
    origin_tag.extend(metadata[0].get("text_extra", []))

    title = best_scheme.get("标题", "欢迎来看我的视频！")
    if len(title) > 80:
        title = title[:70]
        print(f"⚠️ 标题过长，已截断为：{title}")

    human_type2 = best_scheme.get("分区编号", 21)
    topic_json = fetch_bili_topics(config[2], type_pid=human_type2)
    topic_name = "骑行去追夏天的风"
    topic_id = 1313687
    topic_detail = {
        "from_topic_id": 1313687,
        "from_source": "arc.web.recommend",
        "topic_name": "骑行去追夏天的风",
    }
    if isinstance(topic_json, dict) and "data" in topic_json:
        topics = topic_json.get("data", {}).get("topics", [])
        if topics:
            topic_id = topics[0].get("topic_id", human_type2)
            topic_name = topics[0].get("topic_name", "骑行去追夏天的风")
            topic_detail["from_topic_id"] = topic_id
            topic_detail["topic_name"] = topic_name
    else:
        print(f"⚠️ 获取分区 {human_type2} 的话题失败，使用默认值。{topic_json}")

    description_json = best_scheme.get("简介", {})
    target_keys = ["核心看点", "价值承诺", "互动引导", "补充信息"]
    description = "\n".join(str(description_json[k]) for k in target_keys if k in description_json)

    tags = best_scheme.get("标签", ["AI修复", "视频剪辑"])
    origin_tag.extend(tags)
    tags = list(set(origin_tag))
    tags = [tag for tag in tags if len(tag) <= 18]
    tags = tags[:12]
    tags_str = ",".join(tags) if isinstance(tags, list) else str(tags)

    dynamic = best_scheme.get("简介", {}).get("互动引导", "希望大家喜欢")

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


def full_video_info(video_info: Dict[str, Any]) -> Dict[str, Any]:
    """将视频信息补充完整，主要是互动信息和标题信息。"""
    hudong_path = video_info.get("hudong_path")
    if is_valid_target_file_simple(hudong_path):
        hudong_info = read_json(hudong_path)
        video_info["hudong"] = hudong_info

    titles_path = video_info.get("titles_path")
    if is_valid_target_file_simple(titles_path):
        titles_info = read_json(titles_path)
        video_info["title_schemes"] = titles_info

    return video_info


def gen_clean_files(video_path_list: List[str]) -> List[str]:
    """
    根据 video_path_list 生成需要清理的文件列表。
    排除视频本身和固定名单（避免误删结果文件）。
    """
    cleaner_file_list: List[str] = []
    file_names = [
        "optimized_video_plan.json",
        "merged_timestamps.json",
        "hudong.json",
        "title_schemes.json",
        "new_video_script.json",
        "final_scene_info.json",
        "speech_asr_with_owner.json",
        "log.txt",
        "logical_scene_info.json",
        "final_subtitle_box.json",
    ]

    all_files: List[str] = []
    for video_path in video_path_list:
        dir_name = os.path.dirname(video_path) if video_path else ""
        file_name = os.path.basename(video_path) if video_path else ""
        if file_name:
            file_names.append(file_name)
        if dir_name and os.path.exists(dir_name):
            all_sub_files = scan_generated_files(dir_name)
            all_files.extend(all_sub_files)

    all_files = list(set(all_files))  # 去重

    for f in all_files:
        if os.path.basename(f) not in file_names:
            cleaner_file_list.append(f)
        if f in video_path_list and f in cleaner_file_list:
            cleaner_file_list.remove(f)

    print(f"🧹 生成清理文件列表，共 {len(cleaner_file_list)} 个文件。")
    return cleaner_file_list


def check_type(updated_entry: Dict[str, Any]) -> bool:
    """
    检查用户类型与视频题材是否匹配。
    题材映射：
      - 包含 '游戏' -> 'game'
      - 包含 '运动' 或 '体育' -> 'sport'
      - 包含 '搞笑'/'趣味'/'娱乐'/'新闻' -> 'fun'
    """
    user_name = updated_entry.get("userName", "other")
    danmu_info = updated_entry.get("hudong", {}).get("danmu_info", {})
    video_topic = danmu_info.get("视频分析", {}).get("题材", "")

    video_type = "no"
    if video_topic:
        if "游戏" in video_topic:
            video_type = "game"
        elif "运动" in video_topic or "体育" in video_topic:
            video_type = "sport"
        elif "搞笑" in video_topic or "趣味" in video_topic or "娱乐" in video_topic or "新闻" in video_topic:
            video_type = "fun"

    user_type = get_user_type(user_name)
    if user_type != video_type:
        print(
            f"⚠️ 用户 {user_name} 的类型 {user_type} 与视频题材 {video_topic} 的类型 {video_type} 不匹配，跳过上传。"
            f"{updated_entry.get('video_id_list')}"
        )
        return False
    return True


def compute_output_variants(base_video_path: str) -> Dict[str, str]:
    """基于基准视频路径，生成各阶段产物的目标路径。"""
    final_output_path = base_video_path.replace(".mp4", "_final.mp4")
    new_video_path = final_output_path.replace(".mp4", "_new.mp4")
    temp_ending_video_path = final_output_path.replace(".mp4", "_ending.mp4")
    output_watermark_path = final_output_path.replace(".mp4", "_watermark.mp4")
    return {
        "final": final_output_path,
        "new": new_video_path,
        "ending": temp_ending_video_path,
        "watermark": output_watermark_path,
    }


def process_video_batch(
        parent_key: str,
        video_id_list: List[str],
        metadata_cache: Dict[str, Any],
        base_value: Dict[str, Any],
        userName: str,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str], List[Any], Dict[str, float], List[str], List[str], bool]:
    """
    封装“多视频 -> 合并 -> 尾部引导 -> 水印”的完整流水线，支持断点续跑（按产物存在跳过）。

    参数：
      - parent_key: 外层任务 key（用于打印与 persistent 统计）
      - video_id_list: 需要合并的子视频 ID 列表（已排序）
      - metadata_cache: 权威元数据缓存
      - base_value: 外层条目的原始 value（用于读取 base best_scheme）
      - userName: 用户名

    返回：
      - final_output_path: 最终可上传的视频文件路径
      - best_scheme_final: 按分数选出的最佳方案（用于构建上传参数）
      - best_cover_path: 最佳封面路径（增强后）
      - comment_list_top30: 合并后的评论 Top30
      - last_stage_times: 最后一个视频的预处理阶段耗时（保持原日志逻辑）
      - origin_video_path_list: 原始视频路径集合（用于清理）
      - video_path_list: 参与合并的实际视频路径集合
      - had_missing_scheme: 是否存在“无法选取投稿方案”的子视频（用于 persistent 标记）
    """
    this_fun_start_time = time.time()
    video_path_list: List[str] = []
    origin_video_path_list: List[str] = []
    comment_list_all: List[Any] = []
    best_score_max = float("-inf")
    best_scheme_final: Optional[Dict[str, Any]] = None
    best_cover_path: Optional[str] = None
    last_stage_times: Dict[str, float] = {}
    had_missing_scheme = False

    base_video_path_for_naming: Optional[str] = None

    # 逐视频预处理（保持原判定逻辑）
    for video_id in video_id_list:
        video_info = full_video_info(metadata_cache.get(video_id, {}))
        origin_video_path_list.append(video_info.get("video_process_path"))
        origin_video_path_list.append(video_info.get("video_path"))
        comment_list = video_info.get("hudong", {}).get("comment_list", [])
        comment_list_all.extend(comment_list)

        best_scheme = base_value.get("best_scheme") or get_best_plan_by_potential(
            video_info.get("title_schemes", {})
        )
        if not best_scheme:
            print(f"⏭️ 跳过 {parent_key}：无法选取投稿方案。")
            had_missing_scheme = True
            continue

        score = float(best_scheme.get("增长潜力", {}).get("爆款潜力指数", 0))

        print(
            f"\n⏳ {userName} 开始处理子任务 {video_id}，时间：{time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        try:
            video_path, cover_path, stage_times = _preprocess_media_steps(
                video_id, video_info, best_scheme, userName
            )
            last_stage_times = stage_times  # 保持与原逻辑一致：只记录最后一个视频的耗时

            # 选择最终用于上传的标题/封面方案（高分优先）
            if score > best_score_max:
                best_score_max = score
                best_scheme_final = best_scheme
                best_cover_path = cover_path
                video_path_list.insert(0, video_path)  # 高分视频放前面
            else:
                video_path_list.append(video_path)

            base_video_path_for_naming = video_path  # 保持与原逻辑一致：用最后一次的 video_path 生成产物名
        except Exception as e:
            print(f"⚠️  处理媒体过程中出现异常：{e} {video_id} {userName}")
            traceback.print_exc()
            # 替换为安全更新
            update_log_status(video_id, {"status": "error"})
            # 与原逻辑一致：不中断整批，后续步骤依赖文件存在自行决策
            continue

    # 评论取 Top30
    comment_list_top30 = sorted(comment_list_all, key=lambda x: x[1], reverse=True)[:30]

    # 若没有有效视频，直接抛错
    if not video_path_list or not base_video_path_for_naming:
        raise RuntimeError(f"未得到可用的视频路径（{parent_key}，{userName}）。")

    # 产物文件名（与原逻辑一致：源于最后一个 video_path）
    outputs = compute_output_variants(base_video_path_for_naming)
    final_output_path = outputs["final"]
    new_video_path = outputs["new"]
    temp_ending_video_path = outputs["ending"]
    output_watermark_path = outputs["watermark"]

    # 断点续跑：若已有水印产物，直接返回
    if file_valid(output_watermark_path):
        print(f"✅ 发现已存在水印产物，复用：{output_watermark_path}")
        return (
            output_watermark_path,
            best_scheme_final,
            best_cover_path,
            comment_list_top30,
            last_stage_times,
            origin_video_path_list,
            video_path_list,
            had_missing_scheme,
        )

    # 合并（若已存在则跳过）
    if not file_valid(final_output_path):
        print(f"🔗 合并 {len(video_path_list)} 段视频 -> {final_output_path}")
        merge_videos_ffmpeg(video_path_list, output_path=final_output_path)
    else:
        print(f"✅ 发现已存在合并产物，复用：{final_output_path}")

    # 追加尾部引导（< 6000 秒）
    active_path = final_output_path
    try:
        duration = probe_duration(final_output_path)
        if duration is not None and duration < 6000:
            if file_valid(new_video_path):
                print(f"✅ 发现已存在追加结尾产物，复用：{new_video_path}")
                active_path = new_video_path
            else:
                try:
                    origin_ending_video_path = "origin_ending_video.mp4"
                    ending_text = (best_scheme_final or {}).get("简介", {}).get(
                        "结尾语", "感谢观看本视频，欢迎点赞、评论、关注、投币、分享！"
                    )
                    gen_ending_video(ending_text, temp_ending_video_path, origin_ending_video_path)
                    merge_videos_ffmpeg([final_output_path, temp_ending_video_path], output_path=new_video_path)
                    if file_valid(new_video_path):
                        active_path = new_video_path
                        print(f"✅ 已追加结尾引导 -> {new_video_path}")
                except Exception as e:
                    print(f"⚠️ 尾部引导视频失败，继续使用原视频：{e}")
    except Exception as e:
        print(f"⚠️ 检测视频时长失败，跳过追加结尾：{e}")

    # 增加水印（若已存在则跳过）
    try:
        if file_valid(output_watermark_path):
            print(f"✅ 发现已存在水印产物，复用：{output_watermark_path}")
            final_output_path_ready = output_watermark_path
        else:
            user_type = get_user_type(userName)
            wm_path = get_watermark_path(user_type, userName)
            start_time = time.time()
            add_transparent_watermark(active_path, wm_path, output_watermark_path)
            if file_valid(output_watermark_path):
                print(f"✅  耗时 {time.time() - start_time:.2f} 秒 水印增加成功，保存为 {output_watermark_path}")
                final_output_path_ready = output_watermark_path
            else:
                print("⚠️ 水印生成失败，继续使用无水印视频。")
                final_output_path_ready = active_path
    except Exception as e:
        print(f"⚠️ 水印增加失败，继续使用原视频：{e}")
        final_output_path_ready = active_path
    print(f"🎬 处理完成，{userName}，时间：{time.strftime('%Y-%m-%d %H:%M:%S')}：{final_output_path_ready}，总耗时 {time.time() - this_fun_start_time:.2f} 秒。")
    return (
        final_output_path_ready,
        best_scheme_final,
        best_cover_path,
        comment_list_top30,
        last_stage_times,
        origin_video_path_list,
        video_path_list,
        had_missing_scheme,
    )


def get_wait_minutes():
    """
    根据当前时间的小时数，返回一个非线性的等待分钟数。
    - 凌晨和清晨等待时间最长。
    - 白天和傍晚逐渐减少。
    - 深夜等待时间最短。
    - 等待时间以5分钟为单位变化。

    Returns:
        int: 建议的等待分钟数。
    """
    # 1. 获取当前时间的小时数 (0-23)
    current_hour = datetime.datetime.now().hour

    # 2. 根据不同的时间段，返回不同的等待时间
    # 规则：越早时间越长，越晚时间越短
    if current_hour <= 8:  # 清晨 06:00 - 08:59，开始苏醒，等待时间减少
        return 40

    elif current_hour <= 11:  # 上午 09:00 - 11:59，工作时间，等待时间减少
        return 30

    elif current_hour <= 17:  # 中午及下午 12:00 - 17:59，活跃时间
        return 20

    elif current_hour <= 21:  # 傍晚 18:00 - 21:59，晚上休息前
        return 10

    else:  # 深夜 22:00 - 23:59，准备休息，等待时间最短
        return 0


def auto_upload() -> None:
    """
    非阻塞版 auto_upload（主线程负责预处理，投稿提交到每个账号的单线程 executor）：
    - 保留并执行原脚本的全部预处理逻辑
    - 在生成 upload_params 后，使用 account_executors[userName].submit(...) 提交 upload_worker，
      以确保同一用户同一时刻只会有一个上传任务在运行。
    - 视频处理流水线已封装到 process_video_batch，并支持断点续跑（按产物存在跳过）。
    - 新增需求：如果本轮循环因账号冷却或上限未提交任何新视频，则在最后至少有效处理一个视频（耗时>10s），
      以充分利用计算资源。
    """
    global upload_log_global
    # 读取各账号 cookie
    base_config_map = init_config()

    for uid, detail_info in base_config_map.items():
        name = detail_info.get("name", f"user_{uid}")
        sessdata = detail_info.get("SESSDATA", f"SESSDATA")
        bili_jct = detail_info.get("BILI_JCT", f"user_{uid}")
        total_cookie = detail_info.get("total_cookie", f"user_{uid}")
        # 判断total_cookie是否和之前的不一样，如果不一样则更新
        before_total_cookie = config_map.get(name, (None, None, None))[2]
        if before_total_cookie != total_cookie:
            print(f"🔄 检测到用户 {name} 的 total_cookie 发生变化，已更新。")
            # 如果name在error_user_map中，删除对应的错误记录
            if name in error_user_map:
                del error_user_map[name]

        config_map[name] = (sessdata, bili_jct, total_cookie)
    temp_set: Set[str] = set()
    metadata_cache, upload_log = _load_metadata_and_log()
    upload_log_global = upload_log

    if not metadata_cache:
        print(f"❌ 无可用任务：{METADATA_FILE} 为空或不存在。")
        return

    if not upload_log and (len(metadata_cache) - len(upload_log)) > 400:
        print(f"❌ 无可用任务：{UPLOAD_LOG_FILE} 为空或不存在。")
        return
    user_processed_counts = defaultdict(int)
    all_start_time = time.time()

    # ▼▼▼【核心修改点 1：新增任务优先级调度逻辑】▼▼▼
    processed_tasks = []
    unprocessed_tasks = []
    print("🔍 正在进行任务预扫描，优先处理已生成的视频...")
    upload_log_keys = set(upload_log.keys())
    for key, value in metadata_cache.items():
        is_contain = False
        # 判断key是否被包含在upload_log_keys的元素中
        for log_key in upload_log_keys:
            if key in log_key:
                is_contain = True
                break
        if is_contain:
            continue

        task_tuple = (key, value)
        userName = value.get("userName", "other")

        # 模拟 process_video_batch 的文件名确定逻辑
        video_id_list = value.get("video_id_list", [key])
        if not video_id_list:
            unprocessed_tasks.append(task_tuple)
            continue

        # 1. 找到最后一个 video_id 对应的 video_info
        last_video_id = sorted(video_id_list)[-1]
        last_video_info = metadata_cache.get(last_video_id)

        if not last_video_info or not last_video_info.get("video_process_path"):
            # 如果信息不完整，无法预测路径，归为未处理
            unprocessed_tasks.append(task_tuple)
            continue

        # 2. 获取用于命名产物的基础路径
        base_path_for_naming = last_video_info.get("video_process_path")

        # 3. 使用已有的函数预测所有产物路径
        output_variants = compute_output_variants(base_path_for_naming)

        # 4. 检查最终产物（带水印的视频）是否存在
        final_watermark_path = output_variants.get("watermark")
        remake_watermark_path = None  # 初始化为 None

        # 仅当原始路径有效时，才构造 remake 文件的路径
        if final_watermark_path:
            # 1. 获取 final_watermark_path 所在的目录
            directory = os.path.dirname(final_watermark_path)
            # 2. 构造 remake_final_watermark.mp4 的完整路径
            remake_watermark_path = os.path.join(directory, "remake_final_watermark.mp4")

        # 3. 使用 or 条件检查：只要原始文件或 remake 文件中任何一个存在，就视为已处理
        if (final_watermark_path and os.path.exists(final_watermark_path)) or \
                (remake_watermark_path and os.path.exists(remake_watermark_path)):
            # 如果最终带水印的文件或 remake 文件存在，则视为已处理
            processed_tasks.append(task_tuple)
            user_processed_counts[userName] += 1
        else:
            unprocessed_tasks.append(task_tuple)

    # 5. 构建优先队列：已处理的在前，未处理的在后
    prioritized_task_list = processed_tasks + unprocessed_tasks
    sorted_users = sorted(user_processed_counts.items(), key=lambda item: item[1], reverse=True)

    # 构建单行输出字符串
    distribution_summary = ", ".join([f"{user}: {count}" for user, count in sorted_users])

    print(
        f"✅ 发现 {len(processed_tasks)} 个已处理好的任务，将优先上传。一共 {len(prioritized_task_list)} 个任务待处理。📊 已处理任务分布: {distribution_summary}")

    futures: List[concurrent.futures.Future] = []
    error_count = 0
    processed_video_id: List[str] = []
    latest_user = ""

    # --- 新增变量以满足新需求 ---
    # 标志位，记录本轮是否有实际的投稿任务被提交
    submitted_any_uploads = False
    # 收集因账号限制而跳过，但可被预处理的视频任务
    skippable_candidates: List[Dict[str, Any]] = []
    already_upload_users = []
    # --- 变量新增结束 ---

    this_time_upload_count = 0

    remote_upload_dict = {}
    today_start = datetime.datetime.combine(datetime.date.today(), datetime.time.min).timestamp()
    # 检查用户今日上传数量（本地 + 平台）
    try:
        bvid_file_data = read_json(bvid_file_path)
    except Exception as e:
        print(f"❌ 读取 {bvid_file_path} 失败：{e}")
        bvid_file_data = {}
    for userName, user_videos in bvid_file_data.items():
        recent_videos = [v for v in user_videos if v.get("created") and v["created"] >= today_start]
        remote_upload_count = len(recent_videos)
        remote_upload_dict[userName] = remote_upload_count

    user_uploads_info = analyze_user_uploads_by_day(upload_log_global, metadata_cache)
    for user, count in remote_upload_dict.items():
        if user in user_uploads_info:
            user_uploads_info[user]['remote_upload_count'] = count

    save_json(USER_UPLOADS_INFO_FILE, user_uploads_info)

    # ▼▼▼【核心修改点 2：遍历经过优先级排序的列表】▼▼▼
    for key, value in prioritized_task_list:
        # ▲▲▲【核心修改点 2：结束】▲▲▲
        if key in processed_video_id:
            continue
        userName = value.get("userName", "other")

        upload_status = upload_log_global.get(key, {}).get("status", "未处理")
        if upload_status == 'error':
            print(f"⏭️ 跳过 {key} {userName}：之前制作视频处理失败，状态为 error。")
            error_count += 1
            continue

        should_skip = False

        updated_entry = full_video_info(value)  # 补全

        video_id_list = value.get("video_id_list", [key])
        video_id_list = sorted(video_id_list)
        video_id_key = "_".join(video_id_list)
        processed_video_id.extend(video_id_list)

        # 基本检查
        for video_id in video_id_list:
            value_info = full_video_info(metadata_cache.get(video_id, {}))
            should_skip, reason = _basic_task_checks(video_id, value_info, video_id_key)
            if should_skip:
                if "已记录上传成功" in reason:
                    continue
                if "之前处理失败" in reason:
                    print(f"{reason} {userName}")
                    error_count += 1
                    break
                else:
                    print(f"{reason} {userName}")
                    break

        if should_skip:
            continue

        if userName in error_user_map:
            print(f"⚠️ 跳过 {userName} 用户上传：之前上传失败，错误信息：{error_user_map[userName]} {key}")
            error_count += 1
            continue

        # 选择 config
        if userName not in config_map.keys():
            print(f"⚠️ 跳过 {userName} 用户上传 请检查配置数据。video_ids={video_id_list}")
            continue
        config = config_map.get(userName, config_map["base"])

        remote_upload_count = remote_upload_dict.get(userName, 0)

        user_info = user_uploads_info.setdefault(
            userName,
            {
                "uploads_today": 0,
                "uploads_last_hour": 0,
                "latest_upload_time": "无记录",
                "latest_timestamp": None,
            },
        )
        # 将remote_upload_count更新到user_uploads_info中
        user_info['remote_upload_count'] = remote_upload_count

        uploads_today = user_info.get("uploads_today", 0)
        uploads_last_hour = user_info["uploads_last_hour"]
        latest_upload_time = user_info["latest_upload_time"]
        latest_timestamp = user_info["latest_timestamp"]

        wait_minutes = get_wait_minutes()
        generation_options = value.get("generation_options", {}) or {}
        is_real_time = generation_options.get("is_real_time", False)
        if not is_real_time:
            wait_minutes += 10  # 非实时投稿，增加等待时间
        if userName in right_now_user_list:
            wait_minutes = 0

        latest_upload_str = latest_timestamp
        interval_minutes_str = 0

        if latest_upload_str:
            try:
                # 将时间字符串转换为 datetime 对象
                latest_upload_dt = latest_timestamp
                # 计算时间差 (timedelta 对象)
                time_difference = time.time() - latest_upload_dt
                # 将时间差转换为分钟数（整数）
                interval_minutes_str = int(time_difference / 60)
            except (ValueError, TypeError):
                # 如果时间格式不正确或类型错误，则保持为 "N/A"
                interval_minutes_str = 0
        print(
            f"🔍 处理 {key} (用户: {userName}) 今日已本地上传 {uploads_today} 个视频， 实际平台数据：{remote_upload_count}  "
            f"最近一小时上传个数为: {uploads_last_hour}，最近上传时间为：{latest_upload_time}，当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}"
            f"是否实时视频{is_real_time}，建议等待时间：{wait_minutes} 分钟。还需等待时间：{max(0, wait_minutes - interval_minutes_str)} 分钟。"
        )

        # --- 修改跳过逻辑，收集可处理的候选任务 ---
        is_cooldown_or_limit = False
        cooldown_reason = ""
        uploads_today_max = 25
        # if userName in limit_user_list:
        #     uploads_today_max = 25

        if uploads_today >= uploads_today_max or remote_upload_count >= 21:
            is_cooldown_or_limit = True
            cooldown_reason = f"今日已本地上传 {uploads_today} 个视频， 实际平台数据：{remote_upload_count} ，达到上限 {uploads_today_max}。"
        elif latest_timestamp and (all_start_time - latest_timestamp) < wait_minutes * 60 and uploads_last_hour >= 1:
            is_cooldown_or_limit = True
            cooldown_reason = f"距离上次上传少于 {wait_minutes} 分钟。 上次上传时间：{latest_upload_time}，当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}"
        elif userName == latest_user:
            is_cooldown_or_limit = True
            cooldown_reason = "与上一个上传用户相同，避免连续上传。"

        # 判断当前时间是否在 5 点 到 24 点之间
        if not (5 <= datetime.datetime.now().hour < 24) and not is_real_time:
            is_cooldown_or_limit = True
            cooldown_reason = "当前时间不在允许的上传时间段（5点-24点）内。"

        if userName in already_upload_users:
            is_cooldown_or_limit = True
            cooldown_reason = "本轮循环中该用户已提交过上传任务，避免重复提交。"

        if is_cooldown_or_limit:
            print(f"⚠️ 跳过 {userName} 用户上传：{cooldown_reason}")
            # 收集该任务以备后续处理，但不提交上传
            candidate_params = {
                "parent_key": key,
                "video_id_list": video_id_list,
                "metadata_cache": metadata_cache,
                "base_value": value,
                "userName": userName,
            }
            skippable_candidates.append(candidate_params)
            continue
        # --- 逻辑修改结束 ---

        latest_user = userName

        # 执行视频处理流水线（带断点续跑）
        try:
            (
                final_output_path,
                best_scheme_final,
                best_cover_path,
                comment_list_top30,
                last_stage_times,
                origin_video_path_list,
                video_path_list,
                had_missing_scheme,
            ) = process_video_batch(
                parent_key=key,
                video_id_list=video_id_list,
                metadata_cache=metadata_cache,
                base_value=value,
                userName=userName,
            )
        except Exception as e:
            print(f"❌ 视频处理流水线失败：{e} | {key} | {userName}")
            traceback.print_exc()
            error_count += 1
            # 替换为安全更新
            update_log_status(key, {"status": "error"})
            continue

        if had_missing_scheme:
            # 与原逻辑一致：将整个 key 放入 persistent_tasks
            temp_set.add(key)

        # 更新互动评论到 updated_entry
        updated_entry.setdefault("hudong", {})
        updated_entry["hudong"]["comment_list"] = comment_list_top30

        # 构建上传参数（保持原逻辑）
        upload_params = _build_upload_params(
            value, best_scheme_final or {}, best_cover_path or "", final_output_path, config, userName
        )

        # 时长字符串
        try:
            video_duration_sec = probe_duration(final_output_path)  # 原逻辑：返回秒
            video_duration_str = ms_to_time(
                int(video_duration_sec * 1000)) if video_duration_sec is not None else "00:00"
        except Exception:
            video_duration_str = "00:00"

        # --- 更新任务提交状态 ---
        submitted_any_uploads = True
        # --- 状态更新结束 ---

        task_stage_times = dict(last_stage_times)

        # 清理文件（排除最终产物）
        all_files_to_cleanup = gen_clean_files(origin_video_path_list)
        all_files_to_cleanup = [
            f for f in all_files_to_cleanup if os.path.basename(f) != os.path.basename(final_output_path)
        ]
        print(f"🧹 预处理完成，准备清理 {len(all_files_to_cleanup)} 个临时文件。排除{final_output_path}")

        # 按账号单线程执行上传
        if this_time_upload_count < 10:
            print(
                f"🚀 准备为用户 {userName} 后台投稿 {key} (ID: {video_id_key}) - 《{upload_params.get('title')}》（按账号串行）"
            )
            this_time_upload_count += 1

            account_executor = account_executors[userName]
            future = account_executor.submit(
                upload_worker,
                upload_params,
                video_id_key,
                updated_entry,
                all_files_to_cleanup,
                task_stage_times,
                userName,
                video_duration_str,
            )
            futures.append(future)
        already_upload_users.append(userName)

    submitted_any_uploads = False
    user_candidate_counts = defaultdict(int)
    # --- 新增备用处理逻辑 (日志优化 + 进度统计) ---
    if not submitted_any_uploads and skippable_candidates:

        # 1. 启动信息：更详细的启动摘要
        print(
            f"💡 本轮未提交任何新投稿，启动【备用视频预处理】流程以充分利用计算资源。 共 {len(skippable_candidates)} 个候选任务。 当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")

        for c in skippable_candidates:
            user_candidate_counts[c['userName']] += 1

        user_summary = ", ".join([f"{user}: {count}个" for user, count in user_candidate_counts.items()])
        print(f"💡 共发现 {len(skippable_candidates)} 个候选任务，用户分布: {user_summary}")

        effective_task_found = False

        # --- 新增：用于统计跳过任务的数量 ---
        # skippable_candidates = []
        skipped_count = 0
        total_candidates = len(skippable_candidates)
        # --- 新增结束 ---

        # ▼▼▼【核心优化点：按“已处理任务数”对候选任务进行升序排序】▼▼▼
        # 这样可以确保优先处理那些“已处理任务很少”的用户的视频
        print("🔃 正在重新排序候选任务，优先处理存量任务较少的用户...")
        skippable_candidates.sort(key=lambda x: user_processed_counts.get(x['userName'], 0))

        # 2. 处理进度：增加进度指示和详细信息
        for i, candidate in enumerate(skippable_candidates, 1):
            user_name = candidate['userName']
            exist_count = user_processed_counts.get(user_name, 0)
            parent_key = candidate['parent_key']
            video_ids = candidate['video_id_list']

            print(f"\n⏳ [{i}/{total_candidates}] 尝试处理候选任务: {parent_key}")
            print(f"   - 用户: {user_name} (已处理任务数: {exist_count}) 包含视频ID: {video_ids}")

            remaining_count = total_candidates - i

            # --- 改动 1：获取后台上传状态 ---
            pending_uploads_count = sum(1 for f in futures if not f.done())
            is_uploading = pending_uploads_count > 0

            try:
                processing_duration = time.time() - all_start_time

                # --- 改动 2：修改退出判断逻辑 ---
                # 只有当【超时】且【没在上传】时，才退出。
                # 否则（没超时 或 正在上传），直接跳过这个if，继续往下执行 process_video_batch
                if processing_duration > 200 and not is_uploading:
                    print(f"🎉 【有效处理完成】 任务 '{parent_key}' 耗时 {processing_duration:.2f} 秒. 且无后台投稿。")
                    print("   - 目标达成，备用处理流程结束。")
                    effective_task_found = True
                    break

                # 如果超时但正在上传，这里可以补一句简单的日志（可选，不写也不影响逻辑）
                if processing_duration > 200:
                    print(f"   ⚡ [算力压榨] 耗时已超 {processing_duration:.2f}s，利用 {pending_uploads_count} 个后台上传间隙继续处理...")
                else:
                    print(f"   ⚡ 未进行实际的处理 处理太快了 {processing_duration:.2f}s，继续处理...")

                process_video_batch(**candidate)

                user_processed_counts[user_name] += 1

            except Exception as e:
                print(f"❌ 【处理失败】 候选任务 '{parent_key}' 发生错误: {e}")
                traceback.print_exc()
                print(f"   - 📊 进度: 处理失败，剩余 {remaining_count} 个待检查。")
                update_log_status(parent_key, {"status": "error"})
                continue

        # 4. 最终总结：根据是否找到有效任务给出不同的总结
        if not effective_task_found:
            print("\n" + "-" * 50)
            if total_candidates > 0:
                print(f"✅ 已检查全部 {total_candidates} 个候选任务，但未发现需要大量计算的（耗时均≤10秒）。")
            else:
                print("✅ 备用流程结束，没有需要处理的候选任务。")

        print("=" * 50 + "\n")

    # 处理被跳过的 persistent tasks
    if len(temp_set) > 0:
        print(f"⚠️ 跳过了 {len(temp_set)} 个任务：{', '.join(temp_set)}")
        persistent_tasks = load_json(persistent_tasks_file, default={})
        persistent_tasks = set(persistent_tasks) if isinstance(persistent_tasks, list) else set()
        persistent_tasks.update(temp_set)
        save_json(persistent_tasks_file, list(persistent_tasks))
    # 等待所有后台上传完成
    print(
        f"等待所有等待后台上传完成... 本次投稿数量 {this_time_upload_count}  用户{already_upload_users}  当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')} {error_user_map}")
    concurrent.futures.wait(futures, timeout=None)
    print(f"{'用户名':<15} | {'本地':>6} | {'远程':>6} | {'待传':>6} | {'间隔(分)':>7} | {'最近上传时间':<19}")

    now = datetime.datetime.now()
    for user_name, info in sorted(
            user_uploads_info.items(),
            key=lambda item: item[1].get('uploads_today', 0),
            reverse=True
    ):
        # --- 新增的逻辑：计算时间间隔 ---
        latest_upload_str = info.get("latest_upload_time")
        interval_minutes_str = "N/A"  # 默认值

        if latest_upload_str:
            try:
                # 将时间字符串转换为 datetime 对象
                latest_upload_dt = datetime.datetime.strptime(latest_upload_str, "%Y-%m-%d %H:%M:%S")
                # 计算时间差 (timedelta 对象)
                time_difference = now - latest_upload_dt
                # 将时间差转换为分钟数（整数）
                minutes_ago = int(time_difference.total_seconds() / 60)
                interval_minutes_str = str(minutes_ago)
            except (ValueError, TypeError):
                # 如果时间格式不正确或类型错误，则保持为 "N/A"
                interval_minutes_str = "错误格式"

        need_to_upload = user_candidate_counts.get(user_name, 0)
        print(
            f"{user_name:<18} | "
            f"{info.get('uploads_today', 0):>6} | "  # 宽度增加到 6
            f"{remote_upload_dict.get(user_name, 0):>6} | "  # 宽度增加到 6
            f"{need_to_upload:>6} | "  # 宽度增加到 6
            f"{interval_minutes_str:>9} | "  # 宽度增加到 9
            f"{info.get('latest_upload_time', 'N/A'):<19}"  # 指定宽度 19，确保对齐
        )

    # 将user_candidate_counts保存到文件
    save_json(SKIPPED_CANDIDATES_FILE, user_candidate_counts)
    print(f"错误数量为{len(error_user_map)}  全部任务处理完毕。时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")

def clear_proxies():
    """在当前脚本会话中临时清空HTTP/HTTPS代理环境变量。"""
    print("--- 正在清空会话代理 ---")
    # 定义需要被清除的代理环境变量名（包括大小写）
    proxy_keys = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']

    cleared = False
    for key in proxy_keys:
        if key in os.environ:
            # 使用 del 删除环境变量，使其在当前进程中失效
            del os.environ[key]
            print(f"已临时移除环境变量: {key}")
            cleared = True

    if cleared:
        print("代理信息已在当前会话中清空。")
    else:
        print("无需清空，未发现代理环境变量。")
    print("-" * 28 + "\n")


# ---------- CLI ----------
if __name__ == "__main__":
    clear_proxies()
    while True:
        auto_upload()
        time.sleep(60)  # 每分钟运行一次