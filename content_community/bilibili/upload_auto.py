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

import concurrent.futures
import datetime
import hashlib
import json
import os
import random
import threading
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from common_utils.common_utils import (
    get_config,
    format_seconds_to_mmss,
    read_json,
    is_valid_target_file_simple,
    scan_generated_files,
    ms_to_time,
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

# 账号映射
accounts: Dict[str, str] = {
    "tao": "tao",
    "taoxiao": "taoxiao",
    "junxiao": "junxiao",
    "junda": "junda",
    "ruru": "ruru",
    "nana": "nana",
    "jie": "jie",
    "qiqi": "qiqi",
    "mama": "mama",
    "hong": "hong",
    "yan": "yan",
    "xue": "xue",
    "cai": "cai",
    "jun": "jun",
    "xiaosu": "xiaosu",
    "chabian": "chabian",
    "lin": "lin",
    "jj": "jj",
    "hao": "hao",
    "dan": "dan",
    "ning": "ning",
    "yang": "yang",
    "ruruxiao": "ruruxiao",
    "qiqixiao": "qiqixiao",
    "yiyi": "yiyi",
    "xiaodan": "xiaodan",
    "xiaoxue": "xiaoxue",
}

# 读取各账号 cookie
for name, map_key in accounts.items():
    sessdata = get_config(f"{name}_bilibili_sessdata_cookie")
    bili_jct = get_config(f"{name}_bilibili_csrf_token")
    total_cookie = get_config(f"{name}_bilibili_total_cookie")
    config_map[map_key] = (sessdata, bili_jct, total_cookie)

# 题材分组
group_info: Dict[str, List[str]] = {
    "fun": [
        "ruru",
        "jj",
        "chabian",
        "dan",
        "yiyi",
        "qiqixiao",
        "yang",
        "xiaodan",
        "qiqixiao",
        "dahao",
        "lin",
        "xiaohao",
        "xue",
        "jj",
        "ruru",
        "xiaosu",
    ],
    "sport": ["nana", "jun"],
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
    ],
}

# 推荐话题用户列表
video_recommend_user_list = [
    "cai",
    "yang",
    "dahao",
    "ruru",
    "yiyi",
    "lin",
    "mama",
    "hong",
    "yan",
    "jie",
    "qiqi",
    "xiaosu",
    "jun",
    "jj",
    "qiqixiao",
    "xiaoxue",
]

# 错误记录
error_user_map: Dict[str, str] = {}

# ---------- 文件路径常量 ----------
METADATA_FILE = "../../LLM/TikTokDownloader/back_up/metadata_cache.json"  # 权威源
UPLOAD_LOG_FILE = "../../LLM/TikTokDownloader/back_up/metadata_cache_with_uploads.json"  # 上传日志
persistent_tasks_file = "../../LLM/TikTokDownloader/back_up/persistent_tasks.json"
bvid_file_path = "../../LLM/TikTokDownloader/back_up/bvid_file.json"

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


def analyze_user_uploads_by_day(metadata_cache_with_uploads: Any) -> Dict[str, Dict[str, Any]]:
    """
    汇总每个用户当天/近1小时的投稿数量与最近投稿时间。
    """
    user_timestamps: Dict[str, List[float]] = {}

    # 兼容列表封装的情况
    if isinstance(metadata_cache_with_uploads, list) and len(metadata_cache_with_uploads) > 0:
        metadata_cache_with_uploads = metadata_cache_with_uploads[0]

    if not isinstance(metadata_cache_with_uploads, dict):
        return {}

    for _, data in metadata_cache_with_uploads.items():
        user_name = data.get("userName")
        if not user_name:
            continue
        try:
            ts = data["upload_info"]["timestamp"]
        except (KeyError, TypeError):
            continue
        user_timestamps.setdefault(user_name, []).append(ts)

    stats_result: Dict[str, Dict[str, Any]] = {}
    now = datetime.datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    one_hour_ago = now.timestamp() - 3600

    for user_name, timestamps in user_timestamps.items():
        if not timestamps:
            continue
        uploads_today = sum(1 for ts in timestamps if ts >= today_start)
        uploads_last_hour = sum(1 for ts in timestamps if ts > one_hour_ago)
        latest_ts = max(timestamps)
        latest_time_str = datetime.datetime.fromtimestamp(latest_ts).strftime("%Y-%m-%d %H:%M:%S")

        stats_result[user_name] = {
            "uploads_today": uploads_today,
            "uploads_last_hour": uploads_last_hour,
            "latest_upload_time": latest_time_str,
            "latest_timestamp": latest_ts,
        }

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
    print(f"{user_name} ✅ 使用水印图片 {watermark_path} 筛选池大小 {len(filtered_files)}")
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


# ---------- 核心上传工作线程 ----------
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

        # 更新全局 upload_log 并持久化
        with upload_lock:
            upload_log_global[key] = updated_entry
            try:
                save_json(UPLOAD_LOG_FILE, upload_log_global)
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
        with upload_lock:
            upload_log_global[key] = upload_log_global.get(key, {})
            upload_log_global[key]["status"] = "error"
            upload_log_global[key]["error_message"] = err
            try:
                save_json(UPLOAD_LOG_FILE, upload_log_global)
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

    video_path = value.get("video_path")
    if not is_valid_target_file_simple(video_path):
        return True, f"⏭️ 跳过 {key} (ID: {video_id})：视频文件缺失 -> {video_path}"

    best_scheme = value.get("best_scheme") or get_best_plan_by_potential(value.get("title_schemes", {}))
    if not best_scheme:
        return True, f"⏭️ 跳过 {key}：无法选取投稿方案。"

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
      - video_path: 最终用于上传的视频路径
      - cover_path: 最终用于上传的封面路径
      - stage_times: 每一步耗时字典
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
                upload_log_global[key] = upload_log_global.get(key, {})
                upload_log_global[key]["status"] = "error"
                save_json(UPLOAD_LOG_FILE, upload_log_global)
                print("❌ 重制失败")
            stage_times["重制视频"] = time.time() - t0
        except Exception as e:
            stage_times["重制视频"] = time.time() - t0
            upload_log_global[key] = upload_log_global.get(key, {})
            upload_log_global[key]["status"] = "error"
            save_json(UPLOAD_LOG_FILE, upload_log_global)
            print(f"❌ 重制失败：{e}")

    # 封面路径
    meta_cover = metadata[0].get("abs_cover_path", "")
    scheme_cover = best_scheme.get("封面", {}).get("图片路径", "default_cover.jpg")
    cover_path = meta_cover if os.path.exists(meta_cover) else scheme_cover

    # 重复视频使用方案封面
    if value.get("is_duplicate", False):
        cover_path = scheme_cover if os.path.exists(scheme_cover) else meta_cover
        print(f"⚠️ 重复视频，使用方案封面 {cover_path}。")

    # 封面增强处理
    try:
        t0 = time.time()
        output_image_path = cover_path.replace(".jpg", "_enhanced.jpg")
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
        dir_name = os.path.dirname(video_path)
        file_name = os.path.basename(video_path)
        file_names.append(file_name)
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


# ---------- 主流程 ----------
def auto_upload() -> None:
    """
    非阻塞版 auto_upload（主线程负责预处理，投稿提交到每个账号的单线程 executor）：
    - 保留并执行原脚本的全部预处理逻辑
    - 在生成 upload_params 后，使用 account_executors[userName].submit(...) 提交 upload_worker，
      以确保同一用户同一时刻只会有一个上传任务在运行。
    """
    global upload_log_global

    temp_set: Set[str] = set()
    metadata_cache, upload_log = _load_metadata_and_log()
    upload_log_global = upload_log

    if not metadata_cache:
        print(f"❌ 无可用任务：{METADATA_FILE} 为空或不存在。")
        return

    if not upload_log and (len(metadata_cache) - len(upload_log)) > 20:
        print(f"❌ 无可用任务：{UPLOAD_LOG_FILE} 为空或不存在。")
        return

    futures: List[concurrent.futures.Future] = []
    error_count = 0
    processed_video_id: List[str] = []
    latest_user = ""

    # 遍历所有权威元数据任务
    for key, value in metadata_cache.items():
        if key in processed_video_id:
            continue

        userName = value.get("userName", "other")
        today_start = datetime.datetime.combine(datetime.date.today(), datetime.time.min).timestamp()

        user_uploads_info = analyze_user_uploads_by_day(upload_log_global)

        best_score_max = float("-inf")
        should_skip = False

        updated_entry = full_video_info(value)  # 深拷贝 + 补全

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
            print(f"⚠️ 跳过 {userName} 用户上传：之前上传失败，错误信息：{error_user_map[userName]}")
            error_count += 1
            continue

        # 选择 config
        if userName not in config_map.keys():
            # 修正原日志中的可能变量未定义问题，不改变逻辑（仍然 continue）
            print(f"⚠️ 跳过 {userName} 用户上传 请检查配置数据。video_ids={video_id_list}")
            continue
        config = config_map.get(userName, config_map["base"])

        # 检查用户今日上传数量（本地 + 平台）
        try:
            bvid_file_data = read_json(bvid_file_path)
        except Exception as e:
            print(f"❌ 读取 {bvid_file_path} 失败：{e}")
            bvid_file_data = {}
        user_videos = bvid_file_data.get(userName, [])
        recent_videos = [v for v in user_videos if v.get("created") and v["created"] >= today_start]
        remote_upload_count = len(recent_videos)

        user_info = user_uploads_info.setdefault(
            userName,
            {
                "uploads_today": 0,
                "uploads_last_hour": 0,
                "latest_upload_time": "无记录",
                "latest_timestamp": None,
            },
        )

        uploads_today = user_info.get("uploads_today", 0)
        uploads_last_hour = user_info["uploads_last_hour"]
        latest_upload_time = user_info["latest_upload_time"]
        latest_timestamp = user_info["latest_timestamp"]

        print(
            f"🔍 处理 {key} (用户: {userName}) 今日已本地上传 {uploads_today} 个视频， 实际平台数据：{remote_upload_count}  "
            f"最近一小时上传个数为: {uploads_last_hour}，最近上传时间为：{latest_upload_time}，当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        if uploads_today >= 25 or remote_upload_count >= 20:
            print(
                f"⚠️ 跳过 {userName} 用户上传：今日已本地上传 {uploads_today} 个视频， 实际平台数据：{remote_upload_count} ，达到上限。"
            )
            continue
        if latest_timestamp and (time.time() - latest_timestamp) < 1200 and uploads_last_hour >= 1:
            print(
                f"⚠️ 跳过 {userName} 用户上传：距离上次上传少于 20 分钟。 上次上传时间：{latest_upload_time}，当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            continue

        if userName == latest_user:
            print(f"⚠️ 跳过 {userName} 用户上传：与上一个上传用户相同，避免连续上传。")
            continue
        latest_user = userName

        video_path_list: List[str] = []
        origin_video_path_list: List[str] = []
        best_scheme_final: Optional[Dict[str, Any]] = None
        best_cover_path: Optional[str] = None
        comment_list_all: List[Any] = []

        # 预处理每个视频
        for video_id in video_id_list:
            video_info = full_video_info(metadata_cache.get(video_id, {}))
            origin_video_path_list.append(video_info.get("video_process_path"))
            origin_video_path_list.append(video_info.get("video_path"))
            comment_list = video_info.get("hudong", {}).get("comment_list", [])
            comment_list_all.extend(comment_list)

            best_scheme = value.get("best_scheme") or get_best_plan_by_potential(video_info.get("title_schemes", {}))
            if not best_scheme:
                print(f"⏭️ 跳过 {key}：无法选取投稿方案。")
                temp_set.add(key)
                continue

            score = float(best_scheme.get("增长潜力", {}).get("爆款潜力指数", 0))

            print(
                f"\n⏳ {userName} 开始处理任务 {key}，视频ID列表：{video_id_list}，时间：{time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            try:
                video_path, cover_path, stage_times = _preprocess_media_steps(
                    video_id, video_info, best_scheme, userName
                )
                if score > best_score_max:
                    best_score_max = score
                    best_scheme_final = best_scheme
                    best_cover_path = cover_path
                    video_path_list.insert(0, video_path)  # 高分视频放前面
                else:
                    video_path_list.append(video_path)
            except Exception as e:
                print(f"⚠️ 处理媒体过程中出现异常：{e} {video_id} {userName}")
                traceback.print_exc()
                error_count += 1
                break

        # 合并视频
        final_output_path = video_path.replace(".mp4", "_final.mp4")
        # 评论取 Top30
        comment_list_all = sorted(comment_list_all, key=lambda x: x[1], reverse=True)[:30]
        updated_entry["hudong"]["comment_list"] = comment_list_all

        merge_videos_ffmpeg(video_path_list, output_path=final_output_path)
        if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
            duration = probe_duration(final_output_path)

            # 尾部引导视频（小于 6000 秒时添加）
            new_video_path = final_output_path.replace(".mp4", "_new.mp4")
            temp_ending_video_path = final_output_path.replace(".mp4", "_ending.mp4")
            if duration < 6000:
                try:
                    origin_ending_video_path = "origin_ending_video.mp4"
                    ending_text = best_scheme_final.get("简介", {}).get(
                        "结尾语", "感谢观看本视频，欢迎点赞、评论、关注、投币、分享！"
                    )
                    gen_ending_video(ending_text, temp_ending_video_path, origin_ending_video_path)
                    merge_videos_ffmpeg([final_output_path, temp_ending_video_path], output_path=new_video_path)
                    final_output_path = new_video_path
                except Exception as e:
                    print(f"⚠️ 尾部引导视频失败，继续使用原视频：{e}")

        # 增加水印
        try:
            output_watermark_path = final_output_path.replace(".mp4", "_watermark.mp4")
            user_type = get_user_type(userName)
            start_time = time.time()
            watermark_path = get_watermark_path(user_type, userName)
            add_transparent_watermark(final_output_path, watermark_path, output_watermark_path)
            if os.path.exists(output_watermark_path) and os.path.getsize(output_watermark_path) > 0:
                print(f"✅ 水印增加成功，保存为 {output_watermark_path} 耗时 {time.time() - start_time:.2f} 秒")
                final_output_path = output_watermark_path
        except Exception as e:
            print(f"⚠️ 水印增加失败，继续使用原视频：{e}")

        # 构建上传参数
        upload_params = _build_upload_params(value, best_scheme_final, best_cover_path, final_output_path, config, userName)
        video_duration = probe_duration(final_output_path)
        video_duration_str = ms_to_time(video_duration * 1000)

        print(
            f"🚀 准备为用户 {userName} 后台投稿 {key} (ID: {video_id_key}) - 《{upload_params.get('title')}》（按账号串行）"
        )

        task_stage_times = dict(stage_times)

        # 清理文件（排除最终产物）
        all_files_to_cleanup = gen_clean_files(origin_video_path_list)
        all_files_to_cleanup = [
            f for f in all_files_to_cleanup if os.path.basename(f) != os.path.basename(final_output_path)
        ]
        print(f"🧹 预处理完成，准备清理 {len(all_files_to_cleanup)} 个临时文件。排除{final_output_path}")

        # 按账号单线程执行上传
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

    # 等待所有后台上传完成
    print("等待所有后台上传完成...")
    concurrent.futures.wait(futures, timeout=None)

    # 处理被跳过的 persistent tasks
    if len(temp_set) > 0:
        print(f"⚠️ 跳过了 {len(temp_set)} 个任务：{', '.join(temp_set)}")
        persistent_tasks = load_json(persistent_tasks_file, default={})
        persistent_tasks = set(persistent_tasks) if isinstance(persistent_tasks, list) else set()
        persistent_tasks.update(temp_set)
        save_json(persistent_tasks_file, list(persistent_tasks))

    print(f"错误数量为{error_user_map and len(error_user_map) or 0}  全部任务处理完毕。时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")


# ---------- CLI ----------
if __name__ == "__main__":
    while True:
        auto_upload()
        time.sleep(60)  # 每分钟运行一次