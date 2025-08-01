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

import json
import os
import copy
import time
import traceback

from common_utils.common_utils import get_config, format_seconds_to_mmss
from common_utils.video_utils import add_image_to_video_end, get_video_duration_seconds, create_enhanced_cover, \
    merge_videos_ffmpeg
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


# 定义需要处理的账号名及其对应的config_map键名（区分大小写）
accounts = {
    'tao': 'tao',
    'ruru': 'ruru',
    'nana': 'nana',
    'jie': 'jie',
    'qiqi': 'qiqi',
    # 'mama': 'mama',
    # 'hong': 'hong',
    # 'su': 'su',
    'yan': 'yan',
}

for name, map_key in accounts.items():
    sessdata = get_config(f"{name}_bilibili_sessdata_cookie")
    bili_jct = get_config(f"{name}_bilibili_csrf_token")
    total_cookie = get_config(f"{name}_bilibili_total_cookie")
    config_map[map_key] = (sessdata, bili_jct, total_cookie)

error_user_map = {}
from content_community.bilibili.bilibili_uploader import upload_to_bilibili, fetch_bili_topics

# ---------- 文件路径常量 ----------
METADATA_FILE = '../../LLM/TikTokDownloader/metadata_cache.json'           # 权威源
UPLOAD_LOG_FILE = '../../LLM/TikTokDownloader/metadata_cache_with_uploads.json'  # 上传日志
persistent_tasks_file = "../../LLM/TikTokDownloader/persistent_tasks.json"

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
def auto_upload():
    temp_set = set()  # 用于临时存储已处理的任务，避免重复处理
    # 1. 读取元数据 & 上传日志
    metadata_cache: dict = load_json(METADATA_FILE, default={})
    upload_log: dict = load_json(UPLOAD_LOG_FILE, default={})
    if not upload_log:
        print(f"❌ 上传日志文件 {UPLOAD_LOG_FILE} 不存在或为空，请检查。")
        return

    if not metadata_cache:
        print(f"❌ 无可用任务：{METADATA_FILE} 为空或不存在。")
        return


    new_uploads_made = False
    error_count = 0
    # 2. 遍历权威元数据
    for key, value in metadata_cache.items():
        start_time = time.time()
        updated_entry = copy.deepcopy(value)
        status = value.get('status', '未处理')
        if status == 'error':
            error_count += 1
            continue

        # if key in upload_log and upload_log[key].get('status') == 'error':
        #     print(f"⏭️ 跳过 {key}：之前重制失败，已标记")
        #     continue
        # print("-" * 60)

        # 2.1 若日志里已记录成功投稿，则跳过
        if key in upload_log and upload_log[key].get("upload_info"):
            # print(f"✅ 已上传，跳过 {key}")
            continue

        # ---------- 数据合法性检查 ----------
        metadata = value.get('metadata')
        userName = value.get('userName', 'other')
        if userName in error_user_map:
            print(f"⚠️ 跳过 {userName} 用户上传：之前上传失败，错误信息：{error_user_map[userName]}")
            error_count += 1
            continue
        if userName not in config_map.keys():
            print(f"⚠️ 跳过 {userName} 用户上传 请检查配置数据。")
            error_count += 1
            userName = 'base'
            continue
        config = config_map.get(userName, config_map['base'])
        print(f"🔍 处理 {key} (用户: {userName})")
        if not (isinstance(metadata, list) and metadata):
            print(f"⏭️ 跳过 {key}：metadata 字段缺失或格式错误。{metadata}")
            continue

        # ---------- 选择最佳投稿方案 ----------
        best_scheme = value.get('best_scheme') or get_best_plan_by_potential(
            value.get('title_schemes', {})
        )
        if not best_scheme:
            print(f"⏭️ 跳过 {key}：无法选取投稿方案。")
            temp_set.add(key)  # 添加到临时集合，避免重复处理
            continue

        video_id = metadata[0].get('id')
        if not video_id:
            print(f"⏭️ 跳过 {key}：metadata 中缺少 id。")
            continue

        video_path = value.get('video_path')
        duration = metadata[0].get('duration', "00:10")
        duration = time_str_to_seconds(duration)  # 确保 duration 格式正确
        if not video_path or not os.path.exists(video_path):
            print(f"⏭️ 跳过 {key} (ID: {video_id})：视频文件缺失 -> {video_path}")
            continue

        current_video_path = video_path  # 默认新视频路径为原视频路径
        generation_options = value.get('generation_options', {})
        if generation_options.get('remake_video', False):
            # 如果需要重制视频，则调用重制函数
            print(f"🔄 重制视频 {video_path}... userName: {userName}")
            try:
                final_video_path = remake_video_robust(video_path, bgm_library_path='../app/bgm_audio', force_regenerate=True)
                if final_video_path:
                    print(f"✅ 重制视频成功，保存为 {final_video_path}")
                    video_path = final_video_path
                    current_video_path = final_video_path
                else:
                    upload_log[key] = upload_log.get(key, {})
                    upload_log[key]['status'] = 'error'
                    save_json(UPLOAD_LOG_FILE, upload_log)
                    print(f"❌ 重制视频失败")
                    error_count += 1
                    continue
                # 重制后的视频路径仍然是 video_path
            except Exception as e:
                upload_log[key] = upload_log.get(key, {})
                upload_log[key]['status'] = 'error'
                save_json(UPLOAD_LOG_FILE, upload_log)
                print(f"❌ 重制视频失败：{e}")
                error_count += 1
                continue
        # ---------- 预处理：在尾部插入引导图片 ----------
        new_video_path = current_video_path.replace('.mp4', '_new.mp4')
        try:
            image_duration = int(duration / 100)
            image_duration = max(1, image_duration)
            print(f"🔄 尾部插图处理：视频时长 {duration} 秒，插图持续 {image_duration} 秒。 文件路径：{current_video_path} -> {new_video_path}")
            final_jpg_path = f'{userName}_final.jpg'
            if not os.path.exists(final_jpg_path):
                final_jpg_path = 'final.jpg'
                print(f"⚠️ 尾部插图文件 {final_jpg_path} 不存在，使用默认图片。")
            add_image_to_video_end(current_video_path, final_jpg_path, new_video_path, image_duration)
            video_path = new_video_path
        except Exception as e:
            print(f"⚠️  尾部插图失败，继续使用原视频：{e}")

        temp_video_path = video_path.replace('.mp4', '_temp.mp4')
        try:
            if generation_options.get('add_epilogue', False):
                print(f"🔄 添加结尾视频片段到 {video_path}... userName: {userName}")
                copyright_video_path = f'{userName}_final.mp4'
                if not os.path.exists(copyright_video_path):
                    copyright_video_path = 'final.mp4'
                    print(f"⚠️ 版权视频文件 {copyright_video_path} 不存在，使用默认视频。")
                video_path_list = [video_path, copyright_video_path]
                merge_videos_ffmpeg(video_path_list, output_path=temp_video_path)
                if os.path.exists(temp_video_path):
                    video_path = temp_video_path
                    print(f"✅ 合并视频成功，保存为 {temp_video_path}")
        except Exception as e:
            print(f"⚠️ 合并视频失败：{e}")


        try:
            # ---------- 准备投稿参数 ----------
            cover_path = (
                metadata[0].get('abs_cover_path') if os.path.exists(metadata[0].get('abs_cover_path', ''))
                else best_scheme.get('封面', {}).get('图片路径', 'default_cover.jpg')
            )
            output_image_path = cover_path.replace('.jpg', '_enhanced.jpg')
            create_enhanced_cover(
                input_image_path=cover_path,
                output_image_path=output_image_path,
                text_lines=[best_scheme.get('封面', {}).get('配文', '')],
            )
            cover_path = output_image_path if os.path.exists(output_image_path) else cover_path
        except Exception as e:
            traceback.print_exc()
            print(f"⚠️  封面处理失败：{e}")

        origin_tag = metadata[0].get('tag', [])
        origin_tag.extend(metadata[0].get('text_extra', []))
        title = best_scheme.get('标题', '欢迎来看我的视频！')
        human_type2 = best_scheme.get('分区编号', 21)
        topic_json = fetch_bili_topics(config[2], type_pid=human_type2)
        topic_name = '骑行去追夏天的风'  # 默认话题名称
        topic_id = 1313687  # 默认话题ID
        topic_detail = {
            "from_topic_id": 1313687,
            "from_source": "arc.web.recommend",
            'topic_name': '骑行去追夏天的风'
        }
        # 尝试获取data字段下面的topics列表中一个个元素的topic_id
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
        # 只拼接存在的指定字段
        description = "\n".join(
            str(description_json[k]) for k in target_keys if k in description_json
        )
        tags = best_scheme.get('标签', ['AI修复', '视频剪辑'])
        origin_tag.extend(tags)  # 保留原始标签
        # 去重origin_tag
        tags = list(set(origin_tag))
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

        print(f"🚀 开始投稿 {key} (ID: {video_id}) - 《{title}》 userName: {userName}，封面：{cover_path}，分区：{human_type2}，话题：{topic_name}，话题ID：{topic_id}。")

        # ---------- 调用上传接口 ----------
        max_retries = 3
        result = None
        for attempt in range(1, max_retries + 1):
            try:
                result = upload_to_bilibili(**upload_params)
                break
            except Exception as e:
                print(f"❌ 上传接口异常 (第 {attempt} 次重试)：{e}")
                print(upload_params)
                if attempt < max_retries:
                    print("等待 1 分钟后重试…")
                    time.sleep(60)
                else:
                    print("已达最大重试次数，放弃本次上传。")

        # ---------- 结果处理 ----------
        if result and result.get("aid") and result.get("bvid"):
            print(f"🎉 投稿成功！AID={result['aid']}  BVID={result['bvid']} 时间：{time.strftime('%Y-%m-%d %H:%M:%S')} username {userName} 耗时 {time.time() - start_time:.2f} 秒。")
            try:
                final_duration_sec = get_video_duration_seconds(video_path)
                if final_duration_sec is not None:
                    formatted_duration = format_seconds_to_mmss(final_duration_sec)
                    print(f"ℹ️ 获取到最终视频时长: {formatted_duration}，正在更新元数据...")
                    # 确保 metadata 结构符合预期再更新
                    if 'metadata' in updated_entry and isinstance(updated_entry['metadata'], list) and updated_entry[
                        'metadata']:
                        updated_entry['metadata'][0]['duration'] = formatted_duration
                    else:
                        print("⚠️ 无法在 updated_entry 中找到 'metadata' 列表来更新时长。")
                else:
                    print("⚠️ 未能获取最终视频时长，metadata 中的 duration 字段将不被更新。")
                if os.path.exists(video_path):
                    os.remove(video_path)
                if new_video_path and os.path.exists(new_video_path):
                    os.remove(new_video_path)
                if temp_video_path and os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
            except Exception as e:
                print(f"⚠️ 删除视频文件失败：{e}")

            # 把「权威元数据 + upload_info」写入日志字典
            updated_entry["upload_info"] = {
                "upload_params": upload_params,
                "upload_result": result,
            }
            upload_log[key] = updated_entry
            new_uploads_made = True
        else:
            err = result.get("message", "未知错误") if isinstance(result, dict) else str(result)
            error_user_map[userName] = err
            print(f"❌ 投稿失败：{err}")

        # 3. 如有新成功上传，则更新日志文件
        # print("=" * 60)
        if new_uploads_made:
            try:
                save_json(UPLOAD_LOG_FILE, upload_log)
                print(f"✅ 上传日志已更新 -> {UPLOAD_LOG_FILE} 耗时 {time.time() - start_time:.2f} 秒。")
            except IOError as e:
                print(f"🔥 写入日志文件失败：{e}")
        else:
            print("本次运行没有新的成功投稿。")

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