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

from common_utils.common_utils import get_config
config_map = {}

base_SESSDATA = get_config("bilibili_sessdata_cookie")  # 必需。你的B站登录会话 SESSDATA cookie 值。
base_BILI_JCT = get_config("bilibili_csrf_token")
base_total_cookie = get_config("bilibili_total_cookie")

config_map['base'] = (base_SESSDATA, base_BILI_JCT, base_total_cookie)

mama_SESSDATA = get_config("mama_bilibili_sessdata_cookie")  # 可选。妈妈账号的 SESSDATA cookie 值。
mama_BILI_JCT = get_config("mama_bilibili_csrf_token")
mama_total_cookie = get_config("mama_bilibili_total_cookie")
config_map['mama'] = (mama_SESSDATA, mama_BILI_JCT, mama_total_cookie)


from content_community.bilibili.bilibili_uploader import upload_to_bilibili, add_image_to_video_end, fetch_bili_topics

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


def save_json(path: str, data: dict):
    """将 data 保存成 JSON 文件（带缩进、美化）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


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


# ---------- 主逻辑 ----------
def auto_upload():
    temp_set = set()  # 用于临时存储已处理的任务，避免重复处理
    # 1. 读取元数据 & 上传日志
    metadata_cache: dict = load_json(METADATA_FILE, default={})
    upload_log: dict = load_json(UPLOAD_LOG_FILE, default={})

    if not metadata_cache:
        print(f"❌ 无可用任务：{METADATA_FILE} 为空或不存在。")
        return

    new_uploads_made = False

    # 2. 遍历权威元数据
    for key, value in metadata_cache.items():
        updated_entry = copy.deepcopy(value)

        # print("-" * 60)

        # 2.1 若日志里已记录成功投稿，则跳过
        if key in upload_log and upload_log[key].get("upload_info"):
            # print(f"✅ 已上传，跳过 {key}")
            continue

        # ---------- 数据合法性检查 ----------
        metadata = value.get('metadata')
        userName = value.get('userName', 'base')
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
        if not video_path or not os.path.exists(video_path):
            print(f"⏭️ 跳过 {key} (ID: {video_id})：视频文件缺失 -> {video_path}")
            continue

        # ---------- 预处理：在尾部插入引导图片 ----------
        new_video_path = video_path.replace('.mp4', '_new.mp4')
        try:
            add_image_to_video_end(video_path, 'final.png', new_video_path)
            video_path = new_video_path
        except Exception as e:
            print(f"⚠️  尾部插图失败，继续使用原视频：{e}")



        # ---------- 准备投稿参数 ----------
        cover_path = (
            metadata[0].get('abs_cover_path') if os.path.exists(metadata[0].get('abs_cover_path', ''))
            else best_scheme.get('封面', {}).get('图片路径', 'default_cover.jpg')
        )
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
            print(f"⚠️ 获取分区 {human_type2} 的话题失败，使用默认值。")


        description_json = best_scheme.get('简介', {})
        description = "\n".join(description_json.values()) if isinstance(description_json, dict) else str(description_json)
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

        print(f"🚀 开始投稿 {key} (ID: {video_id}) - 《{title}》")

        # ---------- 调用上传接口 ----------
        try:
            result = upload_to_bilibili(**upload_params)
        except Exception as e:
            print(f"❌ 上传接口异常：{e}")
            print(upload_params)
            continue

        # ---------- 结果处理 ----------
        if result and result.get("aid") and result.get("bvid"):
            print(f"🎉 投稿成功！AID={result['aid']}  BVID={result['bvid']}")

            # 把「权威元数据 + upload_info」写入日志字典
            updated_entry["upload_info"] = {
                "upload_params": upload_params,
                "upload_result": result,
            }
            upload_log[key] = updated_entry
            new_uploads_made = True
        else:
            err = result.get("message", "未知错误") if isinstance(result, dict) else str(result)
            print(f"❌ 投稿失败：{err}")

        # 3. 如有新成功上传，则更新日志文件
        # print("=" * 60)
        if new_uploads_made:
            try:
                save_json(UPLOAD_LOG_FILE, upload_log)
                print(f"✅ 上传日志已更新 -> {UPLOAD_LOG_FILE}")
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

    print("全部任务处理完毕。")


# ---------- CLI ----------
if __name__ == "__main__":
    while True:
        auto_upload()
        time.sleep(60)  # 每小时运行一次