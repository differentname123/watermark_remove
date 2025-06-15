#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import requests
import time
import logging
import os
import json
import threading
from queue import Queue, Empty

from common_utils.common_utils import get_config
# 评论相关代码保留，但暂时不使用
from content_community.bilibili.comment import BilibiliCommenter

# --- 1. 全局常量 ---
URL_MODIFY_RELATION = "https://api.bilibili.com/x/relation/modify"

# --- 2. 全局配置 ---
total_cookie = get_config("bilibili_total_cookie")
csrf_token = get_config("bilibili_csrf_token")

CONFIG = {
    "STRATEGIES": {
        "popular": False,  # 热门视频通常不是目标用户，可以关闭
        "following": False,  # 已经关注的UP主不需要再处理
        "search": True,
    },
    "COOKIE": total_cookie,
    "CSRF_TOKEN": csrf_token,
    "TARGET_UIDS": [  # 监控动态时使用，当前已关闭
        "443415885",
        "10330740",
    ],
    "TARGET_KEYWORDS": [  # 用于搜索视频的关键词
        "互关",
        "互粉",
        "互赞",
        "互助",
        "新人UP主",
    ],
    "FOLLOW_KEYWORDS": [  # 用于判断是否要关注的关键词
        "互关",
        "互粉",
        "回关",
        "互赞",
        "互助",
    ],
    "MAX_VIDEOS_PER_SOURCE": 20,  # 每次搜索可以多拉取一些
    "PROCESSED_VIDEOS_FILE": "processed_bvideos.json",
    "PROCESSED_FIDS_FILE": "processed_fids.json",  # 新增：记录已处理的用户ID
    "REQUEST_TIMEOUT": 10,
    "REQUEST_DELAY": 1,
}

# --- 3. 日志与会话配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# 创建一个全局会话对象，用于保持登录状态
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://www.bilibili.com/',
    'Cookie': CONFIG['COOKIE']
})


# --- 4. API请求核心函数 ---
def send_get_request(url, params=None):
    """通用GET请求函数"""
    try:
        # 每次API请求前，随机暂停
        time.sleep(random.uniform(1.5, 3.5))
        response = session.get(url, params=params, timeout=CONFIG['REQUEST_TIMEOUT'])
        response.raise_for_status()
        data = response.json()
        if data.get('code', 0) != 0:
            logging.warning(f"API返回错误: code={data.get('code')}, message={data.get('message')}, url={response.url}")
            return None
        return data.get('data')
    except requests.exceptions.RequestException as e:
        logging.error(f"网络请求失败: {e}")
    except json.JSONDecodeError:
        logging.error("无法解析服务器返回的JSON数据。")
    return None


def modify_relation(fid, action_type, csrf_token):
    """
    修改用户关系 (关注或取消关注)。
    fid: 目标用户的UID
    action_type: 1 为关注, 2 为取消关注
    csrf_token: 从Cookie中获取的bili_jct值
    """
    action_text = "关注" if action_type == 1 else "取消关注"
    payload = {
        "fid": fid,
        "act": action_type,
        "re_src": 11,  # 关系来源，通常用 11
        "csrf": csrf_token
    }
    try:
        response = session.post(URL_MODIFY_RELATION, data=payload, timeout=CONFIG['REQUEST_TIMEOUT'])
        response.raise_for_status()
        result = response.json()
        if result.get('code') == 0:
            logging.info(f"  {'✅' if action_type == 1 else '🗑️'} 成功{action_text} UID: {fid}")
            return True
        # 常见错误码处理
        elif result.get('code') == 22014:  # 对方将你拉黑
            logging.warning(f"  ⚠️ {action_text} UID: {fid} 失败: {result['message']} (可能已被对方拉黑)")
            return True  # 返回True，避免重试
        elif result.get('code') == 22007:  # 已经关注了
            logging.info(f"  ℹ️ {action_text} UID: {fid}: 已经是关注状态。")
            return True  # 返回True，避免重试
        else:
            logging.error(
                f"  ❌ {action_text} UID: {fid} 失败: {result.get('message', '未知错误')} (Code: {result.get('code')})")
            return False
    except requests.exceptions.RequestException as e:
        logging.error(f"  ❌ 请求{action_text} UID: {fid} 失败: {e}")
        return False
    except ValueError:  # 对应 json.JSONDecodeError
        logging.error(f"  ❌ {action_text} UID: {fid} 响应内容不是有效的 JSON。")
        return False


# --- 5. 视频获取策略实现 ---
def fetch_from_popular():
    logging.info("开始执行 [策略一：获取热门视频]...")
    video_list = []
    url = "https://api.bilibili.com/x/web-interface/popular"
    params = {'ps': CONFIG['MAX_VIDEOS_PER_SOURCE'], 'pn': 1}
    data = send_get_request(url, params)
    if data and 'list' in data:
        for item in data['list']:
            if 'bvid' in item:
                item['_source_strategy'] = 'popular'
                video_list.append(item)
        logging.info(f"  > 成功从热门榜单获取 {len(video_list)} 个视频。")
    else:
        logging.warning("  > 从热门榜单获取视频失败。")
    return video_list


def fetch_from_following():
    logging.info("开始执行 [策略二：监控关注的UP主]...")
    if not CONFIG['TARGET_UIDS']:
        logging.warning("  > 未配置目标UID，跳过此策略。")
        return []
    video_list = []
    url_template = "https://api.bilibili.com/x/polymer/web-dynamic/v1/feed/space"
    for uid in CONFIG['TARGET_UIDS']:
        logging.info(f"  > 正在获取UP主(UID: {uid})的最新动态...")
        params = {'host_mid': uid}
        data = send_get_request(url_template, params=params)
        if data and 'items' in data:
            found_count = 0
            for item in data['items']:
                if item.get('type') == 'DYNAMIC_TYPE_AV':
                    major = item.get('modules', {}).get('module_dynamic', {}).get('major')
                    if major and major.get('type') == 'MAJOR_TYPE_ARCHIVE':
                        video_data = major.get('archive')
                        if video_data and 'bvid' in video_data:
                            author_info = item.get('modules', {}).get('module_author', {})
                            video_data['owner'] = {
                                'mid': author_info.get('mid'),
                                'name': author_info.get('name'),
                                'face': author_info.get('face'),
                            }
                            # 补全mid字段，与搜索结果对齐
                            if 'mid' not in video_data:
                                video_data['mid'] = author_info.get('mid')
                            video_data['_source_strategy'] = 'following'
                            video_list.append(video_data)
                            found_count += 1
                            if found_count >= CONFIG['MAX_VIDEOS_PER_SOURCE']: break
            logging.info(f"    - 从UID {uid} 处获取 {found_count} 个新视频。")
    return video_list


def fetch_from_search():
    logging.info("开始执行 [策略三：关键词搜索]...")
    if not CONFIG['TARGET_KEYWORDS']:
        logging.warning("  > 未配置目标关键词，跳过此策略。")
        return []
    video_list = []
    url = "https://api.bilibili.com/x/web-interface/search/type"
    for keyword in CONFIG['TARGET_KEYWORDS']:
        logging.info(f"  > 正在搜索关键词 '{keyword}'...")
        params = {
            'search_type': 'video',
            'keyword': keyword,
            'order': 'pubdate',  # 按最新发布排序
            'page': 1,
            'ps': CONFIG['MAX_VIDEOS_PER_SOURCE']
        }
        data = send_get_request(url, params=params)
        if data and 'result' in data:
            found_count = 0
            search_results = data.get('result', [])
            # 兼容老版本和新版本API的返回格式
            if not isinstance(search_results, list):
                search_results = data.get('result', {}).get('video', [])

            for item in search_results:
                if item.get('type') == 'video' and 'bvid' in item:
                    if 'title' in item:
                        item['title'] = item['title'].replace('<em class="keyword">', '').replace('</em>', '')
                    item['_source_strategy'] = 'search'
                    video_list.append(item)
                    found_count += 1
            logging.info(f"    - 从关键词 '{keyword}' 处获取 {found_count} 个视频。")
    return video_list


# --- 6. 已处理记录管理 (视频BVID和用户FID) ---
def load_processed_set(filepath):
    if not os.path.exists(filepath):
        return set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    except (json.JSONDecodeError, IOError):
        return set()


def save_processed_set(data_set, filepath):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # 将集合转换为列表以便JSON序列化
            json.dump(list(data_set), f, indent=4)
    except IOError as e:
        logging.error(f"保存文件 {filepath} 失败: {e}")


# --- 7. 视频拉取主逻辑 ---
def fetch_videos():
    logging.info("==================== 开始获取待处理视频 ====================")
    processed_bvideos = load_processed_set(CONFIG['PROCESSED_VIDEOS_FILE'])
    logging.info(f"已加载 {len(processed_bvideos)} 个已处理的视频记录。")

    all_found_videos = []
    if CONFIG['STRATEGIES']['popular']:
        all_found_videos.extend(fetch_from_popular())
    if CONFIG['STRATEGIES']['following']:
        all_found_videos.extend(fetch_from_following())
    if CONFIG['STRATEGIES']['search']:
        all_found_videos.extend(fetch_from_search())

    unique_videos_map = {video['bvid']: video for video in reversed(all_found_videos) if 'bvid' in video}
    logging.info(f"所有策略共找到 {len(all_found_videos)} 个视频，去重后剩 {len(unique_videos_map)} 个。")

    videos_to_process = [video for bvid, video in unique_videos_map.items() if bvid not in processed_bvideos]
    logging.info(f"过滤掉已处理的视频后，最终得到 {len(videos_to_process)} 个新视频待处理。")

    newly_processed_bvid_set = {video['bvid'] for video in videos_to_process}
    updated_processed_set = processed_bvideos.union(newly_processed_bvid_set)
    save_processed_set(updated_processed_set, CONFIG['PROCESSED_VIDEOS_FILE'])
    logging.info(f"已处理视频记录已更新，总数: {len(updated_processed_set)}。")

    logging.info("==================== 获取任务完成 ====================")
    return videos_to_process


# --- 8. 并发执行逻辑 ---
videos_queue = Queue()

# (评论功能保留，暂不启用)
comment_list = [
    "如果你喜欢我的内容，不妨关注一下？我也会回关你的！🤝",
    "希望和大家一起进步，关注我，我会回访你的频道。😊",
    "如果你关注我了，请告诉我，我一定会回关的。🙏",
    "想和更多朋友互关，如果你关注我，我也会关注你哦！👍",
    "新朋友互关吗？关注我，我也会支持你！",
    "互相关注，共同发展，我期待你的关注和我的回关。",
    "非常乐意和大家互关，关注我，我立刻回粉！",
    "为了更好的交流，我们互相关注吧？我也会去你的频道。👀",
    "欢迎关注我，我也会关注回来的，一起加油！",
    "如果你订阅了我的频道，留言告诉我，我也会去订阅你的！",
    "一起为梦想努力，关注我，我也会回关帮你点赞。",
    "寻找志同道合的朋友互关，关注我，我必回关！",
    "想扩大圈子，关注我，我也会去你的频道留言并关注。",
    "你的关注是对我最大的支持，我也会用关注回报你！",
    "咱们互相支持，你关注我，我也会关注你。✅",
    "小透明求互关，关注我，我秒回！💯",
    "如果你按下关注键，我也会同样按下你的关注键，一起成长！",
    "互关吗朋友？你点关注，我必回访。"
]


def video_fetcher_worker():
    """视频拉取线程：定期拉取新视频并放入队列。"""
    while True:
        new_videos = fetch_videos()
        if new_videos:
            # 随机打乱顺序，避免行为模式过于固定
            random.shuffle(new_videos)
            for video in new_videos:
                videos_queue.put(video)
        else:
            logging.info("本次未获取到新视频。")
        logging.info(f'本次获取到 {len(new_videos)} 个新视频。队列当前长度：{videos_queue.qsize()}')
        # 每次拉取大循环，随机暂停20到30分钟
        sleep_time = random.uniform(120, 180)
        logging.info(f"视频拉取线程休眠 {int(sleep_time / 60)} 分钟...")
        time.sleep(sleep_time)


# (评论功能保留，暂不启用)
def comment_worker():
    """评论线程：从队列获取视频并发表评论。"""
    commenter = BilibiliCommenter(CONFIG['COOKIE'], CONFIG['CSRF_TOKEN'])
    while True:
        try:
            video = videos_queue.get(timeout=30)
        except Empty:
            continue
        bvid = video.get('bvid')
        if not bvid:
            continue
        comment_text = random.choice(comment_list)
        logging.info(f"准备评论视频：BVID {bvid} | 标题：{video.get('title')}")
        success = commenter.post_comment(bvid, comment_text, 1)
        if success:
            logging.info(f"  > 评论成功: '{comment_text}'")
        else:
            logging.error(f"  > 评论失败。")
        time.sleep(random.uniform(20, 45))


# (新功能)
def follower_worker(csrf_token):
    """关注线程：从队列获取视频，判断是否需要关注作者。"""
    processed_fids = load_processed_set(CONFIG['PROCESSED_FIDS_FILE'])
    logging.info(f"已加载 {len(processed_fids)} 个已处理的用户(fid)记录。")

    while True:
        try:
            video = videos_queue.get(timeout=30)  # 等待30秒，如果没有新视频则继续循环
        except Empty:
            continue

        title = video.get('title', '')
        desc = video.get('description', '')
        # 兼容不同API返回的用户ID字段 ('mid' 或 'owner.mid')
        author_id = video.get('mid')
        if not author_id and 'owner' in video and isinstance(video['owner'], dict):
            author_id = video['owner'].get('mid')

        if not author_id:
            logging.warning(f"视频 BVID {video.get('bvid')} 缺少作者ID，跳过。")
            continue

        # 如果用户ID已经处理过，则跳过
        if author_id in processed_fids:
            logging.debug(f"用户 UID {author_id} 已在处理列表，跳过。")
            continue

        # 检查标题或描述是否包含关注关键词
        text_to_check = f"{title} {desc}".lower()
        should_follow = any(keyword.lower() in text_to_check for keyword in CONFIG['FOLLOW_KEYWORDS'])

        if should_follow:
            author_name = video.get('author') or (video.get('owner') and video['owner'].get('name'))
            logging.info(
                f"发现目标用户: {author_name} (UID: {author_id}) | 来源: BVID {video.get('bvid')} | 标题: {title}")

            # 随机暂停一段时间再执行关注，模拟人类行为
            time.sleep(random.uniform(20, 45))
            success = modify_relation(author_id, 1, csrf_token)

            # 无论成功与否（包括已关注/被拉黑等情况），都将其标记为已处理，避免重复请求
            if success:
                processed_fids.add(author_id)
                save_processed_set(processed_fids, CONFIG['PROCESSED_FIDS_FILE'])
        else:
            # 即使不关注，也标记为已处理，避免重复检查该用户
            processed_fids.add(author_id)
            save_processed_set(processed_fids, CONFIG['PROCESSED_FIDS_FILE'])
            logging.debug(f"视频 BVID {video.get('bvid')} 未匹配到关注关键词，作者UID {author_id} 已标记为无需处理。")

        # 每次处理后都暂停，控制API请求频率
        time.sleep(random.uniform(3, 8))


if __name__ == '__main__':
    if not CONFIG['COOKIE'] or not CONFIG['CSRF_TOKEN']:
        logging.error(
            "错误：请在 common_utils.common_utils.get_config 中配置 bilibili_total_cookie 和 bilibili_csrf_token。")
        exit()

    logging.info("程序启动...")

    # 启动视频拉取线程
    video_thread = threading.Thread(target=video_fetcher_worker, name="VideoFetcherWorker", daemon=True)
    video_thread.start()

    # --- 启动关注线程 ---
    follower_thread = threading.Thread(target=follower_worker, args=(CONFIG['CSRF_TOKEN'],), name="FollowerWorker",
                                       daemon=True)
    follower_thread.start()

    # --- 评论线程已暂停 ---
    # logging.info("评论功能已暂停。如需启用，请取消主程序中的相关代码注释。")
    # comment_thread = threading.Thread(target=comment_worker, name="CommentWorker", daemon=True)
    # comment_thread.start()

    # 保持主线程运行
    try:
        while True:
            logging.info(f"主线程运行中... 当前待处理视频队列长度: {videos_queue.qsize()}")
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n程序被用户中断，正在退出...")