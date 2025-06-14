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
from content_community.bilibili.comment import BilibiliCommenter

# --- 1. 全局配置 ---
total_cookie = get_config("bilibili_total_cookie")
CONFIG = {
    "STRATEGIES": {
        "popular": True,
        "following": True,
        "search": True,
    },
    "COOKIE": total_cookie,  # 请务必替换成你自己的COOKIE！！
    "TARGET_UIDS": [
        "443415885",
        "10330740",
    ],
    "TARGET_KEYWORDS": [
        "炉石传说",
        "互关",
        "必剪创作",
        "生活记录",
        "互关互赞",
        "影视剪辑",
        "互关互助",
        "粉丝",
        "新人",
        "UP主",
        "新人向",
    ],
    "MAX_VIDEOS_PER_SOURCE": 15,
    "PROCESSED_VIDEOS_FILE": "processed_bvideos.json",
    "REQUEST_TIMEOUT": 10,
    "REQUEST_DELAY": 1,
}

# --- 2. 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'  # 明确指定编码
)


# --- 3. API请求核心函数 ---
def send_request(url, params=None):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.bilibili.com/',
        'Cookie': CONFIG['COOKIE']
    }
    try:
        time.sleep(random.uniform(1.5, 3.5))  # 每次API请求前，随机暂停1.5到3.5秒
        response = requests.get(url, headers=headers, params=params, timeout=CONFIG['REQUEST_TIMEOUT'])
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


# --- 4. 视频获取策略实现 (保留完整信息版) ---
def fetch_from_popular():
    logging.info("开始执行 [策略一：获取热门视频]...")
    video_list = []
    url = "https://api.bilibili.com/x/web-interface/popular"
    params = {'ps': CONFIG['MAX_VIDEOS_PER_SOURCE'], 'pn': 1}
    data = send_request(url, params)
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
        data = send_request(url_template, params=params)
        if data and 'items' in data:
            found_count = 0
            for item in data['items']:
                if item.get('type') == 'DYNAMIC_TYPE_AV':
                    major = item.get('modules', {}).get('module_dynamic', {}).get('major')
                    if major and major.get('type') == 'MAJOR_TYPE_ARCHIVE':
                        video_data = major.get('archive')
                        if video_data and 'bvid' in video_data:
                            # 合并作者信息
                            author_info = item.get('modules', {}).get('module_author', {})
                            video_data['owner'] = {
                                'mid': author_info.get('mid'),
                                'name': author_info.get('name'),
                                'face': author_info.get('face'),
                            }
                            video_data['_source_strategy'] = 'following'
                            video_data['_dynamic_raw'] = item
                            video_list.append(video_data)
                            found_count += 1
                            if found_count >= CONFIG['MAX_VIDEOS_PER_SOURCE']:
                                break
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
            'order': 'pubdate',
            'page': 1,
            'ps': CONFIG['MAX_VIDEOS_PER_SOURCE']
        }
        data = send_request(url, params=params)
        if data and 'result' in data:
            found_count = 0
            for item in data['result']:
                if item.get('type') == 'video' and 'bvid' in item:
                    # 简单清除标题中可能存在的HTML标记
                    if 'title' in item:
                        item['title'] = item['title'].replace('<em class="keyword">', '').replace('</em>', '')
                    item['_source_strategy'] = 'search'
                    video_list.append(item)
                    found_count += 1
            logging.info(f"    - 从关键词 '{keyword}' 处获取 {found_count} 个视频。")
    return video_list


# --- 5. 已处理视频记录管理 ---
def load_processed_bvideos():
    filepath = CONFIG['PROCESSED_VIDEOS_FILE']
    if not os.path.exists(filepath):
        return set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    except (json.JSONDecodeError, IOError):
        return set()


def save_processed_bvideos(bvid_set):
    filepath = CONFIG['PROCESSED_VIDEOS_FILE']
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(list(bvid_set), f, indent=4)
    except IOError as e:
        logging.error(f"保存已处理视频文件失败: {e}")


# --- 6. 视频拉取主逻辑 ---
def fetch_videos():
    """
    拉取各个策略下的视频，进行去重和过滤后返回新视频列表，
    同时更新存储文件
    """
    logging.info("==================== 开始获取待处理视频 (保留完整信息) ====================")
    processed_bvideos = load_processed_bvideos()
    logging.info(f"已加载 {len(processed_bvideos)} 个已处理的视频记录。")

    all_found_videos = []
    if CONFIG['STRATEGIES']['popular']:
        all_found_videos.extend(fetch_from_popular())
    # if CONFIG['STRATEGIES']['following']:
    #     all_found_videos.extend(fetch_from_following())
    if CONFIG['STRATEGIES']['search']:
        all_found_videos.extend(fetch_from_search())

    unique_videos_map = {video['bvid']: video for video in reversed(all_found_videos) if 'bvid' in video}
    logging.info(f"所有策略共找到 {len(all_found_videos)} 个视频，去重后剩 {len(unique_videos_map)} 个。")

    videos_to_process = [video for bvid, video in unique_videos_map.items() if bvid not in processed_bvideos]
    logging.info(f"过滤掉已处理的视频后，最终得到 {len(videos_to_process)} 个新视频待处理。")

    newly_processed_bvid_set = {video['bvid'] for video in videos_to_process}
    updated_processed_set = processed_bvideos.union(newly_processed_bvid_set)
    save_processed_bvideos(updated_processed_set)
    logging.info(f"已处理视频记录已更新，总数: {len(updated_processed_set)}。")

    logging.info("==================== 获取任务完成 ====================")
    return videos_to_process


# --- 7. 视频拉取和评论的并发执行 ---
# 定义一个线程安全的全局队列，用于存储待处理的视频
videos_queue = Queue()

# 评论文本列表
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
    "互关吗朋友？你点关注，我必回访。",
    "已关注，等你回关，我们一起成长！🚀",
    "路过的朋友扩列吗？你fo我，我秒回fo。",
    "让我们互相成为粉丝吧！你关注，我回关，就这么简单。",
    "你的关注是我更新的动力，我也会用回关支持你！💪",
    "互粉互赞，有来有往，一起把数据做起来！📈",
    "你好，很高兴刷到你，可以互关一下吗？看到会马上回的。",
    "点个关注交个朋友，我也会去你主页串门回关！👋",
    "同为创作者，深知不易，抱团取暖吧！你关注我，我为你加油！🔥",
    "你的订阅对我意义重大，我也会订阅你的频道作为回报！🔔",
    "朋友，互关走一个？我24小时内必定回关。⏳",
    "我已经朝你迈出了一步（已关注），期待我们的双向奔赴！💕",
    "诚邀你加入我的“互关战队”，入队方式就是点个关注！",
    "轻轻点一下关注，我就会“咻”地一下飞到你的主页回关你！😉",
    "信誉互关，每一个关注我都不会错过，保证回访！",
    "你的主页很棒，已经关注了，期待回关，以后常来常往！",
    "别划走，点个关注，你不会损失什么，但会收获一个铁粉！😜",
    "让我们交换一个关注，作为友谊的开始。🤝",
    "我按下了你的关注按钮，现在球传给你了哦！🏀",
    "互关=回关+常互动+点赞，超值套餐不来一个吗？",
    "茫茫人海，相遇是缘。点个关注，我看到必回。",
    "你的关注，我的回关，是我们故事的开始。",
    "为梦想互关！你帮我，我帮你，一起走花路！🌸",
    "你主动，我们就有故事。关注我，我马上回关你。",
    "滴！收到你的关注信号后，我会立刻发射回关信号！📡",
    "如果你正在寻找互关伙伴，那么你找到了！关注我吧！",
    "我有一个关注名额，想留给你，等你来取！",
    "你的内容很有趣，已关注，坐等回关，以后一起交流！",
    "互关互助，共同进步，期待你的关注。",
    "朋友，你点关注，我回关注，我们就是好朋友。✅",
    "为了让大数据记住我们，互关一下吧！我必回。",
    "关注列表已为你留好位置，等你点击关注！",
    "你的关注，我的荣幸；我的回关，我的承诺。",
    "互关，不只是数字，更是互相学习的机会。",
    "我来邀请你了，互相关注一下，怎么样？😊",
    "已订阅你的频道，留言通知一下，等你回订哦！",
    "小透明求眼熟，你关注我，我回关你，让彼此不再透明。",
    "你关注，我回关，这是我们之间最简单的默契。",
    "寻找长期互关好友，你若不离，我定不弃！",
    "关注我，并给我一个暗号，我立刻回关！",
    "你的才华值得被更多人看到，我们互关，互相引流吧！",
    "你的关注票很珍贵，请投给我，我也会回投给你！",
    "如果你关注了我，记得提醒我，我怕错过你的回关邀请。",
    "一起玩转这个平台，从一个互相关注开始！",
    "你好，可以互关支持一下吗？我看到就会回的，谢谢！🙏",
    "你的关注+我的回关=我们的共同成长。",
    "互相关注，让我们的主页出现在彼此的推荐列表里！",
    "别犹豫，按下关注，我也会光速回关！",
    "你的作品很赞，已经关注，期待你的回关，成为常客！",
    "互相“订阅”，做彼此最忠实的观众！🎬",
    "来了来了，我带着我的关注走来了，你的回关在哪里？",
    "为了友谊，干了这杯“互关”酒！🍻",
    "你按下关注，我为你点亮回关，公平交易！",
    "互相关注，让我们在创作的路上不再孤单。",
    "你的名字，我想让它出现在我的关注列表里，可以吗？",
    "关注我，你将收获一个真诚的回关和持续的支持。",
    "咱们互粉一下，以后就是自己人了！",
    "如果你看到了这条评论，说明我们有缘，互关一下吧！",
    "关注我，不仅回关，我还会去你最新的作品下为你打气！",
    "点击关注，开启我们的互粉之旅！💯"
]


def video_fetcher_worker():
    """
    视频拉取线程：
      - 每隔 60 秒调用一次 fetch_videos() 拉取最新视频，
      - 对新视频进行去重后放入全局队列 videos_queue 中
    """
    while True:
        new_videos = fetch_videos()
        if new_videos:
            for video in new_videos:
                videos_queue.put(video)
                # logging.info(f"视频加入队列：BVID {video.get('bvid')} 标题：{video.get('title')}")
        else:
            logging.info("本次未获取到新视频。")
        logging.info(f'本次获取到 {len(new_videos)} 个新视频，已添加到队列中。队列当前长度：{videos_queue.qsize()}')
        time.sleep(random.uniform(1200, 1800))  # 每次拉取大循环，随机暂停2到3分钟


def comment_worker():
    """
    评论线程：
      - 不断地从 videos_queue 中获取视频，
      - 每次评论后等待 10 秒（确保评论间隔不小于 10 秒）
    """
    while True:
        try:
            video = videos_queue.get(timeout=5)
        except Empty:
            continue  # 如果队列为空，则继续等待
        bvid = video.get('bvid')
        if not bvid:
            continue
        # 此处使用时间戳对评论列表取余选取评论内容
        comment_text = random.choice(comment_list)  # 更简单，更随机

        # comment_text = comment_list[int(time.time()) % len(comment_list)]
        success = commenter.post_comment(bvid, comment_text, 1)
        logging.info(f"开始处理视频评论：BVID {bvid} 标题：{video.get('title')} 评论内容：{comment_text} 成功：{success}")
        # 将video_duration映射到10到30秒之间
        time.sleep(random.uniform(5, 30))  # 每次评论后，随机暂停15到45秒，这个间隔要拉长，评论太快是高危行为


if __name__ == '__main__':
    csrf_token = get_config("bilibili_csrf_token")
    total_cookie = get_config("bilibili_total_cookie")

    if not csrf_token or not total_cookie:
        print("错误：请在 common_utils.common_utils.get_config 中配置 csrf_token 和 total_cookie。")
        exit()

    # --- 实例化评论器 ---
    commenter = BilibiliCommenter(total_cookie=total_cookie, csrf_token=csrf_token)

    # 启动视频拉取线程（后台线程）
    video_thread = threading.Thread(target=video_fetcher_worker, name="VideoFetcherWorker", daemon=True)
    video_thread.start()

    # 启动评论线程（后台线程）
    comment_thread = threading.Thread(target=comment_worker, name="CommentWorker", daemon=True)
    comment_thread.start()

    # 保持主线程运行，直到手动中断
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("程序已终止。")