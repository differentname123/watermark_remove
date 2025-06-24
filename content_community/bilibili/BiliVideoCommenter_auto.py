#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import random
import traceback

import requests
import time
import logging
import os
import json
import threading
from queue import Queue, Empty
from  content_community.bilibili.get_comment import get_bilibili_comments
from common_utils.common_utils import get_config
# 评论相关代码保留，但暂时不使用
from content_community.bilibili.comment import BilibiliCommenter
from content_community.bilibili.get_danmu import gen_proper_comment

# --- 1. 全局常量 ---
URL_MODIFY_RELATION = "https://api.bilibili.com/x/relation/modify"

# --- 2. 全局配置 ---
total_cookie = get_config("bilibili_total_cookie")
csrf_token = get_config("bilibili_csrf_token")

CONFIG = {
    "STRATEGIES": {
        "popular": True,      # 热门视频通常不是目标用户，可以关闭
        "following": False,   # 已经关注的UP主不需要再处理
        "search": False,
        "ranking": True,      # <<< NEW: 新增分区排行榜策略开关
    },
    "COOKIE": total_cookie,
    "CSRF_TOKEN": csrf_token,
    "TARGET_UIDS": [  # 监控动态时使用，当前已关闭
        "443415885",
        "10330740",
    ],
    # <<< NEW: START - 新增分区排行榜相关配置 >>>
    "RANKING_TIDS": { # 目标分区ID (rid) 和名称的映射
        0: "全站",
        1: "动画",
        168: "国创",
        3: "音乐",
        129: "舞蹈",
        4: "游戏",
        36: "知识",
        188: "科技",
        234: "运动",
        223: "汽车",
        160: "生活",
        211: "美食",
        217: "动物圈",
        119: "鬼畜",
        155: "时尚",
        5: "娱乐",
        181: "影视",
    },
    # <<< NEW: END - 新增分区排行榜相关配置 >>>
    "TARGET_KEYWORDS": [
        "互关", "互粉", "互赞", "互助", "新人UP主", "回关", "回粉", "互暖",
        "互评", "互捞", "三连", "求三连", "互三连", "互币", "新人报道", "新人up",
        "小UP主", "萌新UP", "底层UP主", "小透明", "涨粉", "求关注", "求抱团",
        "抱团取暖", "一起加油", "挑战100粉", "冲击千粉", "有粉必回", "有赞必回",
        "在线秒回", "已关求回"
    ],
    "FOLLOW_KEYWORDS": [
        "互关", "互粉", "回关", "互赞", "互助", "回粉", "必回", "必回关",
        "有粉必回", "有访必回", "诚信互关", "诚信互粉", "永不取关", "不取关",
        "赞评必回", "互赞互评", "互三连", "互币", "关我必回", "私信秒回",
        "你关我就关"
    ],
    "MAX_VIDEOS_PER_SOURCE": 20,  # 每次搜索/每个分区排行可以多拉取一些
    "PROCESSED_VIDEOS_FILE": "comment_processed_bvideos.json",
    "GEN_PROCESSED_VIDEOS_FILE": "gen_comment_processed_bvideos.json",
    "COMMENTED_PROCESSED_VIDEOS_FILE": "commented_processed_bvideos.json",

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
def fetch_from_popular(max_count=100):
    """
    循环获取B站热门榜单的视频，直到没有更多数据为止。
    """
    logging.info("开始执行 [策略一：获取热门视频]...")

    # 将 video_list 初始化在循环外部，用于累加所有页的数据
    all_videos = []
    url = "https://api.bilibili.com/x/web-interface/popular"
    page_number = 1  # 从第一页开始

    while True:
        logging.info(f"  > 正在尝试获取热门榜单第 {page_number} 页...")
        params = {'ps': CONFIG['MAX_VIDEOS_PER_SOURCE'], 'pn': page_number}

        data = send_get_request(url, params)

        # 检查API响应是否成功，并且 'list' 键存在且不为空
        if data and 'list' in data and data['list']:
            page_videos = data['list']
            for item in page_videos:
                if 'bvid' in item:
                    item['_source_strategy'] = 'popular'
                    all_videos.append(item)

            logging.info(f"  > 成功从第 {page_number} 页获取 {len(page_videos)} 个视频。")
            if len(all_videos) >= max_count:
                logging.info(f"  > 已达到最大获取数量 {max_count}，停止获取更多数据。")
                break
            # 准备获取下一页
            page_number += 1

            # 增加延时，避免请求过快被API限制。可根据需要调整时间。
            time.sleep(1)
        else:
            # 如果 'list' 不存在、为空，或者API请求失败，则认为没有更多数据了
            logging.info("  > 热门榜单数据已全部获取完毕，或API未返回有效数据，停止获取。")
            break  # 退出循环

    if all_videos:
        logging.info(f"  > [策略一：获取热门视频] 执行完毕。总共获取 {len(all_videos)} 个视频。")
    else:
        logging.warning("  > [策略一：获取热门视频] 执行完毕，但未能获取到任何视频。")

    return all_videos


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

    # 定义每页获取的数据量
    PAGE_SIZE = 20

    for keyword in CONFIG['TARGET_KEYWORDS']:
        logging.info(f"  > 正在搜索关键词 '{keyword}'...")

        current_page = 1
        videos_fetched_for_keyword = 0  # 记录当前关键词已获取的视频数量

        while videos_fetched_for_keyword < CONFIG['MAX_VIDEOS_PER_SOURCE']:
            params = {
                'search_type': 'video',
                'keyword': keyword,
                'order': 'pubdate',  # 按最新发布排序
                'page': current_page,
                'ps': PAGE_SIZE  # 固定每页20个
            }

            logging.info(f"    - 请求第 {current_page} 页，目标获取 {PAGE_SIZE} 个视频...")
            data = send_get_request(url, params=params)

            if not data or 'result' not in data:
                logging.warning(
                    f"      - 未能获取到关键词 '{keyword}' 第 {current_page} 页的数据，或数据格式不正确。停止此关键词的搜索。")
                break  # 无法获取数据，停止当前关键词的搜索

            search_results = data.get('result', [])
            # 兼容老版本和新版本API的返回格式
            if not isinstance(search_results, list):
                search_results = data.get('result', {}).get('video', [])

            if not search_results:
                logging.info(f"      - 关键词 '{keyword}' 第 {current_page} 页没有更多视频了。")
                break  # 当前页没有数据，说明已经到头了

            page_videos_added = 0  # 记录当前页实际添加的视频数量
            for item in search_results:
                if item.get('type') == 'video' and 'bvid' in item:
                    if 'title' in item:
                        item['title'] = item['title'].replace('<em class="keyword">', '').replace('</em>', '')
                    item['_source_strategy'] = 'search'
                    video_list.append(item)
                    videos_fetched_for_keyword += 1
                    page_videos_added += 1

                    # 如果已经达到或超过了目标数量，就停止
                    if videos_fetched_for_keyword >= CONFIG['MAX_VIDEOS_PER_SOURCE']:
                        break  # 跳出 inner loop (for item in search_results)

            logging.info(
                f"      - 从关键词 '{keyword}' 第 {current_page} 页获取 {page_videos_added} 个视频，当前关键词累计 {videos_fetched_for_keyword} 个。")

            # 如果当前页获取的视频数量少于PAGE_SIZE，说明已经是最后一页了，或者没有更多符合条件的视频了
            if page_videos_added < PAGE_SIZE:
                logging.info(f"      - 关键词 '{keyword}' 已获取完所有可用视频（不足 {PAGE_SIZE} 个）。")
                break  # 跳出 outer loop (while videos_fetched_for_keyword < CONFIG.MAX_VIDEOS_PER_SOURCE)

            current_page += 1

            # 添加延迟，避免请求过快被封禁
            time.sleep(1)  # 建议延迟1秒，可根据需要调整

        logging.info(
            f"  > 关键词 '{keyword}' 搜索完成，总共获取 {videos_fetched_for_keyword} 个视频 (目标 {CONFIG['MAX_VIDEOS_PER_SOURCE']})。")
        logging.info("-" * 50)  # 分隔线
    CONFIG['MAX_VIDEOS_PER_SOURCE'] = 20 # 重置为每页20个，避免影响后续搜索，因为不会更新这么快速
    return video_list

# <<< NEW: START - 新增分区排行榜获取函数 >>>
def fetch_from_ranking():
    """
    从指定分区的排行榜获取视频。
    """
    logging.info("开始执行 [策略四：获取分区排行榜视频]...")
    if not CONFIG['RANKING_TIDS']:
        logging.warning("  > 未配置目标分区ID (RANKING_TIDS)，跳过此策略。")
        return []

    video_list = []
    url = "https://api.bilibili.com/x/web-interface/ranking/v2"

    for tid, name in CONFIG['RANKING_TIDS'].items():
        logging.info(f"  > 正在获取分区 '{name}' (TID: {tid}) 的排行榜...")
        params = {
            'rid': tid,
            'type': 'all',  # 获取全部分类，可根据需求改为 'rookie' 或 'origin'
        }

        data = send_get_request(url, params=params)

        if data and 'list' in data and data['list']:
            # API返回最多100个视频，我们根据配置取前N个
            ranking_videos = data['list']
            for item in ranking_videos:
                if 'bvid' in item:
                    item['_source_strategy'] = 'ranking'
                    video_list.append(item)
            logging.info(f"    - 成功从分区 '{name}' 获取 {len(ranking_videos)} 个视频。")
        else:
            logging.warning(f"    - 未能从分区 '{name}' 获取到视频数据，或数据为空。")

    if video_list:
        logging.info(f"  > [策略四：获取分区排行榜视频] 执行完毕。总共获取 {len(video_list)} 个视频。")
    else:
        logging.warning("  > [策略四：获取分区排行榜视频] 执行完毕，但未能获取到任何视频。")

    return video_list
# <<< NEW: END - 新增分区排行榜获取函数 >>>


# --- 6. 已处理记录管理 (视频BVID和用户FID) ---
def load_processed_set(filepath):
    if not os.path.exists(filepath):
        return set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    except (json.JSONDecodeError, IOError):
        return set()

def load_processed_dict(filepath):
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def save_processed_set(data_set, filepath):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # 将集合转换为列表以便JSON序列化
            json.dump(list(data_set), f, indent=4)
    except IOError as e:
        logging.error(f"保存文件 {filepath} 失败: {e}")

def save_processed_dict(data_dict, filepath):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # 关键改动：添加 ensure_ascii=False
            json.dump(data_dict, f, indent=4, ensure_ascii=False)
        print(f"数据已成功保存到 {filepath}")
    except IOError as e:
        logging.error(f"保存文件 {filepath} 失败: {e}")

# --- 7. 视频拉取主逻辑 ---
def fetch_videos():
    logging.info("==================== 开始获取待处理视频 ====================")
    processed_bvideos = load_processed_set(CONFIG['PROCESSED_VIDEOS_FILE'])
    # processed_bvideos = set()
    logging.info(f"已加载 {len(processed_bvideos)} 个已处理的视频记录。")

    all_found_videos = []
    if CONFIG['STRATEGIES']['popular']:
        all_found_videos.extend(fetch_from_popular())
    if CONFIG['STRATEGIES']['following']:
        all_found_videos.extend(fetch_from_following())
    if CONFIG['STRATEGIES']['search']:
        all_found_videos.extend(fetch_from_search())
    # <<< MODIFIED: START - 集成新的获取策略 >>>
    if CONFIG['STRATEGIES']['ranking']:
        all_found_videos.extend(fetch_from_ranking())
    # <<< MODIFIED: END - 集成新的获取策略 >>>

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
comment_videos_queue = Queue()

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
        sleep_time = random.uniform(1200, 1800)
        logging.info(f"视频拉取线程休眠 {int(sleep_time / 60)} 分钟...")
        time.sleep(sleep_time)


# (评论功能保留，暂不启用)
def comment_worker():
    """评论线程：从队列获取视频并发表评论。"""
    base_commenter = BilibiliCommenter(CONFIG['COOKIE'], CONFIG['CSRF_TOKEN'])
    nana_total_cookie = get_config("nana_bilibili_total_cookie")
    nana_csrf_token = get_config("nana_bilibili_csrf_token")
    nana_commenter = BilibiliCommenter(nana_total_cookie, nana_csrf_token)

    mama_total_cookie = get_config("mama_bilibili_total_cookie")
    mama_csrf_token = get_config("mama_bilibili_csrf_token")
    mama_commenter = BilibiliCommenter(mama_total_cookie, mama_csrf_token)

    ruru_total_cookie = get_config("ruru_bilibili_total_cookie")
    ruru_csrf_token = get_config("ruru_bilibili_csrf_token")
    ruru_commenter = BilibiliCommenter(ruru_total_cookie, ruru_csrf_token)

    commenter_list = [base_commenter, mama_commenter, nana_commenter]

    commented_video = load_processed_set(CONFIG['COMMENTED_PROCESSED_VIDEOS_FILE'])
    detail_video_info_map = load_processed_dict(CONFIG['GEN_PROCESSED_VIDEOS_FILE'])
    detail_video_info_map = {bvid: info for bvid, info in detail_video_info_map.items() if info.get('gen_comment')}

    for video_info in detail_video_info_map.values():
        bvid = video_info.get('BVID')
        if bvid and bvid not in commented_video:
            comment_videos_queue.put(video_info)
    logging.info(f"已加载 {len(commented_video)} 个已评论的视频记录。还需要 {comment_videos_queue} 个视频待评论。总共 {len(detail_video_info_map)} 个视频生成记录。")


    while True:
        try:
            for commenter in commenter_list:
                valid_video = None
                start_time = time.time()
                # 尝试在最多30秒内获取一条有效视频
                while time.time() - start_time < 30:
                    try:
                        candidate = comment_videos_queue.get(timeout=5)
                        publish_time = candidate.get('发布时间', None)  # 示例 为 '2025-03-15 21:11:23'

                        # 将字符串时间转为 datetime 对象
                        if publish_time:
                            publish_time = datetime.datetime.strptime(publish_time, '%Y-%m-%d %H:%M:%S')

                        # 获取当前时间并计算一周前的时间
                        one_week_ago = datetime.datetime.now() - datetime.timedelta(weeks=1)

                        # 如果发布时间不在最近一周内，则跳过
                        if publish_time and publish_time >= one_week_ago:
                            commented_video.add(candidate.get('BVID', '未知BVID'))
                            save_processed_set(commented_video, CONFIG['COMMENTED_PROCESSED_VIDEOS_FILE'])
                        else:
                            logging.info(f"发布时间 {publish_time} 超过一周，跳过该视频。")
                            continue
                    except Empty:
                        logging.info("评论视频队列为空，本评论者暂时跳过。")
                        break
                    # 判断视频是否有效
                    bvid = candidate.get('BVID')
                    if not bvid:
                        logging.info("获取视频无效，bvid为空，跳过该视频。")
                        # 可选：如果认为该视频以后可能恢复，就放回队列
                        # comment_videos_queue.put(candidate)
                        continue
                    else:
                        valid_video = candidate
                        break

                # 如果没有获取到有效视频则跳过当前评论者
                if not valid_video:
                    continue

                # 准备评论
                bvid = valid_video.get('BVID')
                comment_list = valid_video.get('gen_comment', [])
                comment_text = random.choice(comment_list)
                # 删除comment_list中的comment_text
                comment_list.remove(comment_text)
                title = valid_video.get('标题', '无标题')

                success = commenter.post_comment(bvid, comment_text, 1)
                if success:
                    logging.info(f"  > 主评论成功✅: '{comment_text}' BVID {bvid} | 标题：{title}")

                    available_replies = comment_list.copy()
                    random.shuffle(available_replies)

                    # 2. 筛选出需要进行回复的评论者 (排除主评论者自己)
                    sub_commenters_to_reply = [sc for sc in commenter_list if sc != commenter]

                    for sub_commenter, reply_message in zip(sub_commenters_to_reply, available_replies):
                        # 其他评论者回复主评论
                        reply_rpid = sub_commenter.reply_to_comment(
                            bvid=bvid,
                            message_content=reply_message,  # <-- 使用配对好的、不重复的回复
                            root_rpid=success,
                            parent_rpid=success,
                            type_code=1
                        )
                        if reply_rpid:
                            # 优化日志：记录实际回复的内容，而不是主评论内容
                            logging.info(f"  >  回复成功: '{reply_message}' BVID {bvid} | 标题：{title}")
                        else:
                            logging.error(f"  > 回复失败: '{reply_message}' BVID {bvid} | 标题：{title}")

                else:
                    logging.error(f"  > 主评论失败❌。BVID {bvid} | 标题：{title}")
                    time.sleep(random.uniform(200, 400))  # 主评论失败后稍作等待

            # 每轮所有评论者执行完后随机休眠一段时间
            time.sleep(random.uniform(100, 200))
        except Exception as e:
            logging.info("评论线程被用户中断，正在退出...")
            traceback.print_exc()
            continue

def get_comment_user(bvid):
    result_id_list = []
    try:
        comments = get_bilibili_comments(bvid)
        for i, reply in enumerate(comments):
            UID = reply['member']['mid']
            message = reply['content']['message']
            should_follow = any(keyword.lower() in message for keyword in CONFIG['FOLLOW_KEYWORDS'])
            if should_follow:
                result_id_list.append(UID)
    except Exception as e:
        logging.error(f"获取评论失败: {e}")
        return result_id_list
    return result_id_list


# (新功能)
def gen_comment():
    """关注线程：从队列获取视频，判断是否需要关注作者。"""
    detail_video_info_map = load_processed_dict(CONFIG['GEN_PROCESSED_VIDEOS_FILE'])
    processed_bvideos = load_processed_set(CONFIG['PROCESSED_VIDEOS_FILE'])
    # 只保留processed_bvideos中gen_comment不为空的视频
    detail_video_info_map = {bvid: info for bvid, info in detail_video_info_map.items() if info.get('gen_comment')}

    for bvid in processed_bvideos:
        if bvid not in detail_video_info_map:
            temp_dict = {}
            temp_dict['bvid'] = bvid
            videos_queue.put(temp_dict)


    logging.info(f"已加载 {len(detail_video_info_map)} 个已生成的记录。")

    while True:
        try:
            video = videos_queue.get(timeout=30)  # 等待30秒，如果没有新视频则继续循环
            logging.info(f"获取到新视频 BVID: {video.get('bvid', '未知')}，开始处理...")
        except Empty:
            continue

        bvid = video.get('bvid')
        if bvid in detail_video_info_map.keys():
            logging.info(f"视频 BVID {bvid} 已经处理过，跳过。")
            continue
        else:
            video_info = gen_proper_comment(bvid)
            if video_info:
                detail_video_info_map[bvid] = video_info
                save_processed_dict(detail_video_info_map, CONFIG['GEN_PROCESSED_VIDEOS_FILE'])
                logging.info(f"视频 BVID {bvid} 处理完成，已保存生成信息。")
                comment_videos_queue.put(video_info)



if __name__ == '__main__':
    if not CONFIG['COOKIE'] or not CONFIG['CSRF_TOKEN']:
        logging.error(
            "错误：请在 common_utils.common_utils.get_config 中配置 bilibili_total_cookie 和 bilibili_csrf_token。")
        exit()

    logging.info("程序启动...")

    # 启动视频拉取线程
    video_thread = threading.Thread(target=video_fetcher_worker, name="VideoFetcherWorker", daemon=True)
    video_thread.start()

    # --- 启动生成评论线程 ---
    follower_thread = threading.Thread(target=gen_comment, name="FollowerWorker",
                                       daemon=True)
    follower_thread.start()

    # --- 评论线程已暂停 ---
    logging.info("评论功能已暂停。如需启用，请取消主程序中的相关代码注释。")
    comment_thread = threading.Thread(target=comment_worker, name="CommentWorker", daemon=True)
    comment_thread.start()

    # 保持主线程运行
    try:
        while True:
            logging.info(f"主线程运行中... 当前待处理视频队列长度: {videos_queue.qsize()} comment_videos_queue长度: {comment_videos_queue.qsize()}")
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n程序被用户中断，正在退出...")