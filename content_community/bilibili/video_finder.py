import requests
import time
import logging
import os
import json

from common_utils.common_utils import get_config
from content_community.bilibili.comment import BilibiliCommenter

total_cookie = get_config("bilibili_total_cookie")

# --- 1. 全局配置 ---
CONFIG = {
    "STRATEGIES": {
        "popular": True,
        "following": True,
        "search": True,
    },
    "COOKIE": total_cookie,  # <--- ！！！请务必替换成你自己的COOKIE！！！
    "TARGET_UIDS": [
        "443415885",
        "10330740",
    ],
    "TARGET_KEYWORDS": [
        "炉石传说",
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
        time.sleep(CONFIG['REQUEST_DELAY'])
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
    """策略一：从热门榜单获取视频，保留完整原始信息"""
    logging.info("开始执行 [策略一：获取热门视频]...")
    video_list = []
    url = "https://api.bilibili.com/x/web-interface/popular"
    params = {'ps': CONFIG['MAX_VIDEOS_PER_SOURCE'], 'pn': 1}
    data = send_request(url, params)

    if data and 'list' in data:
        for item in data['list']:
            if 'bvid' in item:
                item['_source_strategy'] = 'popular'  # 注入来源信息
                video_list.append(item)
        logging.info(f"  > 成功从热门榜单获取 {len(video_list)} 个视频。")
    else:
        logging.warning("  > 从热门榜单获取视频失败。")

    return video_list


def fetch_from_following():
    """策略二：从关注的UP主最新动态获取视频，保留完整原始信息"""
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
                        # 动态接口返回的视频信息在 major.archive 字段中
                        video_data = major.get('archive')
                        if video_data and 'bvid' in video_data:
                            # 为了信息完整，将动态的作者信息也合并进去
                            author_info = item.get('modules', {}).get('module_author', {})
                            video_data['owner'] = {
                                'mid': author_info.get('mid'),
                                'name': author_info.get('name'),
                                'face': author_info.get('face'),
                            }
                            video_data['_source_strategy'] = 'following'
                            video_data['_dynamic_raw'] = item  # 可选：保留整个动态的原始信息
                            video_list.append(video_data)
                            found_count += 1
                            if found_count >= CONFIG['MAX_VIDEOS_PER_SOURCE']:
                                break
            logging.info(f"    - 从UID {uid} 处获取 {found_count} 个新视频。")

    return video_list


def fetch_from_search():
    """策略三：根据关键词搜索最新视频，保留完整原始信息"""
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
                    # 清理标题中的HTML标签，这是一个有用的预处理
                    if 'title' in item:
                        item['title'] = item['title'].replace('<em class="keyword">', '').replace('</em>', '')
                    item['_source_strategy'] = 'search'
                    video_list.append(item)
                    found_count += 1
            logging.info(f"    - 从关键词 '{keyword}' 处获取 {found_count} 个视频。")

    return video_list


# --- 5. 主逻辑：执行与去重 ---

def load_processed_bvideos():
    filepath = CONFIG['PROCESSED_VIDEOS_FILE']
    if not os.path.exists(filepath): return set()
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


def main():
    if CONFIG['COOKIE'] == "YOUR_COOKIE_HERE":
        logging.error("错误：请在CONFIG中配置你的B站Cookie！")
        return []

    logging.info("==================== 开始获取待处理视频(v3 - 保留完整信息) ====================")

    processed_bvideos = load_processed_bvideos()
    logging.info(f"已加载 {len(processed_bvideos)} 个已处理的视频记录。")

    all_found_videos = []

    if CONFIG['STRATEGIES']['popular']:
        all_found_videos.extend(fetch_from_popular())
    # if CONFIG['STRATEGIES']['following']:
    #     all_found_videos.extend(fetch_from_following())
    if CONFIG['STRATEGIES']['search']:
        all_found_videos.extend(fetch_from_search())

    # 使用字典来去重，保留第一次出现的视频信息
    unique_videos_map = {video['bvid']: video for video in reversed(all_found_videos) if 'bvid' in video}
    logging.info(f"\n所有策略共找到 {len(all_found_videos)} 个视频，去重后剩 {len(unique_videos_map)} 个。")

    # 筛选出尚未处理的新视频
    videos_to_process = [video for bvid, video in unique_videos_map.items() if bvid not in processed_bvideos]
    logging.info(f"过滤掉已处理的视频后，最终得到 {len(videos_to_process)} 个新视频待处理。")

    # 更新已处理列表
    newly_processed_bvid_set = {video['bvid'] for video in videos_to_process}
    updated_processed_set = processed_bvideos.union(newly_processed_bvid_set)
    save_processed_bvideos(updated_processed_set)
    logging.info(f"已处理视频记录已更新，总数: {len(updated_processed_set)}。")

    logging.info("==================== 获取任务完成 ====================")

    print("\n--- 待处理的视频完整信息列表 ---")
    return videos_to_process


if __name__ == '__main__':
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
    "如果你按下关注键，我也会同样按下你的关注键，一起成长！"
]
    csrf_token = get_config("bilibili_csrf_token")
    total_cookie = get_config("bilibili_total_cookie")

    if not csrf_token or not total_cookie:
        print("错误：请在 common_utils.common_utils.get_config 中配置 csrf_token 和 total_cookie。")
        exit()

    # --- 实例化评论器 ---
    commenter = BilibiliCommenter(total_cookie=total_cookie, csrf_token=csrf_token)

    # 再次强调，请勿将此脚本用于违反B站社区规则的用途
    videos_queue = main()
    # 遍历视频列表，逐个处理评论
    if videos_queue:
        for video in videos_queue:
            bvid = video.get('bvid')
            if bvid:
                print(f"处理视频 BVID: {bvid}   {video.get('title')}")
                # 随机选择comment_text中的一行
                comment_text = comment_list[int(time.time()) % len(comment_list)]

                # --- 执行评论 ---
                success = commenter.post_comment(bvid, comment_text, 1)

                if success:
                    print("\n评论操作完成：成功。")
                else:
                    print("\n评论操作完成：失败。")
    else:
        print("没有找到待处理的视频。")