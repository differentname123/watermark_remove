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

from content_community.bilibili.bili_utils import update_bili_user_sign, modify_relation
from  content_community.bilibili.get_comment import get_bilibili_comments
from common_utils.common_utils import get_config
# 评论相关代码保留，但暂时不使用
from content_community.bilibili.comment import BilibiliCommenter

# --- 1. 全局常量 ---
URL_MODIFY_RELATION = "https://api.bilibili.com/x/relation/modify"

# --- 2. 全局配置 ---
total_cookie = get_config("dahao_bilibili_total_cookie")
csrf_token = get_config("dahao_bilibili_csrf_token")

commenter_list = []
cookie_list = []


user_name_list = ['tao', 'xiaoxue', 'jie', 'qiqi', 'mama', 'xiaosu', 'jun', 'jj', 'ruru']
for name in user_name_list:
    if not get_config(f"{name}_bilibili_total_cookie") or not get_config(f"{name}_bilibili_csrf_token"):
        logging.error(f"请在配置文件中设置{name}_bilibili_total_cookie和{name}_bilibili_csrf_token")
        exit(1)
    commenter_list.append(BilibiliCommenter(get_config(f"{name}_bilibili_total_cookie"), get_config(f"{name}_bilibili_csrf_token")))
    cookie_list.append(get_config(f"{name}_bilibili_total_cookie"))


CONFIG = {
    "STRATEGIES": {
        "popular": True,  # 热门视频通常不是目标用户，可以关闭
        "following": False,  # 已经关注的UP主不需要再处理
        "search": True,
    },
    "COOKIE": total_cookie,
    "CSRF_TOKEN": csrf_token,
    "TARGET_UIDS": [  # 监控动态时使用，当前已关闭
        "443415885",
        "10330740",
    ],
    "TARGET_KEYWORDS": [
        "互关",
        "互粉",
        "互赞",
        "互助",
        "新人UP主",
        "回关",
        "回粉",
        "互暖",
        "互评",
        "互捞",
        "三连",
        "求三连",
        "互三连",
        "互币",
        "新人报道",
        "新人up",
        "小UP主",
        "萌新UP",
        "底层UP主",
        "小透明",
        "涨粉",
        "求关注",
        "求抱团",
        "抱团取暖",
        "一起加油",
        "挑战100粉",
        "冲击千粉",
        "有粉必回",
        "有赞必回",
        "在线秒回",
        "已关求回"
    ],
    "FOLLOW_KEYWORDS": [
        "互关",
        "互粉",
        "回关",
        "互赞",
        "互助",
        "回粉",
        "必回",
        "必回关",
        "有粉必回",
        "有访必回",
        "诚信互关",
        "诚信互粉",
        "永不取关",
        "不取关",
        "赞评必回",
        "互赞互评",
        "互三连",
        "互币",
        "关我必回",
        "私信秒回",
        "你关我就关"
    ],
    "MAX_VIDEOS_PER_SOURCE": 20,  # 每次搜索可以多拉取一些
    "DISCOVERED_VIDEOS_FILE":"DISCOVERED_VIDEOS_FILE.json",
    "PROCESSED_VIDEOS_FILE": "processed_bvideos.json",
    "TARGET_PROCESSED_FIDS_FILE": "target_processed_fids.json",
    "PROCESSED_FIDS_FILE": "processed_fids.json",  # 新增：记录已处理的用户ID
    "REQUEST_TIMEOUT": 10,
    "REQUEST_DELAY": 1,
}

# --- 3. 日志与会话配置 ---
logging.basicConfig(
    level=logging.INFO,
    # 在格式中同时加入进程ID(%(process)d)和线程ID(%(thread)d)
    format=f'%(asctime)s - [PID:%(process)d Thread:%(thread)d] - %(levelname)s - %(message)s'
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


# --- 6. 已处理记录管理 (视频BVID和用户FID) ---
def load_processed_set(filepath):
    if not os.path.exists(filepath):
        return set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            temp_set = set(json.load(f))
            temp_set = {str(fid) for fid in temp_set}
            return temp_set
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

    # --- 步骤 1: 加载并更新全局视频发现库 ---

    # 加载所有历史上发现过的视频
    discovered_videos_map = load_processed_dict(CONFIG['DISCOVERED_VIDEOS_FILE'])
    logging.info(f"已加载 {len(discovered_videos_map)} 个历史已发现视频。")

    # 从所有策略中获取当前批次的视频
    all_found_videos = []
    if CONFIG['STRATEGIES']['popular']:
        all_found_videos.extend(fetch_from_popular())
    if CONFIG['STRATEGIES']['following']:
        all_found_videos.extend(fetch_from_following())
    if CONFIG['STRATEGIES']['search']:
        all_found_videos.extend(fetch_from_search())

    # 对本轮获取的视频进行去重
    unique_newly_found_map = {video['bvid']: video for video in reversed(all_found_videos) if 'bvid' in video}
    logging.info(f"所有策略本轮共找到 {len(all_found_videos)} 个视频，去重后剩 {len(unique_newly_found_map)} 个。")

    # 将新发现的视频合并到全局发现库中
    new_videos_added_count = 0
    for bvid, video in unique_newly_found_map.items():
        if bvid not in discovered_videos_map:
            discovered_videos_map[bvid] = video
            new_videos_added_count += 1

    # 如果有新视频加入，则保存更新后的发现库
    if new_videos_added_count > 0:
        logging.info(f"发现 {new_videos_added_count} 个全新视频，正在更新全局发现库...")
        save_processed_dict(discovered_videos_map, CONFIG['DISCOVERED_VIDEOS_FILE'])
        logging.info(f"全局发现库已更新，总数: {len(discovered_videos_map)}。")
    else:
        logging.info("本轮未发现任何新视频。")

    # --- 步骤 2: 从更新后的发现库中筛选待处理视频 ---

    # 加载已处理的视频 bvid 集合
    processed_bvideos = load_processed_set(CONFIG['PROCESSED_VIDEOS_FILE'])
    logging.info(f"已加载 {len(processed_bvideos)} 个已处理的视频记录。")

    # 从完整的“已发现视频库”中，筛选出“未处理”的视频
    videos_to_process = [
        video for bvid, video in discovered_videos_map.items()
        if bvid not in processed_bvideos
    ]
    logging.info(
        f"从 {len(discovered_videos_map)} 个已发现视频中，过滤掉已处理的视频后，最终得到 {len(videos_to_process)} 个待处理视频。")

    return videos_to_process


# --- 8. 并发执行逻辑 ---
videos_queue = Queue()
comment_videos_queue = Queue()

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

danmu_list = [
    "视频质量太高了，已三连！希望我的努力也能被看到~",
    "发现宝藏UP主！果断关注，也希望自己的小作品能被发现。",
    "干货满满，已三连！同为创作者，一起加油！",
    "太用心了，必须支持！我们互相“充电”吧！",
    "制作精良，已点赞关注。也欢迎有空来我这儿坐坐。",
    "这是什么神仙视频！已三连，希望能沾沾大佬的欧气！",
    "学到了很多，感谢UP主！已关注，什么时候我也能做出这种质量啊。",
    "这质量，不点赞关注说不过去。大家一起努力，让好内容发光！",
    "从你的视频里看到了热爱与坚持，已支持！这也激励了我继续创作。",
    "已三连，不解释。同道中人，共勉！",
    "大数据求求了，多推一些这样的优质内容！已关注，也希望我的视频能被推到。",
    "大佬带带我！视频太牛了，默默三连，向您学习！",
    "这才是值得关注的UP主！已三连，希望大家都能为优质内容发电。",
    "关注了，期待UP主更多佳作！也希望我的坚持未来能有回报。",
    "视频做得真好，忍不住三连了。创作者都不容易，一起加油呀！",
    "一键三连！看完感觉自己又充满了创作的动力！",
    "UP主YYDS！已三连，向大佬看齐，努力更新中！",
    "太强了，感觉自己的技能点又提升了！已关注，希望我的分享也能帮到别人。",
    "为你的才华点赞，也为自己的梦想加油。已三连支持！",
    "如此高质量的视频必须三连支持！希望我的主页也能迎来像你一样的观众。"
]

interactive_comment_list = [
    "弹幕已奉上！从我的【视频页面】关注我吧——无论是谁关注，我都会回关。",
    "点赞评论弹幕都到位，求关注～记得从我的【视频页面】点，我保证一律回关！",
    "支持已送达：已赞已评已发弹幕。到我的【视频页面】关注，我会回关并留言确认。",
    "任务完成：弹幕已发！请从我的【视频页面】关注我，我会马上回关，绝不失约。",
    "已点赞评论弹幕，互帮互助走起来～从我的【视频页面】关注，我一定会回关你。",
    "已经帮忙打气并发弹幕！关注请走【视频页面】，我承诺对每位粉丝回关。",
    "支持全套已完成，求个关注～去我的【视频页面】点关注，我会回关并回访主页。",
    "点赞、评论、弹幕都做好了～从我的【视频页面】关注我，我保证回关并回访你主页。",
    "弹幕已送出！去我的【视频页面】点关注吧，我一定会回关报答支持。",
    "已点赞已评论已弹幕，互助模式开启：请到我的【视频页面】关注，我承诺必回关。",
    "弹幕打卡完成！从我的【视频页面】关注我，我必回关，互相扶持一起成长。",
    "安排好了，弹幕也发了，求个关注～一定要从我的【视频页面】点，我保证回关。",
    "全部支持动作已完成，来我的【视频页面】关注我，我会第一时间回关并留言感谢。",
    "弹幕搞定，拜托去我的【视频页面】点关注，我一定会回关并去你主页看看。",
    "支持已送达！请到我的【视频页面】关注我，我保证回关，一起把账号做大！",
    "点赞评论弹幕齐活～从我的【视频页面】点关注，我一定回关，让我们互相见证成长。",
    "已完成弹幕与互动，诚意满满！请从我的【视频页面】关注，我必回关并回访你主页。",
    "任务打卡：已赞、已评、已发弹幕。记得走我的【视频页面】关注，我会回关不食言。",
    "弹幕已办好～去我的【视频页面】点关注，我承诺对每一位关注者一一回关！",
    "已赞评弹幕齐全，等待关注回馈！请务必从我的【视频页面】点，我一定会回关。"
]



def video_fetcher_worker():
    """视频拉取线程：定期拉取新视频并放入队列。"""
    while True:
        new_videos = fetch_videos()
        if new_videos:
            # 随机打乱顺序，避免行为模式过于固定
            random.shuffle(new_videos)
            # 清空videos_queue中的旧视频
            while not videos_queue.empty():
                try:
                    videos_queue.get_nowait()
                    comment_videos_queue.get_nowait()  # 如果评论功能启用，这里也可以放入评论队列
                except Empty:
                    break
            for video in new_videos:
                videos_queue.put(video)
                comment_videos_queue.put(video)  # 如果评论功能启用，这里也可以放入评论队列
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
    for cookie in cookie_list:
        result = update_bili_user_sign(cookie,
                                       "只会回关通过我视频关注我的粉丝，请一定通过我的视频页面来关注我，不然会认为是异常粉丝的")
        print(f"更新用户签名结果: {result}")

    while True:
        # 尝试在最多 30 秒内从队列中获取一个有效的视频
        start_time = time.time()
        valid_video = None
        while time.time() - start_time < 30:
            try:
                candidate = comment_videos_queue.get(timeout=5)
            except Empty:
                logging.warning("评论视频队列为空，暂时没有视频可处理。")
                break
            bvid = candidate.get('bvid')
            if not bvid:
                logging.warning("获取视频无效，bvid 为空，跳过该条视频。")
                # 如果希望以后再次处理，可以重新放回队列：comment_videos_queue.put(candidate)
                continue
            valid_video = candidate
            break

        # 没拿到有效视频则短暂休眠后重试整个流程
        if not valid_video:
            time.sleep(random.uniform(5, 10))
            continue

        bvid = valid_video.get('bvid')
        title = valid_video.get('title', '无标题')
        desc = valid_video.get('description', '')
        text_to_check = f"{title} {desc}".lower()
        source = valid_video.get('_source_strategy', 'unknown')
        should_comment = any(keyword.lower() in text_to_check for keyword in CONFIG['FOLLOW_KEYWORDS'])
        # should_comment = True
        if not should_comment and source != 'popular':
            logging.info(f"视频 BVID {bvid} 标题和描述均不包含关注关键词，跳过评论。")
            continue
        logging.info(f"开始处理视频：BVID {bvid} | 标题：{title}，将由所有评论者逐一评论并发送弹幕。")
        # 打乱commenter_list顺序，避免行为模式过于固定
        random.shuffle(commenter_list)
        success_count = 0
        # 对同一视频，所有 commenter 都要评论并发弹幕一次
        for commenter in commenter_list:
            if success_count > 3:
                logging.info(f"本视频评论者成功数已达上限({success_count})，跳过剩余评论者。")
                break
            commenter_name = getattr(commenter, "username", None) or getattr(commenter, "name", None) or str(commenter)
            try:
                # 评论
                comment_text = random.choice(interactive_comment_list)
                logging.info(f"评论者 {commenter_name} -> 准备评论: '{comment_text}'")
                success = commenter.post_comment(bvid, comment_text, 1, like_video=True)
                if success:
                    logging.info(f"  > {commenter_name} 评论成功: '{comment_text}'")
                else:
                    logging.error(f"  > {commenter_name} 评论失败。")

                # 弹幕
                danmaku_text = random.choice(danmu_list)
                danmaku_sent = commenter.send_danmaku(
                    bvid=bvid,
                    msg=danmaku_text,
                    progress=2000
                )
                if danmaku_sent:
                    logging.info(f"  > {commenter_name} 弹幕发送成功: '{danmaku_text}'")
                else:
                    logging.error(f"  > {commenter_name} 弹幕发送失败。")
                success_count += 1
                # 每个评论者间加一个短延迟，避免瞬时并发造成风控
                time.sleep(random.uniform(1.0, 3.0))


            except Exception as e:
                logging.exception(f"处理评论者 {commenter_name} 时发生异常：{e}")

        logging.info(f"视频 {bvid} 已由所有评论者处理完毕。")

        # 处理完一个视频后，短暂休眠再取下一个视频（按需调整）
        time.sleep(random.uniform(100, 110))

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
def follower_worker(csrf_token):
    """关注线程：从队列获取视频，判断是否需要关注作者。"""
    processed_bvideos = load_processed_set(CONFIG['PROCESSED_VIDEOS_FILE'])
    processed_fids = load_processed_set(CONFIG['PROCESSED_FIDS_FILE'])
    target_processed_bvideos = load_processed_set(CONFIG['TARGET_PROCESSED_FIDS_FILE'])

    logging.info(f"已加载 {len(processed_fids)} 个已处理的用户(fid)记录。")

    while True:
        try:
            video = videos_queue.get(timeout=30)  # 等待30秒，如果没有新视频则继续循环
            processed_bvideos.add(video.get('bvid', '未知'))
            save_processed_set(processed_bvideos, CONFIG['PROCESSED_VIDEOS_FILE'])
            logging.info(f"获取到新视频 BVID: {video.get('bvid', '未知')}，开始处理...")
        except Empty:
            continue

        title = video.get('title', '')
        desc = video.get('description', '')
        source = video.get('_source_strategy', 'unknown')
        # 兼容不同API返回的用户ID字段 ('mid' 或 'owner.mid')
        author_id = video.get('mid')
        if not author_id and 'owner' in video and isinstance(video['owner'], dict):
            author_id = video['owner'].get('mid')

        if not author_id:
            logging.info(f"视频 BVID {video.get('bvid')} 缺少作者ID，跳过。")
            continue
        if 'popular' != source:
            logging.info(f"视频 BVID {video.get('bvid')} 来源于 '{source}'，跳过。")
            continue

        # 检查标题或描述是否包含关注关键词
        text_to_check = f"{title} {desc}".lower()
        should_follow = any(keyword.lower() in text_to_check for keyword in CONFIG['FOLLOW_KEYWORDS'])
        should_follow = True
        result_id_list = [author_id]
        random_value = random.random() < 0.1  # True ~0.1, False ~0.9
        if should_follow or random_value:
            result_id_list.extend(get_comment_user(bvid=video.get('bvid')))
            author_name = video.get('author') or (video.get('owner') and video['owner'].get('name'))
            result_id_list = list(set(result_id_list))
            logging.info(f"发现目标用户: {author_name} (UID: {author_id}) | 来源: BVID {video.get('bvid')} | 标题: {title} 连带评论有 {len(result_id_list)} 个用户需要关注。")
            for fid in result_id_list:
                fid = str(fid)
                if fid in processed_fids:
                    logging.info(f"用户 UID {fid} 已在处理列表，跳过。")
                    continue
                else:
                    target_processed_bvideos.add(fid)
                    processed_fids.add(fid)

                    # 随机暂停一段时间再执行关注，模拟人类行为
                    for cookie in cookie_list:
                        success = modify_relation(fid, 1, cookie)
                    time.sleep(random.uniform(40, 60))


            save_processed_set(processed_fids, CONFIG['PROCESSED_FIDS_FILE'])
            save_processed_set(target_processed_bvideos, CONFIG['TARGET_PROCESSED_FIDS_FILE'])
        else:
            logging.info(f"视频 BVID {video.get('bvid')} 未匹配到关注关键词，作者UID {author_id} 不需要关注。")
            # 即使不关注，也标记为已处理，避免重复检查该用户
            processed_fids.add(author_id)
            save_processed_set(processed_fids, CONFIG['PROCESSED_FIDS_FILE'])

        # 每次处理后都暂停，控制API请求频率
        time.sleep(random.uniform(3, 8))


if __name__ == '__main__':
    if False:
        # 清楚DISCOVERED_VIDEOS_FILE.json中的数据
        if os.path.exists(CONFIG['DISCOVERED_VIDEOS_FILE']):
            os.remove(CONFIG['DISCOVERED_VIDEOS_FILE'])
            logging.info(f"已删除旧的 {CONFIG['DISCOVERED_VIDEOS_FILE']} 文件，重新开始。")
        if os.path.exists(CONFIG['PROCESSED_VIDEOS_FILE']):
            os.remove(CONFIG['PROCESSED_VIDEOS_FILE'])
            logging.info(f"已删除旧的 {CONFIG['PROCESSED_VIDEOS_FILE']} 文件，重新开始。")
        if os.path.exists(CONFIG['PROCESSED_FIDS_FILE']):
            os.remove(CONFIG['PROCESSED_FIDS_FILE'])
            logging.info(f"已删除旧的 {CONFIG['PROCESSED_FIDS_FILE']} 文件，重新开始。")
        if os.path.exists(CONFIG['TARGET_PROCESSED_FIDS_FILE']):
            os.remove(CONFIG['TARGET_PROCESSED_FIDS_FILE'])
            logging.info(f"已删除旧的 {CONFIG['TARGET_PROCESSED_FIDS_FILE']} 文件，重新开始。")

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