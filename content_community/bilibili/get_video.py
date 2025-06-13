import requests
import time
from datetime import datetime, timedelta
import random
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
# --- 配置参数 ---
# Bilibili API 端点 (这些是非官方接口的示例，可能会变动)
# 请查阅 bilibili-API-collect 项目以获取最新和更详细的接口
API_ENDPOINTS = {
    "video_info": "https://api.bilibili.com/x/web-interface/view",
    "ranking_all": "https://api.bilibili.com/x/web-interface/ranking/v2",  # 综合榜单
    "region_new_videos": "https://api.bilibili.com/x/web-interface/ranking/region",  # 区域最新榜单 (通常按时间排序)
    "video_comments": "https://api.bilibili.com/x/v2/reply/main",  # 视频评论 (oid是视频的aid)
    # 更多API... 例如通过关键词搜索: "https://api.bilibili.com/x/web-interface/search/web/v2"
}

# 模拟浏览器行为的User-Agent列表
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0"
]

# --- 筛选参数 ---
FRESHNESS_WINDOW_HOURS = 30  # 视频发布时间必须在过去X小时内
MIN_INITIAL_VIEWS = 2000  # 视频至少要达到此播放量才能被考虑
MIN_INITIAL_LIKES = 100  # 视频至少要达到此点赞量才能被考虑
MIN_TOTAL_COMMENTS = 20  # 视频总评论数至少要达到此数量，表明有一定活跃度

# LLM相关参数
LLM_RELEVANCE_THRESHOLD = 0.7  # LLM判断视频内容与你账号主题的匹配度阈值 (0-1)

# --- 自动化运行参数 ---
SCRAPE_INTERVAL_MINUTES = 15  # 每隔多久运行一次视频发现循环
REQUEST_DELAY_SECONDS = (1, 3)  # 每个API请求之间的随机延迟 (防止被B站限流)
MAX_PROCESSED_VIDEOS_CACHE = 1000  # 内存中存储的已处理视频ID的最大数量，防止重复处理


# --- LLM 集成占位符 (你需要实现这部分) ---
def call_llm_for_analysis(video_title: str, video_desc: str, hot_comments: list, latest_comments: list) -> dict:
    """
    占位函数：调用你的语言模型 (LLM) 进行视频内容和评论区分析。
    这个函数应该：
    1. 根据视频标题、简介判断视频内容与你的账号主题是否相关。
    2. 分析热门评论和最新评论，了解讨论热点、用户情绪和可能的评论切入点。

    参数:
    - video_title: 视频标题
    - video_desc: 视频简介
    - hot_comments: 热门评论列表 (字典形式，包含 'message', 'like' 等)
    - latest_comments: 最新评论列表 (字典形式，包含 'message', 'like' 等)

    返回: 一个包含分析结果的字典，例如：
    {"relevance_score": 0.9, "sentiment": "positive", "discussion_points": ["引发共鸣点", "提供独特见解"]}
    """
    print(f"--- 模拟LLM分析视频: 《{video_title}》 ---")

    # ⚠️ 实际开发时，这里会是调用你的LLM API的代码，例如:
    # client = OpenAI() # 或其他LLM客户端
    # response = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "你是一个B站视频分析专家，评估视频与用户账号内容的匹配度并找出评论机会。"},
    #         {"role": "user", "content": f"视频标题: {video_title}\n简介: {video_desc}\n热门评论: {hot_comments}\n最新评论: {latest_comments}\n请评估其与'人工智能教程/应用'主题的关联度(0-1分)，并总结评论区讨论的亮点和切入点。"}
    #     ]
    # )
    # llm_output = response.choices[0].message.content
    # 然后你需要解析 llm_output 来提取 relevance_score 和 discussion_points

    # --- 模拟LLM响应 ---
    # 假设你的账号专注于"技术/教程"内容
    simulated_relevance = random.uniform(0.5, 1.0)
    if "教程" in video_title or "技术" in video_title or "AI" in video_title or "编程" in video_title:
        simulated_relevance = min(1.0, simulated_relevance * 1.2)  # 如果标题包含关键词，提高相关性
    else:
        simulated_relevance = simulated_relevance * 0.8  # 否则降低

    simulated_sentiment = random.choice(["positive", "neutral"])
    simulated_discussion_points = ["对视频内容进行延伸讨论", "提出相关问题引导回复", "分享个人见解"]

    time.sleep(random.uniform(1, 2))  # 模拟LLM API调用延迟

    return {
        "relevance_score": simulated_relevance,
        "sentiment": simulated_sentiment,
        "discussion_points": simulated_discussion_points
    }


# --- Bilibili API 客户端类 ---
class BilibiliAPIClient:
    def __init__(self):
        self.session = requests.Session()
        # 用于避免重复处理视频，内存缓存，生产环境应使用数据库
        self.processed_bvideos = set()
        self.last_api_request_time = time.time()

    def _get_headers(self):
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "Connection": "keep-alive"
        }

    def _make_request(self, url, params=None):
        headers = self._get_headers()
        # 确保请求之间有最小延迟
        time_since_last_request = time.time() - self.last_api_request_time
        min_delay = random.uniform(REQUEST_DELAY_SECONDS[0], REQUEST_DELAY_SECONDS[1])
        if time_since_last_request < min_delay:
            time.sleep(min_delay - time_since_last_request)

        try:
            response = self.session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()  # 对 4xx 或 5xx 状态码抛出 HTTPError
            self.last_api_request_time = time.time()
            data = response.json()
            if data.get('code') != 0:
                print(f"API Error for {url}, params: {params}: {data.get('message', 'Unknown error')}")
                return None
            return data
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {url}, params: {params}: {e}")
            return None

    def get_video_details(self, bvid: str):
        """获取视频的详细信息 (标题、简介、UP主、统计数据等)"""
        params = {"bvid": bvid}
        data = self._make_request(API_ENDPOINTS["video_info"], params)
        if data and data.get('data'):
            return data['data']
        return None

    def get_ranking_videos(self, rid: int = 0, day: int = 3):
        """
        获取B站排行榜视频
        :param rid: 分区ID (0为全站，例如1为动画，4为游戏，36为知识)
        :param day: 榜单天数 (1, 3, 7)
        """
        params = {"rid": rid, "day": day}
        data = self._make_request(API_ENDPOINTS["ranking_all"], params)
        if data and data.get('data') and data['data'].get('list'):
            videos = []
            for item in data['data']['list']:
                # pubdate 是 Unix 时间戳，需要转换为 datetime 对象
                pubdate_dt = datetime.fromtimestamp(item.get('pubdate'))
                videos.append({
                    'bvid': item.get('bvid'),
                    'aid': item.get('aid'),  # 评论API通常使用aid
                    'title': item.get('title'),
                    'pubdate': pubdate_dt,
                    'owner_name': item.get('owner', {}).get('name'),
                    'stat_view': item.get('stat', {}).get('view'),
                    'stat_like': item.get('stat', {}).get('like'),
                    'stat_reply': item.get('stat', {}).get('reply'),
                    'desc': item.get('desc', '')  # 简介可能不完整或需要额外调用get_video_details
                })
            return videos
        return []

    def get_region_new_videos(self, rid: int = 1, day: int = 1):
        """
        获取特定分区最新榜单的视频 (通常比综合榜单更侧重“新”视频)
        :param rid: 分区ID
        :param day: 榜单天数 (通常用1获取最新)
        """
        params = {"rid": rid, "day": day}
        data = self._make_request(API_ENDPOINTS["region_new_videos"], params)
        if data and data.get('data') and data['data'].get('list'):
            videos = []
            for item in data['data']['list']:
                pubdate_dt = datetime.fromtimestamp(item.get('pubdate'))
                videos.append({
                    'bvid': item.get('bvid'),
                    'aid': item.get('aid'),
                    'title': item.get('title'),
                    'pubdate': pubdate_dt,
                    'owner_name': item.get('owner', {}).get('name'),
                    'stat_view': item.get('stat', {}).get('view'),
                    'stat_like': item.get('stat', {}).get('like'),
                    'stat_reply': item.get('stat', {}).get('reply'),
                    'desc': item.get('desc', '')
                })
            return videos
        return []

    def get_video_comments(self, oid: int, page: int = 1, page_size: int = 20):
        """
        获取视频评论
        :param oid: 视频的 aid (通常是整数ID)
        :param page: 评论页码
        :param page_size: 每页评论数量 (最大值可能受API限制，通常为20或30)
        """
        params = {"oid": oid, "type": 1, "pn": page, "ps": page_size}
        data = self._make_request(API_ENDPOINTS["video_comments"], params)
        if data and data.get('data'):
            # 获取热门评论和普通评论
            replies = data['data'].get('replies', [])
            hot_replies = data['data'].get('hots', [])

            parsed_comments = []
            # 优先处理热门评论
            for r in hot_replies:
                if r and r.get('content') and r['content'].get('message'):
                    parsed_comments.append({
                        "message": r['content']['message'],
                        "like": r.get('like', 0),
                        "member_name": r.get('member', {}).get('uname')
                    })
            # 接着处理最新评论 (从普通评论中取前几条)
            for r in replies:
                if r and r.get('content') and r['content'].get('message'):
                    parsed_comments.append({
                        "message": r['content']['message'],
                        "like": r.get('like', 0),
                        "member_name": r.get('member', {}).get('uname')
                    })

            # 去重并限制数量，只返回一部分热门和最新评论作为LLM的输入
            unique_comments = []
            seen_messages = set()
            for comment in parsed_comments:
                if comment['message'] not in seen_messages:
                    unique_comments.append(comment)
                    seen_messages.add(comment['message'])

            return unique_comments[:20], data['data'].get('page', {}).get('count', 0)  # 返回20条，以及总评论数
        return [], 0

    def add_to_processed(self, bvid: str):
        """将视频标记为已处理，避免在当前会话中重复检查"""
        if len(self.processed_bvideos) >= MAX_PROCESSED_VIDEOS_CACHE:
            # 简单的清除机制：当缓存满时，移除最老的一个，实际应用中可用LRU
            # 注意：set是无序的，这里只是示意
            self.processed_bvideos.pop()  # 移除任意一个
        self.processed_bvideos.add(bvid)


# --- 主发现与筛选流程函数 ---
def run_video_discovery_pipeline():
    api_client = BilibiliAPIClient()

    # 存储通过筛选的潜在评论视频
    potential_videos_for_commenting = []

    print("--- 启动B站视频发现与筛选管道 ---")

    while True:
        current_time = datetime.now()
        print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] 开始新的视频发现周期...")

        # --- 阶段一：视频数据源采集 ---
        all_discovered_videos_meta = []

        # 示例：从全站热门榜单获取视频 (day=1 倾向于获取较新的视频)
        print("-> 正在从全站热门榜单获取视频...")
        all_discovered_videos_meta.extend(api_client.get_ranking_videos(rid=0, day=1))
        # time.sleep(random.uniform(REQUEST_DELAY_SECONDS[0], REQUEST_DELAY_SECONDS[1]))

        # # 示例：从特定分区（例如知识区 rid=36）获取最新视频
        # print("-> 正在从知识区最新榜单获取视频 (rid=36)...")
        # all_discovered_videos_meta.extend(api_client.get_region_new_videos(rid=36, day=1))
        # time.sleep(random.uniform(REQUEST_DELAY_SECONDS[0], REQUEST_DELAY_SECONDS[1]))

        # 对采集到的视频进行去重，并筛选出本次周期内未处理过的视频
        unique_bvideos = {}
        for video_meta in all_discovered_videos_meta:
            if video_meta and video_meta.get('bvid'):
                unique_bvideos[video_meta['bvid']] = video_meta

        newly_discovered_videos = []
        for bvid, meta in unique_bvideos.items():
            if bvid not in api_client.processed_bvideos:
                newly_discovered_videos.append(meta)
                api_client.add_to_processed(bvid)  # 标记为已发现，本周期不再重复处理

        print(f"本周期内发现 {len(newly_discovered_videos)} 个新的独特视频待筛选。")

        # --- 阶段二 & 阶段三：初步筛选 & 二次筛选 ---
        for video_summary in newly_discovered_videos:
            bvid = video_summary.get('bvid')
            aid = video_summary.get('aid')
            pubdate = video_summary.get('pubdate')
            title = video_summary.get('title')

            if not bvid or not aid or not pubdate:
                continue  # 数据不完整，跳过

            # 1. 新鲜度过滤
            if current_time - pubdate > timedelta(hours=FRESHNESS_WINDOW_HOURS):
                print(f"  跳过视频 {bvid}: 发布时间过旧 (发布于 {pubdate.strftime('%Y-%m-%d %H:%M')})")
                continue

            # 获取视频的详细信息 (包括完整的简介和实时统计数据)
            video_details = api_client.get_video_details(bvid)
            if not video_details:
                print(f"  未能获取视频 {bvid} 的详细信息，跳过。")
                continue

            video_desc = video_details.get('desc', '')
            stat_view = video_details.get('stat', {}).get('view', 0)
            stat_like = video_details.get('stat', {}).get('like', 0)
            stat_reply = video_details.get('stat', {}).get('reply', 0)  # 当前评论总数

            # 2. 初期数据指标评估 (播放量和点赞)
            if stat_view < MIN_INITIAL_VIEWS or stat_like < MIN_INITIAL_LIKES:
                # print(f"  跳过视频 {bvid}: 初期播放量/点赞量过低 ({stat_view} 播放, {stat_like} 点赞)")
                continue

            # --- 增长速度监测 (重要提示：此部分在单次运行中是概念性的) ---
            # 真正的增长速度监测需要：
            # a. 一个持久化存储 (数据库)，记录视频首次发现时的播放量、点赞量、评论数。
            # b. 在后续周期中，再次获取这些数据，与首次数据对比，计算增长率。
            # 例如： (current_stat_view - initial_stat_view) / (current_time - initial_discovery_time).total_seconds()

            # 3. 评论区活跃度与内容匹配 (结合LLM)
            hot_comments = []
            latest_comments = []
            total_comments = 0
            if aid:  # 确保有aid才能获取评论
                # 获取前几页的评论，用于LLM分析热门和最新讨论
                comments_for_analysis, total_comments = api_client.get_video_comments(aid, page=1, page_size=20)
                # 简单区分热门和最新，实际可以根据点赞数或发布时间更精确地筛选
                hot_comments = [c for c in comments_for_analysis if c.get('like', 0) > 50]  # 示例：点赞大于50为热门
                latest_comments = comments_for_analysis[:5]  # 示例：列表前5条为最新

            if total_comments < MIN_TOTAL_COMMENTS:
                # print(f"  跳过视频 {bvid}: 评论总数过少 ({total_comments})，活跃度不足。")
                continue

            print(f"  正在分析潜在视频: 《{title}》 (BV: {bvid}) - 当前评论数: {stat_reply}")

            # 调用LLM进行内容相关性及评论区分析
            llm_analysis_results = call_llm_for_analysis(title, video_desc, hot_comments, latest_comments)

            if llm_analysis_results['relevance_score'] < LLM_RELEVANCE_THRESHOLD:
                print(
                    f"    ❌ 视频《{title}》与主题相关性不足 (得分: {llm_analysis_results['relevance_score']:.2f})，跳过。")
                continue

            # 如果所有筛选条件都满足，则加入到待评论视频列表
            potential_videos_for_commenting.append({
                "bvid": bvid,
                "aid": aid,
                "title": title,
                "description": video_desc,
                "pubdate": pubdate,
                "stats": {"views": stat_view, "likes": stat_like, "comments": stat_reply},
                "llm_analysis": llm_analysis_results,
                "hot_comments_sample": hot_comments,  # 存储以备后续评论生成使用
                "latest_comments_sample": latest_comments  # 存储以备后续评论生成使用
            })
            print(
                f"    ✅ 发现高潜力视频: 《{title}》 (BV: {bvid}) - 相关性: {llm_analysis_results['relevance_score']:.2f}")

        # --- 阶段四：视频优先级队列与调度 ---
        # 根据LLM分析的相关性得分或其他指标进行排序
        potential_videos_for_commenting.sort(key=lambda x: x['llm_analysis']['relevance_score'], reverse=True)

        print(f"\n--- 本周期共筛选出 {len(potential_videos_for_commenting)} 个高潜力视频 ---")
        if potential_videos_for_commenting:
            print("--- 排名前5的视频 (基于LLM相关性得分) ---")
            for i, video in enumerate(potential_videos_for_commenting[:5]):
                print(f"  Top {i + 1}: 《{video['title']}》 (BV: {video['bvid']})")
                print(f"    - 发布时间: {video['pubdate'].strftime('%Y-%m-%d %H:%M')}")
                print(
                    f"    - 播放量: {video['stats']['views']}, 点赞: {video['stats']['likes']}, 评论: {video['stats']['comments']}")
                print(f"    - LLM相关性得分: {video['llm_analysis']['relevance_score']:.2f}")
                print(f"    - LLM建议讨论点: {', '.join(video['llm_analysis']['discussion_points'])}")
        else:
            print("  本周期未发现符合条件的视频。")

        # 在实际应用中，你会将 `potential_videos_for_commenting` 传递给下一个阶段
        # 例如：将它们加入一个评论任务队列，由另一个模块来执行评论发布。

        # 清空当前周期发现的视频列表，准备下一个周期
        potential_videos_for_commenting = []

        print(f"\n下一个视频发现周期将在 {SCRAPE_INTERVAL_MINUTES} 分钟后开始...")
        time.sleep(SCRAPE_INTERVAL_MINUTES * 60)  # 等待直到下一个周期


# --- 运行主函数 ---
if __name__ == "__main__":
    try:
        run_video_discovery_pipeline()
    except KeyboardInterrupt:
        print("\n用户中断，管道已停止。")
    except Exception as e:
        print(f"发生了一个未预期的错误: {e}")