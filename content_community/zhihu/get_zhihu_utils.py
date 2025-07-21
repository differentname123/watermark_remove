import json
import os
import pathlib
import random
import time
from bs4 import BeautifulSoup, Tag, NavigableString

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import requests
from typing import List, Dict

from common_utils.common_utils import save_json, download_public_image

# --- 配置区域 ---
AUTH_FILE = "zhihu_auth_state.json"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/115.0.5790.110 Safari/537.36"
)
ZHIHU_COOKIE_STRING = "_xsrf=EGbVA6NHTaM3dXlCMEiWj9aRBvWl4inW; _zap=34c3bc6c-ebae-4e0d-9a06-5bfb7b8a9548; d_c0=APARO_5-wBmPTvibmi_6NacNH42miN-ERZY=|1735194921; Hm_lvt_98beee57fd2ef70ccdd5ca52b9740c49=1752912686; HMACCOUNT=14EFD85132347319; DATE=1752912688356; crystal=U2FsdGVkX19HZsSWXHhQvCmTy0IhrzSuQUWjAmBNS9sgZFRn8/yuh3qnHWWPtqhs6Fx+lpP4dATHFISdyemFNM4nXvmwy0g7ICamPZ7CB/IU+sI4WwbiZS83jULObdwxNXvxwM1lrWOA06h6rgMPFiBn+qiMLtZXpkprUsDW2FNS69MOpODrf6kUIyRL/QXg9IJ08veQmhzEr8G7YQHsdz07b0Wmj/50nQPJAtdug/kh+RlGpYjjx8ALS6jGD2Gq; __snaker__id=RQbe8vWX2MrZo7OW; cmci9xde=U2FsdGVkX184glWsyX1mezYXZa3qDzIhjRBe9TQQVEY+LLFTFMDXP9nSs9RkgMzxPSZtU4lOrgeYZefNb02KEA==; pmck9xge=U2FsdGVkX1/T3lTFMBSW1MNHTt55yQ7uWCI2fzMhaBs=; assva6=U2FsdGVkX1803CTSuJgB3mtcX8vMBRJ4mrNXHxoktsA=; assva5=U2FsdGVkX18s7ilkblADh/oGQosbZLgtP7rzonbhC4T0Gf8YWm1GZf8atIJSu1QVH69xU8U+rNeMHgJRf8dEEQ==; vmce9xdq=U2FsdGVkX1+9L5kRV9p5NPBm2ZFxevxsXB601UTW1lO8oB1sywbxX35uCVgDtPFrCRoAhGz6Qw95IDt5HxZKgRgHcwII5jsliZ9AEeEMx0QyMg0NlFbFg2/No39rs8FcREB1wxx4Hg7ZLGq+HBSZ6UbnPSt1xTw3YsTv3I27GXM=; z_c0=2|1:0|10:1752912935|4:z_c0|92:Mi4xemV3ZkR3QUFBQUFBOEJFN19uN0FHU1lBQUFCZ0FsVk5KNkpvYVFCQlUzLUlBR3dQNGcwQkhEOWNXcmsyZ0VFZnJB|55035f8a3b883e94b09b952267d56c00a1a59a98cc43712feb238eb4056739b8; q_c1=e3e3832501fe4a17ba8bf7b5b47f6e60|1752912935000|1752912935000; __zse_ck=004_9njDuYcKusVXiMuD5SZ7LomPNAudnA3idCp0Ig/GY0=5gYSDmi0TNqJ7FjyXoqYArffzF1BhUy=GuO4ecVqJEGHl8QUnfJi3U5WYI/Y1QFhwF7Gz7I7xqjnm05LC1vY2-1G/2qpHnazSH76s+360GqWHjozS9rp18hQPVk+DOgjdq/n5leUgxJ+237tPuguqC5x1a1EjhuQ/RyoAp0+8lSPskVcy16jac+kELW9lwJayh1jscDZfo7NsPnlZfJUWL; gdxidpyhxdE=tdTqBn8olmH5j1Mvzgb2xuBP1JV%5CEOlNCeWjTWLag7PU1kAgwZc0iYoJNXHP4pesZ9c6pt6sCy9XvWRvcCih4H3wlMve85vPDsuwZGH0niB0eBEbOdwyCMBakjZYqhkOUjVurq3zUjP5bbW9MM0av5%2Fc9lNsShTacGcfNJ14hrRJYUyE%3A1752924069740; tst=h; SESSIONID=g7XUl1IInUF9AXT8Qhzu67qCg6YIsGbsJhzvVkXxO1E; JOID=V1kQAE26VE5LfxtUC7zz1qk37Fgd2WcQGxAuGUTXBQx0O0AEN73oNiV4G1YISQCL874KJt7dAvWPk3H3dDUAq3I=; osd=W1kVBkq2VEtNeBdUDrr02qky6l8R2WIWHBwuHELQCQxxPUcIN7juMSl4HlAPRQCO9bkGJtvbBfmPlnfweDUFrXU=; Hm_lpvt_98beee57fd2ef70ccdd5ca52b9740c49=1752950198; BEC=5ee33e0856ed13c879689106c041a08d"


def process_content_format(content_list: list) -> list:
    """
    通用函数，处理 _format 结尾的列表，并合并连续的同类型内容。
    """
    if not content_list:
        return []

    temp_list = []
    for item in content_list:
        item_type = item.get("type")
        if item_type == "text":
            content_value = item.get("content", "").strip()
            if content_value:
                temp_list.append({"type": "text", "value": content_value})
        elif item_type == "image":
            image_path = item.get("image_path", None)
            if image_path:
                temp_list.append({
                    "type": "image",
                    "image_name": image_path
                })

    if not temp_list:
        return []

    # 合并逻辑
    # 从第一个元素开始
    merged_list = [temp_list[0]]
    for i in range(1, len(temp_list)):
        current_item = temp_list[i]
        last_item_in_merged = merged_list[-1]

        # 检查当前项是否与合并列表中的最后一项类型相同
        if current_item["type"] == last_item_in_merged["type"] and current_item["type"] == "text":
            # 如果是连续的文本，则合并内容
            last_item_in_merged["value"] += "\n\n" + current_item["value"]
        else:
            # 如果类型不同，或不是文本类型，直接添加
            merged_list.append(current_item)

    return merged_list


def process_comments_nested(comments_data: list, upvote_threshold: int = 10) -> list:
    """
    处理评论列表，根据回复关系和时间戳构建嵌套结构，并过滤低赞评论。

    Args:
        comments_data (list): 原始评论数据列表。
        upvote_threshold (int): 评论被包含所需的最低点赞数。
    """
    if not comments_data:
        return []

    # Pass 1: 创建所有评论的映射，并为作者建立时间戳索引
    comment_map = {}
    author_map = {}

    # *** 新增：先过滤低赞评论 ***
    # 只有点赞数达到阈值的评论才会被处理
    filtered_comments_data = [
        c for c in comments_data if c.get("vote_count", 0) >= upvote_threshold
    ]

    for i, comment_raw in enumerate(filtered_comments_data):
        author_info = comment_raw.get("author", {}).get("member", {})
        author_name = author_info.get("name", "匿名用户")
        comment_id_str = f"{comment_raw.get('id')}"
        created_time = comment_raw.get("created_time")

        text_parts = [
            part.get("content", "")
            for part in comment_raw.get("content_format", []) if part.get("type") == "text"
        ]
        comment_text = "".join(text_parts).strip()
        if not comment_text:
            continue

        processed_comment = {
            "comment_id": comment_id_str,
            "author": author_name,
            "upvotes": comment_raw.get("vote_count", 0),
            "text": comment_text,
            "replies": [],
            "_created_time": created_time,
            "_is_child": False
        }

        reply_to = comment_raw.get("reply_to_author")
        if reply_to and reply_to.get("member"):
            processed_comment["reply_to_author"] = reply_to["member"].get("name", "未知用户")

        comment_map[comment_id_str] = processed_comment

        if author_name not in author_map:
            author_map[author_name] = []
        author_map[author_name].append({"id": comment_id_str, "time": created_time})

    # Pass 2: 建立父子关系
    for comment_id, comment in comment_map.items():
        if "reply_to_author" in comment:
            reply_to_name = comment["reply_to_author"]

            if reply_to_name in author_map:
                potential_parents = author_map[reply_to_name]
                valid_parents = [
                    p for p in potential_parents if p["time"] < comment["_created_time"]
                ]

                if valid_parents:
                    parent = max(valid_parents, key=lambda p: p["time"])
                    parent_comment = comment_map.get(parent["id"])
                    # 确保父评论在过滤后仍然存在
                    if parent_comment:
                        parent_comment["replies"].append(comment)
                        comment["_is_child"] = True

    # Pass 3: 筛选顶级评论并按时间排序
    root_comments = [
        comment for comment in comment_map.values() if not comment["_is_child"]
    ]
    root_comments.sort(key=lambda c: c["_created_time"])

    # Pass 4: 递归清理临时键并排序
    def cleanup_and_sort_replies(comments_list):
        for comment in comments_list:
            if "_created_time" in comment:
                del comment["_created_time"]
            if "_is_child" in comment:
                del comment["_is_child"]
            if comment["replies"]:
                comment["replies"].sort(key=lambda r: r.get("_created_time", 0))
                cleanup_and_sort_replies(comment["replies"])

    cleanup_and_sort_replies(root_comments)

    return root_comments


def transform_zhihu_to_video_script(
        input_json_path: str,
        comment_upvote_threshold: int = 10
) -> dict:
    """
    将知乎问答JSON文件转换为视频文案脚本格式。

    Args:
        input_json_path (str): 输入的知乎JSON文件路径。
        comment_upvote_threshold (int): 评论被包含所需的最低点赞数。
    """
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：文件未找到于 {input_json_path}")
        return {}
    except json.JSONDecodeError:
        print(f"错误：文件 {input_json_path} 不是有效的JSON格式。")
        return {}

    question_data = data.get("question", {})
    answers_data = data.get("answers", [])

    output = {
        "question_title": question_data.get("title", "无标题"),
        "question_description": process_content_format(question_data.get("detail_format", [])),
        "tags": [topic.get("name") for topic in question_data.get("topics", []) if topic.get("name")],
        "answers": []
    }

    for i, answer in enumerate(answers_data):
        processed_answer = {
            "answer_id": f"{answer.get('id', i + 1)}",
            "author": answer.get("author", {}).get("name", "匿名用户"),
            "upvotes": answer.get("voteupCount", 0),
            "content": process_content_format(answer.get("content_format", [])),
            "comments": process_comments_nested(
                answer.get("comments", []),
                upvote_threshold=comment_upvote_threshold
            )
        }
        output["answers"].append(processed_answer)

    return output


def parse_zhihu_hot_list(html_content):
    """
    解析知乎热榜的HTML文件，并提取出问题、描述、图片、热度等信息。

    :param html_file_path: 包含知乎热榜内容的HTML文件路径。
    :return: 一个包含解析后数据的列表，每个元素是一个字典，代表一个热榜条目。
    """
    soup = BeautifulSoup(html_content, 'lxml')

    # 存储所有解析出的热榜条目
    parsed_data = []

    # 定位到包含所有热榜条目的主容器
    hot_list_container = soup.find('div', class_='HotList-list')
    if not hot_list_container:
        print("错误: 未能找到热榜列表容器。HTML结构可能已更改。")
        return []

    # 找到所有的热榜条目，每个条目都是一个<section>标签
    hot_items = hot_list_container.find_all('section', class_='HotItem')

    for item in hot_items:
        # 初始化一个字典来存储当前条目的信息
        item_data = {}

        # --- 提取排名 ---
        rank_tag = item.find('div', class_='HotItem-rank')
        item_data['rank'] = rank_tag.text.strip() if rank_tag else 'N/A'

        # --- 提取标签（如“新”） ---
        label_tag = item.find('div', class_='HotItem-label')
        item_data['label'] = label_tag.text.strip() if label_tag else 'N/A'

        # --- 提取内容容器 ---
        content_container = item.find('div', class_='HotItem-content')
        if content_container:
            # --- 提取问题标题 ---
            title_tag = content_container.find('h2', class_='HotItem-title')
            item_data['title'] = title_tag.text.strip() if title_tag else 'N/A'

            # --- 提取问题描述/摘要 ---
            excerpt_tag = content_container.find('p', class_='HotItem-excerpt')
            item_data['excerpt'] = excerpt_tag.text.strip() if excerpt_tag else ''

            # --- 提取链接 ---
            link_tag = content_container.find('a')
            item_data['link'] = link_tag['href'] if link_tag else 'N/A'

            # --- 提取热度信息 ---
            metrics_tag = content_container.find('div', class_='HotItem-metrics')
            item_data['metrics'] = metrics_tag.text.strip().replace('​', '') if metrics_tag else 'N/A'

        # --- 提取图片链接 ---
        image_link_tag = item.find('a', class_='HotItem-img')
        if image_link_tag:
            image_tag = image_link_tag.find('img')
            item_data['image_url'] = image_tag['src'] if image_tag and image_tag.has_attr('src') else 'N/A'
        else:
            item_data['image_url'] = 'N/A'

        parsed_data.append(item_data)

    return parsed_data

def fetch_zhihu_hot(cookie_string: str = None):
    """
    严格复制指定的 fetch 请求，以获取知乎热榜页面。
    现在可以接受一个 cookie 字符串来模拟登录状态。
    """
    # 目标 URL
    url = "https://www.zhihu.com/hot"

    # 严格复制所有请求头
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "cache-control": "max-age=0",
        "priority": "u=0, i",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\", \"Google Chrome\";v=\"138\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
    }

    cookies = parse_cookie_string(cookie_string)

    session = requests.Session()

    try:
        # 发送 GET 请求，这次带上了 cookies 参数
        response = session.get(url, headers=headers, cookies=cookies)

        response.raise_for_status()

        print("请求成功!")
        print(f"状态码: {response.status_code}")
        hot_list_data = parse_zhihu_hot_list(response.text)

        return hot_list_data

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None

def parse_cookie_string(cookie_string: str) -> Dict[str, str]:
    """将浏览器Cookie字符串解析成字典格式。"""
    cookies = {}
    if not cookie_string or cookie_string.lower() == 'your_cookie_string_here':
        return cookies
    for item in cookie_string.split(';'):
        item = item.strip()
        if '=' in item:
            key, value = item.split('=', 1)
            cookies[key] = value
    return cookies


# --- 同步获取回答评论函数 ---
def fetch_zhihu_answer_comments(answer_id: str, limit: int = 100) -> List[Dict]:
    # --- 此处代码保持不变，与原始功能一致 ---
    print(f"--- 开始获取回答ID: {answer_id} 的评论 (上限: {limit}条) ---")
    cookies = parse_cookie_string(ZHIHU_COOKIE_STRING)

    if not cookies:
        print("错误：全局变量 ZHIHU_COOKIE_STRING 为空或格式无效，无法进行请求。")
        return []

    api_url = f"https://www.zhihu.com/api/v4/answers/{answer_id}/comments"
    per_page_limit = 20
    params = {
        "include": "data[*].author,collapsed,reply_to_author,disliked,content,voting,vote_count,is_parent_author,is_author",
        "limit": per_page_limit,
        "offset": 0,
        "status": "open"
    }
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": f"https://www.zhihu.com/question/123/answer/{answer_id}",
        "Accept": "application/json, text/plain, */*"
    }
    all_comments = []

    try:
        with requests.Session() as session:
            session.headers.update(headers)
            session.cookies.update(cookies)

            while len(all_comments) < limit:
                if len(all_comments) + per_page_limit > limit:
                    params['limit'] = limit - len(all_comments)

                response = session.get(api_url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                comments_data = data.get('data', [])
                if not comments_data:
                    print("信息：API未返回更多评论数据，已获取全部内容。")
                    break
                for comment in comments_data:
                    comment['content_format'] = parse_content(comment.get('content', ''))
                all_comments.extend(comments_data)
                print(f"成功获取 {len(comments_data)} 条评论，当前总数: {len(all_comments)}")

                paging_info = data.get('paging', {})
                if paging_info.get('is_end', True):
                    print("信息：API响应is_end=true，所有评论已加载完毕。")
                    break

                params['offset'] += len(comments_data)
                time.sleep(random.uniform(0.8, 2.0))

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code == 404:
            print(f"错误：获取评论失败，API返回404 Not Found。可能原因：回答ID({answer_id})无效或评论已关闭。")
        elif status_code in [401, 403]:
            print(f"错误：获取评论失败，API返回 {status_code} (Unauthorized/Forbidden)。请检查Cookie是否正确且未过期。")
        else:
            print(f"错误：HTTP请求失败，状态码: {status_code}，响应内容: {e.response.text}")
        return []
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"错误：发生网络请求或JSON解析错误: {e}")
        return all_comments[:limit]
    except Exception as e:
        print(f"未知错误: {e}")
        return []

    print(f"--- 任务完成，共获取 {len(all_comments)} 条评论 ---")
    return all_comments[:limit]


def parse_content(html_string: str):
    """
    解析包含文本、媒体或纯文本的字符串，按顺序提取内容。
    V2版本特性:
    - 正确处理 <br> 标签，将其转换成换行符。
    - 能够处理不含任何HTML标签的纯文本输入。
    - 更加健壮，能处理标签与纯文本混合的情况。
    """
    # 如果输入为空或仅包含空白，直接返回空列表
    if not html_string or not html_string.strip():
        return []

    soup = BeautifulSoup(html_string, 'html.parser')
    results = []

    # 遍历所有顶层子元素，可能是标签(Tag)也可能是纯文本(NavigableString)
    for element in soup.children:
        # --- 情况1: 元素是一个HTML标签 ---
        if isinstance(element, Tag):
            # --- 处理文本段落 <p> ---
            if element.name == 'p':
                # 使用 separator='\n' 来保留 <br> 带来的换行
                text = element.get_text(separator='\n', strip=True)
                if text:
                    results.append({
                        'type': 'text',
                        'content': text
                    })

            # --- 处理媒体 <figure> ---
            elif element.name == 'figure':
                img_tag = element.find('img')
                if img_tag:
                    url = img_tag.get('data-original') or img_tag.get('data-actualsrc') or img_tag.get('src')
                    if url and 'data:image/svg+xml' not in url:  # 过滤掉占位符SVG
                        media_type = 'gif' if url.lower().endswith('.gif') else 'image'
                        results.append({
                            'type': media_type,
                            'url': url
                        })
            elif element.name == 'a':
                text = element.get_text(strip=True)
                href = element.get('href')
                if text and href:
                    results.append({
                        'type': 'link',
                        'content': text,
                        'url': href
                    })

        # --- 情况2: 元素是纯文本字符串 ---
        elif isinstance(element, NavigableString):
            text = str(element).strip()
            if text:
                results.append({
                    'type': 'text',
                    'content': text
                })

    return results

def download_image(real_final_result):
    # 获取question的detail_format字段
    question = real_final_result.get('question', {})
    question_detail = question.get('detail_format', [])
    question_id = question.get('id', 'unknown')
    image_dir = question_id
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    count = 0
    # 找到类型为'image'的内容
    for item in question_detail:
        if item.get('type') == 'image':
            count += 1
            image_url = item.get('url')
            if image_url:
                base_name = f"question_image_{question_id}_{count}.jpg"
                save_path = os.path.join(image_dir, base_name)
                save_path = pathlib.Path(save_path)
                if os.path.exists(save_path):
                    print(f"文件已存在，跳过下载: {save_path}")
                else:
                    download_public_image(image_url, save_path)
                if os.path.exists(save_path):
                    item['image_abs_path'] = str(save_path.resolve())
                    item['image_path'] = base_name

    # 获取answers的content_format字段
    answers = real_final_result.get('answers', [])
    for answer in answers:
        content_format = answer.get('content_format', [])
        if not content_format:
            continue
        count = 0
        for item in content_format:
            if item.get('type') == 'image':
                count += 1
                image_url = item.get('url')
                if image_url:
                    base_name = f"answer_image_{answer.get('id', 'unknown')}_{count}.jpg"
                    save_path = os.path.join(image_dir, base_name)
                    save_path = pathlib.Path(save_path)
                    if os.path.exists(save_path):
                        print(f"文件已存在，跳过下载: {save_path}")
                    else:
                        download_public_image(image_url, save_path)
                    if os.path.exists(save_path):
                        item['image_abs_path'] = str(save_path.resolve())
                        item['image_path'] = base_name
    return real_final_result

def add_image_desc(real_final_result):
    """
    为问题和回答中的图片添加描述信息。
    目前仅为图片添加了占位符描述，实际应用中可以根据具体内容进行更详细的描述。
    """
    real_final_result_copy = real_final_result.copy()
    # 去除real_final_result_copy中的answers在的comments
    for answer in real_final_result_copy.get('answers', []):
        answer.pop('comments', None)
    question = real_final_result.get('question', {})
    question_detail = question.get('detail_format', [])

    for item in question_detail:
        if item.get('type') == 'image':
            item['description'] = "这是一个图片"

    answers = real_final_result.get('answers', [])
    for answer in answers:
        content_format = answer.get('content_format', [])
        for item in content_format:
            if item.get('type') == 'image':
                item['description'] = "这是一个回答中的图片"

    return real_final_result


# --- 同步获取问题回答并补充评论 ---
def fetch_question_answers(question_id: str, output_filename: str, desired_answers: int = 5, max_no_increase: int = 3):
    print(f"--- 目标问题ID: {question_id}, 期望获取 {desired_answers} 个回答 ---")
    all_answers_data = []
    no_increase_count = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-infobars"],
            slow_mo=50
        )
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1920, "height": 1080},
            locale="zh-CN",
            storage_state=AUTH_FILE,
        )
        context.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN','en-US'] });
            window.chrome = { runtime: {} };
            """
        )

        page = context.new_page()

        def handle_response(response):
            if "/api/v4/questions/" in response.url and "/feeds" in response.url:
                try:
                    data = response.json()
                    new_answers = data.get('data', [])
                    print(f"--- 捕获到 {len(new_answers)} 条新回答的API响应 ---")
                    for new_answer in new_answers:
                        all_answers_data.append(new_answer.get('target', {}))
                except Exception as e:
                    print(f"解析API响应失败: {e}, URL: {response.url}")

        page.on("response", handle_response)

        try:
            question_url = f"https://www.zhihu.com/question/{question_id}"
            print(f">>> 正在访问问题页面: {question_url}")
            page.goto(question_url, wait_until="networkidle", timeout=60000)
            print(">>> 等待问题标题加载...")
            page.wait_for_selector('h1.QuestionHeader-title', timeout=15000)
            print(">>> 页面加载成功，问题标题已出现。")

            script_tag = page.query_selector('#js-initialData')
            if script_tag:
                json_data_str = script_tag.inner_text()
                initial_data = json.loads(json_data_str)
                questions = initial_data.get('initialState', {}).get('entities', {}).get('questions', {}).get(question_id, {})
                questions['detail_format'] = parse_content(questions.get('detail', ''))

                initial_answers = initial_data.get('initialState', {}).get('entities', {}).get('answers', {})
                if initial_answers:
                    all_answers_data.extend(initial_answers.values())
                    print(f"--- 成功从初始HTML中提取 {len(initial_answers)} 个回答 ---")
            else:
                print("警告: 页面加载后未找到 #js-initialData 标签，可能无法获取第一页回答。")

            while len(all_answers_data) < desired_answers:
                prev_count = len(all_answers_data)
                page.evaluate("window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })")
                time.sleep(random.uniform(2.5, 4.0))
                curr_count = len(all_answers_data)
                if curr_count > prev_count:
                    no_increase_count = 0
                    print(f"回答数量增加：{prev_count} → {curr_count}，继续滚动。")
                else:
                    no_increase_count += 1
                    print(f"警告：滚动后回答数量未增加（{no_increase_count}/{max_no_increase}）。")
                    if no_increase_count >= max_no_increase:
                        print("连续多次未增，停止抓取。")
                        break

        except PlaywrightTimeoutError as e:
            print(f"在页面加载或操作过程中发生超时: {e}")
            page.screenshot(path="error_screenshot.png", full_page=True)
            print("已保存错误截图至 error_screenshot.png")
        except Exception as e:
            print(f"在页面加载或操作过程中发生错误: {e}")
            page.screenshot(path="error_screenshot.png", full_page=True)
            print("已保存错误截图至 error_screenshot.png")
        finally:
            page.remove_listener("response", handle_response)
            browser.close()

    print(f"\n--- 抓取完成，共捕获 {len(all_answers_data)} 条回答数据 ---")
    final_data = all_answers_data[:desired_answers]
    real_final_result = {'question': questions, 'answers': final_data}
    save_json(output_filename, real_final_result)
    print(f"原始API响应数据已保存至文件: {output_filename}")

    print(f"\n--- 开始补充评论信息获取 ---")
    for answer in final_data:
        answer_id = answer.get("id")
        if not answer_id:
            print(f"警告: 跳过无效回答数据: {answer}")
            continue
        comments = fetch_zhihu_answer_comments(answer_id, limit=50)
        answer['content_format'] = parse_content(answer.get('content', ''))
        answer["comments"] = comments
        print(f"回答ID {answer_id} 的评论数量: {len(comments)}")
    real_final_result = {'question': questions, 'answers': final_data}
    save_json(output_filename, real_final_result)

    real_final_result = download_image(real_final_result)
    save_json(output_filename, real_final_result)
    video_script_data = transform_zhihu_to_video_script(
        output_filename,
        comment_upvote_threshold=10
    )
    save_json(output_filename, video_script_data)

    print(f"最终结果已保存至文件: {output_filename}")


if __name__ == "__main__":
    question_id = "1929871927457080536"
    fetch_question_answers(question_id, f"{question_id}/zhihu_answers_{question_id}.json", desired_answers=20)
    # hot_list_data = fetch_zhihu_hot(ZHIHU_COOKIE_STRING)
    #
    # with open('zhihu_hot_list.json', 'w', encoding='utf-8') as f:
    #     json.dump(hot_list_data, f, ensure_ascii=False, indent=4)
    # print("结果已保存到 zhihu_hot_list.json 文件中。")