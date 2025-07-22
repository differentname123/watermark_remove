import copy
import json
import os
import pathlib
import random
import re
import time
from bs4 import BeautifulSoup, Tag, NavigableString

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import requests
from typing import List, Dict

from LLM.gemini import analyze_images_gemini
from common_utils.common_utils import save_json, download_public_image, read_json, string_to_object

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
                    "image_name": image_path,
                    "image_path": item.get("image_abs_path", None)
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
            "upvotes": answer.get("voteup_count", 0),
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


# 1. 常量定义
MAX_IMAGE_BATCH_SIZE = 20
MAX_RETRIES = 3
BASE_PROMPT = """
            你是一位精通多模态分析和视频脚本策划的AI专家。你的任务是接收一个包含文本上下文的JSON文件和一系列对应的图片文件，为每一张图片生成一段深度结合其视觉内容与文本上下文的、可用于视频制作的结构化描述。

            我将为你提供两部分核心输入：
            一个完整的JSON对象：其中包含了所有图文的排版结构和文字内容。
            一系列实际的图片文件：这些图片的文件名与JSON中的image_name一一对应。

            你的分析必须同时基于这两部分输入进行。
            我将提供给你一个完整的JSON对象，该对象完整记录了一段知乎问答的所有内容，其结构如下：
            - 它包含`question_title`, `question_description`等顶层字段。
            - question_description和每个answer的content字段都是一个数组，由"type": "text"和"type": "image"的对象交错组成。这个顺序精准地反映了原文的图文排版。

            **你的任务是：**

            1.  **解析完整数据**：接收并理解我提供的整个JSON对象。
            2.  **自动定位与关联**：
                * 先根据我上传的图片名称一一定位json数据中`type`为`image`的对象。
                * 对于你找到的**每一张图片**，自动从找到图片的上下文，也就是上一个type为`text`的对象和下一个type为`text`的对象中提取相关信息。如果图片是数组的第一个元素，则其‘前文’为空；如果它是最后一个元素，则其‘后文’为空。
            3.  **批量生成描述**：
                * 结合每张图片的**直接上下文**以及该回答的**整体论点**。
                * 逐一生成描述: 对定位到的每一张图片，运用以下“三步分析法”生成其描述：
                  第一步：客观画面分析: 详细客观说明图片的基本内容和类型（如网页截图、数据图表、新闻配图、表情包等）。
                  第二步：上下文功能解读: 结合你提取的上下文文字，分析这张图片在此处的核心作用。它是在为上文观点提供证据、补充信息、进行讽刺/调侃，还是用于视觉总结或情感渲染？揭示作者放置这张图片的意图。
                  第三步：生成最终描述: 将前两步的分析融合成一段精炼、流畅的综合描述。这段描述必须能直接指导视频创作者，让他们明白“画面是什么”以及“为什么要在解说词的这个节点插入这个画面”。
            4.  **整合输出**：
                * 请只返回一个纯粹的JSON对象，不要包含任何额外的欢迎语、解释或总结。
                * 这个JSON对象的键 (key) 必须是图片的名称 (image_name)。
                * 这个JSON对象的值 (value) 必须是按照上述规则为该图片生成的最终综合描述。
                * 结果中需要包含所有在输入数据中找到的图片及其描述。

            **下面是你需要处理的完整JSON数据：**
"""


def _create_answer_batches(real_final_result: dict) -> list[list[dict]]:
    """
    筛选出有图片的回答，并将它们打包成批次。
    每个批次的图片总数不超过 MAX_IMAGE_BATCH_SIZE。
    """
    answers_with_image_info = []

    # 步骤1: 筛选有图片的回答，并收集信息
    for i, answer in enumerate(real_final_result.get('answers', [])):
        images_in_answer = []
        for item in answer.get('content', []):
            if item.get('type') == 'image':
                images_in_answer.append({
                    'image_path': item['image_path'],
                    'image_name': item['image_name']
                })

        if images_in_answer:
            answer_copy = copy.deepcopy(answer)
            answer_copy.pop('comments', None)  # 移除不必要的字段
            answers_with_image_info.append({
                'data': answer_copy,
                'images': images_in_answer,
                'image_count': len(images_in_answer)
            })

    if not answers_with_image_info:
        return []

    # 步骤2: 将回答打包成批次
    all_batches = []
    current_batch = []
    image_count_in_current_batch = 0

    for answer_info in answers_with_image_info:
        # 如果当前批次为空，或者加入新回答后图片总数不超过限制，则加入当前批次
        # 特殊情况：如果一个回答自身的图片数就超过了限制，它也必须自成一个批次
        if not current_batch or (image_count_in_current_batch + answer_info['image_count']) <= MAX_IMAGE_BATCH_SIZE:
            current_batch.append(answer_info)
            image_count_in_current_batch += answer_info['image_count']
        else:
            # 否则，当前批次已满，将其存入总列表，并用当前回答开启新批次
            all_batches.append(current_batch)
            current_batch = [answer_info]
            image_count_in_current_batch = answer_info['image_count']

    # 不要忘记将最后一个正在构建的批次加入总列表
    if current_batch:
        all_batches.append(current_batch)

    return all_batches


def validate_image_analysis(base_name_list, image_analysis_result):
    """
    检查 base_name_list 中的图片名称是否和 image_analysis_result 的键完全一致。
    如果不一致，返回 False 并打印出差异；一致则返回 True。
    """
    expected = set(base_name_list)
    actual = set(image_analysis_result.keys())

    missing = expected - actual  # 在 expected 中但不在 actual 中
    extra = actual - expected  # 在 actual 中但不在 expected 中

    if missing or extra:
        if missing:
            print(f"缺少分析结果的图片：{missing}")
        if extra:
            print(f"多余的分析结果键：{extra}")
        return False

    return True


def _process_answer_batch(batch_of_answers: list[dict], real_final_result: dict) -> dict:
    """
    处理单个包含多个回答的批次。
    """
    # 1. 收集批次内所有图片路径
    image_paths_for_api = []
    for question_info in real_final_result.get('question_description', []):
        if question_info['type'] == 'image':
            image_paths_for_api.append(question_info['image_path'])

    for answer_info in batch_of_answers:
        for img in answer_info['images']:
            image_paths_for_api.append(img['image_path'])

    base_name_list = [os.path.basename(path) for path in image_paths_for_api]

    # 2. 构建本次API请求的上下文JSON
    context_data = {k: v for k, v in real_final_result.items() if k != 'answers'}
    context_data['answers'] = [info['data'] for info in batch_of_answers]

    # 3. 生成最终的Prompt
    prompt = f"{BASE_PROMPT}\n{json.dumps(context_data, ensure_ascii=False, indent=2)}"

    total_images = len(image_paths_for_api)
    total_answers = len(batch_of_answers)
    print(f"--- 开始处理批次，包含 {total_answers} 个回答和 {total_images} 张图片 ---")

    # 4. 调用API并重试
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = analyze_images_gemini(
                prompt=prompt,
                image_paths=image_paths_for_api
            )
            image_analysis_result = string_to_object(raw)

            # 确保base_name_list中的图片名称与分析结果的键一致
            if validate_image_analysis(base_name_list, image_analysis_result):
                print(f"批次处理成功 (尝试 {attempt}/{MAX_RETRIES})")
                return image_analysis_result
            else:
                print(
                    f"[警告][尝试 {attempt}] 分析结果数量({len(image_analysis_result)}) 图片数量({total_images})，重试中...{image_analysis_result}")
        except Exception as e:
            print(f"[错误][尝试 {attempt}] 调用API时出现异常：{e}，重试中...")

    print(f"[严重错误] 批次处理失败 {MAX_RETRIES} 次，跳过此批次。")
    return {}


def add_image_desc_by_answer_batching(real_final_result: dict) -> dict:
    """
    为图片生成描述的最终优化版。
    - 筛选出有图片的回答。
    - 将这些回答打包成批次（每批图片总数<=20）。
    - 为每个批次构建包含问题和当前批次回答的上下文。
    - 汇总所有结果并更新到最终数据中。
    """
    # 1. 检查问题描述中是否有图片，单独处理 (这是一个简化，也可以融入批次)
    # 为简化逻辑，我们假设主要处理目标在answers中，问题区的图片可单独或优先处理
    # 此处我们主要关注answers的分批
    question_images = []
    for item in real_final_result.get('question_description', []):
        if item.get('type') == 'image':
            question_images.append(item)

    # 2. 创建基于回答的批次
    answer_batches = _create_answer_batches(real_final_result)

    if not answer_batches and not question_images:
        print("--- 未发现任何图片，无需处理 ---")
        return real_final_result

    print(f"--- 发现有图片的回答，已创建 {len(answer_batches)} 个处理批次 ---")

    all_image_descriptions = {}

    for i, batch in enumerate(answer_batches):
        print(f"\n>>> 处理批次 {i + 1}/{len(answer_batches)}")
        batch_results = _process_answer_batch(batch, real_final_result)
        if batch_results:
            all_image_descriptions.update(batch_results)

    print("\n--- 所有批次处理完成，开始更新最终结果 ---")

    # 4. 将汇总后的描述更新回原始数据结构中
    updated_result = copy.deepcopy(real_final_result)
    for answer in updated_result.get('answers', []):
        for item in answer.get('content', []):
            if item.get('type') == 'image':
                image_name = item['image_name']
                desc = all_image_descriptions.get(image_name)
                if desc:
                    item['image_desc'] = desc
    for question_info in updated_result.get('question_description', []):
        if question_info['type'] == 'image':
            image_name = question_info['image_name']
            desc = all_image_descriptions.get(image_name)
            if desc:
                question_info['image_desc'] = desc

    print("--- 图片描述更新完成！ ---")
    return updated_result


def gen_video_info(real_final_result):
    """
    根据回答的信息生成视频信息。主要包括使用的图片，和对应的文案
    """
    # 深度拷贝
    video_info = copy.deepcopy(real_final_result)
    # 获取question_description
    question_description = video_info.get('question_description', [])
    for desc_dict in question_description:
        if desc_dict.get('type') == 'image':
            desc_dict['image_name'] = desc_dict.get('image_path', '').split('/')[-1]
            desc_dict['image_desc'] = desc_dict.get('content', '')


def extract_image(real_final_result):
    """
    将图片专门抽取出来，并删除原始的图片描述
    """
    images = []

    # 获取question_description
    question_description = real_final_result.get('question_description', [])
    for desc_dict in question_description[:]:
        if desc_dict.get('type') == 'image':
            images.append({
                'image_name': desc_dict.get('image_name'),
                'image_desc': desc_dict.get('image_desc', '')
            })
            question_description.remove(desc_dict)  # 删除已处理的desc_dict

    # 获取answers中的图片
    answers = real_final_result.get('answers', [])
    for answer in answers:
        content_format = answer.get('content', [])
        for item in content_format[:]:
            if item.get('type') == 'image':
                images.append({
                    'image_name': item.get('image_name'),
                    'image_desc': item.get('image_desc', '')
                })
                content_format.remove(item)  # 删除已处理的item

    # 添加到real_final_result中
    real_final_result['image_lib'] = images
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
                questions = initial_data.get('initialState', {}).get('entities', {}).get('questions', {}).get(
                    question_id, {})
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
    # 补充voteup_count字段，如果不存在就尝试从'voteupCount'获取
    for answer in all_answers_data:
        if 'voteupCount' in answer:
            answer['voteup_count'] = answer.pop('voteupCount')
        else:
            answer['voteup_count'] = 0

        # 如果 voteup_count 是 0，尝试从 matrix_tips 中提取
        if answer['voteup_count'] == 0 and 'matrix_tips' in answer:
            tips = answer['matrix_tips']
            match = re.search(r'(\d+)\s*赞同', tips)
            if match:
                answer['voteup_count'] = int(match.group(1))

    # 按照'voteup_count'降序排序回答数据
    all_answers_data.sort(key=lambda x: x.get('voteup_count', 0), reverse=True)

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
    print(f"回答评论数据已保存至文件: {output_filename}")

    real_final_result = download_image(real_final_result)
    save_json(output_filename, real_final_result)
    print(f"图片已下载并更新至文件: {output_filename}")
    video_script_data = transform_zhihu_to_video_script(
        output_filename,
        comment_upvote_threshold=10
    )
    real_final_result = add_image_desc_by_answer_batching(video_script_data)
    save_json(output_filename, real_final_result)
    real_final_result = extract_image(real_final_result)
    save_json(output_filename, real_final_result)

    print(f"带图片描述结果保存至文件: {output_filename}")


if __name__ == "__main__":
    question_id = "1930699864796280471"
    # fetch_question_answers(question_id, f"{question_id}/zhihu_answers_{question_id}.json", desired_answers=20)

    # hot_list_data = fetch_zhihu_hot(ZHIHU_COOKIE_STRING)
    # with open('zhihu_hot_list.json', 'w', encoding='utf-8') as f:
    #     json.dump(hot_list_data, f, ensure_ascii=False, indent=4)
    # print("结果已保存到 zhihu_hot_list.json 文件中。")

    output_file = f"{question_id}/zhihu_answers_{question_id}.json"
    real_final_result = read_json(output_file)
    # add_image_desc_by_answer_batching(real_final_result)
    real_final_result = extract_image(real_final_result)
    save_json(output_file, real_final_result)