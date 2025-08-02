import copy
import json
import multiprocessing
import os
import pathlib
import random
import re
import time

from PIL import Image, UnidentifiedImageError

from bs4 import BeautifulSoup, Tag, NavigableString

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import requests
from typing import List, Dict

from LLM.gemini import analyze_images_gemini, analyze_videos_gemini
from common_utils.common_utils import save_json, download_public_image, read_json, string_to_object, find_key_values, \
    download_public_video, ms_to_time, time_to_ms
from common_utils.split_scenes import find_and_split_scenes
from common_utils.video_utils import probe_duration
from content_community.zhihu.gen_video_by_video_info import gen_video_by_video_info
from content_community.zhihu.gen_zhihu_video_info import gen_video_final_info

# --- 配置区域 ---
AUTH_FILE = "zhihu_auth_state.json"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/115.0.5790.110 Safari/537.36"
)
ZHIHU_COOKIE_STRING = "_xsrf=EGbVA6NHTaM3dXlCMEiWj9aRBvWl4inW; _zap=34c3bc6c-ebae-4e0d-9a06-5bfb7b8a9548; d_c0=APARO_5-wBmPTvibmi_6NacNH42miN-ERZY=|1735194921; Hm_lvt_98beee57fd2ef70ccdd5ca52b9740c49=1752912686; HMACCOUNT=14EFD85132347319; DATE=1752912688356; crystal=U2FsdGVkX19HZsSWXHhQvCmTy0IhrzSuQUWjAmBNS9sgZFRn8/yuh3qnHWWPtqhs6Fx+lpP4dATHFISdyemFNM4nXvmwy0g7ICamPZ7CB/IU+sI4WwbiZS83jULObdwxNXvxwM1lrWOA06h6rgMPFiBn+qiMLtZXpkprUsDW2FNS69MOpODrf6kUIyRL/QXg9IJ08veQmhzEr8G7YQHsdz07b0Wmj/50nQPJAtdug/kh+RlGpYjjx8ALS6jGD2Gq; __snaker__id=RQbe8vWX2MrZo7OW; cmci9xde=U2FsdGVkX184glWsyX1mezYXZa3qDzIhjRBe9TQQVEY+LLFTFMDXP9nSs9RkgMzxPSZtU4lOrgeYZefNb02KEA==; pmck9xge=U2FsdGVkX1/T3lTFMBSW1MNHTt55yQ7uWCI2fzMhaBs=; assva6=U2FsdGVkX1803CTSuJgB3mtcX8vMBRJ4mrNXHxoktsA=; assva5=U2FsdGVkX18s7ilkblADh/oGQosbZLgtP7rzonbhC4T0Gf8YWm1GZf8atIJSu1QVH69xU8U+rNeMHgJRf8dEEQ==; vmce9xdq=U2FsdGVkX1+9L5kRV9p5NPBm2ZFxevxsXB601UTW1lO8oB1sywbxX35uCVgDtPFrCRoAhGz6Qw95IDt5HxZKgRgHcwII5jsliZ9AEeEMx0QyMg0NlFbFg2/No39rs8FcREB1wxx4Hg7ZLGq+HBSZ6UbnPSt1xTw3YsTv3I27GXM=; z_c0=2|1:0|10:1752912935|4:z_c0|92:Mi4xemV3ZkR3QUFBQUFBOEJFN19uN0FHU1lBQUFCZ0FsVk5KNkpvYVFCQlUzLUlBR3dQNGcwQkhEOWNXcmsyZ0VFZnJB|55035f8a3b883e94b09b952267d56c00a1a59a98cc43712feb238eb4056739b8; q_c1=e3e3832501fe4a17ba8bf7b5b47f6e60|1752912935000|1752912935000; __zse_ck=004_9njDuYcKusVXiMuD5SZ7LomPNAudnA3idCp0Ig/GY0=5gYSDmi0TNqJ7FjyXoqYArffzF1BhUy=GuO4ecVqJEGHl8QUnfJi3U5WYI/Y1QFhwF7Gz7I7xqjnm05LC1vY2-1G/2qpHnazSH76s+360GqWHjozS9rp18hQPVk+DOgjdq/n5leUgxJ+237tPuguqC5x1a1EjhuQ/RyoAp0+8lSPskVcy16jac+kELW9lwJayh1jscDZfo7NsPnlZfJUWL; gdxidpyhxdE=tdTqBn8olmH5j1Mvzgb2xuBP1JV%5CEOlNCeWjTWLag7PU1kAgwZc0iYoJNXHP4pesZ9c6pt6sCy9XvWRvcCih4H3wlMve85vPDsuwZGH0niB0eBEbOdwyCMBakjZYqhkOUjVurq3zUjP5bbW9MM0av5%2Fc9lNsShTacGcfNJ14hrRJYUyE%3A1752924069740; tst=h; SESSIONID=g7XUl1IInUF9AXT8Qhzu67qCg6YIsGbsJhzvVkXxO1E; JOID=V1kQAE26VE5LfxtUC7zz1qk37Fgd2WcQGxAuGUTXBQx0O0AEN73oNiV4G1YISQCL874KJt7dAvWPk3H3dDUAq3I=; osd=W1kVBkq2VEtNeBdUDrr02qky6l8R2WIWHBwuHELQCQxxPUcIN7juMSl4HlAPRQCO9bkGJtvbBfmPlnfweDUFrXU=; Hm_lpvt_98beee57fd2ef70ccdd5ca52b9740c49=1752950198; BEC=5ee33e0856ed13c879689106c041a08d"

def sort_and_filter_by_score(
        items,
        min_score: float = 0
):
  """
  1. 过滤掉 relevance.score < min_score 的项
  2. 按 score 从高到低排序

  :param items: 待处理的字典列表
  :param min_score: 分数阈值，低于该值的项会被丢弃
  :return: 过滤并排序后的新列表
  """

  def get_score(item) -> float:
    container = item.get('image_desc') or item.get('image_info') or {}
    relevance = container.get('relevance') or {}
    return relevance.get('score', 0)

  # 先过滤
  filtered = [item for item in items if get_score(item) >= min_score]
  # 再排序
  return sorted(filtered, key=get_score, reverse=True)

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
    video_lib = data.get("video_lib", [])
    output = {
        "question_title": question_data.get("title", "无标题"),
        "question_description": process_content_format(question_data.get("detail_format", [])),
        "tags": [topic.get("name") for topic in question_data.get("topics", []) if topic.get("name")],
        "answers": [],
        "video_lib": video_lib
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
    V3版本特性:
    - (新) 能够识别特定 class 的 <a> 标签作为视频元素，并提取标题、封面和链接。
    - (新) 能够处理标准的 <video> 标签。
    - 正确处理 <br> 标签，将其转换成换行符。
    - 能够处理不含任何HTML标签的纯文本输入。
    - 更加健壮，能处理标签与纯文本混合的情况。
    """
    if not html_string or not html_string.strip():
        return []

    soup = BeautifulSoup(html_string, 'html.parser')
    results = []

    for element in soup.children:
        if isinstance(element, Tag):
            if element.name == 'p':
                text = element.get_text(separator='\n', strip=True)
                if text:
                    results.append({
                        'type': 'text',
                        'content': text
                    })

            elif element.name == 'figure':
                img_tag = element.find('img')
                if img_tag:
                    url = img_tag.get('data-original') or img_tag.get('data-actualsrc') or img_tag.get('src')
                    if url and 'data:image/svg+xml' not in url:
                        media_type = 'gif' if url.lower().endswith('.gif') else 'image'
                        results.append({
                            'type': media_type,
                            'url': url
                        })

            # --- MODIFIED: 修改对 <a> 标签的处理逻辑 ---
            elif element.name == 'a':
                # 首先检查它是否是一个视频链接 (根据class判断)
                if 'video-box' in element.get('class', []):
                    video_title_tag = element.find('span', class_='title')
                    video_title = video_title_tag.get_text(strip=True) if video_title_tag else ''

                    video_thumb_tag = element.find('img', class_='thumbnail')
                    video_thumb_url = video_thumb_tag.get('src') if video_thumb_tag else ''

                    video_page_url = element.get('href')

                    if video_page_url:
                        results.append({
                            'type': 'video',
                            'title': video_title,
                            'thumbnail_url': video_thumb_url,
                            'url': video_page_url
                        })
                # 如果不是视频链接，则按原来的方式处理为普通链接
                else:
                    text = element.get_text(strip=True)
                    href = element.get('href')
                    if text and href:
                        results.append({
                            'type': 'link',
                            'content': text,
                            'url': href
                        })

            # --- NEW: 新增对标准 <video> 标签的处理 ---
            elif element.name == 'video':
                video_url = element.get('src')
                # poster 属性通常用于存放视频封面图
                poster_url = element.get('poster')
                if video_url:
                    results.append({
                        'type': 'video',
                        'title': element.get('title', ''),  # 尝试获取title属性作为标题
                        'thumbnail_url': poster_url or '',
                        'url': video_url
                    })

        elif isinstance(element, NavigableString):
            text = str(element).strip()
            if text:
                results.append({
                    'type': 'text',
                    'content': text
                })

    return results


def ensure_file_is_jpg(file_path: str | pathlib.Path) -> bool:
    """
    确保指定路径的文件是一个真正的、标准的JPEG图片。

    - 如果文件已经是标准JPEG (格式为JPEG, 模式为RGB)，则不执行任何操作。
    - 如果文件不是JPEG格式或不是RGB模式 (例如，一个被重命名为.jpg的PNG文件)，
      它将被转换为真正的JPEG并覆盖原始文件。
    - 如果文件路径不存在或文件不是一个有效的图片，则会报告错误。

    Args:
        file_path (str | pathlib.Path): 指向声称为JPG的图片文件路径。

    Returns:
        bool: 如果操作成功或无需操作，返回 True。如果发生错误，返回 False。
    """
    # 1. 统一路径对象并检查文件是否存在
    path = pathlib.Path(file_path)
    if not path.is_file():
        print(f"❌ 错误: 文件不存在 -> {path}")
        return False

    try:
        # 2. 打开图片并检查其元信息
        with Image.open(path) as img:
            # 获取真实的格式和色彩模式
            original_format = img.format
            original_mode = img.mode

            # 3. 判断是否需要转换
            # 条件：真实格式不是'JPEG' 或者 色彩模式不是'RGB'
            if original_format != 'JPEG' or original_mode != 'RGB':
                # 为了安全地覆盖原文件，我们先将图片数据加载到内存
                img.load()

                # 关闭文件句柄后，进行转换和保存
                # 确保转换为RGB模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 使用临时文件进行保存，防止程序中断导致原文件损坏
                temp_path = path.with_suffix(f"{path.suffix}.tmp")
                img.save(temp_path, 'JPEG', quality=100)  # 使用高质量保存

                # 用转换好的临时文件替换原始文件 (这是一个原子操作，更安全)
                temp_path.replace(path)

                print(f"✅ 成功: 文件已转换为真正的JPG并保存 -> {path}")
            else:
                pass

        return True

    except UnidentifiedImageError:
        print(f"❌ 错误: 文件不是一个有效的图片格式 -> {path}")
        return False
    except Exception as e:
        print(f"❌ 发生未知错误: {e} -> {path}")
        return False

def download_image(real_final_result):
    # 获取question的detail_format字段
    question = real_final_result.get('question', {})
    question_detail = question.get('detail_format', [])
    question_id = question.get('id', 'unknown')
    image_dir = f'{question_id}/images'
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
                if os.path.exists(save_path) and ensure_file_is_jpg(save_path):
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
                    if os.path.exists(save_path) and ensure_file_is_jpg(save_path):
                        item['image_abs_path'] = str(save_path.resolve())
                        item['image_path'] = base_name
    return real_final_result


# 1. 常量定义
MAX_IMAGE_BATCH_SIZE = 20
MAX_RETRIES = 3

VIDEO_BASE_PROMPT = """
### **1. 角色与目标 (Role & Goal)**
你是一位顶尖的多模态分析专家和视频脚本策划AI。你的核心目标是接收一个话题背景和**一组预先切分好场景的视频数据**，为每一个视频生成一段深度结构化的分析JSON。这份JSON必须精准、高效地解构**每一个给定场景**内的视听内容，并对整个视频与给定话题的匹配度进行可量化、可解释的评估，最终服务于“将视频素材与文案进行智能匹配、排序与筛选”的终极需求。
-----

### **2. 核心输入 (Core Inputs)**

1.  `topic_context` (话题背景): 一个字符串，描述这批视频共同的主题或应用场景（例如：“展现城市科技创新”、“描绘家庭温馨瞬间”）。你必须将此背景作为所有分析和评估的核心“滤镜”与最高准则。
2.  `video_scene_data` (视频场景数据): 一个JSON对象。其键为视频文件名，值为该视频预先切分好的场景列表 (`scenes`)。每个场景都精确定义了 `scene_name`、`start_time` (毫秒) 和 `end_time` (毫秒)。你必须严格基于此结构进行分析。
    **输入格式示例：**
    ```json
    {
      "video_1934271416687100245.mp4": {
        "scenes": [
          {"scene_name": "场景1", "start_time": 0, "end_time": 12687},
          {"scene_name": "场景2", "start_time": 12687, "end_time": 36496},
          {"scene_name": "场景3", "start_time": 36496, "end_time": 42880}
        ]
      },
      "video_1934286389123474140.mp4": {
        "scenes": [
          {"scene_name": "场景1", "start_time": 0, "end_time": 7709},
          {"scene_name": "场景2", "start_time": 7709, "end_time": 44690}
        ]
      }
    }
    ```

-----

### **3. 核心任务与分析框架 (Core Task & Framework)**

你需要为 `video_scene_data` 中的每一个视频，严格按照下面定义的分析框架，生成一个完整的 JSON 分析对象。

#### **分析框架定义**

```json
{
  "video_filename.mp4": {
    "video_level_analysis": {
      "topic_relevance": {
        "score": 0,
        "reasoning": "...",
        "matching_keywords": ["..."]
      },
      "desc": "...",
      "semantic_tags": {
        "Subject": ["..."],
        "Scene_Action": ["..."],
        "Emotion_Atmosphere": ["..."],
        "Symbol_Concept": ["..."]
      },
      "narrative_arc": ["..."],
      "dynamic_tags": ["..."],
      "audio_tags": ["..."]
    },
    "scene_level_analysis": [
      {
        "scene_name": "...",
        "start_time": 0,
        "end_time": 0,
        "desc": "...",
        "semantic_tags": { "...": ["..."] },
        "narrative_arc": ["..."],
        "dynamic_tags": ["..."],
        "audio_tags": ["..."]
      }
    ]
  }
}
```

#### **`topic_relevance` 字段详解**

  * `score` (整数, 0-100): 量化视频与 `topic_context` 的相关度。
      * **85-100 (高度相关):** 核心主题、情感、象征与话题完美契合。
      * **50-84 (中度相关):** 主要内容与话题相关，但次要方面匹配度一般。
      * **1-49 (低度相关):** 仅含零散相关元素，整体关联弱。
      * **0 (完全无关):** 与话题无任何可识别联系。
  * `reasoning` (字符串): 必须引用分析标签来支撑分数判断的简洁说明。
  * `matching_keywords` (字符串数组): 从视频中提炼的、与话题直接相关的关键词。

-----

### **4. 关键规则与思维链 (Key Rules & Chain of Thought)**

1.  **上下文优先原则 (Context-First Principle)**

      * **核心指令：** 必须将 `topic_context` 作为所有解读的首要依据。
      * **示例：** “环保”背景下，工厂是 `Symbol_Concept: ["工业污染"]`；“经济发展”背景下，则是 `Symbol_Concept: ["工业基础"]`。

2.  **综合相关性评估原则 (Integrated Relevance Assessment Principle)**

      * **步骤一：** 完成对所有预设场景的全面内容分析。
      * **步骤二：** 综合所有场景的分析结果，在 `video_level_analysis` 层面评估与 `topic_context` 的整体契合度。
      * **步骤三：** 严格按照评分标准赋分，并提炼 `reasoning` 和 `matching_keywords`。

3.  **为匹配而生的标签原则 (Tagging-for-Matching Principle)**

      * **核心指令：** 标签必须具备明确的匹配价值，拒绝模糊或冗余的标签。
      * **`dynamic_tags`：** 需聚焦视觉效果，如 `节奏加快`, `镜头聚焦`。
      * **`audio_tags`：** 需使用 `类型:描述` 扁平格式，如 `BGM:激昂史诗`, `VO:专业男声`, `SFX:心跳声`。

4.  **场景忠实原则 (Scene Fidelity Principle) -【最高优先级】**

      * **1. 认知与遵循 (Acknowledge & Follow):** 你必须认知到输入数据已包含精确的场景列表 (`scenes`)。你的任务是填充分析，而非切分。
      * **2. 绝对禁止 (Forbid Modification):** **严禁**自行创建、合并、删除或修改任何场景的边界。
      * **3. 精确对应 (Ensure 1:1 Correspondence):** 输出的 `scene_level_analysis` 数组，其元素数量必须与输入 `scenes` 数组的数量**完全一致**。
      * **4. 数据映射 (Map & Analyze):** 对于每一个场景，必须将输入的 `scene_name`, `start_time`, `end_time` **原样复制**到输出的对应对象中，并**仅在该时间范围内**进行内容描述和打标。

-----

### **5. 输出格式 (Output Format)**

  * **唯一输出：** 仅返回一个结构规整、不含任何注释或多余文本的单一JSON对象。
  * **结构：** JSON的顶级键应为视频文件名，其值为该视频的完整分析对象。

-----

### **6. 输出格式示例**

**假设 `topic_context` 为：“展现都市脉搏与奋斗精神”**

**假设 `video_scene_data` 输入为：**

```json
{
  "city_pulse_01.mp4": {
    "scenes": [
      {"scene_name": "场景1-序幕", "start_time": 0, "end_time": 8000},
      {"scene_name": "场景2-发展", "start_time": 8000, "end_time": 15000},
      {"scene_name": "场景3-高潮", "start_time": 15000, "end_time": 22000}
    ]
  }
}
```

**你的输出应为：**

```json
{
  "city_pulse_01.mp4": {
    "video_level_analysis": {
      "topic_relevance": {
        "score": 95,
        "reasoning": "视频通过其核心意象(Symbol_Concept: ['都市脉搏']), 叙事弧光(narrative_arc: ['从序幕到高潮'])及音乐氛围(audio_tags: ['BGM:史诗感电子乐'])，与'都市脉搏与奋斗精神'主题在内容、情感和象征层面均高度契合。",
        "matching_keywords": ["都市脉搏", "奋斗精神", "活力", "现代感", "时间流逝", "宏大"]
      },
      "desc": "一段描绘城市从黄昏到深夜的延时摄影航拍，节奏由平缓逐渐加快，配以史诗感的电子乐。整体讲述了'都市脉搏'的故事，从宁静的序幕过渡到充满活力的发展，最终聚焦于奋斗者的象征——摩天大楼。",
      "semantic_tags": {
        "Subject": ["城市", "建筑群", "车流", "天空"],
        "Scene_Action": ["延时摄影", "日夜更替", "航拍"],
        "Emotion_Atmosphere": ["宏大", "壮观", "活力", "现代感", "奋斗"],
        "Symbol_Concept": ["都市脉搏", "时间流逝", "人类文明", "奋斗精神"]
      },
      "narrative_arc": ["从序幕到高潮"],
      "dynamic_tags": ["航拍", "延时摄影", "节奏加快"],
      "audio_tags": ["BGM:史诗感电子乐"]
    },
    "scene_level_analysis": [
      {
        "scene_name": "场景1-序幕",
        "start_time": 0,
        "end_time": 8000,
        "desc": "广角航拍，展示城市在黄昏下的宁静天际线。",
        "semantic_tags": {"Subject": ["城市天际线", "夕阳"], "Emotion_Atmosphere": ["宁静", "壮美"], "Symbol_Concept": ["故事的开始"]},
        "narrative_arc": ["序幕"],
        "dynamic_tags": ["静态广角", "节奏平缓"],
        "audio_tags": ["BGM:舒缓前奏"]
      },
      {
        "scene_name": "场景2-发展",
        "start_time": 8000,
        "end_time": 15000,
        "desc": "延时加速，车流变为光轨，城市灯光逐一点亮。",
        "semantic_tags": {"Subject": ["车流光轨", "城市灯光"], "Emotion_Atmosphere": ["活力", "流动感"], "Symbol_Concept": ["都市活力", "时间加速"]},
        "narrative_arc": ["发展与变化"],
        "dynamic_tags": ["延时摄影", "快节奏"],
        "audio_tags": ["BGM:节奏加强"]
      },
      {
        "scene_name": "场景3-高潮",
        "start_time": 15000,
        "end_time": 22000,
        "desc": "镜头缓慢推向一栋灯火通明的摩天大楼。",
        "semantic_tags": {"Subject": ["摩天大楼", "窗户灯光"], "Emotion_Atmosphere": ["焦点", "坚持", "希望"], "Symbol_Concept": ["奋斗中心", "不眠的追求"]},
        "narrative_arc": ["高潮与点题"],
        "dynamic_tags": ["镜头聚焦", "节奏放缓"],
        "audio_tags": ["BGM:高潮旋律"]
      }
    ]
  }
}
```
"""

BASE_PROMPT = """
你是一位精通多模态分析和视频脚本策划的 AI 专家。你的任务是接收一个包含文本上下文的 JSON 文件和一系列对应的图片文件，为每一张图片生成一段深度结合其视觉内容与文本上下文的、可用于视频制作的**结构化分析对象**。

---

## 核心输入

1. **上下文 JSON**

   * 一个完整的 JSON 对象，精准反映原文的图文排版结构。

2. **图片文件**

   * 多张图片，文件名与 JSON 中的 `image_name` 一一对应。

> 你的分析必须同时基于视觉内容和文本上下文进行。

---

## 任务流程

1. **解析完整数据**

   * 接收并理解我提供的整个上下文 JSON 对象。

2. **自动定位与关联**

   * 根据上传的图片文件名，逐一在 JSON 数据中定位 `type: "image"` 的对象。
   * 对每张图片，识别并理解其紧邻的上下文（前后最近的 `type: "text"` 对象），以及图片所在回答或文章的核心论点。

3. **应用“四步分析框架”批量生成分析对象**

   ### 第一步：视觉与功能定性 (Visual & Functional Analysis)

   * **视觉分析**：图片类型（如网页截图、数据图表、新闻照片、插画等）。
   * **功能分析**：结合上下文，作者使用图片的核心意图（如提供证据、可视化数据、情绪渲染、叙事转折等）。

   ### 第二步：结构化语义标签 (Structured Semantic Tagging)

   基于第一步结果，为图片生成 `semantic_tags` 对象，包含：

   * **Subject**：最关键的名词实体，如 `["医生","孩子","长城"]`。
   * **Scene_Action**：核心行为或环境，如 `["救援","奔跑","家庭聚会"]`。
   * **Emotion_Atmosphere**：传达的情感基调，如 `["温暖","紧张","宏大"]`。
   * **Symbol_Concept**：深层含义或抽象概念，如 `["团结","牺牲","内卷"]`。

   ### 第三步：综合描述 (Synthesized Prose Description)

   * 将前两步分析结果融合，生成可读性强的 `image_desc` 文本，说明画面内容及其作用，用于人工审核和理解。

   ### 第四步：上下文相关性评估 (Contextual Relevance Assessment)

   生成 `relevance` 对象，评估图片与上下文关联紧密度，包含：

   * **level**：关联等级
   * **score**：数值评分
   * **reasoning**：一句话核心理由

   **评分标准**：

   * **核心证据**（score = 5）：图片是论点的关键证明，缺失会严重影响说服力。
   * **强力支撑**（score = 4）：图片提供强有力的具象化例证，极大增强表现力。
   * **辅助说明**（score = 3）：图片与主题相关，补充解释或丰富视觉，但可替代。
   * **氛围渲染**（score = 2）：图片逻辑关联弱，主要为营造情绪或视觉调剂。
   * **弱相关/装饰**（score = 1）：纯装饰，与内容逻辑几乎无关。

4. **整合输出**

   * 仅返回一个纯 JSON 对象，不包含额外说明。
   * JSON 键为 `image_name`，值为完整分析对象，且 `relevance` 字段位于首位。

---

## 最终示例格式

```json
{
  "image_chart.png": {
    "relevance": {
      "level": "核心证据",
      "score": 5,
      "reasoning": "该图表是上文关于增长放缓论点的直接核心数据可视化证据。"
    },
    "image_desc": "这是一张数据图表的截图，用可视化方式证明人口增长放缓论点，核心在于展示曲线平缓趋势。",
    "semantic_tags": {
      "Subject": ["图表","曲线","数据"],
      "Scene_Action": ["展示趋势","可视化"],
      "Emotion_Atmosphere": ["客观","冷静","学术"],
      "Symbol_Concept": ["证据","人口问题","增长放缓","科学论证"]
    }
  },
  "image_加班.jpg": {
    "relevance": {
      "level": "强力支撑",
      "score": 4,
      "reasoning": "图片通过场景渲染，有力例证“职场内卷”主题。"
    },
    "image_desc": "这是一张办公室深夜加班的纪实照片，用于渲染职场压力氛围。",
    "semantic_tags": {
      "Subject": ["办公室","员工","电脑"],
      "Scene_Action": ["加班","深夜工作"],
      "Emotion_Atmosphere": ["疲惫","压力","紧张","内卷"],
      "Symbol_Concept": ["职场压力","奋斗","996"]
    }
  }
}
```
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

def merge_short_scenes(scenes_by_video: dict, min_duration_ms: int) -> dict:
    """
    合并持续时间小于 min_duration_ms 的场景，并为每个场景增加 start_time 和 end_time 键。
    """
    merged = {}
    for video, scenes in scenes_by_video.items():
        # 按时间顺序获取所有场景时间对
        ordered_times = [(time_to_ms(s), time_to_ms(e)) for s, e in scenes.values()]
        # 合并短场景
        merged_times = []
        for start_ms, end_ms in ordered_times:
            if merged_times:
                prev_start, prev_end = merged_times[-1]
                # 如果新段不足阈值，则合并到上一段
                if (end_ms - start_ms) < min_duration_ms:
                    merged_times[-1] = (prev_start, max(prev_end, end_ms))
                    continue
            # 否则新增场景段
            merged_times.append((start_ms, end_ms))
        # 构建输出格式
        scenes_list = []
        for idx, (start_ms, end_ms) in enumerate(merged_times, start=1):
            scenes_list.append({
                'scene_name': f"场景{idx}",
                'start_time': start_ms,
                'end_time': end_ms
            })
        merged[video] = {'scenes': scenes_list}
    return merged


def gen_scene_info(video_abs_path_list):
    """
    生成场景信息
    """
    scene_info_map = {}
    for video_path in video_abs_path_list:
        base_name = os.path.basename(video_path)
        # 确保视频路径是绝对路径
        video_path = pathlib.Path(video_path).resolve()
        if not video_path.is_file():
            print(f"视频文件不存在：{video_path}")
            continue
        scene_info_dict = find_and_split_scenes(str(video_path), max_scenes=10,high_threshold=10)
        if not scene_info_dict:
            start_time = ms_to_time(0)
            print("未能成功获取视频场景信息。", scene_info_dict)
            duration_s = probe_duration(video_path)
            end_time = ms_to_time(duration_s * 1000)
            scene_info_dict = {'场景1':(start_time, end_time)}
        scene_info_map[base_name] = scene_info_dict

    merged_scene_info_map = merge_short_scenes(scene_info_map, min_duration_ms=2000)
    return merged_scene_info_map


def add_video_desc_by_question(real_final_result) -> dict:
    """
    为视频生成相应描述，只提供问题背景
    """
    # 深度拷贝
    updated_result = real_final_result
    # 获取问题描述
    question_title = updated_result.get('question_title', '')
    question_description = updated_result.get('question_description', '')
    full_prompt = f"{VIDEO_BASE_PROMPT}\n 问题标题为：{question_title}\n 问题描述为：{question_description}"
    video_lib = updated_result.get('video_lib', [])
    # video_lib = video_lib[:2]  # 仅保留前2个视频

    video_abs_path_list = [video.get('video_abs_path', '') for video in video_lib if video.get('video_abs_path')]
    scene_info_map = gen_scene_info(video_abs_path_list)
    batch_size = 5
    # 将video_lib分成最多5个视频为一个批次
    if not video_abs_path_list:
        print("--- 未发现任何视频，无需处理 ---")
        return updated_result
    if len(video_abs_path_list) > batch_size:
        print(f"--- 发现 {len(video_abs_path_list)} 个视频，将分批处理 ---")
        video_batches = [video_abs_path_list[i:i + batch_size] for i in range(0, len(video_abs_path_list), batch_size)]
    else:
        print(f"--- 发现 {len(video_abs_path_list)} 个视频，无需分批处理 ---")
        video_batches = [video_abs_path_list]
    video_analysis_result_all = {}
    for video_batch in video_batches:
        base_name_video_paths = [os.path.basename(path) for path in video_batch]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # 生成视频描述
                raw = analyze_videos_gemini(
                    prompt=f'{full_prompt}\n 场景信息为：{scene_info_map}',
                    video_paths=video_batch
                )
                video_analysis_result = string_to_object(raw)
                if validate_image_analysis(base_name_video_paths, video_analysis_result):
                    print(f"视频描述生成成功，处理了 {len(video_batch)} 个视频")
                    video_analysis_result_all.update(video_analysis_result)
                    break
                else:
                    print(f"[警告] 视频描述生成失败，处理了 {len(video_batch)} 个视频，结果不一致：{video_analysis_result}")
                    continue
            except Exception as e:
                print(f"[错误] 视频描述生成失败，处理了 {len(video_batch)} 个视频，异常：{e}")
                continue
    # 更新所有的视频描述
    for video in video_lib:
        video_name = video.get('video_name', '')
        video_desc = video_analysis_result_all.get(video_name, {})
        if video_desc:
            video['video_desc'] = video_desc

    return updated_result

def add_image_desc_by_answer_batching(real_final_result: dict) -> dict:
    """
    为图片生成描述的最终优化版（多进程版）。
    - 筛选出有图片的回答。
    - 将这些回答打包成批次。
    - 使用多进程池并行处理每个批次。
    - 汇总所有结果并更新到最终数据中。
    """
    # 1. 此处逻辑保持不变
    question_images = []
    for item in real_final_result.get('question_description', []):
        if item.get('type') == 'image':
            question_images.append(item)

    # 2. 创建基于回答的批次
    answer_batches = _create_answer_batches(real_final_result)

    if not answer_batches and not question_images:
        print("--- 未发现任何图片，无需处理 ---")
        return real_final_result

    print(f"--- 发现有图片的回答，已创建 {len(answer_batches)} 个处理批次，准备启动多进程处理 ---")

    all_image_descriptions = {}

    tasks = [(batch, real_final_result) for batch in answer_batches]
    with multiprocessing.Pool() as pool:
        list_of_results = pool.starmap(_process_answer_batch, tasks)

    print("\n--- 所有并行批次处理完成，开始汇总结果 ---")

    # 将所有进程返回的结果字典合并到一个大字典中
    for batch_results in list_of_results:
        if batch_results:
            all_image_descriptions.update(batch_results)
    print("--- 开始更新最终结果 ---")
    updated_result = copy.deepcopy(real_final_result)
    # 更新回答中的图片描述
    for answer in updated_result.get('answers', []):
        for item in answer.get('content', []):
            if item.get('type') == 'image':
                image_name = item['image_name']
                desc = all_image_descriptions.get(image_name)
                if desc:
                    item['image_desc'] = desc
    # 更新问题中的图片描述
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
                    'image_info': item.get('image_desc', '')
                })
                content_format.remove(item)  # 删除已处理的item

    # 添加到real_final_result中
    real_final_result['image_lib'] = sort_and_filter_by_score(images)
    return real_final_result

def download_video(real_final_result):
    """
    统一下载相应的视频
    """
    question = real_final_result.get('question', {})
    question_id = question.get('id', 'unknown')
    video_dir = f'{question_id}/videos'
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    video_info_list = []

    play_info_list = real_final_result.get('play_info_list', [])
    for play_info in play_info_list:
        video_id = find_key_values(play_info, "id")[0]
        video_url_list = find_key_values(play_info, "url")
        if video_url_list:
            for video_url in video_url_list[0]:
                base_name = f'video_{video_id}.mp4'
                save_path = os.path.join(video_dir, base_name)
                save_path = pathlib.Path(save_path)
                if not save_path.exists():
                    print(f"开始下载视频: {video_url} 到 {save_path}")
                    download_public_video(video_url, save_path)
                else:
                    print(f"视频已存在，跳过下载: {save_path}")
                if save_path.exists():
                    item = {
                        'video_name': base_name,
                        'video_abs_path': str(save_path.resolve()),
                    }
                    video_info_list.append(item)
                    break
    # 将视频信息添加到 real_final_result 中
    if video_info_list:
        real_final_result['video_lib'] = video_info_list
    return real_final_result



# --- 同步获取问题回答并补充评论 ---
def fetch_question_answers(question_id: str, output_filename: str, desired_answers: int = 5, max_no_increase: int = 3):
    print(f"--- 目标问题ID: {question_id}, 期望获取 {desired_answers} 个回答 ---")
    all_answers_data = []
    play_info_list = []  # 用于保存 play_info 接口返回
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
            url = response.url
            if "/api/v4/questions/" in url and "/feeds" in url:
                try:
                    data = response.json()
                    new_answers = data.get('data', [])
                    print(f"--- 捕获到 {len(new_answers)} 条新回答的API响应 ---")
                    for new_answer in new_answers:
                        all_answers_data.append(new_answer.get('target', {}))
                except Exception as e:
                    print(f"解析API响应失败: {e}, URL: {url}")
            # 监控 play_info 请求
            if "/api/v4/video/play_info" in url:
                try:
                    info = response.json()
                    play_info_list.append(info)
                except Exception as e:
                    print(f"解析 play_info 响应失败: {e}, URL: {url}")

        page.on("response", handle_response)

        try:
            question_url = f"https://www.zhihu.com/question/{question_id}"
            print(f">>> 正在访问问题页面: {question_url}")
            page.goto(question_url, wait_until="networkidle", timeout=60000)
            print(">>> 等待问题标题加载...")
            page.wait_for_selector('h1.QuestionHeader-title', timeout=15000)
            print(">>> 页面加载成功，问题标题已出现。")

            # 初始数据提取
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

            # 滚动加载更多回答
            while len(all_answers_data) < desired_answers:
                prev_count = len(all_answers_data)
                page.evaluate("window.scrollTo({ top: document.body.scrollHeight * 0.9, behavior: 'smooth' })")
                time.sleep(random.uniform(1.5, 2.0))

                page.evaluate("window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })")
                time.sleep(random.uniform(1.5, 3.0))

                page.keyboard.press("ArrowUp")
                time.sleep(random.uniform(1.5, 2.5))

                page.keyboard.press("ArrowDown")
                time.sleep(random.uniform(1.5, 2.5))

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

    # 后处理回答数据
    for answer in all_answers_data:
        if 'voteupCount' in answer:
            answer['voteup_count'] = answer.pop('voteupCount')
        else:
            answer['voteup_count'] = 0

        if answer['voteup_count'] == 0 and 'matrix_tips' in answer:
            tips = answer['matrix_tips']
            match = re.search(r'(\d+)\s*赞同', tips)
            if match:
                answer['voteup_count'] = int(match.group(1))

    all_answers_data.sort(key=lambda x: x.get('voteup_count', 0), reverse=True)
    final_data = all_answers_data[:desired_answers]

    # 初次保存结果并加入 play_info_list
    real_final_result = {
        'question': questions,
        'answers': final_data,
        'play_info_list': play_info_list
    }
    save_json(output_filename, real_final_result)
    print(f"原始API响应数据及 play_info_list 已保存至文件: {output_filename}")

    # 补充评论及后续处理
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
    real_final_result = {'question': questions, 'answers': final_data, 'play_info_list': play_info_list}
    save_json(output_filename, real_final_result)
    print(f"回答评论数据已保存至文件: {output_filename}")

    real_final_result = download_image(real_final_result)
    save_json(output_filename, real_final_result)
    print(f"图片已下载并更新至文件: {output_filename}")

    real_final_result = download_video(real_final_result)
    save_json(output_filename, real_final_result)
    print(f"视频已下载并更新至文件: {output_filename}")

    video_script_data = transform_zhihu_to_video_script(
        output_filename,
        comment_upvote_threshold=10
    )
    real_final_result = add_image_desc_by_answer_batching(video_script_data)
    save_json(output_filename, real_final_result)
    real_final_result = extract_image(real_final_result)
    save_json(output_filename, real_final_result)

    print(f"带图片描述结果保存至文件: {output_filename}")

    add_video_desc_by_question(real_final_result)
    save_json(output_filename, real_final_result)
    print(f"视频描述已添加并保存至文件: {output_filename}")



def gen_video(question_id):
    fetch_question_answers(question_id, f"{question_id}/zhihu_answers_{question_id}.json", desired_answers=100)
    final_video_info = gen_video_final_info(question_id)
    final_video_path = gen_video_by_video_info( f"{question_id}/zhihu_answers_{question_id}_video_info_op.json")
    return final_video_path, final_video_info

if __name__ == "__main__":
    question_id = "1933988735948678767"
    # fetch_question_answers(question_id, f"{question_id}/zhihu_answers_{question_id}.json", desired_answers=100)
    # final_video_info = gen_video_final_info(question_id)
    # gen_video_by_video_info( f"{question_id}/zhihu_answers_{question_id}_video_info_op.json")

    # hot_list_data = fetch_zhihu_hot(ZHIHU_COOKIE_STRING)
    # with open('zhihu_hot_list.json', 'w', encoding='utf-8') as f:
    #     json.dump(hot_list_data, f, ensure_ascii=False, indent=4)
    # print("结果已保存到 zhihu_hot_list.json 文件中。")

    output_filename = f"{question_id}/zhihu_answers_{question_id}.json"
    real_final_result = read_json(output_filename)
    # add_image_desc_by_answer_batching(real_final_result)
    # real_final_result = extract_image(real_final_result)
    # save_json(output_filename, real_final_result)

    add_video_desc_by_question(real_final_result)
    save_json(output_filename, real_final_result)


    # add_video_desc_by_question(real_final_result)
    # save_json(output_filename, real_final_result)

    # real_final_result = download_video(real_final_result)
    # save_json(output_filename, real_final_result)
    # print(f"视频已下载并更新至文件: {output_filename}")
    #
    # video_script_data = transform_zhihu_to_video_script(
    #     output_filename,
    #     comment_upvote_threshold=10
    # )
    # real_final_result = add_image_desc_by_answer_batching(video_script_data)
    # save_json(output_filename, real_final_result)
    # real_final_result = extract_image(real_final_result)
    # save_json(output_filename, real_final_result)
    #
    # print(f"带图片描述结果保存至文件: {output_filename}")