import requests
import xml.etree.ElementTree as ET
import re
import ast

from collections import Counter
import math
from datetime import datetime
import json

from LLM.gemini import get_llm_content
from content_community.bilibili.get_comment import get_bilibili_comments

# --- 配置项 ---
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
HEADERS = {'User-Agent': USER_AGENT}

# Bilibili API 地址
BVID_TO_CID_API = "https://api.bilibili.com/x/web-interface/view"
DANMAKU_API = "https://api.bilibili.com/x/v1/dm/list.so"

# 弹幕类型映射 (方便理解)
DANMAKU_TYPE_MAP = {
    1: '滚动弹幕', 4: '底部弹幕', 5: '顶部弹幕',
    6: '逆向弹幕', 7: '特殊弹幕', 8: '高级弹幕', 9: '脚本弹幕',
}

TRANSLATION_MAP = {
    # --- 顶层核心信息 ---
    "bvid": "BVID",
    "aid": "稿件ID (aid)",
    "videos": "视频分P总数",
    "tid": "分区ID",
    "tname": "分区名称",
    "copyright": {
        "key": "版权类型",
        "handler": lambda x: "原创" if x == 1 else "转载"
    },
    "pic": "封面图片URL",
    "title": "标题",
    "pubdate": {
        "key": "发布时间",
        "handler": lambda ts: datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    },
    "ctime": {
        "key": "创建时间",
        "handler": lambda ts: datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    },
    "desc": "视频简介",
    "state": "视频状态",
    "duration": {
        "key": "总时长",
        "handler": lambda
            s: f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"
    },
    "dynamic": "视频动态文字",
    "cid": "弹幕ID (cid)",

    # --- UP主信息 (owner) ---
    "owner": {
        "key": "UP主信息",
        "handler": {
            "mid": "UP主MID",
            "name": "UP主昵称",
            "face": "UP主头像URL"
        }
    },

    # --- 数据统计 (stat) ---
    "stat": {
        "key": "数据统计",
        "handler": {
            "aid": "稿件ID (aid)",
            "view": "播放数",
            "danmaku": "弹幕数",
            "reply": "评论数",
            "favorite": "收藏数",
            "coin": "投币数",
            "share": "分享数",
            "now_rank": "当前排名",
            "his_rank": "历史最高排名",
            "like": "点赞数",
            "dislike": "点踩数"
        }
    },

    # --- 视频分P信息 (pages) ---
    "pages": {
        "key": "分P信息列表",
        "handler": lambda pages: [
            {
                "分P序号": p['page'],
                "分P标题": p['part'],
                "弹幕ID (cid)": p['cid'],
                "时长": f"{p['duration'] // 60:02d}:{p['duration'] % 60:02d}"
            } for p in pages
        ]
    },

    # --- 合作成员信息 (staff) ---
    "staff": {
        "key": "合作成员",
        "handler": lambda staff_list: [
            {
                "成员MID": s['mid'],
                "成员昵称": s['title'],
                "职位": s['name'],
                "头像URL": s['face'],
            } for s in staff_list
        ] if staff_list else "无"
    },

    # --- 视频标签 (tags) ---
    "tags": {
        "key": "视频标签",
        "handler": lambda tags_list: [tag['tag_name'] for tag in tags_list] if tags_list else []
    }
}


def translate_info(data: dict, translation_map: dict) -> dict:
    """
    根据翻译映射表，递归地翻译和处理API返回的数据。
    """
    translated_dict = {}
    for key, value in data.items():
        if key in translation_map:
            rule = translation_map[key]
            # 如果规则是字典，表示需要进一步处理
            if isinstance(rule, dict):
                new_key = rule['key']
                handler = rule['handler']
                # 如果处理器是函数，直接调用
                if callable(handler):
                    translated_dict[new_key] = handler(value)
                # 如果处理器是字典，表示是嵌套对象，递归处理
                elif isinstance(handler, dict):
                    translated_dict[new_key] = translate_info(value, handler)
            # 如果规则是简单的字符串，直接用作新的键
            else:
                translated_dict[rule] = value

    return translated_dict

def get_bilibili_video_info_full(bvid: str):
    """
    通过Bilibili API获取视频的完整详细信息，并进行中文化和格式化。

    :param bvid: 视频的BV号 (例如: "BV1hK4y19799")
    :return: 包含视频完整信息的字典，如果失败则返回None
    """
    url = "https://api.bilibili.com/x/web-interface/view"
    params = {"bvid": bvid}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': f'https://www.bilibili.com/video/{bvid}'
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        if data['code'] == 0:
            video_data = data['data']
            # 使用翻译函数处理整个数据对象
            return translate_info(video_data, TRANSLATION_MAP)
        else:
            print(f"API请求失败: {data['message']} (code: {data['code']})")
            # 特殊处理视频不存在的情况
            if data['code'] == -404 or data['code'] == 62002:
                print(f"错误原因：视频 (BVID: {bvid}) 不存在或已被删除。")
            return None

    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"解析返回数据时出错: {e}，可能是API结构已更改或返回内容非标准JSON。")
        return None

# --- 辅助函数：获取CID 和 弹幕 ---
def get_cid_from_bvid(bvid: str, p_index: int = 0) -> int | None:
    """
    通过视频的 BVID 获取指定分P的 CID。
    :param bvid: 视频的 BVID (例如: 'BV1vx411c7Xh')
    :param p_index: 分P的索引，0表示第一个P，1表示第二个P，以此类推。
    :return: 对应的 CID 或 None (如果获取失败)
    """
    params = {'bvid': bvid}
    print(f"正在获取 BVID '{bvid}' 的视频信息以获取 CID...")
    try:
        response = requests.get(BVID_TO_CID_API, params=params, headers=HEADERS)
        response.raise_for_status()
        data = response.json()

        if data['code'] == 0:
            pages = data['data']['pages']
            if p_index < len(pages):
                cid = pages[p_index]['cid']
                print(f"成功获取到 BVID '{bvid}' 第 {p_index + 1} 个分P的 CID: {cid}")
                return cid
            else:
                print(f"错误: BVID '{bvid}' 没有第 {p_index + 1} 个分P。该视频共有 {len(pages)} 个分P。")
                return None
        else:
            print(f"获取视频信息失败，错误码: {data['code']}, 消息: {data['message']}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
        return None
    except KeyError as e:
        print(f"解析视频信息JSON失败，缺少键: {e}. 响应内容: {response.text}")
        return None

def get_video_info_from_bvid(bvid: str, p_index: int = 0) -> int | None:
    """
    通过视频的 BVID 获取指定分P的 CID。
    :param bvid: 视频的 BVID (例如: 'BV1vx411c7Xh')
    :param p_index: 分P的索引，0表示第一个P，1表示第二个P，以此类推。
    :return: 对应的 CID 或 None (如果获取失败)
    """
    params = {'bvid': bvid}
    print(f"正在获取 BVID '{bvid}' 的视频信息以获取 CID...")
    try:
        response = requests.get(BVID_TO_CID_API, params=params, headers=HEADERS)
        response.raise_for_status()
        data = response.json()

        if data['code'] == 0:
            pages = data['data']['pages']
            if p_index < len(pages):
                cid = pages[p_index]['cid']
                print(f"成功获取到 BVID '{bvid}' 第 {p_index + 1} 个分P的 CID: {cid}")
                return cid
            else:
                print(f"错误: BVID '{bvid}' 没有第 {p_index + 1} 个分P。该视频共有 {len(pages)} 个分P。")
                return None
        else:
            print(f"获取视频信息失败，错误码: {data['code']}, 消息: {data['message']}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
        return None
    except KeyError as e:
        print(f"解析视频信息JSON失败，缺少键: {e}. 响应内容: {response.text}")
        return None


def int_to_hex_color(decimal_color: int) -> str:
    """将十进制颜色值转换为十六进制颜色字符串。"""
    return f"#{decimal_color:06X}"


def get_danmaku(cid: int) -> list[dict]:
    """
    通过视频的 CID 获取弹幕列表。
    :param cid: 视频的 CID
    :return: 包含弹幕信息的字典列表
    """
    params = {'oid': cid}
    print(f"正在获取 CID '{cid}' 的弹幕...")
    danmaku_list = []
    try:
        response = requests.get(DANMAKU_API, params=params, headers=HEADERS)
        response.raise_for_status()

        root = ET.fromstring(response.content)

        for d_element in root.findall('d'):
            p_attr = d_element.get('p')
            if p_attr:
                parts = p_attr.split(',')
                try:
                    danmaku_info = {
                        'text': d_element.text,
                        'video_time': float(parts[0]),
                        'type': DANMAKU_TYPE_MAP.get(int(parts[1]), '未知类型'),
                        'font_size': int(parts[2]),
                        'color': int_to_hex_color(int(parts[3])),
                        'send_timestamp': int(parts[4]),
                        'pool': int(parts[5]),
                        'sender_id_hash': parts[6],
                        'row_id': int(parts[7]),
                    }
                    if danmaku_info['text'] is not None:  # 确保文本不为空
                        danmaku_list.append(danmaku_info)
                except (ValueError, IndexError) as ve:
                    # print(f"解析弹幕属性失败: {p_attr}, 错误: {ve}")
                    continue
        print(f"成功获取到 {len(danmaku_list)} 条弹幕。")
        return danmaku_list

    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
        return []
    except ET.ParseError as e:
        print(f"解析弹幕XML失败: {e}. 响应内容: {response.content[:500]}...")
        return []
    except Exception as e:
        print(f"发生未知错误: {e}")
        return []


# --- 弹幕相似度分析相关函数 ---

def normalize_danmaku_text(text: str) -> str:
    """
    标准化弹幕文本：
    - 转换为小写（对中文影响小）
    - 移除常用标点符号和特殊字符
    - 规范化连续的相同字符（如 "哈哈哈" -> "哈"）
    - 规范化常见的笑声表达（如 "2333" -> "笑", "hhh" -> "笑"）
    - 移除空格
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()

    # 移除标点符号 (保留中文汉字、数字、字母)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)

    # 规范化连续的相同字符 (例如 "哈哈哈" -> "哈")
    text = re.sub(r'(.)\1+', r'\1', text)

    # 规范化常见笑声表达 (可根据需要扩展)
    text = re.sub(r'(哈|h|呵|233)+', '笑', text)
    text = text.replace('草', '笑')  # 弹幕中的"草"很多表示笑

    return text


def get_char_ngrams(text: str, n: int = 2) -> set[str]:
    """
    获取文本的字符 n-gram 集合。
    例如: "你好世界", n=2 -> {"你好", "好世", "世界"}
    """
    if len(text) < n:
        return {text}  # 对于短于n的文本，整个文本作为唯一的n-gram
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(text1: str, text2: str, n_gram_size: int = 2) -> float:
    """
    计算两个文本的Jaccard相似度 (基于字符n-gram)。
    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    if not text1 or not text2:
        return 0.0

    set1 = get_char_ngrams(text1, n_gram_size)
    set2 = get_char_ngrams(text2, n_gram_size)

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union == 0:
        return 0.0
    return intersection / union


def analyze_similar_danmaku_frequency(
        danmaku_data: list[dict],
        similarity_threshold: float = 0.7,
        ngram_size: int = 2,
        min_danmaku_length: int = 1  # 过滤掉标准化后过短的弹幕
) -> list[tuple[str, int]]:
    """
    分析弹幕频率，并将相似意义的弹幕进行分组。
    :param danmaku_data: 原始弹幕数据列表。
    :param similarity_threshold: 判断弹幕是否相似的Jaccard相似度阈值。
    :param ngram_size: 用于Jaccard相似度计算的字符n-gram大小。
    :param min_danmaku_length: 标准化后弹幕的最小长度，低于此长度的将被忽略。
    :return: 包含 (代表弹幕, 频率) 的元组列表，按频率降序排列。
    """
    print("\n--- 开始分析相似弹幕频率 ---")

    # 存储分组后的弹幕信息
    # 每个元素是一个字典: {'representative': '代表弹幕文本', 'count': 数量, 'members': Counter (存储组成员的原始文本计数)}
    grouped_clusters = []

    for dm in danmaku_data:
        original_text = dm['text']
        normalized_text = normalize_danmaku_text(original_text)

        if not normalized_text or len(normalized_text) < min_danmaku_length:
            continue  # 忽略空或过短的弹幕

        found_group = False
        for cluster in grouped_clusters:
            # 计算当前标准化弹幕与组代表弹幕的相似度
            sim = jaccard_similarity(normalized_text, cluster['representative'], ngram_size)

            # 如果相似度达到阈值，则加入该组
            if sim >= similarity_threshold:
                cluster['count'] += 1
                cluster['members'][original_text] += 1
                # 动态更新组的代表弹幕为该组内出现频率最高的原始弹幕
                cluster['representative'] = cluster['members'].most_common(1)[0][0]
                found_group = True
                break

        if not found_group:
            # 如果没有找到匹配的组，则创建一个新组
            new_cluster = {
                'representative': original_text,  # 初始代表为第一个原始弹幕
                'count': 1,
                'members': Counter({original_text: 1})  # 记录原始文本计数
            }
            grouped_clusters.append(new_cluster)

    # 按数量降序排序
    sorted_clusters = sorted(grouped_clusters, key=lambda x: x['count'], reverse=True)

    result = []
    for cluster in sorted_clusters:
        result.append((cluster['representative'], cluster['count']))

    print(f"分析完成，共识别出 {len(result)} 组相似弹幕。")
    return result

def get_sorted_danmu(cid):
    top_similar_danmakus = []
    all_danmakus = get_danmaku(cid)
    if all_danmakus:
        # 分析并获取频率最高的相似弹幕
        top_similar_danmakus = analyze_similar_danmaku_frequency(
            all_danmakus,
            similarity_threshold=0.5,  # 调整这个阈值可以控制相似度宽松程度
            ngram_size=2,  # 调整n-gram大小，2-3对中文短语较好
            min_danmaku_length=1  # 过滤掉标准化后长度小于1的弹幕
        )
    # 最多只获取前100个top_similar_danmakus
    top_similar_danmakus = top_similar_danmakus[:100] if top_similar_danmakus else []
    return top_similar_danmakus


def string_to_list(input_str: str) -> list:
    """
    将带有 Markdown 标记的字符串转换为 Python 列表对象。

    该函数支持：
      - 去除包含```json或```等 Markdown 代码块标记的部分；
      - 从包含其它文本的字符串中提取出列表内容（通过查找类似 "[...]" 的子串）。

    如果解析失败，则抛出 ValueError 异常。
    """
    # 使用正则表达式匹配 Markdown 代码块中被 ``` 或 ```json 包裹的列表部分
    pattern = re.compile(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```")
    match = pattern.search(input_str)
    if match:
        list_str = match.group(1)
    else:
        # 如果找不到 Markdown 代码块，则尝试从整个字符串中提取第一个以 [ 开头、以 ] 结尾的子串
        start = input_str.find('[')
        end = input_str.rfind(']')
        if start != -1 and end != -1 and start < end:
            list_str = input_str[start:end + 1]
        else:
            raise ValueError("未找到可以解析的列表数据")

    # 首先尝试使用 ast.literal_eval 解析（优点是可以解析 Python 的字面量）
    try:
        result = ast.literal_eval(list_str)
        if isinstance(result, list):
            return result
        else:
            raise ValueError("解析结果不是列表类型")
    except Exception as e:
        # 如果 ast.literal_eval 失败，则回退到使用 json.loads 解析 (适用于标准 JSON 格式)
        try:
            result = json.loads(list_str)
            if isinstance(result, list):
                return result
            else:
                raise ValueError("解析结果不是列表类型")
        except Exception as e2:
            raise ValueError(f"字符串无法被解析为列表: {e2}")

def gen_proper_comment(bvid):
    """
    生成适合的评论内容。
    这里可以根据需要实现更复杂的逻辑。
    """
    video_info = get_bilibili_video_info_full(bvid)

    if video_info:
        print("\n--- 视频信息获取成功 ---")
        title = video_info.get('标题', '未知标题').replace('\n', '\\n').replace('\r', '\\r')
        desc = video_info.get('视频简介', '无简介').replace('\n', '\\n').replace('\r', '\\r')
        tag = video_info.get('视频标签', [])
        tname = video_info.get('分区名称', '未知分区').replace('\n', '\\n').replace('\r', '\\r')
        comment_count = video_info.get('数据统计', {}).get('评论数', 0)
        comments = []
        top_similar_danmakus = []
        if comment_count > 2:
            comments = get_bilibili_comments(bvid)
            comments = [(comment['content']['message'], comment['like']) for comment in comments]
        video_info['已有评论'] = comments

        danmu_count = video_info.get('数据统计', {}).get('弹幕数', 0)
        if danmu_count > 2:
            cid = video_info.get('弹幕ID (cid)', None)
            top_similar_danmakus = get_sorted_danmu(cid)
        video_info['已有弹幕'] = top_similar_danmakus
    else:
        print("\n--- 视频信息获取失败 ---")

    prompt_template = """下面是一个视频的相关信息，请你根据这些信息，给我生成几个有趣能够吸引眼球的评论(不在乎是引起共鸣还是引起反感，目标就是有流量)，包括但不限于：制造对立、身份认同、玩梗、抖机灵、挑衅、发布暴论等。你的唯一目标是为我给出的视频信息生成能风格各异、能引爆流量的神评，返回结果示例如下，严格按照下面的格式,最有趣的放在最前面:
    [
        "评论1",
        "评论2",
        ...
    ]
    视频信息如下:
    标题: "{video_title}"
    描述: "{video_description}"
    分区: "{video_partition}"
    标签: "{video_tags}"
    已有评论(数字代表点赞数): {existing_comments}
    已有弹幕(数字代表出现的次数): {top_danmakus}
    """

    # 填充数据
    filled_prompt = prompt_template.format(
        video_title=title,
        video_description=desc,
        video_partition=tname,
        video_tags=tag,  # 列表会直接显示为 [item1, item2, ...]
        existing_comments=comments,  # 列表会直接显示为 [item1, item2, ...]
        top_danmakus=top_similar_danmakus  # 列表会直接显示为 [(item1, count1), ...]
    )

    start_time = datetime.now()
    result = get_llm_content(prompt=filled_prompt)
    print(f"\n--- LLM 生成评论内容耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒 ---")
    if result:
        result = string_to_list(result)
    video_info['gen_comment'] = result
    return video_info

# --- 主执行部分 ---
if __name__ == "__main__":
    # 替换为你想要获取弹幕的视频 BVID
    video_bvid = 'BV1MrNtzoECN'  # 一个有较多弹幕的鬼畜视频

    result = gen_proper_comment(video_bvid)
    # video_bvid = 'BV1Vf4y1P7oD' # 另一个例子
    part_index = 0  # 视频分P，0表示第一个P

    cid_result = get_cid_from_bvid(video_bvid, part_index)

    if cid_result:
        all_danmakus = get_danmaku(cid_result)

        if all_danmakus:
            # 分析并获取频率最高的相似弹幕
            top_similar_danmakus = analyze_similar_danmaku_frequency(
                all_danmakus,
                similarity_threshold=0.5,  # 调整这个阈值可以控制相似度宽松程度
                ngram_size=2,  # 调整n-gram大小，2-3对中文短语较好
                min_danmaku_length=1  # 过滤掉标准化后长度小于1的弹幕
            )

            print("\n--- 频率最高的10组相似弹幕 ---")
            if top_similar_danmakus:
                for i, (text, count) in enumerate(top_similar_danmakus[:10]):
                    print(f"{i + 1}. '{text}' (出现次数: {count})")
            else:
                print("没有找到足够的数据进行相似弹幕分析。")
        else:
            print("未能获取到任何弹幕。")
    else:
        print(f"无法获取 BVID '{video_bvid}' 第 {part_index + 1} 个分P的 CID，无法继续获取弹幕。")