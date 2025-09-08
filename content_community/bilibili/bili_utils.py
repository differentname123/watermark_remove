# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/8/7 2:52
:last_date:
    2025/8/7 2:52
:description:
    
"""
import json
import re
from collections import defaultdict

import requests
import time
import random

from LLM.gemini import get_llm_content
from common_utils.common_utils import get_config, read_json, time_to_ms, string_to_object, save_json, init_config

URL_MODIFY_RELATION = "https://api.bilibili.com/x/relation/modify"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


base_prompt = """
            # 角色
            你是一个智能的视频内容查重助手。
            # 任务
            你的任务是根据我提供的【原始视频元数据】，在【目标视频列表】中，判断是否存在实际上是同一个视频的条目。目标是帮助我确认这个视频是否已经在新平台上发布过。
            # 匹配逻辑
            判断标准不是要求所有字段都100%完全相同，而是基于以下逻辑进行综合判断：
            1.  **标题 (title)**: 内容高度一致，大部分内容都是重复的。
            2.  **作者 (author)**: 应基本一致。这是个非常强的匹配信号。
            3.  **时长 (duration)**: 应该非常接近，允许有几秒钟的误差（例如在 ±1 秒内），因为视频转码或添加片头片尾可能导致微小变化。
            4.  **判定规则**: 一个可靠的匹配，通常需要 **至少满足上述两条标准**。例如，“标题和时长都高度吻合” 或 “标题和作者都高度吻合”。如果只满足一项，则不算匹配。请在找到第一个最可靠的匹配项后立即停止搜索并返回结果。
            
            # 输出要求
            1.  返回结果**必须是**一个纯净的、不包含任何前后说明文字的JSON对象。
            2.  **如果找到匹配项**:
                - `bvid` 字段为匹配视频的BVID字符串。
                - `reason` 字段应简要说明匹配的依据，例如："标题核心内容一致，且时长在误差范围内"。
            3.  **如果未找到匹配项**:
                - `bvid` 字段为 `null`。
                - `reason` 字段应说明未找到的原因，例如："没有找到标题和时长足够相似的视频"。
            4.  返回的JSON格式示例：
                - 找到时: `{ "bvid": "BV177tnzzEHR", "reason": "标题和作者名称匹配." }`
                - 未找到时: `{ "bvid": null, "reason": "没有找到满足匹配逻辑的视频." }`
                
            # 输入数据
    """

def add_goods_to_selection(cookie: str, goods: list, operate_source: int = 4, from_type: int = 18):
    """
    将抓取到的商品批量加入选品车

    :param cookie: 登录后的 Cookie 字符串
    :param goods: 要加入选品车的商品列表，每项结构需与接口 body 中 goods 字段保持一致
    :param operate_source: 操作来源标识，默认为4（可根据实际场景调整）
    :param from_type: 来源类型，默认为18（可根据实际场景调整）
    :return: 接口返回的 JSON 响应
    """
    url = "https://cm.bilibili.com/dwp/api/web_api/v1/selection/car/item/add"
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "content-type": "application/json;charset=UTF-8",
        "priority": "u=1, i",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\", \"Microsoft Edge\";v=\"138\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "referer": "https://cm.bilibili.com/quests/",
        "cookie": cookie,
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "\
                      "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
    }

    payload = {
        "goods": goods,
        "operateSource": operate_source,
        "bizExtraInfo": "",
        "fromType": from_type
    }
    goodsName = ', '.join([item.get('goodsName', '未知商品') for item in goods])
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        if result.get("code") == 0:
            print(f"[Success] 已成功加入 {len(goods)} 个商品到选品车 goodsName: {goodsName}")
        else:
            print(f"[Warning] 接口返回非 0 状态: {result.get('code')} - {result.get('message')} goodsName: {goodsName}")
        return result
    except Exception as e:
        print(f"[Error] 添加商品到选品车失败: {e}")
        return None


def fetch_goods(cookie: str, max_count: int, goodsName: str = '', sourceTypes: int = 1):
    """
    抓取商品信息，直到获取到 max_count 条或者没有更多数据为止。

    :param cookie: 登录后的 Cookie 字符串
    :param max_count: 想要获取的商品总数
    :param goodsName: 商品关键词（默认空）
    :param sourceTypes: 来源类型（默认为1）
    :return: 所有抓取到的商品数据列表
    """
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "priority": "u=1, i",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\", \"Microsoft Edge\";v=\"138\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "referer": "https://cm.bilibili.com/quests/",
        "cookie": cookie,
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
    }

    goods = []
    page = 1
    page_size = 20

    while len(goods) < max_count:
        params = {
            "cmcFirstCatNames": "",
            "goodsName": goodsName,
            "page": page,
            "size": page_size,
            "sourceTypes": sourceTypes,
            "sortType": 0
        }

        url = "https://cm.bilibili.com/dwp/api/web_api/v1/item/list"
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            items = data.get("data", {}).get("data", {})

            if not items:
                # print(f"[Info] 第 {page} 页无更多数据，提前结束")
                break

            goods.extend(items)
            # 每个商品添加额外信息sourceTypes
            for item in goods:
                item['sourceTypes'] = sourceTypes

            if len(goods) >= max_count:
                break

            page += 1
            time.sleep(random.uniform(0.8, 1.5))  # 随机等待，模拟真实请求
        except Exception as e:
            print(f"[Error] 第 {page} 页请求失败: {e}")
            break

    return goods

import requests
import time
import random

def list_selection_car_items(cookie: str, target_count: int = 10):
    """
    获取选品车中商品列表，直到达到目标数量或无更多数据

    :param cookie: 登录后的 Cookie 字符串
    :param target_count: 希望拉取的商品总数，默认为10
    :return: 包含商品的列表，最多 target_count 条
    """
    items = []
    page = 1
    size = min(10, target_count)

    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "priority": "u=1, i",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\", \"Microsoft Edge\";v=\"138\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "referer": "https://cm.bilibili.com/quests/",
        "cookie": cookie,
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
        )
    }

    while len(items) < target_count:
        # 构建请求 URL
        url = (
            f"https://cm.bilibili.com/dwp/api/web_api/v1/selection/car/item/list"
            f"?page={page}&size={size}&sourceType=-1&promotionCampaigns=&"
            f"selectionCarItemType=1&windowShelveStatus=-1&goodsName=&requestFrom=-1"
        )

        # 模拟随机延迟
        time.sleep(random.uniform(0.5, 1.2))

        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            if result.get("code") == 0:
                data = result.get("data", {}).get("data", [])
                if not data:
                    print("[Info] 无更多数据，提前结束")
                    break

                items.extend(data)
                print(f"[Success] 分页 {page} 获取 {len(data)} 条，共累计 {len(items)} 条")

                # 如果本次返回少于请求数量，说明已无更多数据
                if len(data) < size:
                    break

                # 准备下一页
                page += 1
                remaining = target_count - len(items)
                size = min(10, remaining)
            else:
                print(f"[Warning] 接口返回非 0 状态: {result.get('code')} - {result.get('message')}")
                break
        except Exception as e:
            print(f"[Error] 获取选品车列表失败: {e}")
            break

    # 最终返回不超过 target_count 条数据
    return items


import time

def update_short_url(cookie, goods, max_retries=5):
    """
    生成商品的短链接，每个 good 必须匹配到短链才算成功，否则最多重试 max_retries 次。

    Args:
        cookie (str): 认证所需的 cookie。
        goods (list of dict): 每个 dict 至少包含 'outerId' 和 'goodsName' 键。
        max_retries (int): 最大重试次数，默认 3 次。

    Returns:
        list of dict: 原 goods 列表，每个 dict 新增 'shortUrl' 字段（若匹配到则为链接，否则为空字符串）。
    """
    for attempt in range(1, max_retries + 1):
        print(f"[尝试 {attempt}/{max_retries}] 开始生成短链接...{goods[0]['goodsName']} 等 {len(goods)} 个商品")

        # 每次重置 shortUrl 字段
        for good in goods:
            good['shortUrl'] = ''

        # 加入选品车
        add_goods_to_selection(cookie=cookie, goods=goods)
        time.sleep(2)  # 等待选品车更新

        # 获取选品车当前列表
        car_items = list_selection_car_items(cookie)

        # 匹配 shortUrl
        for good in goods:
            for item in car_items:
                if good['outerId'] in item.get('outerId', '') or item.get('outerId', '') in good['outerId']:
                    good['shortUrl'] = item.get('shortUrl', '')
                    break  # 找到就跳出内层循环

        # 检查是否全部匹配成功
        unmatched = [g for g in goods if not g['shortUrl']]
        if not unmatched:
            print("所有商品均已成功生成短链接。")
            return goods

        print(f"未匹配到短链的商品还有 {len(unmatched)} 个，将进行下一次重试。")

    print("达到最大重试次数，以下商品未获取到短链：")
    for g in [g for g in goods if not g['shortUrl']]:
        print(f" - {g['goodsName']} (outerId={g['outerId']})")
    return goods

import time
import random
import requests
from urllib.parse import quote_plus, urlencode


def fetch_from_search(key_word, target_count=20, timeout=10, page_size=20):
    """
    从 B 站关键词搜索接口获取视频数据（不带 Cookie 版本）

    Args:
        key_word (str): 搜索关键词
        target_count (int): 目标数量
        timeout (int): 请求超时（秒）
        page_size (int): 每页数量（B 站默认 20）

    Returns:
        list: 视频信息列表（可能少于 target_count）
    """
    if not key_word:
        print("[提示] 未提供关键词。")
        return []

    url = "https://api.bilibili.com/x/web-interface/search/type"
    referer = f"https://search.bilibili.com/all?keyword={quote_plus(key_word)}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Origin": "https://www.bilibili.com",
        "Referer": referer,
        "Connection": "keep-alive",
        "Cookie": get_config("jie_bilibili_total_cookie"),  # 不带 Cookie
    }

    video_list, current_page, fetched = [], 1, 0

    while fetched < target_count:
        time.sleep(random.uniform(1.2, 2.2))  # 延时，防止触发风控

        params = {
            "search_type": "video",
            "keyword": key_word,
            "order": "pubdate",
            "page": current_page,
            "ps": page_size
        }

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 412:
                print("[错误] 触发 412 Precondition Failed（无 Cookie 可能被拦截）。")
                break

            resp.raise_for_status()
            data = resp.json()

        except requests.exceptions.RequestException as e:
            print(f"[错误] 网络请求失败: {e}")
            break
        except ValueError:
            print(f"[错误] JSON 解析失败，响应片段: {resp.text[:200]!r}")
            break

        if data.get("code", 0) != 0:
            print(f"[警告] API 返回错误: code={data.get('code')}, message={data.get('message')}")
            break

        payload = data.get("data", {})
        search_results = payload.get("result", [])
        if not isinstance(search_results, list):
            search_results = payload.get("result", {}).get("video", [])

        if not search_results:
            break

        page_added = 0
        for item in search_results:
            if item.get("type") == "video" and "bvid" in item:
                if "title" in item:
                    item["title"] = item["title"].replace('<em class="keyword">', '').replace('</em>', '')
                item["_source_strategy"] = "search"
                video_list.append(item)
                fetched += 1
                page_added += 1
                if fetched >= target_count:
                    break

        if page_added < min(page_size, target_count - (fetched - page_added)):
            break

        current_page += 1

    return video_list

def check_duplicate_video(meta_data):
    """
    通过元数据信息检查该视频是否在b站上已经存在
    """
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            douyin_username = meta_data.get("nickname", "")
            douyin_full_title = meta_data.get("full_title", "")
            douyin_duration = meta_data.get("duration", '0:0')
            douyin_duration = time_to_ms(douyin_duration) / 1000
            if douyin_duration < 60:
                return False
            user_map_file = "douyin_bilibili_user_map.json"
            user_map_info = read_json(user_map_file)

            if douyin_username in user_map_info:
                bilibili_username = user_map_info[douyin_username].get("bilibili_username", "")
                if bilibili_username:
                    print(f"[提示] 用户 {douyin_username} 在 B 站已有视频{douyin_full_title}，b站昵称: {bilibili_username}")
                    return True

            douyin_info = {
                'title': douyin_full_title,
                'author': douyin_username,
                'duration': douyin_duration,
            }
            data_list = fetch_from_search(key_word=douyin_full_title)
            bilibili_key_list = ['author', 'bvid', 'title', 'description', 'duration', 'mid']
            result_list = []
            for data in data_list:
                # 只保留指定的键
                filtered_data = {key: data.get(key, '') for key in bilibili_key_list}
                temp_duration = time_to_ms(filtered_data.get('duration', '0:0')) / 1000
                if abs(temp_duration - douyin_duration) <= 10:
                    filtered_data['duration'] = temp_duration
                    result_list.append(filtered_data)
            if not result_list:
                return False
            prompt = base_prompt
            prompt = f'{prompt}原始视频元数据:{douyin_info}\n目标视频列表:{result_list}'
            raw = get_llm_content(prompt=prompt, model_name="gemini-2.5-flash")
            result = string_to_object(raw)
            target_bvid = result.get("bvid")
            target_value = None
            for item in result_list:
                if item.get('bvid') == target_bvid:
                    target_value = item
                    break

            if target_value:
                print(f"[提示] 检查到重复视频: {target_value.get('title')} (BVID: {target_value.get('bvid')}) {target_value.get('author', '')}")
                user_map_info[douyin_username] = {
                    "bilibili_username": target_value.get('author', ''),
                    "bilibili_bvid": target_value.get('bvid', ''),
                    "mid": target_value.get('mid', '')
                }
                save_json(user_map_file, user_map_info)
                block_all_author([target_value.get('mid', '')])
                return True

            # 未检测到重复
            return False

        except Exception as e:
            print(f"[错误] 检查重复视频时发生异常（尝试 {attempt} / {max_retries}）：{e}")
            if attempt < max_retries:
                sleep_time = 2 ** (attempt - 1)
                try:
                    import time
                    time.sleep(sleep_time)
                except Exception:
                    pass
                continue
            else:
                return False

def get_bilibili_income_detail(cookie_string: str) -> dict | None:
    """
    通过提供的 cookie 获取 Bilibili 创作激励收入明细。

    Args:
        cookie_string (str): 从浏览器中复制的完整用户 cookie 字符串。

    Returns:
        dict | None: 如果请求成功，返回解析后的 JSON 数据 (一个字典)。
                      如果请求失败或解析失败，返回 None。
    """
    # 目标 URL
    # 注意：URL中的 csrf token 可能与你的 cookie 绑定，如果请求失败，
    # 可能需要同时更新 cookie 和 URL 中的 csrf token。
    url = "https://api.bilibili.com/studio/growup/up/income/detail?biz=1&csrf=36f468b198245a940bd9c19957ed1736&from=0&limit=8&page=1&s_locale=zh_CN&type=0"

    # 构造请求头
    headers = {
        "Cookie": cookie_string,
        "accept": "application/json, text/plain, */*",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "priority": "u=1, i",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\", \"Microsoft Edge\";v=\"138\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "Referer": "https://member.bilibili.com/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0"
    }

    try:
        # 发送 GET 请求
        response = requests.get(url, headers=headers, timeout=10)  # 设置10秒超时

        # 检查 HTTP 状态码是否表示成功
        response.raise_for_status()

        # 解析并返回 JSON 数据
        return response.json()

    except requests.exceptions.HTTPError as e:
        print(f"HTTP 错误: {e}")
        print(f"服务器响应内容: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"请求失败，发生网络错误: {e}")
        return None
    except json.JSONDecodeError:
        print("解析 JSON 失败，服务器返回的可能不是有效的 JSON 格式。")
        print(f"服务器原始响应内容: {response.text}")
        return None


BILI_REPLY_URL = "https://api.vc.bilibili.com/x/im/auto_reply/set_reply_text"

def set_bili_reply(title: str, reply: str, key1: str, key2: str, cookie_str: str, csrf: str | None = None, timeout: int = 10, replay_type=2):
    """
    向 B 站自动回复接口提交设置（模拟 fetch 请求）。
    必填参数:
      - title, reply, key1, key2: 字符串（函数会自动进行表单编码）
      - cookie_str: 完整的 Cookie 字符串（例如 "SESSDATA=...; bili_jct=...; ..."）
    可选:
      - csrf: 如果你已经单独拿到 csrf（bili_jct），可以传入覆盖从 cookie 的提取
      - timeout: requests 超时时间（秒）
    返回:
      - response 的 JSON（若服务器返回 JSON）；否则返回 response.text
    """

    if not cookie_str or not isinstance(cookie_str, str):
        raise ValueError("请提供完整的 cookie_str 字符串（必填）")

    # 尝试从 cookie 中提取 bili_jct（csrf token）
    if csrf is None:
        m = re.search(r"(?:^|;\s*)bili_jct=([^;]+)", cookie_str)
        if m:
            csrf_token = m.group(1)
        else:
            raise ValueError("cookie 中未发现 bili_jct，且未提供 csrf 参数。请提供包含 bili_jct 的 cookie 或手动传入 csrf。")
    else:
        csrf_token = csrf

    # 构造 headers（尽量真实）
    headers = {
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "Content-Type": "application/x-www-form-urlencoded",
        # 常见 UA（你可以替换为真实浏览器 UA）
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0",
        "Origin": "https://message.bilibili.com",
        "Referer": "https://message.bilibili.com/",
        "Sec-CH-UA": "\"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\", \"Microsoft Edge\";v=\"138\"",
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": "\"Windows\"",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        # 把 cookie 放在 header 中
        "Cookie": cookie_str,
    }

    # 构造表单数据（与 fetch 的 body 保持一致字段）
    payload = {
        "type": replay_type,
        "reply": reply,
        "id": "0",
        "title": title,
        "key1": key1,
        "key2": key2,
        "build": "0",
        "mobi_app": "web",
        "csrf": csrf_token
    }

    # 使用 urlencode 确保表单被正确编码（Content-Type: application/x-www-form-urlencoded）
    body = urlencode(payload, doseq=False)

    # 发起请求
    try:
        resp = requests.post(BILI_REPLY_URL, headers=headers, data=body, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"请求失败: {e}")

    # 尝试返回 JSON，否则返回文本
    try:
        return resp.json()
    except ValueError:
        return resp.text


LINK_SETTING_URL = "https://api.vc.bilibili.com/link_setting/v1/link_setting/set"

def set_bili_keys_reply(keys_reply: str,
                        cookie_str: str,
                        timeout: int = 10):
    """
    提交 link_setting 的 keys_reply 设置（模拟浏览器 fetch POST）。

    参数:
      - keys_reply: 要提交的值，例如 "0" 或其它字符串
      - cookie_str: 完整的 Cookie 字符串（例如 "SESSDATA=...; bili_jct=...; ..."）
      - csrf: 可选，若已单独拿到 bili_jct，可传入以覆盖从 cookie 中提取
      - timeout: requests 超时时间（秒）
    返回:
      - 若响应可解析为 JSON，则返回 resp.json()，否则返回 resp.text
    """
    csrf = None
    if not cookie_str or not isinstance(cookie_str, str):
        raise ValueError("请提供完整的 cookie_str 字符串（必填）")

    # 提取 csrf (bili_jct)
    if csrf is None:
        m = re.search(r"(?:^|;\s*)bili_jct=([^;]+)", cookie_str)
        if m:
            csrf_token = m.group(1)
        else:
            raise ValueError("cookie 中未发现 bili_jct，且未提供 csrf 参数。请提供包含 bili_jct 的 cookie 或手动传入 csrf。")
    else:
        csrf_token = csrf

    # 构造 headers（尽量真实）
    headers = {
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Content-Type": "application/x-www-form-urlencoded",
        # 可替换为你的真实 UA
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
        "Origin": "https://message.bilibili.com",
        "Referer": "https://message.bilibili.com/",
        "Sec-CH-UA": "\"Not;A=Brand\";v=\"99\", \"Brave\";v=\"139\", \"Chromium\";v=\"139\"",
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": "\"Windows\"",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "Sec-GPC": "1",
        # 将 cookie 放入 Header，requests 也会处理 cookies，但放在 header 更接近浏览器行为
        "Cookie": cookie_str,
    }

    # 表单数据
    payload = {
        "keys_reply": keys_reply,
        "recv_reply": keys_reply,
        "build": "0",
        "mobi_app": "web",
        "csrf": csrf_token
    }

    body = urlencode(payload, doseq=False)

    try:
        resp = requests.post(LINK_SETTING_URL, headers=headers, data=body, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"请求失败: {e}")

    # 返回 JSON 或文本
    try:
        return resp.json()
    except ValueError:
        return resp.text


GET_REPLY_TEXT_URL = "https://api.vc.bilibili.com/x/im/auto_reply/get_reply_text"

def get_bili_reply_text(cookie_str: str,
                       types = None,
                       build: str = "0",
                       mobi_app: str = "web",
                       web_location: str = "333.40164",
                       timeout: int = 10):
    """
    发送与浏览器 fetch 等效的 GET 请求，且将 cookie 放到 Header 中。

    参数:
      - cookie_str: 完整 Cookie 字符串（必填），例如 "SESSDATA=...; bili_jct=...; ..."
      - types: 要在查询中提交的 type[] 列表，默认 [2]
      - build, mobi_app, web_location: 查询参数，默认复刻你给出的值
      - timeout: requests 超时时间（秒）

    返回:
      - 若响应可解析为 JSON，则返回 resp.json()，否则返回 resp.text
    """
    if not cookie_str or not isinstance(cookie_str, str):
        raise ValueError("请提供完整的 cookie_str 字符串（必填）")

    if types is None:
        types = [2]

    # 构造 headers（尽量模拟真实浏览器）
    headers = {
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        # 常见 UA（可替换为你的真实 UA）
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
        "Sec-CH-UA": "\"Not;A=Brand\";v=\"99\", \"Brave\";v=\"139\", \"Chromium\";v=\"139\"",
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": "\"Windows\"",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "Sec-GPC": "1",
        "Referer": "https://message.bilibili.com/",
        # 把 cookie 放到 Header 中（按你的要求）
        "Cookie": cookie_str,
    }

    # 构造 query params，注意 type[] 为数组形式
    params = []
    for t in types:
        params.append(("type[]", str(t)))
    params.extend([
        ("build", build),
        ("mobi_app", mobi_app),
        ("web_location", web_location)
    ])

    try:
        resp = requests.get(GET_REPLY_TEXT_URL, headers=headers, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"请求失败: {e}")

    # 返回 JSON 或文本
    try:
        return resp.json()
    except ValueError:
        return resp.text

DEL_REPLY_URL = "https://api.vc.bilibili.com/x/im/auto_reply/del_reply_text"

def del_bili_reply_text(id: str,
                       cookie_str: str,
                       csrf = None,
                       timeout: int = 10):
    """
    删除指定 id 的自动回复文本（模拟浏览器 fetch 的 POST 请求）。
    参数:
      - id: 要删除的回复 id（字符串或可以转为字符串的类型）
      - cookie_str: 完整的 Cookie 字符串（必填），例如 "SESSDATA=...; bili_jct=...; ..."
      - csrf: 可选，如果你已单独拿到 bili_jct（csrf token），可传入覆盖从 cookie 中提取
      - timeout: 请求超时时间（秒）
    返回:
      - 如果响应能解析为 JSON，则返回 resp.json()，否则返回 resp.text
    抛出:
      - 在请求异常时抛出 RuntimeError；在缺少 bili_jct 且未提供 csrf 时抛出 ValueError
    """

    if not cookie_str or not isinstance(cookie_str, str):
        raise ValueError("请提供完整的 cookie_str 字符串（必填）")

    # 提取 csrf (bili_jct)
    if csrf is None:
        m = re.search(r"(?:^|;\s*)bili_jct=([^;]+)", cookie_str)
        if m:
            csrf_token = m.group(1)
        else:
            raise ValueError("cookie 中未发现 bili_jct，且未提供 csrf 参数。请提供包含 bili_jct 的 cookie 或手动传入 csrf。")
    else:
        csrf_token = csrf

    # 构造 headers（尽量模拟真实浏览器）
    headers = {
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
        "Origin": "https://message.bilibili.com",
        "Referer": "https://message.bilibili.com/",
        "Sec-CH-UA": "\"Not;A=Brand\";v=\"99\", \"Brave\";v=\"139\", \"Chromium\";v=\"139\"",
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": "\"Windows\"",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "Sec-GPC": "1",
        # 将 cookie 放到 Header 中（按你要求）
        "Cookie": cookie_str,
    }

    # 表单数据（与 fetch body 保持一致）
    payload = {
        "id": str(id),
        "build": "0",
        "mobi_app": "web",
        "csrf": csrf_token
    }

    body = urlencode(payload, doseq=False)

    try:
        resp = requests.post(DEL_REPLY_URL, headers=headers, data=body, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"请求失败: {e}")

    # 返回 JSON 或文本
    try:
        return resp.json()
    except ValueError:
        return resp.text


SIGN_UPDATE_URL = "https://api.bilibili.com/x/member/web/sign/update"


def update_bili_user_sign(cookie_str: str, user_sign: str, timeout: int = 10):
    """
    更新 B 站个性签名 (user_sign) —— 模拟浏览器 fetch 请求。

    参数:
      - cookie_str: 完整 Cookie 字符串 (必须包含 bili_jct)
      - user_sign: 新的个性签名

    返回:
      - JSON (若能解析)，否则返回文本
    """

    if not cookie_str or not isinstance(cookie_str, str):
        raise ValueError("必须提供完整 cookie_str")

    # 提取 bili_jct (csrf token)
    m = re.search(r"(?:^|;\s*)bili_jct=([^;]+)", cookie_str)
    if not m:
        raise ValueError("cookie 中未发现 bili_jct，请确保 cookie 包含 bili_jct")
    csrf_token = m.group(1)

    headers = {
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0",
        "Origin": "https://space.bilibili.com",
        "Referer": "https://space.bilibili.com/",
        "Sec-CH-UA": "\"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\", \"Microsoft Edge\";v=\"138\"",
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": "\"Windows\"",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "Cookie": cookie_str,
    }

    payload = {
        "user_sign": user_sign,
        "csrf": csrf_token,
    }
    body = urlencode(payload, doseq=False)

    try:
        resp = requests.post(SIGN_UPDATE_URL, headers=headers, data=body, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"请求失败: {e}")

    try:
        return resp.json()
    except ValueError:
        return resp.text


def _parse_cookie_string(cookie_string):
    """返回 (cookies_dict, bili_jct_value)"""
    parsed = {}
    bili_jct = ""
    for part in cookie_string.split(';'):
        part = part.strip()
        if not part:
            continue
        if '=' in part:
            k, v = part.split('=', 1)
            parsed[k] = v
            if k == 'bili_jct':
                bili_jct = v
    return parsed, bili_jct

def modify_relation(fid, action_type, cookie_str, url: str = URL_MODIFY_RELATION, timeout: int = 10, retries: int = 1):
    """
    精简版 modify_relation（使用 print 而非 logging）。
    参数:
      - fid: 目标 UID（字符串或数字）
      - action_type: 1=关注, 2=取消关注, 5=拉黑（按你原逻辑）6=解除拉黑（按你原逻辑）
      - cookie_str: 完整 Cookie 字符串（如 "SESSDATA=...; bili_jct=...; ..."），若为 None 则使用模块内 FULL_COOKIE_STRING
      - url, timeout, retries: 可选
    返回: (success: bool, result: dict or str)
    """
    cookie_string = cookie_str
    if not cookie_string:
        print("ERROR: 未提供 cookie_str（请传入或在模块中定义 FULL_COOKIE_STRING 或 环境变量 BILI_COOKIE）。")
        return False, "no_cookie"

    cookies, csrf = _parse_cookie_string(cookie_string)
    if not cookies.get("SESSDATA") or not csrf:
        print("ERROR: cookie 中缺少 SESSDATA 或 bili_jct（CSRF token）。")
        return False, "missing_sessdata_or_csrf"

    if action_type == 1:
        action_text = "关注"
    elif action_type == 5:
        action_text = "拉黑"
    else:
        action_text = "取消关注"

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://space.bilibili.com/",
        "Origin": "https://space.bilibili.com"
    }

    payload = {
        "fid": str(fid),
        "act": int(action_type),
        "re_src": 11,
        "csrf": csrf
    }

    last_err = None
    for attempt in range(1, retries + 2):  # 总尝试次数 = retries + 1
        try:
            print(f"INFO: 尝试 {action_text} UID:{fid}（第 {attempt} 次）...")
            resp = requests.post(url, data=payload, headers=headers, cookies=cookies, timeout=timeout)
            resp.raise_for_status()
            try:
                j = resp.json()
            except ValueError:
                print(f"ERROR: {action_text} UID:{fid} — 响应不是 JSON，status={resp.status_code}")
                return False, f"non_json_status_{resp.status_code}"

            if j.get("code") == 0:
                print(f"SUCCESS: 成功{action_text} UID: {fid}")
                return True, j
            else:
                print(f"ERROR: {action_text} UID:{fid} 失败: {j.get('message')} (Code: {j.get('code')})")
                return False, j

        except requests.exceptions.RequestException as e:
            last_err = e
            print(f"WARN: 第 {attempt} 次请求失败: {e}")
            if attempt <= retries:
                wait = random.uniform(0.5, 1.5) * attempt
                print(f"INFO: 等待 {wait:.2f}s 后重试...")
                time.sleep(wait)
            else:
                print(f"ERROR: 达到最大重试次数，操作失败: {e}")
                return False, str(e)

    return False, str(last_err or "unknown_error")

def block_all_author(mid_list=None):
    """
    拉黑所有原作者用户
    """
    start_time = time.time()
    # ---------- 文件路径常量 ----------

    config_map = init_config()

    if not mid_list:
        user_map_info_file = '../../LLM/TikTokDownloader/douyin_bilibili_user_map.json'  # 权威源
        user_map_info = read_json(user_map_info_file)
        mid_list = []
        for key, value in user_map_info.items():
            mid = value.get("mid")
            if mid and mid not in mid_list:
                mid_list.append(mid)
    print(f"[提示] 共有 {len(mid_list)} 个不同的 B 站用户需要拉黑  用户数量为{len(config_map)}")

    for mid in mid_list:
        for uid, value in config_map.items():
            total_cookie = value.get("total_cookie", "")
            name = value.get("name", "")
            if not total_cookie:
                print(f"[跳过] 用户 {uid} 未配置 total_cookie，无法拉黑")
                continue
            if mid == uid:
                print(f"[跳过] 用户 {uid} 不能拉黑自己")
                continue
            print(f"[提示] 使用用户 {uid} 的账号尝试拉黑 {mid}")
            success, result = modify_relation(fid=mid, action_type=5, cookie_str=total_cookie)
            if success:
                print(f"[成功] 用户 {name} 成功拉黑 {mid}")
        time.sleep(2)  # 每个用户间隔 2 秒
    print(f"[完成] 所有用户拉黑操作完成，耗时 {time.time() - start_time:.2f} 秒")

def get_all_income():
    config_map = init_config()

    # 先收集所有 name
    all_names = set()
    user_data = {}

    for uid, value in config_map.items():
        total_cookie = value.get("total_cookie", "")
        user_name = value.get("name", "")
        income_data = get_bilibili_income_detail(total_cookie)

        if not income_data:
            print(f"\n用户 {user_name} 未能获取数据，请检查 cookie 是否过期。")
            user_data[user_name] = {}
            continue

        # print(f"\n成功获取 {user_name} 的收入明细数据：")
        # print(json.dumps(income_data, indent=4, ensure_ascii=False))

        # 存储该用户的收入
        user_income = {}
        for item in income_data.get("data", {}).get("list", []):
            name = item["name"]
            amt = item["incentive_amt"]
            user_income[name] = amt
            all_names.add(name)

        user_data[user_name] = user_income

    # 统一补齐：没有的补 0
    result_dict = {}
    for name in all_names:
        details = []
        total = 0
        for user, udata in user_data.items():
            amt = udata.get(name, 0)
            total += amt
            details.append({"user": user, "incentive_amt": amt})

        # detail 按收益降序排列
        details.sort(key=lambda x: x["incentive_amt"], reverse=True)

        result_dict[name] = {"total": total, "details": details}

    # 降序排序（按照日期）
    def parse_date_from_name(n):
        import re, datetime
        m = re.match(r"(\d{4})年(\d{2})月(\d{2})日", n)
        if m:
            y, mth, d = map(int, m.groups())
            return datetime.date(y, mth, d)
        return datetime.date.min

    sorted_result = dict(sorted(result_dict.items(), key=lambda x: parse_date_from_name(x[0]), reverse=False))

    # 打印结果
    print("\n===== 最终统计结果 =====")
    for name, data in sorted_result.items():
        print(f"{name}: 总和={data['total']}")
        for detail in data["details"]:
            print(f"  - {detail['user']}: {detail['incentive_amt']}")

    return sorted_result



if __name__ == '__main__':
    get_all_income()


    COOKIE = get_config("jie_bilibili_total_cookie")


    # # 进行拉黑 关注 取消关注
    # block_all_author()


    # 更新个性签名
    # result = update_bili_user_sign(COOKIE, "测试个性签名")
    # print(result)

    # # 删除指定的自动回复文本
    # result= del_bili_reply_text('1101615', COOKIE)
    # print(result)
    #
    # # 查看已经设置好了的自动回复的提示词
    # result = get_bili_reply_text(COOKIE)
    # print(result)
    #
    #
    # # 设置自动回复的提示词
    # title = "游戏"
    # reply = "https://docs.qq.com/sheet/DTmZWVWh3WnpsbE5Q?no_promotion=1&tab=BB08J2\n已经汇总到该文档中，请自行查看"
    # key1 = "游戏"
    # key2 = "好玩，无敌，破解"
    #
    # result = set_bili_reply(title=title, reply=reply, key1=key1, key2=key2, cookie_str=COOKIE)
    # print(result)
    #
    #
    # # 打开自动回复
    # keys_reply_value = '1'
    # result = set_bili_keys_reply(keys_reply=keys_reply_value, cookie_str=COOKIE)
    # print(result)




    #
    #
    # meta_data =       {
    #             "collection_time": "2025-08-20 00:51:36",
    #             "id": "7447738126698548537",
    #             "desc": "十分钟搞笑合集让你一次笑个够 #搞笑",
    #             "full_title": "十分钟搞笑合集让你一次笑个够 #搞笑",
    #             "create_timestamp": 1734061672,
    #             "create_time": "2024-12-13 11:47:52",
    #             "text_extra": [
    #                 "搞笑"
    #             ],
    #             "type": "视频",
    #             "height": 720,
    #             "width": 1280,
    #             "downloads": "https://www.douyin.com/aweme/v1/play/?video_id=v1e00fgi0000ctdqr2nog65qpsfq50c0&line=0&file_id=7e869e5ac3ee446dad4d866f228d4e78&sign=685ca2e8037bac7b39df99cb64b27d82&is_play_url=1&source=PackSourceEnum_AWEME_DETAIL",
    #             "duration": "10:17",
    #             "uri": "v1e00fgi0000ctdqr2nog65qpsfq50c0",
    #             "dynamic_cover": "https://p9-pc-sign.douyinpic.com/obj/tos-cn-i-0813c000-ce/oQExAmeeFFBAXBJAEOED0FfAIAQvghicJ84Uw7?lk3s=138a59ce&x-expires=1756828800&x-signature=Wtpj09IDNz6tM98hQfpK%2Fn9KRvQ%3D&from=327834062_large&s=PackSourceEnum_AWEME_DETAIL&se=false&sc=dynamic_cover&biz_tag=pcweb_cover&l=20250820005134973D8C6C66D6D5366F59",
    #             "static_cover": "https://p9-pc-sign.douyinpic.com/image-cut-tos-priv/5e17015760a2572804264e356e8ca476~tplv-dy-resize-origshort-autoq-75:330.jpeg?lk3s=138a59ce&x-expires=2070979200&x-signature=gwGU%2F7qBDUV6L47BzaLZ%2BY9hy6o%3D&from=327834062&s=PackSourceEnum_AWEME_DETAIL&se=false&sc=cover&biz_tag=pcweb_cover&l=20250820005134973D8C6C66D6D5366F59",
    #             "uid": "729720776557711",
    #             "sec_uid": "MS4wLjABAAAAOff9GwYDbDMkDI-xCGAlEYcNzcv2mz2cE1HewplyNeo",
    #             "unique_id": "xy11471015",
    #             "signature": "❤️橱窗里面啥都有",
    #             "user_age": 124,
    #             "nickname": "小歪同学",
    #             "mark": "小歪同学",
    #             "music_author": "小歪同学",
    #             "music_title": "@小歪同学创作的原声",
    #             "music_url": "https://lf26-music-east.douyinstatic.com/obj/ies-music-hj/7447738300279900988.mp3",
    #             "digg_count": 139906,
    #             "comment_count": 1179,
    #             "collect_count": 32707,
    #             "share_count": 35500,
    #             "play_count": -1,
    #             "tag": [
    #                 "随拍",
    #                 "生活记录",
    #                 "日常vlog"
    #             ],
    #             "extra": "",
    #             "share_url": "https://www.douyin.com/video/7447738126698548537",
    #             "abs_cover_path": "W:\\project\\python_project\\watermark_remove\\LLM\\TikTokDownloader\\Download\\cover\\7447738126698548537.jpg"
    #         }
    #
    # result = check_duplicate_video(meta_data)
    # print(f"检查结果: {'重复' if result else '不重复'}")
    #
    # #
    # # total_cookie = get_config("ruru_bilibili_total_cookie")
    # # # car_items = list_selection_car_items(total_cookie, 100)
    # #
    # # cookie = total_cookie
    # # result = fetch_goods(cookie=cookie, max_count=20, goodsName="零食")
    # # print(f"共获取到 {len(result)} 个商品")
    # #
    # # # add_goods_to_selection(cookie=cookie, goods=result[:10], operate_source=4, from_type=18)
    # # goods = update_short_url(cookie=cookie, goods=result)
    # # print(goods)