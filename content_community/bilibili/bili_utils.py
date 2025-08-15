# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/8/7 2:52
:last_date:
    2025/8/7 2:52
:description:
    
"""
import requests
import time
import random

from LLM.gemini import get_llm_content
from common_utils.common_utils import get_config, read_json, time_to_ms, string_to_object, save_json

base_prompt = """
            # 角色
            你是一个智能的视频内容查重助手。
            # 任务
            你的任务是根据我提供的【原始视频元数据】，在【目标视频列表】中，判断是否存在实际上是同一个视频的条目。目标是帮助我确认这个视频是否已经在新平台上发布过。
            # 匹配逻辑
            判断标准不是要求所有字段都100%完全相同，而是基于以下逻辑进行综合判断：
            1.  **标题 (title)**: 核心内容一致或高度相似。请忽略前后缀、空格、标点符号、大小写等微小差异。例如，“我的VLOG日常” 和 “【VLOG】我的日常！” 应被视为相似。
            2.  **作者 (author)**: 应基本一致。这是个非常强的匹配信号。
            3.  **时长 (duration)**: 应该非常接近，允许有几秒钟的误差（例如在 ±5 秒内），因为视频转码或添加片头片尾可能导致微小变化。
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
from urllib.parse import quote_plus

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
        "Cookie": get_config("dahao_bilibili_total_cookie"),  # 不带 Cookie
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
            bilibili_key_list = ['author', 'bvid', 'title', 'description', 'duration']
            result_list = []
            for data in data_list:
                # 只保留指定的键
                filtered_data = {key: data.get(key, '') for key in bilibili_key_list}
                filtered_data['duration'] = time_to_ms(filtered_data.get('duration', '0:0')) / 1000
                result_list.append(filtered_data)

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
                print(f"[提示] 检查到重复视频: {target_value.get('title')} (BVID: {target_value.get('bvid')})")
                user_map_info[douyin_username] = {
                    "bilibili_username": target_value.get('author', ''),
                    "bilibili_bvid": target_value.get('bvid', '')
                }
                save_json(user_map_file, user_map_info)
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

if __name__ == '__main__':
    meta_data = {
  "collection_time": "2025-08-15 17:32:43",
  "id": "7535033987593440527",
  "desc": "kpl历史最贵的一箭，价值350万，一诺季赛 #AG战胜狼队 #一诺 #巅峰对决",
  "full_title": "kpl历史最贵的一箭，价值350万，一诺黄牌射手加身，昭君王诞生，ag战胜狼队，#成都AG超玩会 #2025KPL夏季赛 #AG战胜狼队 #一诺 #巅峰对决",
  "create_timestamp": 1754386826,
  "create_time": "2025-08-05 17:40:26",
  "text_extra": [
    "成都AG超玩会",
    "2025KPL夏季赛",
    "AG战胜狼队",
    "一诺",
    "巅峰对决"
  ],
  "type": "视频",
  "height": 2160,
  "width": 3840,
  "downloads": "https://www.douyin.com/aweme/v1/play/?video_id=v0200fg10000d28srffog65p5jo24g10&line=0&file_id=b150de3c54344f7cad9680f27845298e&sign=7b2fce5b22729e9292545e78aac26805&is_play_url=1&source=PackSourceEnum_AWEME_DETAIL",
  "duration": "00:04:04",
  "uri": "v0200fg10000d28srffog65p5jo24g10",
  "dynamic_cover": "https://p9-pc-sign.douyinpic.com/obj/tos-cn-i-dy/8e8b8c05ce924c3794cf89daa9bfa167?lk3s=138a59ce&x-expires=1756458000&x-signature=Ny5mLlXH2uxlLU3N0lB1Db9BjwI%3D&from=327834062_large&s=PackSourceEnum_AWEME_DETAIL&se=false&sc=dynamic_cover&biz_tag=pcweb_cover&l=20250815173241F6379A2DB08EF5553359",
  "static_cover": "https://p9-pc-sign.douyinpic.com/image-cut-tos-priv/f9905d41f8910211abc51416a3222641~tplv-dy-resize-origshort-autoq-75:330.jpeg?lk3s=138a59ce&x-expires=2070608400&x-signature=SR5ow8kTklKtG84HlYz78RJsZBg%3D&from=327834062&s=PackSourceEnum_AWEME_DETAIL&se=false&sc=cover&biz_tag=pcweb_cover&l=20250815173241F6379A2DB08EF5553359",
  "uid": "2848214190926880",
  "sec_uid": "MS4wLjABAAAANFpaR5OMSmEOzSw3AhoaSB0I0qJFfOApIyVLMno3gJnFve32R8DJ7eZhluCc4_KO",
  "unique_id": "62304446290",
  "signature": "各路顶尖高手操作意识讲解\n王者荣耀各大赛事精彩剪辑解说\nKpl官方合作账号\n新人博主，欢迎大家的关注",
  "user_age": 23,
  "nickname": "小璇超燃王者解说",
  "mark": "小璇超燃王者解说",
  "music_author": "小璇超燃王者解说",
  "music_title": "@小璇超燃王者解说创作的原声",
  "music_url": "https://sf5-hl-cdn-tos.douyinstatic.com/obj/ies-music/7535034122350430986.mp3",
  "digg_count": 26162,
  "comment_count": 642,
  "collect_count": 1322,
  "share_count": 497,
  "play_count": -1,
  "tag": [
    "游戏",
    "竞技游戏",
    ""
  ],
  "extra": "",
  "share_url": "https://www.douyin.com/video/7535033987593440527",
  "abs_cover_path": "W:\\project\\python_project\\watermark_remove\\LLM\\TikTokDownloader\\Download\\cover\\7535033987593440527.jpg"
}

    result = check_duplicate_video(meta_data)
    print(f"检查结果: {'重复' if result else '不重复'}")

    #
    # total_cookie = get_config("ruru_bilibili_total_cookie")
    # # car_items = list_selection_car_items(total_cookie, 100)
    #
    # cookie = total_cookie
    # result = fetch_goods(cookie=cookie, max_count=20, goodsName="零食")
    # print(f"共获取到 {len(result)} 个商品")
    #
    # # add_goods_to_selection(cookie=cookie, goods=result[:10], operate_source=4, from_type=18)
    # goods = update_short_url(cookie=cookie, goods=result)
    # print(goods)