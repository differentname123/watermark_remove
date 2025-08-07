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

from common_utils.common_utils import get_config

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

def list_selection_car_items(cookie: str):
    """
    获取选品车中商品列表（默认第一页 10 条）

    :param cookie: 登录后的 Cookie 字符串
    :return: 接口返回的 JSON 响应
    """
    url = (
        "https://cm.bilibili.com/dwp/api/web_api/v1/selection/car/item/list"
        "?page=1&size=10&sourceType=-1&promotionCampaigns=&"
        "selectionCarItemType=1&windowShelveStatus=-1&goodsName=&requestFrom=-1"
    )
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

    # 随机延迟，模拟人为操作节奏
    time.sleep(random.uniform(0.5, 1.2))

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        if result.get("code") == 0:
            items = result.get("data", {}).get("data", [])
            print(f"[Success] 已获取 {len(items)} 条选品车商品")
            return items
        else:
            print(f"[Warning] 接口返回非 0 状态: {result.get('code')} - {result.get('message')}")
            return None
    except Exception as e:
        print(f"[Error] 获取选品车列表失败: {e}")
        return None

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


if __name__ == '__main__':
    total_cookie = get_config("ruru_bilibili_total_cookie")

    cookie = total_cookie
    result = fetch_goods(cookie=cookie, max_count=20, goodsName="零食")
    print(f"共获取到 {len(result)} 个商品")

    # add_goods_to_selection(cookie=cookie, goods=result[:10], operate_source=4, from_type=18)
    goods = update_short_url(cookie=cookie, goods=result)
    print(goods)