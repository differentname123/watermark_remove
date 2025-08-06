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
                print(f"[Info] 第 {page} 页无更多数据，提前结束")
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

if __name__ == '__main__':
    total_cookie = get_config("ruru_bilibili_total_cookie")

    cookie = total_cookie
    result = fetch_goods(cookie=cookie, max_count=30, goodsName="零食")
    print(f"共获取到 {len(result)} 个商品")
