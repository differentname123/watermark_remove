# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/8/11 3:59
:last_date:
    2025/8/11 3:59
:description:
    
"""
import requests
import time
import json
import re
from urllib.parse import quote  # <--- 1. 引入 quote 函数

def create_favorites(cookie_string: str, search_keyword: str = "牛肉干"):
    """
    复制 Alimama 的 fetch 请求。

    :param cookie_string: 从浏览器中获取的完整 cookie 字符串。
    :param search_keyword: 你想要搜索的商品关键词。
    :return: 成功时返回解析后的 JSON 数据 (dict)，失败时返回 None。
    """

    # 1. 构造请求头 (Headers)
    # 尽可能与浏览器保持一致，包括添加一个匹配的 User-Agent
    headers = {
        "accept": "*/*",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "bx-v": "2.5.11",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "priority": "u=1, i",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\", \"Google Chrome\";v=\"138\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "x-requested-with": "XMLHttpRequest",
        # 从你的 fetch 请求中复制 Referer
        "Referer": "https://pub.alimama.com/portal/v2/pages/promo/goods/index.htm?pageNum=1&pageSize=30&filters=%257B%257D&fn=search&q=%E7%89%9B%E8%82%89%E5%B9%B2&sort=max_tk_rate%3Ades&selected=%257B%257D&floorId=80674",
        # 添加一个匹配的 User-Agent
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        # 将你的 cookie 字符串放入请求头
        "Cookie": cookie_string
    }

    # 2. 构造 URL 参数 (Query Parameters)

    # 尝试从 cookie 中提取 _tb_token_
    tb_token_match = re.search(r'_tb_token_=(\w+);?', cookie_string)
    if not tb_token_match:
        print("警告：无法从 Cookie 字符串中找到 '_tb_token_'。请求可能会失败。")
        tb_token = "e0847b7001a9b"  # 使用你提供的默认值作为后备
    else:
        tb_token = tb_token_match.group(1)

    # 准备 _data_ 参数的原始内容
    data_payload = {
        "floorId": 31401,
        "refpid": "mm_328750149_0_0",
        "variableMap": {
            "title": search_keyword
        }
    }

    # 将 data_payload 转换为紧凑的 JSON 字符串
    data_json_string = json.dumps(data_payload, separators=(',', ':'))

    params = {
        "t": int(time.time() * 1000),  # 生成当前的13位毫秒时间戳
        "_tb_token_": tb_token,
        "_data_": data_json_string
    }

    # 3. 发送 GET 请求
    base_url = "https://pub.alimama.com/openapi/json2/1/gateway.unionpub/xt.entry.json"

    try:
        response = requests.get(base_url, headers=headers, params=params, timeout=10)

        # 检查请求是否成功
        response.raise_for_status()  # 如果状态码不是 2xx，则会抛出异常

        # 尝试解析 JSON 响应
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {e}")
    except json.JSONDecodeError:
        print("解析 JSON 响应失败，返回原始文本内容:")
        print(response.text)

    return None

def fetch_alimama_data(cookie_string: str, pid: str, search_query: str, target_num: int = 100):
    """
    模拟请求阿里妈妈商品搜索接口，并持续获取数据直到满足目标数量或无更多数据。

    Args:
        cookie_string (str): 从浏览器获取的完整Cookie字符串。
        pid (str): 你的阿里妈妈推广位ID (例如 "mm_xxxx_xxxx_xxxx")。
        search_query (str): 要搜索的商品关键词。
        target_num (int): 期望获取的商品总数量。函数会持续翻页直到达到此数量或无更多结果。

    Returns:
        list: 成功时返回包含商品信息的列表，如果未获取到任何数据或失败则返回空列表[]。
    """
    # --- 1. 参数校验 (只需执行一次) ---
    if "在此处粘贴你的完整Cookie字符串" in cookie_string or not cookie_string:
        print("错误：请在 'MY_COOKIE_STRING' 变量中填入你自己的有效 Cookie！")
        return []  # 改为返回空列表，保持返回类型一致性
    if "mm_xxxx_xxxx_xxxx" in pid or not pid:
        print("错误：请在 'MY_PID' 变量中填入你自己的推广位 PID！")
        return []

    # --- 2. 准备固定的Headers和URL (只需执行一次) ---
    url = "https://pub.alimama.com/openapi/param2/1/gateway.unionpub/skyleap.distribution.site.json"
    headers = {
        "accept": "*/*",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "origin": "https://pub.alimama.com",
        "priority": "u=1, i",
        "referer": f"https://pub.alimama.com/portal/v2/pages/promo/goods/index.htm?q={quote(search_query)}",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\", \"Google Chrome\";v=\"138\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        "x-requested-with": "XMLHttpRequest",
        "cookie": cookie_string
    }

    # --- 3. 从Cookie中动态提取 _tb_token_ (只需执行一次) ---
    match = re.search(r'_tb_token_\s*=\s*([^;]+)', cookie_string)
    if not match:
        print("错误：无法从你的Cookie字符串中找到 '_tb_token_'。请检查Cookie是否完整且正确。")
        return []
    tb_token = match.group(1).strip()

    # --- 4. 循环获取数据 ---
    all_products = []
    current_page = 1

    while len(all_products) < target_num:
        print(f"\n正在尝试获取第 {current_page} 页的数据...")

        # --- 4.1 准备每一页的动态Payload ---
        # 注意：extras_dict 必须在循环内部创建，因为 pageNum 在每次循环时都会改变
        extras_dict = {
            "sceneCode": "pub_selection_navigation",
            "floorId": 80674,
            "pageNum": current_page,  # API的页码是从1开始的 (经测试确认)
            "pageSize": "30",  # 每页获取30个
            "pid": pid,
            "variableMap": {
                "fn": "search",
                "resultCanBeEmpty": True,
                "q": search_query,
                "curSelected": {},
                "pubFloorId": 80674,
                "sort": "max_tk_rate:des",
                "tk_navigator": "true",
                "union_lens": "b_pvid:a219t._portal_v2_pages_promo_goods_index_htm_1754819807788_16588432743269266_ccPDH",
                "lensScene": "PUB",
                "spmB": "_portal_v2_pages_promo_goods_index_htm"
            }
        }

        payload = {
            "t": int(time.time() * 1000),
            "_tb_token_": tb_token,
            "siteCode": "selectionPlaza_site",
            "appCode": "pub",
            "extras": json.dumps(extras_dict, separators=(',', ':'))
        }

        # --- 4.2 发送POST请求并处理响应 ---
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=15)
            response.raise_for_status()
            result_json = response.json()

            if str(result_json.get('resultCode', 400)) == '200':
                resultList = result_json.get('data', {}).get('siteData', {}).get('resultList', [])

                # 如果请求成功，但没有返回任何商品，说明已经到底了
                if not resultList:
                    print("API返回数据为空，已获取所有商品，停止翻页。")
                    break  # 退出while循环

                all_products.extend(resultList)
                print(f"成功获取 {len(resultList)} 条商品。当前总数: {len(all_products)} / {target_num}")

                # 准备请求下一页
                current_page += 1

                # 加个延时，避免请求过于频繁
                time.sleep(1)

            else:
                error_message = result_json.get("info", {}).get("message", "未知错误")
                print(f"请求失败，API返回信息：{error_message}")
                if "login" in error_message.lower():
                    print("提示：您的Cookie可能已过期，请从浏览器重新获取并更新到代码中。")
                break  # 发生API错误，退出while循环

        except requests.exceptions.RequestException as e:
            print(f"网络请求发生异常: {e}")
            break  # 发生网络错误，退出while循环
        except json.JSONDecodeError:
            print("解析返回的JSON数据失败，服务器可能返回了非JSON格式的内容。")
            print("原始响应内容:", response.text)
            break  # 发生解析错误，退出while循环

    # --- 5. 返回最终结果 ---
    # 对结果进行切片，确保最多只返回 target_num 个商品
    final_results = all_products
    print(f"\n抓取完成！共获取 {len(final_results)} 条商品。")
    return final_results


def add_to_favorites_final(
        cookie_string: str,
        item_id_list,  # 接收加密的 itemID
        union_lens_str: str,  # 新增：接收 union_lens 字符串
        destination_folder_id: int,
        destination_rule_id: int,
        source_floor_id: int = 80674,
        refpid: str = "mm_328750149_0_0"
):
    """
    将一个或多个商品添加到淘宝联盟的收藏夹 (最终精确版)。
    此版本完全模拟原始请求，包括所有跟踪参数。

    :param cookie_string: 完整 cookie 字符串。
    :param item_id_list: 需要收藏的商品【加密ID】列表。
    :param union_lens_str: 从原始请求中复制的 union_lens 字符串。
    :param destination_folder_id: 目标收藏夹的 ID (finalFloorId)。
    :param destination_rule_id: 目标收藏夹配对的规则 ID (finalZsRuleId)。
    :param source_floor_id: 商品来源的 floorId。
    :param refpid: 你的推广位 ID。
    :return: 成功时返回解析后的 JSON 数据 (dict)，失败时返回 None。
    """
    headers = {
        "accept": "*/*",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "bx-v": "2.5.11",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "priority": "u=1, i",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\", \"Google Chrome\";v=\"138\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "x-requested-with": "XMLHttpRequest",
        "Referer": "https://pub.alimama.com/portal/v2/pages/promo/goods/index.htm",
        "Origin": "https://pub.alimama.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
        "Cookie": cookie_string
    }

    tb_token_match = re.search(r'_tb_token_=(\w+);?', cookie_string)
    if not tb_token_match:
        print("错误：无法从 Cookie 字符串中找到 '_tb_token_'。请求中止。")
        return None
    tb_token = tb_token_match.group(1)

    # 这里的 item_id 使用你提供的加密ID
    item_list_for_json = [{"itemId": item_id, "floorId": source_floor_id} for item_id in item_id_list]
    item_list_string = json.dumps(item_list_for_json, separators=(',', ':'))

    # --- 关键修正：完整复制 variableMap 结构 ---
    data_payload = {
        "floorId": 70519,
        "refpid": refpid,
        "variableMap": {
            "firstZsRuleId": [],
            "firstFloorId": [],
            "itemList": item_list_string,
            "finalZsRuleIdList": [destination_rule_id],
            "finalFloorIdList": [destination_folder_id],
            # 将跟踪参数加回来
            "union_lens": union_lens_str,
            "lensScene": "PUB",
            "spmB": "_portal_v2_pages_promo_goods_index_htm"
        }
    }

    data_json_string = json.dumps(data_payload, separators=(',', ':'))

    post_body = {
        "t": int(time.time() * 1000),
        "_tb_token_": tb_token,
        "_data_": data_json_string
    }

    url = "https://pub.alimama.com/openapi/json2/1/gateway.unionpub/xt.entry.json"

    try:
        response = requests.post(url, headers=headers, data=post_body, timeout=10)
        response.raise_for_status()
        json_response = response.json()

        if json_response.get("info", {}).get("ok"):
            print(f"成功将 {len(item_id_list)} 个商品添加到收藏夹！消息: {json_response.get('info', {}).get('message')}")
            return json_response
        else:
            print("添加到收藏夹操作失败。请检查所有参数，特别是 Cookie 和 union_lens 是否为最新。")
            print("服务器响应:")
            print(json.dumps(json_response, indent=2, ensure_ascii=False))
            return None

    except Exception as e:
        print(f"请求或解析过程中发生严重错误: {e}")
    return None

# ==============================================================================
# ---                        ↓↓↓ 在这里修改你的信息 ↓↓↓                         ---
# ==============================================================================
if __name__ == "__main__":

    MY_COOKIE_STRING = "cna=ccPDH0JbtmsCAWckGUfpiloU; lid=%E4%B8%8D%E5%90%8C%E5%90%8D%E5%AD%97123; wk_unb=UNN79H%2FqRhK1kg%3D%3D; wk_cookie2=19ec7d0497f9c798f71663b36ddae082; dnk=%5Cu4E0D%5Cu540C%5Cu540D%5Cu5B57123; tracknick=%5Cu4E0D%5Cu540C%5Cu540D%5Cu5B57123; _l_g_=Ug%3D%3D; unb=3365477897; lgc=; cookie1=W5w2OF288oV%2BtWjJFbZNhuMaU%2FI4uoNXm4fklxnMplE%3D; login=true; cookie17=UNN79H%2FqRhK1kg%3D%3D; cookie2=2dbc15e660671f2e6ef995d9d1ac89d0; _nk_=%5Cu4E0D%5Cu540C%5Cu540D%5Cu5B57123; cancelledSubSites=empty; sg=378; t=01e52fbaf67e9ab5ec694e335e7bb09b; csg=8c0f6d3e; sn=; _tb_token_=e0847b7001a9b; t_alimama=f6e837d245b3665e74a1f618e66bc89c; cookie2_alimama=1034ab7bcacc95395cc4a348751af259; v_alimama=0; arms_uid=d0fc7bca-b8e0-4f34-bc04-958ce8a70029; uc1=cookie16; havana_lgc_exp=1785776618227; sgcookie=E100GW%2FifobMliykmYdTLoanZvPLj%2BJOjl3Q1K18TcEKSAy86U7DWYTr%2FWOKBF%2BRmZK5vIBG%2Fqc73v0bW1dY%2F%2BplqE%2FvYvTA27UtkM%2Ber1vRspGsokAV5Q8x0%2F9zoIRenFMR; _m_h5_c=df3f68f10dc7b692edfb8e4d70e40647_1754792928595%3B89a37c17732d4653247d2d435e439393; isg=BLa23dIvwpadRLbysLEeqtGNB-y41_oR8lBAniCfORkpY1b9iGPGIRkQfz8PS_Ip; rurl=aHR0cHM6Ly9wdWIuYWxpbWFtYS5jb20vP2ZvcndhcmQ9aHR0cCUzQSUyRiUyRnB1Yi5hbGltYW1hLmNvbSUyRnBvcnRhbCUyRnYyJTJGZWZmZWN0JTJGb3JkZXIlMkZvdmVydmlld09yZGVyJTJGcGFnZSUyRmluZGV4Lmh0bQ%3D%3D; JSESSIONID=254EFDF77205FAB8D537780731860AE0; x5sectag=513267; x5sec=7b22733b32223a2264633930366339373737303935363536222c22617365727665723b33223a22307c434b375a34635147454937327767556144444d7a4e6a55304e7a63344f5463374d7a43366e7650392b2f2f2f2f2f3842227d; xlly_s=1; tfstk=gCsE8AARNkEesFG_A1KrgcSWHU-pT3Pb-gOWETXkdBAHVpTP7_1Gdeadd37NZ1LBt3tBrLf5sB0WrHXk4sfqABxWP_SPsCCBOHNJZ15C1g_IVTDyE1tHywOWOQ-Pesy_GoZfp9KJZSNbclmFMAtjK0xoNAcMVLkh8s_Rp9K-BAGulNXKUyCM1LvkZR-MUp0kqemujFA9E3mHrD0isCpkq3AHKfDMFp0nr203QOA9E3AlZe2wjCpkqQfkqAgVqMowTe2i9i8fno3MJIXH_0mqHLL3GT0SmmjwLeSlTCJ6CGJe8IY9zE5fx16lAMLT5VKRCZ5Pr_UjWHWl-Hvfpkowm9b14ds7F4Rh9GWGAFknxe7kUExk7YmlQHbMoGK0mfJOS9_lv6krqpICcURv78mJPhjXoZfEe-BMxLfJkgNxvBXlHiQX0lgHmO7lgaSPUbpicRSR8b0y-dpwGRywz5XjolBcTn3-yF29QIw4343J-dpwGRyZy4LG6dRb3-5.."

    # 步骤 2: 将下面的字符串替换为你自己的【推广位ID】
    MY_PID = "mm_328750149_0_0"

    search_term = "牛肉干"
    page_to_fetch = 1

    print(f"准备搜索 '{search_term}' 的第 {page_to_fetch} 页商品...")

    search_result = fetch_alimama_data(
        cookie_string=MY_COOKIE_STRING,
        pid=MY_PID,
        search_query=search_term
    )
    print(f"\n共获取到 {len(search_result)} 条商品数据。")