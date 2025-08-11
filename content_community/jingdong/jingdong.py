import requests
import json
import hashlib
from datetime import datetime
import requests
import json
import time
from urllib.parse import urlencode

# 配置应用信息
APP_KEY = "a8b099d7d7c2c1c4802a131725f81b2f"  # 请替换为您自己的真实app_key
APP_SECRET = "b58181748de34f73a37a74c2a803512d"  # 请替换为您自己的真实app_secret

# API网关地址
API_URL = "https://api.jd.com/routerjson"


def call_jd_union_api(method, app_key, app_secret, business_params_wrapper):
    """
    调用京东联盟开放平台的通用函数。
    """
    sys_params = {
        'method': method,
        'app_key': app_key,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'format': 'json',
        'v': '1.0',
        'sign_method': 'md5',
    }

    json_business_params = json.dumps(business_params_wrapper, separators=(',', ':'))

    params_to_sign = sys_params.copy()
    params_to_sign['360buy_param_json'] = json_business_params

    sorted_keys = sorted(params_to_sign.keys())
    concatenated_str = "".join([key + str(params_to_sign[key]) for key in sorted_keys])
    string_to_sign = app_secret + concatenated_str + app_secret

    m = hashlib.md5()
    m.update(string_to_sign.encode('utf-8'))
    sign = m.hexdigest().upper()

    final_request_params = params_to_sign.copy()
    final_request_params['sign'] = sign

    try:
        response = requests.post(API_URL, data=final_request_params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求API时发生错误: {e}")
        return None


def get_promotion_link(material_id, site_id, sub_union_id):
    """
    生成推广链接，并返回京口令
    """
    api_method = 'jd.union.open.promotion.common.get'

    # 请求参数
    promotion_code_req = {
        'materialId': material_id,  # 商品链接
        'siteId': site_id,  # 网站ID/APP ID
        # 'subUnionId': sub_union_id,  # 子渠道ID
        'sceneId': 1,  # 场景ID: 1 表示商品推广
        'command': 1,  # 生成京口令
    }

    business_params_wrapper = {'promotionCodeReq': promotion_code_req}

    result = call_jd_union_api(api_method, APP_KEY, APP_SECRET, business_params_wrapper)

    if not result:
        return None

    # 获取返回数据
    response_data = result.get('jd_union_open_promotion_common_get_responce', {})
    if 'getResult' in response_data:
        data = json.loads(response_data['getResult'])

        # 返回推广链接和京口令
        click_url = data.get('data', {}).get('clickURL')
        kouling = data.get('data', {}).get('jCommandInfo', {}).get('command')

        return {
            'click_url': click_url,
            'kouling': kouling
        }

    return None

def query_goods(goods_req):
    """
    调用 jd.union.open.goods.query 接口查询商品信息。
    :param goods_req: 一个字典，包含所有业务参数，对应文档中的 goodsReqDTO。
                      例如: {'keyword': '手机', 'pageSize': 20, 'sceneId': 1}
    :return: 查询到的商品列表（字典列表）或 None
    """
    api_method = 'jd.union.open.goods.query'

    # 根据API文档，所有业务参数都封装在 'goodsReqDTO' 对象中
    business_params_wrapper = {'goodsReqDTO': goods_req}

    result = call_jd_union_api(api_method, APP_KEY, APP_SECRET, business_params_wrapper)

    if not result:
        return None

    # 检查是否有错误响应
    if 'error_response' in result:
        error_info = result['error_response']
        print(f"API返回错误: {error_info.get('zh_desc', '未知错误')}")
        print(f"错误代码: {error_info.get('code', 'N/A')}")
        return None

    # 获取返回数据
    # 响应的key通常是方法名把点替换成下划线，再加上 "_responce"
    response_data = result.get('jd_union_open_goods_query_responce', {})

    # 根据文档和实践，实际数据在 'queryResult' 字段中，它是一个JSON字符串
    if 'queryResult' in response_data:
        try:
            # 解析 'queryResult' 字符串
            data_str = response_data['queryResult']
            if not data_str:
                return []  # 可能查询结果为空

            data = json.loads(data_str)

            # 返回 'data' 字段中的商品列表
            return data.get('data', [])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"解析商品查询结果时发生错误: {e}")
            print(f"原始queryResult: {response_data.get('queryResult')}")
            return None

    return None


def get_jd_promo_link(materialId: int):
    """
    请求京东联盟API以获取推广链接。
    此版本将cookie、h5st等所有验证信息都固定在函数内部。

    !!! 警告 !!!
    函数内固定的cookie和h5st等参数会很快过期，导致函数失效。
    一旦失效，需要从浏览器手动抓取新值并更新此函数的代码。

    Args:
        materialId (int): 要推广的商品或物料的ID。

    Returns:
        dict: 如果请求成功，返回API响应的JSON数据解析后的字典。
        None: 如果请求失败，返回None。
    """
    # --- 所有验证信息都固定在此处 ---
    fixed_cookie = "__jdu=17547806811201466311166; shshshfpa=1856b9ee-7a24-e0a1-c359-00c748ffecb3-1732041338; shshshfpx=1856b9ee-7a24-e0a1-c359-00c748ffecb3-1732041338; areaId=32; ipLoc-djd=32-2768-53509-54291; mba_muid=17547806811201466311166; unpl=JF8EALNnNSttWkldUhgLTBVDH1QEWw0LHh9QOjRRB1tcGwAATlUaEUB7XlVdWRRKHh9vZxRUWVNJUA4YACsSF3teVVxZD00fBm5jNWRaWEIZRElPKxEQe1xkXlsMThEKbmAMVF1bSlQAHAUZEhBLWlNuXDhMFwpfZwRVXFlJUAQfAhgSIHtcZF9tCXtBbW9mBFVcUEhSBBtsTk5JBl1SWlgOQhYEZmcFV1xYTlMCGQIbEhdMbVVuXg; __jdv=181111935|lianmeng__10__kong|t_2035679405_|tuiguang|338f28f6be8a4a2d9bdbeb75ae4df02a|1754899084780; RT=\"z=1&dm=jd.com&si=l3cpvw8vurp&ss=me6ssu39&sl=1&tt=0&obo=1&ld=t05w&r=fdb0bd86180097911ec0282242e45bbe&ul=t05x&hd=t05z\"; mba_sid=17549006549572011051926.2; 3AB9D23F7A4B3C9B=5N4KGSIGGXI35AIENP3MXLOIFSFSLNQKJDKPWIXPH4QKLTDPFU5ZVAPTZXHVJR6T4ZWKMXKLBMJLBCCQHZI7XQWYZE; __jd_ref_cls=Mnpm_ComponentApplied; wlfstk_smdl=a85v3cacd9sabpuik3ivjipgms1toduy; TrackID=1_f1jDfw1AhP7VGdNUslfFJVdfjypZMdLIRIMAYWZUKEqEozw3SCg-ANGtRfH7yIcDK-hQtd21jZsMToL2C78o_KMJzRhQX_JPfCo3QmrCLY; thor=CCFFF4944A0B1C21F5FBDB71F0D571797BBE630757392507012D249B26F3510E25EA2E787443F1A4B86CCEFB0F2D5FAE0A9B6451A54C95F23D955D285B0C80B27199A3B3B5C1A581C9C3D18B87F3244B95DF7E188A85B8B63416E779090C16BBD727A012E200D28E14FA77D3E5CFB036B21605D39A1573F0E4DA7B5B1E6A1DFD8B116FF2289AEBF81CE7B826395D66D204922C45B21118B88AA42B1B88AD5043; light_key=AASBKE7rOxgWQziEhC_QY6yacntM3TgYsypBuul8u68KyHpbwcAErW5H-MvPngT4JyUstZpV; pinId=wb0T8DPxyq1vjGyzhOIQ6rV9-x-f3wj7; pin=jd_5ef0febc362f9; unick=jd_ie2lrl2l549dae; ceshi3.com=203; _tp=0blb7w59TSs5PR4V4GOC9tOKGoiD8c%2Fo47mVHK5Ra%2B0%3D; _pst=jd_5ef0febc362f9; __jda=209449046.17547806811201466311166.1754780681.1754866672.1754896155.9; __jdc=209449046; shshshfpb=BApXSweNPm_1AoBf-iGMOFXGQabDzOV1pBnBFFBd59xJ1MsfhIYG2; sdtoken=AAbEsBpEIOVjqTAKCQtvQu17QfG_cDTZBcp8Z3edhFghwUPM33Zg2fzgswCd_OrEye_Z9g85rL1rz1Bx8mNdM-oZu_xjWR8uRKvFD9Lhc669BLJhWZh2PLjzGin48qIbDHEqw-Z7L0BZPkXOFZnL644ZjukwhQ; flash=3_yBacWm2812zSFjd1M7fgfApZBfcleH9k_bLfBv_HfO-dbC4_HPQOYPI2VKb6Z3ltaSgmljZ8JnWZuf68yBR_YGvdtkP1Vq-vYlUzTEozVbHKYoPbVh09WGBLqE3NFOQS4KTuXyj7lZRAnk0DM5_tl1MELNhln4e4XvFSthM6xXinbwCntQ0rmq**; __jdb=209449046.118.17547806811201466311166|9.1754896155; 3AB9D23F7A4B3CSS=jdd035N4KGSIGGXI35AIENP3MXLOIFSFSLNQKJDKPWIXPH4QKLTDPFU5ZVAPTZXHVJR6T4ZWKMXKLBMJLBCCQHZI7XQWYZEAAAAMYTBGUDDQAAAAADYJJMSPJFLT4HIX; _gia_d=1"
    fixed_h5st = "20250811155857359;w6mmgaazz6hdpqq7;586ae;tk03wa97d1bc218nhdyYL9tgyLhWDC5sPYehu-Tr2Ue7P8bkhA7X7KzVEGMUqAXW5IQ5-VDturbt-3_ZHrfBrc-AIBZr;b2e712855edaf81ae956af4276db180a;5.2;1754899135359;fZRCXZvUwcaF9A7Un4KJAELG0IuE-h-T-h6I-hfZXx-Vuh-T-prJ_YfZB5hW-V_I_I7IuZLU8QLJqhuU8U_J9crUxduJ_gOJ88_VrhOJ-h-T-h6Q1E7J8E6ZBh-f1ZPJAI7V9E_IxV7UAMbV-cLT9c7JpFuV8I7UwN_Uph_IrZfZnZvFAI6GAU7ZBh-f1ZfV-h-T-ROE-YfZB5hW-h_WvpPUrkMI187ICMeH-h-T-J6ZBh-f1ZPHuA7VAELW4Y8ZB5_Z0kbIzc7F-hfZXx-ZvV_G4E8ZB5_Z7g6ZBh-f1taZB5BZ2I9ZB5_ZudOE-YfZBhfZXx-VB5_ZwdOE-YfZBhfZXxfUwh-T-hOVsY7ZBhfZB5hWptfZnZ-VwN6J-hfZBh-f1ZfHQYeVrUrKvROEvEsM-h-T-trG9oLJvYfZBhfZXxfVB5_ZpN6J-hfZBh-f1heZnZvUsY7ZBhfZB5hWsdeZnZ-UsY7ZBhfZB5hWxh-T-NOE-YfZBhfZXxfVB5_ZtN6J-hfZBh-f1ZuVwh-T-VOE-YfZBhfZXx-ZspPVzh_ZB5_ZwN6J-hfZBh-f1heZnZvHqYfZBhfZXxPUB5_Zuw7ZBhfZB5hWxh-T-x7ZBhfZB5hWxh-T-RrE-hfZBh-fmg-T-R7G8QaD8YfZB5hWkgfZXZPT7Y_UuV7J8IbV7MLUCQ7H-h-T-ZeF-hfZBh-fmg-T-haF-hfZXx-ZtJeDB1eUrpLHKgvTxpfVwhfMTgvFqkbIz8rM-h-T-dLEuYfZB5xD;b0778249b5bfc6c62113baf7f8524a72;eVxh989Gy8bE_oLE7wPD9k7J1RLHxgKJ"
    fixed_x_api_eid_token = "jdd035N4KGSIGGXI35AIENP3MXLOIFSFSLNQKJDKPWIXPH4QKLTDPFU5ZVAPTZXHVJR6T4ZWKMXKLBMJLBCCQHZI7XQWYZEAAAAMYTAHEQUYAAAAADL7HVMF65J62DAX"
    fixed_uuid = "17547806811201466311166"

    base_url = "https://api.m.jd.com/api"
    params = {
        'functionId': 'unionPromoteLinkService', 'appid': 'unionpc', '_': str(int(time.time() * 1000)),
        'loginType': '3', 'uuid': fixed_uuid, 'x-api-eid-token': fixed_x_api_eid_token, 'h5st': fixed_h5st,
    }
    api_url = f"{base_url}?{urlencode(params)}"

    headers = {
        "Cookie": fixed_cookie, 'accept': 'application/json, text/plain, */*',
        'content-type': 'application/x-www-form-urlencoded', 'origin': 'https://union.jd.com',
        'referer': 'https://union.jd.com/',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
    }

    # 修正1：恢复完整的 payload 参数
    payload_dict = {
        "funName": "getCode",
        "param": {
            "isPinGou": 0, "materialId": materialId, "materialType": 1, "needAutoVerifyPlan": None,
            "planId": 3521164587, "promotionType": 15, "receiveType": "cps", "isSmartGraphics": 0,
            "requestId": None, "command": 1, "ext1": "618|pc|"
        },
        "clientPageId": "jingfen_pc"
    }
    body_json_string = json.dumps(payload_dict, separators=(',', ':'))
    data = {'body': body_json_string}

    try:
        session = requests.Session()
        response = session.post(api_url, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        result_data = response.json()

        # 修正2：增加对API返回码的判断
        if result_data.get('code') == 200:
            print("业务请求成功!")
            print(json.dumps(result_data, indent=2, ensure_ascii=False))
            return result_data
        else:
            print("业务请求失败: 服务器返回错误。")
            print(f"错误码: {result_data.get('code')}, 错误信息: {result_data.get('message')}")
            return None

    except requests.exceptions.HTTPError as e:
        print(f"网络请求失败: HTTP错误 {e.response.status_code}")
        print("这极有可能是因为函数内固定的Cookie或h5st等参数已过期。")
        print("服务器响应内容:", e.response.text)
        return None
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None

# 使用示例
if __name__ == '__main__':
    # MY_SITE_ID = "4101771077"  # 替换为你自己的网站/APP/流量媒体ID
    # MY_SUB_UNION_ID = "618_18_c35***e6a"  # 替换为你自己的subUnionId
    #
    # product_url = "jingfen.jd.com/detail/JCiNsS4Euw3Qbo2ZKGdGFPdq_3JKSxG35zuXsO7IWqj.html" # 示例商品URL
    #
    # promotion_data = get_promotion_link(product_url, MY_SITE_ID, MY_SUB_UNION_ID)
    #
    # if promotion_data:
    #     print(f"【推广链接】: {promotion_data.get('click_url')}")
    #     print(f"【京口令】: {promotion_data.get('kouling')}")
    # else:
    #     print("获取推广链接失败。")


    # goods_request_params = {
    #     'keyword': '牛肉',  # 查询关键词
    #     'pageSize': 5,  # 每页数量
    #     'pageIndex': 1,  # 页码
    #     'isCoupon': 1,  # 只查询有优惠券的商品
    #     'sortName': 'inOrderCount30Days',  # 按30天引单量排序
    #     'sort': 'desc',  # 降序
    #     'sceneId': 1  # 场景ID，1-常规商品，2-京东链接/ID/长链/短链等
    # }
    #
    # print("正在查询商品信息...")
    # goods_list = query_goods(goods_request_params)
    # print(goods_list)


    my_material_id = 100050906852
    print("--- 开始调用修正后的函数 ---")
    promo_data = get_jd_promo_link(materialId=my_material_id)

    if promo_data:
        print("\n--- 函数调用成功，并获得有效数据 ---")
    else:
        print("\n--- 函数调用失败 ---")