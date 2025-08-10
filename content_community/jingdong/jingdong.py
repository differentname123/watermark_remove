import requests
import json
import hashlib
from datetime import datetime

# --- 1. 配置您的应用信息 ---
# 警告：请勿将App Key和App Secret硬编码在生产代码中，建议从安全的环境变量或配置中心获取。
APP_KEY = "a8b099d7d7c2c1c4802a131725f81b2f"
APP_SECRET = "b58181748de34f73a37a74c2a803512d"

# API网关地址
API_URL = "https://api.jd.com/routerjson"


def call_jd_union_api(method, app_key, app_secret, business_params):
    """
    调用京东联盟开放平台的通用函数。

    :param method: API接口名称, e.g., 'jd.union.open.goods.jingfen.query'
    :param app_key: 你的App Key
    :param app_secret: 你的App Secret
    :param business_params: 业务参数字典 (注意：这里传入的是 goodsReq 内部的参数，函数会自动包装)
    :return: API响应的JSON数据
    """
    # --- 2. 准备系统参数 ---
    sys_params = {
        'method': method,
        'app_key': app_key,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'format': 'json',
        'v': '1.0',
        'sign_method': 'md5',
    }

    # --- 3. 准备完整的请求参数（系统参数 + 业务参数） ---
    # 业务参数需要包装在'goodsReq'下，并转换为JSON字符串
    # 根据文档，业务参数最外层参数名叫 360buy_param_json
    # 而对于 jd.union.open.goods.jingfen.query 接口，其入参是 goodsReq
    # 所以最终的json结构是 {"goodsReq": {...}}
    full_business_params = {'goodsReq': business_params}

    # 使用 separators 去除JSON字符串中的空格，确保签名一致性
    json_business_params = json.dumps(full_business_params, separators=(',', ':'))

    # 将所有需要签名的参数放入一个字典
    params_to_sign = sys_params.copy()
    params_to_sign['360buy_param_json'] = json_business_params

    # --- 4. 生成签名 ---
    # 1) 将所有请求参数名按照字母先后顺序排列
    sorted_keys = sorted(params_to_sign.keys())

    # 2) 把所有参数名和参数值进行拼接
    sign_str_parts = []
    for key in sorted_keys:
        sign_str_parts.append(key + str(params_to_sign[key]))
    concatenated_str = "".join(sign_str_parts)

    # 3) 把appSecret夹在字符串的两端
    string_to_sign = app_secret + concatenated_str + app_secret

    # 4) 使用MD5进行加密，再转化成大写
    m = hashlib.md5()
    m.update(string_to_sign.encode('utf-8'))
    sign = m.hexdigest().upper()

    # --- 5. 组装最终的HTTP请求参数 ---
    final_request_params = params_to_sign.copy()
    final_request_params['sign'] = sign

    # --- 6. 发起HTTP POST请求 ---
    try:
        response = requests.post(API_URL, data=final_request_params, timeout=10)
        response.raise_for_status()  # 如果请求失败(状态码非2xx)，则抛出HTTPError异常
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求API时发生错误: {e}")
        return None

import requests
import json
import hashlib
from datetime import datetime

# --- 1. 配置您的应用信息 ---
# 警告：请勿将App Key和App Secret硬编码在生产代码中，建议从安全的环境变量或配置中心获取。
APP_KEY = "a8b099d7d7c2c1c4802a131725f81b2f"  # 请替换为您自己的真实app_key
APP_SECRET = "b58181748de34f73a37a74c2a803512d"  # 请替换为您自己的真实app_secret

# API网关地址
API_URL = "https://api.jd.com/routerjson"


def call_jd_union_api(method, app_key, app_secret, business_params_wrapper):
    """
    调用京东联盟开放平台的通用函数。
    这是一个底层函数，负责签名和发送请求。

    :param method: API接口名称, e.g., 'jd.union.open.goods.jingfen.query'
    :param app_key: 你的App Key
    :param app_secret: 你的App Secret
    :param business_params_wrapper: 已经按接口要求包装好的业务参数字典, e.g., {'goodsReq': {...}}
    :return: API响应的JSON数据
    """
    # --- 2. 准备系统参数 ---
    sys_params = {
        'method': method,
        'app_key': app_key,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'format': 'json',
        'v': '1.0',
        'sign_method': 'md5',
    }

    # --- 3. 准备完整的请求参数（系统参数 + 业务参数） ---
    # 将业务参数转换为JSON字符串
    json_business_params = json.dumps(business_params_wrapper, separators=(',', ':'))

    # 将所有需要签名的参数放入一个字典
    params_to_sign = sys_params.copy()
    params_to_sign['360buy_param_json'] = json_business_params

    # --- 4. 生成签名 ---
    # 1) 将所有请求参数名按照字母先后顺序排列
    sorted_keys = sorted(params_to_sign.keys())

    # 2) 把所有参数名和参数值进行拼接
    concatenated_str = "".join([key + str(params_to_sign[key]) for key in sorted_keys])

    # 3) 把appSecret夹在字符串的两端
    string_to_sign = app_secret + concatenated_str + app_secret

    # 4) 使用MD5进行加密，再转化成大写
    m = hashlib.md5()
    m.update(string_to_sign.encode('utf-8'))
    sign = m.hexdigest().upper()

    # --- 5. 组装最终的HTTP请求参数 ---
    final_request_params = params_to_sign.copy()
    final_request_params['sign'] = sign

    # --- 6. 发起HTTP POST请求 ---
    try:
        response = requests.post(API_URL, data=final_request_params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求API时发生错误: {e}")
        return None


# ==============================================================================
#  新的、专门用于生成推广链接的函数
# ==============================================================================
def get_promotion_link(material_id, site_id, **kwargs):
    """
    调用 jd.union.open.promotion.common.get 接口，生成推广链接和口令。

    :param material_id: (必填) 推广物料url，如商品链接。
    :param site_id: (必填) 您的网站ID/APP ID。
    :param kwargs: (可选) 其他业务参数，如 positionId, subUnionId, couponUrl等。
    :return: 包含推广链接和口令的字典，或在失败时返回None。
    """
    api_method = 'jd.union.open.promotion.common.get'

    promotion_code_req = {
        'materialId': material_id,
        'siteId': site_id,
        'sceneId': 1,
        'command': 1,
    }
    promotion_code_req.update(kwargs)
    business_params_wrapper = {'promotionCodeReq': promotion_code_req}

    print(f"\n正在调用转链接口: {api_method}")
    print(f"业务参数详情: {json.dumps(business_params_wrapper, indent=2, ensure_ascii=False)}")

    result = call_jd_union_api(api_method, APP_KEY, APP_SECRET, business_params_wrapper)

    if not result:
        return None  # 请求失败

    # 解析结果
    try:
        # 检查是否有京东层面的错误返回
        if 'error_response' in result:
            print("API返回错误:", result['error_response'])
            return None

        # 【关键修改】兼容拼写错误的 'responce' 和正确的 'response'
        response_key = 'jd_union_open_promotion_common_get_response'
        if response_key not in result:
            response_key = 'jd_union_open_promotion_common_get_responce'  # 使用错误的拼写

        # 检查我们找到的键是否存在
        if response_key not in result:
            raise KeyError("无法在API响应中找到 'response' 或 'responce' 键。")

        # 获取包含业务数据的部分
        response_data = result[response_key]

        # 检查业务数据是否被包裹在 'getResult' JSON字符串中
        if 'getResult' in response_data:
            res_str = response_data['getResult']
            real_result = json.loads(res_str)  # 解析这个字符串

            # 检查业务逻辑层面的返回码
            if real_result.get('code') != 200:
                print("API业务逻辑错误:", real_result)
                return None
            data = real_result.get('data', {})  # 使用 .get 避免 data 不存在时报错
        else:
            # 如果数据不在 'getResult' 中，则直接从 'result' 键获取
            data = response_data.get('result', {}).get('data', {})

        click_url = data.get('clickURL')
        j_command_info = data.get('jCommandInfo', {})
        kouling = j_command_info.get('command')

        # 即使没有口令，也认为是成功
        if not click_url:
            print("成功调用API，但未能从返回数据中提取到 clickURL。")
            print("解析出的数据部分(data):", data)

        return {
            'click_url': click_url,
            'kouling': kouling,
            'raw_response': result  # 同时返回原始响应，方便调试
        }
    except (KeyError, TypeError, IndexError, json.JSONDecodeError) as e:
        print(f"解析推广链接和口令失败。错误: {e}")
        print("原始响应:", result)
        return None


def get_promotion_link_by_unionid(material_id, union_id, **kwargs):
    """
    调用 jd.union.open.promotion.byunionid.get 接口，生成推广链接和口令。

    :param material_id: (必填) 推广物料url，如商品链接。
    :param union_id: (必填) 您的联盟ID (UnionId)，是一个纯数字ID。
    :param kwargs: (可选) 其他业务参数，如 positionId, subUnionId, couponUrl等。
    :return: 包含推广链接和口令的字典，或在失败时返回None。
    """
    api_method = 'jd.union.open.promotion.byunionid.get'

    # 准备 promotionCodeReq 参数
    promotion_code_req = {
        'materialId': material_id,
        'unionId': union_id,
        'sceneId': 1,  # 默认为场景1
        'command': 1,  # 明确要求生成口令
    }

    # 将其他可选参数更新到 promotion_code_req 中
    promotion_code_req.update(kwargs)

    # 按接口要求，将所有业务参数包装在 'promotionCodeReq' key下
    business_params_wrapper = {'promotionCodeReq': promotion_code_req}

    print(f"\n正在调用新接口: {api_method}")
    print(f"业务参数详情: {json.dumps(business_params_wrapper, indent=2, ensure_ascii=False)}")

    result = call_jd_union_api(api_method, APP_KEY, APP_SECRET, business_params_wrapper)

    if not result:
        return None

    # 解析结果 (与上一个函数类似，但需要适配新的接口名)
    try:
        if 'error_response' in result:
            print("API返回错误:", result['error_response'])
            return None

        # 兼容拼写错误的 'responce' 和正确的 'response'
        response_key = 'jd_union_open_promotion_byunionid_get_response'
        if response_key not in result:
            response_key = 'jd_union_open_promotion_byunionid_get_responce'

        if response_key not in result:
            raise KeyError(f"无法在API响应中找到 {response_key} 或其变体。")

        response_data = result[response_key]

        if 'getResult' in response_data:
            res_str = response_data['getResult']
            real_result = json.loads(res_str)
            if real_result.get('code') != 200:
                print("API业务逻辑错误:", real_result)
                return None
            data = real_result.get('data', {})
        else:
            data = response_data.get('result', {}).get('data', {})

        click_url = data.get('clickURL')
        j_command_info = data.get('jCommandInfo', {})
        kouling = j_command_info.get('command')

        if not click_url:
            print("成功调用API，但未能从返回数据中提取到 clickURL。")
            print("解析出的数据部分(data):", data)

        return {
            'click_url': click_url,
            'kouling': kouling,
            'raw_response': result
        }
    except (KeyError, TypeError, IndexError, json.JSONDecodeError) as e:
        print(f"解析推广链接和口令失败。错误: {e}")
        print("原始响应:", result)
        return None



# --- 使用示例 ---
if __name__ == '__main__':
    # 【重要】请在这里替换成您自己的推广位ID (Site ID)
    # 入口：京东联盟-推广管理-网站管理/APP管理/流量媒体管理-查看网站ID/APP ID
    MY_SITE_ID = "4101771077"

    # 检查配置
    if "YOUR_" in APP_KEY or "YOUR_" in APP_SECRET or "YOUR_" in MY_SITE_ID:
        print("错误：请在代码顶部或 `if __name__ == '__main__':` 部分")
        print("      将 APP_KEY, APP_SECRET, 和 MY_SITE_ID 替换为您自己的真实信息！")
    else:
        # --- 场景1: 普通商品转链，生成链接和口令 ---
        print("\n" + "=" * 20 + " 场景1: 普通商品转链 " + "=" * 20)
        product_url = "jingfen.jd.com/detail/R0kUhUY1WCvQrmzLJVs1SGym_3OPKaCs8gWfHPVYdhG.html"  # 示例商品URL

        promotion_data = get_promotion_link(material_id=product_url, site_id=MY_SITE_ID)

        if promotion_data:
            print("\n--- 推广信息提取成功 ---")
            print(f"【推广链接】: {promotion_data.get('click_url')}")
            print(f"【京 口 令】: {promotion_data.get('kouling')}")
            print("--------------------------")
        #
        # # --- 场景2: 商品和优惠券二合一转链 ---
        # print("\n" + "=" * 20 + " 场景2: 二合一转链 " + "=" * 20)
        # product_url_with_coupon = "https://item.jd.com/100078298285.html"  # 另一个示例商品
        # coupon_url = "https://coupon.m.jd.com/coupons/show.action?key=ca99x9g7t1o14d57b066b535d4f6b245&roleId=59963665"  # 示例优惠券链接
        #
        # # 通过 **kwargs 传入额外参数 couponUrl
        # promotion_data_2in1 = get_promotion_link(
        #     material_id=product_url_with_coupon,
        #     site_id=MY_SITE_ID,
        #     couponUrl=coupon_url
        # )
        #
        # if promotion_data_2in1:
        #     print("\n--- 推广信息提取成功 ---")
        #     print(f"【推广链接】: {promotion_data_2in1.get('click_url')}")
        #     print(f"【京 口 令】: {promotion_data_2in1.get('kouling')}")
        #     print("--------------------------")
        #
        # # --- 场景3: 转链并附加自定义跟踪参数 ---
        # print("\n" + "=" * 20 + " 场景3: 附加自定义参数 " + "=" * 20)
        # product_url_for_tracking = "https://item.jd.com/10082699380727.html"
        # my_custom_tracker = "user_123_abc"  # 自定义参数，用于订单追踪
        #
        # # 通过 **kwargs 传入额外参数 subUnionId
        # promotion_data_tracked = get_promotion_link(
        #     material_id=product_url_for_tracking,
        #     site_id=MY_SITE_ID,
        #     subUnionId=my_custom_tracker
        # )
        #
        # if promotion_data_tracked:
        #     print("\n--- 推广信息提取成功 ---")
        #     print(f"【推广链接】: {promotion_data_tracked.get('click_url')}")
        #     print(f"【京 口 令】: {promotion_data_tracked.get('kouling')}")
        #     print(f"【自定义参数】: {my_custom_tracker} (将在订单报表中体现)")
        #     print("--------------------------")

# --- 使用示例 ---
if __name__ == '__main__':
    # 检查是否已填写 App Key 和 App Secret
    if APP_KEY == "在此处替换为您自己的app_key" or APP_SECRET == "在此处替换为您自己的app_secret":
        print("请先在代码中填写您的 APP_KEY 和 APP_SECRET。")
    else:
        # 定义要调用的接口方法
        api_method = 'jd.union.open.goods.jingfen.query'

        # 定义业务参数 (goodsReq 内部的参数)
        # 例如：查询 "实时热销榜" (eliteId=22) 的前5个商品
        jingfen_goods_params = {
            'eliteId': 22,  # 频道ID: 22-实时热销榜
            'pageSize': 5,  # 每页数量
            'pageIndex': 1  # 页码
        }

        print(f"正在调用接口: {api_method}")
        print(f"业务参数: {jingfen_goods_params}")

        # 调用函数并发起请求
        result = call_jd_union_api(api_method, APP_KEY, APP_SECRET, jingfen_goods_params)

        # --- 7. 解析和打印结果 ---
        if result:
            # 使用json.dumps美化输出
            pretty_result = json.dumps(result, indent=4, ensure_ascii=False)
            print("\nAPI响应结果:")
            print(pretty_result)