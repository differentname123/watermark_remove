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