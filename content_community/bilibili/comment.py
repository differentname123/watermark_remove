# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/5/26 22:51
:last_date:
    2025/5/26 22:51
:description:
    
"""
import requests

from common_utils.common_utils import get_config

# --- 需要您填写的参数 ---
oid = "BV18e7pzgE49"  # 例如: "860478472" (这是一个示例oid)
type_code = 1  # 目标类型，1 一般代表视频
message_content = "这个视频质量真高，特别喜欢"


csrf_token = get_config("bilibili_csrf_token")
sessdata_cookie = get_config("bilibili_sessdata_cookie")

# --- API端点 ---
url = "https://api.bilibili.com/x/v2/reply/add"

# --- 请求头，包含Cookie ---
headers = {
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Cookie": f"SESSDATA={sessdata_cookie}; bili_jct={csrf_token}",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36" # 模拟浏览器
}

# --- POST请求的数据 ---
data = {
    "oid": oid,
    "type": type_code,
    "message": message_content,
    "plat": 1,  # 通常 Web 端为 1
    "jsonp": "jsonp",
    "csrf": csrf_token
}

# --- 发送POST请求 ---
try:
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()  # 如果请求失败 (状态码 4xx 或 5xx), 会抛出异常
    result = response.json()

    # --- 处理返回结果 ---
    if result.get("code") == 0:
        print("评论发送成功！")
        if result.get("data") and result["data"].get("reply"):
            print(f"评论ID (rpid): {result['data']['reply']['rpid']}")
    else:
        print(f"评论发送失败，错误码：{result.get('code')}, 错误信息：{result.get('message')}")
        # 可以根据错误码进一步排查问题，具体错误码含义可以参考bilibili-API-collect文档 [6]
except requests.exceptions.RequestException as e:
    print(f"请求发生错误：{e}")
except Exception as e:
    print(f"发生未知错误：{e}")