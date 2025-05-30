import requests
import time

# --- 需要您填写的参数 ---
bvid = "BV1Mq7pzMEW3" # 您提供的BV号

# --- 请求头，模拟浏览器 ---
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": f"https://www.bilibili.com/video/{bvid}" # 有时加上 Referer 也有帮助
}

# 1. 通过 bvid 获取 aid
try:
    view_url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    # 在请求中加入 headers
    view_response = requests.get(view_url, headers=headers)
    view_response.raise_for_status() # 这会捕获 4xx 和 5xx 错误
    view_data = view_response.json()
    if view_data.get("code") == 0 and view_data.get("data"):
        oid_val = view_data["data"]["aid"]
        print(f"获取到视频 aid: {oid_val}")
    else:
        print(f"获取 aid 失败: {view_data.get('message', '未知错误，未返回message')}")
        print(f"完整返回数据: {view_data}") # 打印完整返回数据以便调试
        exit()
except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")  # HTTP错误
    print(f"Response content: {view_response.content}") # 打印服务器返回的原始错误内容
    exit()
except requests.exceptions.RequestException as e:
    print(f"请求 aid 时发生错误: {e}")
    exit()
except Exception as e:
    print(f"解析 aid 数据时发生错误: {e}")
    exit()

# --- 后续获取评论的代码保持不变 ---
type_code = 1
sort_mode = 3
reply_url = "https://api.bilibili.com/x/v2/reply/main"

params = {
    "oid": oid_val,
    "type": type_code,
    "mode": sort_mode,
    "next": 0,
}

# --- 发送GET请求获取评论 ---
try:
    response = requests.get(reply_url, headers=headers, params=params) # 复用上面的headers
    response.raise_for_status()
    data = response.json()

    if data.get("code") == 0:
        print(f"成功获取评论 (模式: {sort_mode})")
        if data.get("data") and data["data"].get("replies"):
            replies_list = data["data"]["replies"]
            print(f"\n--- 评论列表 (第 {data.get('data', {}).get('cursor', {}).get('current_pn', 1)} 页) ---")
            for i, reply in enumerate(replies_list):
                print(f"\n评论 #{i+1}:")
                print(f"  用户: {reply['member']['uname']} (UID: {reply['member']['mid']})")
                print(f"  内容: {reply['content']['message']}")
                print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(reply['ctime']))}")
                print(f"  点赞数: {reply['like']}")
                print(f"  回复数: {reply['count']}")
                if reply.get("replies") and len(reply["replies"]) > 0:
                    print("    --- 楼中楼回复 ---")
                    for j, sub_reply in enumerate(reply["replies"]):
                        print(f"      回复 #{j+1}:")
                        print(f"        用户: {sub_reply['member']['uname']}")
                        print(f"        内容: {sub_reply['content']['message']}")
                        print(f"        点赞数: {sub_reply['like']}")
                    print("    ------------------")
            cursor = data["data"].get("cursor", {})
            if not cursor.get("is_end"):
                print(f"\n可以获取下一页，将 next 参数设置为: {cursor.get('next')}")
            else:
                print("\n已经是最后一页评论了。")
        elif data.get("data") and data["data"].get("top_replies"):
            print("\n--- 置顶/热门评论 ---")
            # (处理 top_replies 的逻辑，与上面 replies_list 类似)
        else:
            print("该视频可能还没有评论，或者返回的数据结构未能解析。")
            if data.get("data") and data["data"].get("notice"):
                print(f"评论区提示: {data['data']['notice']['content']}")
    else:
        print(f"获取评论失败，错误码：{data.get('code')}, 错误信息：{data.get('message')}")
        print(f"完整返回数据: {data}")

except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred while fetching comments: {http_err}")
    print(f"Response content: {response.content}")
except requests.exceptions.RequestException as e:
    print(f"请求评论时发生错误：{e}")
except Exception as e:
    print(f"发生未知错误：{e}")
    if 'data' in locals():
        print(f"原始返回数据 (评论): {data}")