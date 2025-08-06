import requests
import time


def get_bilibili_comments(bvid: str):
    """
    根据 Bilibili BV号获取视频评论。

    参数：
      bvid: 视频对应的 BV号（例如 "BV1ecMnzQEUX"）

    返回：
      如果成功，返回评论列表（列表中的每个元素为单条评论的字典数据）；
      如果没有评论或者发生错误，则返回 None。
    """
    # 模拟浏览器请求头
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/91.0.4472.124 Safari/537.36"),
        "Referer": f"https://www.bilibili.com/video/{bvid}"
    }

    # --- 1. 通过 bvid 获取视频的 aid ---
    view_url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    try:
        view_response = requests.get(view_url, headers=headers)
        view_response.raise_for_status()  # 捕获 HTTP 错误
        view_data = view_response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while fetching video info: {http_err}")
        print(f"Response: {view_response.content}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"请求视频信息时发生错误: {e}")
        return None
    except Exception as e:
        print(f"解析视频信息时发生错误: {e}")
        return None

    if view_data.get("code") != 0 or not view_data.get("data"):
        print(f"获取 aid 失败: {view_data.get('message', '未知错误')}")
        print(f"完整返回数据: {view_data}")
        return None

    aid = view_data["data"]["aid"]
    print(f"获取到视频 aid: {aid}")

    # --- 2. 根据 aid 获取评论 ---
    type_code = 1  # 具体含义参考 API 文档
    sort_mode = 3  # 具体含义参考 API 文档
    reply_url = "https://api.bilibili.com/x/v2/reply/main"
    params = {
        "oid": aid,
        "type": type_code,
        "mode": sort_mode,
        "next": 0,
    }

    try:
        comment_response = requests.get(reply_url, headers=headers, params=params)
        comment_response.raise_for_status()
        comment_data = comment_response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"获取评论时发生 HTTP 错误: {http_err}")
        print(f"Response: {comment_response.content}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"请求评论时发生错误: {e}")
        return None
    except Exception as e:
        print(f"解析评论数据时发生错误: {e}")
        return None

    if comment_data.get("code") != 0:
        print(f"获取评论失败，错误码：{comment_data.get('code')}, 错误信息：{comment_data.get('message')}")
        print(f"完整返回数据: {comment_data}")
        return None

    # --- 3. 解析评论数据 ---
    if comment_data.get("data"):
        replies = comment_data["data"].get("replies") or []
        top_replies = comment_data["data"].get("top_replies") or []

        # 简单合并两个列表，顺序：先 top_replies 后 replies
        merged_comments = top_replies + replies

        return merged_comments if merged_comments else None

    print("该视频可能还没有评论，或者返回的数据结构未能解析。")
    if comment_data.get("data") and comment_data["data"].get("notice"):
        print(f"评论区提示: {comment_data['data']['notice']['content']}")
    return None


if __name__ == "__main__":
    # 示例：输入 BV号 获取对应视频的评论
    bvid = "BV1ZNtAz9E5q"  # 请替换成需要查询的 BV 号
    comments = get_bilibili_comments(bvid)

    if comments:
        print("成功获取评论，下面为评论详情：")
        for i, reply in enumerate(comments):
            print(f"\n评论 #{i + 1}:")
            print(f"  用户: {reply['member']['uname']} (UID: {reply['member']['mid']})")
            print(f"  内容: {reply['content']['message']}")
            print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(reply['ctime']))}")
            print(f"  点赞数: {reply['like']}")
            print(f"  回复数: {reply['count']}")
            if reply.get("replies"):
                print("    --- 楼中楼回复 ---")
                for j, sub_reply in enumerate(reply["replies"]):
                    print(f"      回复 #{j + 1}:")
                    print(f"        用户: {sub_reply['member']['uname']}")
                    print(f"        内容: {sub_reply['content']['message']}")
                    print(f"        点赞数: {sub_reply['like']}")
                print("    ------------------")

        # 如果需要获取分页信息，可查看 comment_data 中的 cursor 字段
    else:
        print("没有获取到评论或者获取评论失败。")