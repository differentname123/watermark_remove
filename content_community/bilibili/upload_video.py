import requests
import base64
import os
import math
import json
import time
import urllib.parse

from common_utils.common_utils import get_config

# --- 配置参数 ---
SESSDATA = get_config("bilibili_sessdata_cookie")  # 必需。你的B站登录会话 SESSDATA cookie 值。
BILI_JCT = get_config("bilibili_csrf_token")

# 视频和封面文件路径
COVER_IMAGE_PATH = "inpainted_image.jpg"
VIDEO_FILE_PATH = "test1_inpainted_with_audio.mp4"

# 视频投稿信息
VIDEO_TITLE = "我的AI修复视频与精彩瞬间"
VIDEO_DESCRIPTION = "这是一个使用AI技术修复后的视频，并加入了有趣的音频，希望大家喜欢！"
VIDEO_TAGS = "AI修复,视频剪辑,有趣,科技,日常生活"  # 多个标签用英文逗号分隔，最多10个
VIDEO_COPYRIGHT = 1  # 1: 自制, 2: 转载
VIDEO_TID = 21  # 分区 ID，例如：21-日常, 122-野生技能协会, 65-网络游戏
VIDEO_RECREATE = -1  # -1: 允许二创(默认), 1: 不允许
VIDEO_DYNAMIC = "我的第一个B站投稿，希望大家喜欢！"  # 粉丝动态
VIDEO_NO_REPRINT = 1  # 1: 允许转载, 0: 不允许

# --- 常量 ---
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}


def get_session():
    """创建一个requests session并设置cookie"""
    session = requests.Session()
    session.headers.update(HEADERS)
    session.cookies.set("SESSDATA", SESSDATA)
    session.cookies.set("bili_jct", BILI_JCT)
    return session


session = get_session()


def upload_cover(image_path: str) -> str:
    """
    上传视频封面
    :param image_path: 封面图片文件路径
    :return: 封面图片的 URL
    """
    print(f"--- 1. 上传封面: {image_path} ---")
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Base64 编码图片数据
        encoded_image = base64.b64encode(image_data).decode("utf-8")
        cover_base64 = f"data:image/jpeg;base64,{encoded_image}"

        url = "https://member.bilibili.com/x/vu/web/cover/up"
        params = {
            "ts": int(time.time() * 1000)
        }
        data = {
            "csrf": BILI_JCT,
            "cover": cover_base64
        }

        response = session.post(url, params=params, data=data)
        response.raise_for_status()  # 检查HTTP响应状态码

        result = response.json()
        if result["code"] == 0:
            cover_url = result["data"]["url"]
            print(f"封面上传成功，URL: {cover_url}")
            return cover_url
        else:
            print(f"封面上传失败，错误码: {result['code']}, 信息: {result['message']}")
            raise Exception(f"封面上传失败: {result['message']}")

    except Exception as e:
        print(f"上传封面发生异常: {e}")
        raise


def pre_upload_video(video_path: str) -> dict:
    """
    获取上传视频的元数据 (预上传)
    :param video_path: 视频文件路径
    :return: 包含上传认证信息、分块大小等信息的字典
    """
    print(f"--- 2. 预上传视频: {video_path} ---")
    video_filename = os.path.basename(video_path)
    video_file_size = os.path.getsize(video_path)

    url = "https://member.bilibili.com/preupload"
    params = {
        "name": video_filename,
        "r": "upos",
        "profile": "ugcfx/bup",  # 普通视频
        "size": video_file_size,
        "probe_version": "20221109",  # 额外参数，文档有提到
        "webVersion": "2.13.0",  # 额外参数，文档有提到
    }

    try:
        response = session.get(url, params=params)
        response.raise_for_status()

        result = response.json()
        if result.get("OK") == 1:
            print(f"预上传成功，biz_id: {result['biz_id']}, endpoint: {result['endpoint']}")
            return result
        else:
            print(f"预上传失败，响应: {result}")
            raise Exception(f"预上传失败: {result}")

    except Exception as e:
        print(f"预上传发生异常: {e}")
        raise


def post_video_meta(preupload_data: dict) -> dict:
    """
    上传视频元数据
    :param preupload_data: 预上传返回的数据
    :return: 包含 upload_id 的字典
    """
    print("--- 3. 上传视频元数据 ---")

    endpoint = preupload_data["endpoint"]
    upos_uri = preupload_data["upos_uri"]
    auth = preupload_data["auth"]
    chunk_size = preupload_data["chunk_size"]
    biz_id = preupload_data["biz_id"]

    # 构造完整上传URL
    upload_url = f"https:{endpoint}{upos_uri.replace('upos:/', '')}"

    params = {
        "uploads": "",  # 必须为空字符串
        "output": "json",
        "profile": "ugcfx/bup",
        "filesize": os.path.getsize(VIDEO_FILE_PATH),
        "partsize": chunk_size,
        "biz_id": biz_id,
    }

    headers_with_auth = {
        "X-Upos-Auth": auth,
        "Content-Length": "0",  # POST请求体为空，Content-Length为0
    }

    try:
        response = session.post(upload_url, params=params, headers=headers_with_auth)
        response.raise_for_status()

        result = response.json()
        if result.get("OK") == 1:
            print(f"视频元数据上传成功，upload_id: {result['upload_id']}")
            return result
        else:
            print(f"视频元数据上传失败，响应: {result}")
            raise Exception(f"视频元数据上传失败: {result}")

    except Exception as e:
        print(f"上传视频元数据发生异常: {e}")
        raise


def upload_video_chunks(video_path: str, preupload_data: dict, post_meta_data: dict) -> list:
    """
    分片上传视频文件
    :param video_path: 视频文件路径
    :param preupload_data: 预上传返回的数据
    :param post_meta_data: 上传元数据返回的数据
    :return: 已上传分块的列表，包含partNumber和eTag
    """
    print("--- 4. 分片上传视频文件 ---")

    endpoint = preupload_data["endpoint"]
    upos_uri = preupload_data["upos_uri"]
    auth = preupload_data["auth"]
    chunk_size = preupload_data["chunk_size"]  # 每次上传的分块大小
    upload_id = post_meta_data["upload_id"]

    video_file_size = os.path.getsize(video_path)
    total_chunks = math.ceil(video_file_size / chunk_size)

    # 构造完整上传URL (与post_video_meta相同的基础URL)
    upload_url_base = f"https:{endpoint}{upos_uri.replace('upos:/', '')}"

    uploaded_parts = []

    try:
        with open(video_path, "rb") as f:
            for i in range(total_chunks):
                part_number = i + 1  # 从1开始
                chunk_data = f.read(chunk_size)
                current_chunk_size = len(chunk_data)

                start_byte = i * chunk_size
                end_byte = start_byte + current_chunk_size

                params = {
                    "partNumber": part_number,
                    "uploadId": upload_id,
                    "chunk": i,  # 从0开始
                    "chunks": total_chunks,
                    "size": current_chunk_size,
                    "start": start_byte,
                    "end": end_byte,
                    "total": video_file_size,
                }

                headers_with_auth = {
                    "X-Upos-Auth": auth,
                    "Content-Type": "application/octet-stream",
                }

                print(f"  - 上传分块 {part_number}/{total_chunks}, 大小: {current_chunk_size} 字节...")
                response = session.put(upload_url_base, params=params, headers=headers_with_auth, data=chunk_data)
                response.raise_for_status()

                # Bilibili此处返回的是纯文本 "MULTIPART_PUT_SUCCESS"
                # 文档中的eTag示例值是"etag"，所以我们也用这个占位符
                if response.text.strip() == "MULTIPART_PUT_SUCCESS":
                    uploaded_parts.append({
                        "partNumber": part_number,
                        "eTag": "etag"  # Bilibili似乎不返回实际eTag，用文档示例的占位符
                    })
                    print(f"    分块 {part_number} 上传成功。")
                else:
                    print(f"    分块 {part_number} 上传失败，响应: {response.text}")
                    raise Exception(f"分块上传失败: {response.text}")

        print(f"所有 {len(uploaded_parts)} 个分块上传完成。")
        return uploaded_parts

    except Exception as e:
        print(f"分片上传视频文件发生异常: {e}")
        raise


def end_upload_video(preupload_data: dict, post_meta_data: dict, uploaded_parts: list) -> dict:
    """
    结束上传视频文件
    :param preupload_data: 预上传返回的数据
    :param post_meta_data: 上传元数据返回的数据
    :param uploaded_parts: 已上传分块的列表
    :return: 结束上传的响应数据
    """
    print("--- 5. 结束上传视频文件 ---")

    endpoint = preupload_data["endpoint"]
    upos_uri = preupload_data["upos_uri"]
    auth = preupload_data["auth"]
    biz_id = preupload_data["biz_id"]
    upload_id = post_meta_data["upload_id"]
    video_filename = os.path.basename(VIDEO_FILE_PATH)

    # 构造完整上传URL (与post_video_meta和upload_video_chunks相同的基础URL)
    upload_url = f"https:{endpoint}{upos_uri.replace('upos:/', '')}"

    params = {
        "output": "json",
        "name": video_filename,
        "profile": "ugcfx/bup",
        "uploadId": upload_id,
        "biz_id": biz_id,
    }

    headers_with_auth = {
        "X-Upos-Auth": auth,
        "Content-Type": "application/json",
    }

    # 构造请求体
    body = {
        "parts": uploaded_parts
    }

    try:
        response = session.post(upload_url, params=params, headers=headers_with_auth, json=body)
        response.raise_for_status()

        result = response.json()
        if result.get("OK") == 1:
            print(f"结束上传成功，文件位置: {result['location']}")
            return result
        else:
            print(f"结束上传失败，响应: {result}")
            raise Exception(f"结束上传失败: {result}")

    except Exception as e:
        print(f"结束上传视频文件发生异常: {e}")
        raise


def submit_video_post(
        cover_url: str,
        preupload_data: dict,
        video_filename: str,
        video_cid: int,
        title: str,
        description: str,
        tags: str,
        copyright_type: int,
        tid: int,
        recreate: int,
        dynamic: str,
        no_reprint: int
) -> dict:
    """
    投递视频稿件
    :param cover_url: 视频封面 URL
    :param preupload_data: 预上传返回的数据 (用于获取 biz_id 作为 cid)
    :param video_filename: 视频文件名 (从预上传 upos_uri 中提取的 filename)
    :param video_cid: 视频的 biz_id (从预上传返回)
    :param title: 视频标题
    :param description: 视频简介
    :param tags: 视频标签
    :param copyright_type: 1: 自制, 2: 转载
    :param tid: 分类 ID
    :param recreate: 是否允许二创 (-1: 允许, 1: 不允许)
    :param dynamic: 粉丝动态
    :param no_reprint: 是否允许转载 (1: 允许, 0: 不允许)
    :return: 投稿成功的 aid 和 bvid
    """
    print("--- 6. 投递视频稿件 ---")

    url = "https://member.bilibili.com/x/vu/web/add/v3"
    params = {
        "ts": int(time.time() * 1000),
        "csrf": BILI_JCT,
    }

    # 构造视频文件信息
    videos_info = [
        {
            "filename": video_filename,
            "title": "P1",  # 默认为P1，如果是多P视频需要根据需求修改
            "desc": "",
            "cid": video_cid,
        }
    ]

    # 构造请求体
    payload = {
        "videos": videos_info,
        "cover": cover_url,
        "cover43": "",  # 文档说明可为空
        "title": title,
        "copyright": copyright_type,
        "tid": tid,
        "tag": tags,
        "desc_format_id": 9999,  # 纯文本
        "desc": description,
        "recreate": recreate,
        "dynamic": dynamic,
        "interactive": 0,
        "act_reserve_create": 0,
        "no_disturbance": 0,
        "no_reprint": no_reprint,
        "subtitle": {
            "open": 0,  # 0: 启用字幕投稿(默认), 1: 不启用
            "lan": "",  # 字幕投稿语言，可为空
        },
        "dolby": 0,  # 杜比音效 (0: 否, 1: 是)
        "lossless_music": 0,  # 无损音乐 (0: 否, 1: 是)
        "up_selection_reply": False,  # 精选评论
        "up_close_reply": False,  # 关闭评论
        "up_close_danmu": False,  # 关闭弹幕
        "web_os": 3,  # 平台类型
        "csrf": BILI_JCT,  # 文档中说明需要在 JSON 体中也包含
    }

    headers = {
        "Content-Type": "application/json; charset=utf-8",
    }

    try:
        response = session.post(url, params=params, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        if result["code"] == 0:
            aid = result["data"]["aid"]
            bvid = result["data"]["bvid"]
            print(f"稿件投递成功！AID: {aid}, BVID: {bvid}")
            print(f"请前往：https://www.bilibili.com/video/{bvid} 查看")
            return {"aid": aid, "bvid": bvid}
        else:
            print(f"稿件投递失败，错误码: {result['code']}, 信息: {result['message']}")
            raise Exception(f"稿件投递失败: {result['message']}")

    except Exception as e:
        print(f"投递视频稿件发生异常: {e}")
        raise


# --- 主执行流程 ---
def main():
    if not SESSDATA or SESSDATA == "YOUR_SESSDATA_HERE" or \
            not BILI_JCT or BILI_JCT == "YOUR_BILI_JCT_HERE":
        print("错误: 请在脚本顶部配置 SESSDATA 和 BILI_JCT。")
        return

    if not os.path.exists(COVER_IMAGE_PATH):
        print(f"错误: 封面图片文件 '{COVER_IMAGE_PATH}' 不存在。")
        return
    if not os.path.exists(VIDEO_FILE_PATH):
        print(f"错误: 视频文件 '{VIDEO_FILE_PATH}' 不存在。")
        return

    try:
        # 1. 上传封面
        cover_url = upload_cover(COVER_IMAGE_PATH)

        # 2. 获取上传元数据 (预上传)
        preupload_data = pre_upload_video(VIDEO_FILE_PATH)
        # --- 关键修改 START ---
        full_filename_with_ext = preupload_data["upos_uri"].split('/')[-1]
        video_filename_from_upos = os.path.splitext(full_filename_with_ext)[0] # 去掉文件后缀名
        # --- 关键修改 END ---
        video_biz_id = preupload_data["biz_id"]

        # 3. 上传视频元数据
        post_meta_data = post_video_meta(preupload_data)

        # 4. 分片上传视频文件
        uploaded_parts = upload_video_chunks(VIDEO_FILE_PATH, preupload_data, post_meta_data)

        # 5. 结束上传视频文件
        end_upload_video(preupload_data, post_meta_data, uploaded_parts)

        # 6. 投递视频稿件
        submit_video_post(
            cover_url=cover_url,
            preupload_data=preupload_data,
            video_filename=video_filename_from_upos, # 使用无后缀名的文件名
            video_cid=video_biz_id,
            title=VIDEO_TITLE,
            description=VIDEO_DESCRIPTION,
            tags=VIDEO_TAGS,
            copyright_type=VIDEO_COPYRIGHT,
            tid=VIDEO_TID,
            recreate=VIDEO_RECREATE,
            dynamic=VIDEO_DYNAMIC,
            no_reprint=VIDEO_NO_REPRINT
        )
        print("\n所有步骤完成，视频已成功上传并投递！")

    except Exception as e:
        print(f"\n视频上传或投递过程中发生致命错误: {e}")


if __name__ == "__main__":
    main()