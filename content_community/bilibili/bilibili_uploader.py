import json

import requests
import base64
import os
import math
import time
import urllib.parse

from common_utils.common_utils import get_config

# --- 配置参数 ---
SESSDATA = get_config("bilibili_sessdata_cookie")  # 必需。你的B站登录会话 SESSDATA cookie 值。
BILI_JCT = get_config("bilibili_csrf_token")

# 默认投稿设置
DEFAULT_COPYRIGHT = 1       # 1: 自制, 2: 转载
DEFAULT_TID = 21            # 21-日常
DEFAULT_RECREATE = -1       # -1: 允许二创, 1: 不允许
DEFAULT_DYNAMIC = "我的第一个B站投稿，希望大家喜欢！"
DEFAULT_NO_REPRINT = 1     # 1: 允许转载, 0: 不允许

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}


def get_session() -> requests.Session:
    """
    创建并返回带有登录 Cookie 的 requests Session
    """
    sess = requests.Session()
    sess.headers.update(HEADERS)
    sess.cookies.set("SESSDATA", SESSDATA)
    sess.cookies.set("bili_jct", BILI_JCT)
    return sess


def upload_cover(session: requests.Session, image_path: str) -> str:
    """
    上传封面图片，返回封面 URL。
    """
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    cover_base64 = f"data:image/jpeg;base64,{data}"
    resp = session.post(
        "https://member.bilibili.com/x/vu/web/cover/up",
        params={"ts": int(time.time() * 1000)},
        data={"csrf": BILI_JCT, "cover": cover_base64}
    )
    resp.raise_for_status()
    result = resp.json()
    if result.get("code") != 0:
        raise RuntimeError(f"封面上传失败：{result.get('message')}")
    return result["data"]["url"]


def preupload_video(session: requests.Session, video_path: str) -> dict:
    """
    预上传视频，获取上传参数。
    """
    size = os.path.getsize(video_path)
    name = os.path.basename(video_path)
    resp = session.get(
        "https://member.bilibili.com/preupload",
        params={
            "name": name,
            "r": "upos",
            "profile": "ugcfx/bup",
            "size": size,
            "probe_version": "20221109",
            "webVersion": "2.13.0",
        }
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("OK") != 1:
        raise RuntimeError(f"预上传失败：{data}")
    return data


def post_video_meta(session: requests.Session, pre: dict, video_path: str) -> dict:
    """
    提交视频元数据，获取 upload_id。
    """
    endpoint = pre["endpoint"]
    upos_uri = pre["upos_uri"].replace("upos:/", "")
    auth = pre["auth"]
    biz_id = pre["biz_id"]
    size = os.path.getsize(video_path)

    url = f"https:{endpoint}{upos_uri}"
    resp = session.post(
        url,
        params={
            "uploads": "",
            "output": "json",
            "profile": "ugcfx/bup",
            "filesize": size,
            "partsize": pre["chunk_size"],
            "biz_id": biz_id,
        },
        headers={"X-Upos-Auth": auth, "Content-Length": "0"}
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("OK") != 1:
        raise RuntimeError(f"元数据上传失败：{data}")
    return data


def upload_chunks(session: requests.Session, video_path: str, pre: dict, meta: dict) -> list:
    """
    分片上传视频，返回分片信息列表。
    """
    endpoint = pre["endpoint"]
    uri = pre["upos_uri"].replace("upos:/", "")
    auth = pre["auth"]
    chunk_size = pre["chunk_size"]
    upload_id = meta["upload_id"]
    total_size = os.path.getsize(video_path)
    total_chunks = math.ceil(total_size / chunk_size)
    url_base = f"https:{endpoint}{uri}"

    parts = []
    with open(video_path, "rb") as f:
        for i in range(total_chunks):
            part = f.read(chunk_size)
            resp = session.put(
                url_base,
                params={
                    "partNumber": i + 1,
                    "uploadId": upload_id,
                    "chunk": i,
                    "chunks": total_chunks,
                    "size": len(part),
                    "start": i * chunk_size,
                    "end": i * chunk_size + len(part),
                    "total": total_size,
                },
                headers={"X-Upos-Auth": auth, "Content-Type": "application/octet-stream"},
                data=part,
            )
            resp.raise_for_status()
            if resp.text.strip() != "MULTIPART_PUT_SUCCESS":
                raise RuntimeError(f"分片{i+1}上传失败: {resp.text}")
            parts.append({"partNumber": i + 1, "eTag": "etag"})
    return parts


def finalize_upload(session: requests.Session, pre: dict, meta: dict, parts: list) -> dict:
    """
    完成上传合并视频。
    """
    endpoint = pre["endpoint"]
    uri = pre["upos_uri"].replace("upos:/", "")
    auth = pre["auth"]
    biz_id = pre["biz_id"]
    upload_id = meta["upload_id"]
    filename = os.path.basename(pre["upos_uri"])

    resp = session.post(
        f"https:{endpoint}{uri}",
        params={
            "output": "json",
            "name": filename,
            "profile": "ugcfx/bup",
            "uploadId": upload_id,
            "biz_id": biz_id,
        },
        headers={"X-Upos-Auth": auth, "Content-Type": "application/json"},
        json={"parts": parts},
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("OK") != 1:
        raise RuntimeError(f"结束上传失败：{data}")
    return data


def submit_post(
    session: requests.Session,
    cover_url: str,
    biz_id: int,
    filename: str,
    title: str,
    description: str,
    tags: str,
    copyright_type: int,
    tid: int,
    recreate: int,
    dynamic: str,
    no_reprint: int,
) -> dict:
    """
    投递稿件，返回 aid 和 bvid。
    """
    videos = [{"filename": filename, "title": "P1", "desc": "", "cid": biz_id}]
    payload = {
        "videos": videos,
        "cover": cover_url,
        "title": title,
        "copyright": copyright_type,
        "tid": tid,
        "tag": tags,
        "desc_format_id": 9999,
        "desc": description,
        "recreate": recreate,
        "dynamic": dynamic,
        "no_reprint": no_reprint,
        "interactive": 0,
        "act_reserve_create": 0,
        "no_disturbance": 0,
        "subtitle": {"open": 0, "lan": ""},
        "dolby": 0,
        "lossless_music": 0,
        "up_selection_reply": False,
        "up_close_reply": False,
        "up_close_danmu": False,
        "web_os": 3,
        "csrf": BILI_JCT,
    }
    resp = session.post(
        "https://member.bilibili.com/x/vu/web/add/v3",
        params={"ts": int(time.time() * 1000), "csrf": BILI_JCT},
        headers={"Content-Type": "application/json; charset=utf-8"},
        json=payload,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"投递失败：{data.get('message')}")
    return data["data"]


def upload_to_bilibili(
    video_path: str,
    cover_path: str,
    title: str,
    description: str,
    tags: str,
    copyright_type: int = DEFAULT_COPYRIGHT,
    tid: int = DEFAULT_TID,
    recreate: int = DEFAULT_RECREATE,
    dynamic: str = DEFAULT_DYNAMIC,
    no_reprint: int = DEFAULT_NO_REPRINT,
) -> dict:
    """
    一步完成B站投稿流程，返回投稿结果。
    """
    if not all([video_path, cover_path, title, description, tags]):
        raise ValueError("缺少必要参数：视频路径/封面路径/标题/简介/标签")
    if not os.path.exists(video_path) or not os.path.exists(cover_path):
        raise FileNotFoundError("视频或封面文件不存在")

    sess = get_session()
    cover_url = upload_cover(sess, cover_path)
    pre = preupload_video(sess, video_path)
    biz_id = pre["biz_id"]
    filename = os.path.splitext(os.path.basename(pre["upos_uri"]))[0]
    meta = post_video_meta(sess, pre, video_path)
    parts = upload_chunks(sess, video_path, pre, meta)
    finalize_upload(sess, pre, meta, parts)
    return submit_post(
        sess, cover_url, biz_id, filename,
        title, description, tags,
        copyright_type, tid, recreate, dynamic, no_reprint
    )


if __name__ == "__main__":
    # 读取LLM/TikTokDownloader/metadata_cache.json
    with open('../../LLM/TikTokDownloader/metadata_cache.json', 'r', encoding='utf-8') as f:
        metadata_cache = json.load(f)
    for key, value in metadata_cache.items():
        video_path = value.get('video_path')

    result = upload_to_bilibili(
        video_path=video_path,
        cover_path="inpainted_image.jpg",
        title="我的AI修复视频与精彩瞬间",
        description="这是一个使用AI技术修复后的视频，并加入了有趣的音频，希望大家喜欢！",
        tags="AI修复,视频剪辑,有趣,科技,日常生活",
    )
    print(f"投稿成功！AID={result['aid']}, BVID={result['bvid']}")
