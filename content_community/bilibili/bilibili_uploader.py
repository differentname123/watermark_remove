import json

import requests
import base64
import os
import math
import time
import urllib.parse

import os
import subprocess
import json


def probe_video(path):
    """用 ffprobe 返回字典：{width, height, fps}"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json",
        path
    ]
    # 注意：ffprobe 已经使用了 -v error，所以它本身就很安静，无需修改。
    out = subprocess.check_output(cmd)
    info = json.loads(out)["streams"][0]
    num, den = map(int, info["r_frame_rate"].split("/"))
    return {
        "width": info["width"],
        "height": info["height"],
        "fps": num / den
    }


def probe_duration(path):
    """返回视频时长（秒）"""
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ])
    return float(out)

def add_image_to_video_end(
    video_path: str,
    image_path: str,
    output_path: str,
    image_duration: float = 1.0,
    max_retries: int = 3
) -> None:
    """
    将一张图片拼接到视频末尾。如果输出文件小于输入视频文件，最多重试 max_retries 次。

    :param video_path: 输入视频路径
    :param image_path: 输入图片路径
    :param output_path: 输出视频路径
    :param image_duration: 图片在视频末尾的持续时长（秒）
    :param max_retries: 最大重试次数
    :raises RuntimeError: 如果所有重试均失败
    """
    # 初次探测视频元信息、时长
    meta = probe_video(video_path)
    video_dur = probe_duration(video_path)
    total_dur = video_dur + image_duration

    # 构造 filter_complex 字符串
    filter_complex = (
        f"[1:v]fps={meta['fps']:.2f},"
        f"scale={meta['width']}:{meta['height']},"
        "format=yuv420p[img];"
        "[0:v][img]concat=n=2:v=1:a=0[v]"
    )

    # 公共 ffmpeg 参数
    base_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-loop", "1", "-t", str(image_duration), "-i", image_path,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?",            # 如果有音频就映射
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",             # 必须重新编码才能用 apad
        "-af", "apad",             # 自动补零
        "-t", str(total_dur),      # 强制输出时长
    ]

    input_size = os.path.getsize(video_path)

    for attempt in range(1, max_retries + 1):
        try:
            # 每次都重新生成输出文件
            cmd = base_cmd + [output_path]
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            # ffmpeg 出错，记录并重试
            print(f"[警告] 第 {attempt} 次合成失败：{e}")
        else:
            # 检查输出文件大小
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path)
                if output_size >= input_size:
                    # 成功且大小正常，退出循环
                    return
                else:
                    print(f"[警告] 第 {attempt} 次生成的视频大小 ({output_size}) 小于输入视频大小 ({input_size})，重试中...")
            else:
                print(f"[警告] 第 {attempt} 次没有生成输出文件，重试中...")

    # 如果循环结束仍未成功，则抛出异常
    raise RuntimeError(f"多次尝试后仍未生成有效视频（{max_retries} 次）")

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


def get_session(sessdata, bili_jct) -> requests.Session:
    """
    创建并返回带有登录 Cookie 的 requests Session
    """
    sess = requests.Session()
    sess.headers.update(HEADERS)
    sess.cookies.set("SESSDATA", sessdata)
    sess.cookies.set("bili_jct", bili_jct)
    return sess


def upload_cover(session: requests.Session, image_path: str, bili_jct=BILI_JCT) -> str:
    """
    上传封面图片，返回封面 URL。
    """
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    cover_base64 = f"data:image/jpeg;base64,{data}"
    resp = session.post(
        "https://member.bilibili.com/x/vu/web/cover/up",
        params={"ts": int(time.time() * 1000)},
        data={"csrf": bili_jct, "cover": cover_base64}
    )
    resp.raise_for_status()
    result = resp.json()
    if result.get("code") != 0:
        raise RuntimeError(f"封面上传失败：{result.get('message')}")
    return result["data"]["url"]

def fetch_bili_topics(cookie: str, type_pid=1029, type_id=21):
    """
    使用给定的 Cookie 获取 B 站创作中心话题类型信息。

    参数:
        cookie (str): 用于请求的完整 Cookie。

    返回:
        dict: 返回接口返回的 JSON 数据，如果请求失败则返回 None。
    """
    base_url = "https://member.bilibili.com/x/vupre/web/topic/type/v2"

    # 请求参数
    params = {
        "pn": 0,
        "ps": 200,
        "platform": "pc",
        "type_id": 21,
        "type_pid": type_pid,
    }

    # 构造请求头
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "sec-ch-ua": "\"Microsoft Edge\";v=\"137\", \"Chromium\";v=\"137\", \"Not/A)Brand\";v=\"24\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "Referer": "https://member.bilibili.com/platform/upload/video/frame?page_from=creative_home_top_upload",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
        "Cookie": cookie
    }

    try:
        session = requests.Session()
        response = session.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        if 'response' in locals() and response is not None:
            print(f"状态码: {response.status_code}")
            print(f"响应文本: {response.text}")
        return None

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
    bili_jct: str = BILI_JCT,
    human_type2=1002,
    topic_detail={"from_topic_id": 1313687, "from_source": "arc.web.recommend"},
    topic_id: int = 1313687

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
        "human_type2": human_type2,
        "topic_detail": topic_detail,
        "topic_id": topic_id,
        "web_os": 3,
        "csrf": bili_jct,
    }
    resp = session.post(
        "https://member.bilibili.com/x/vu/web/add/v3",
        params={"ts": int(time.time() * 1000), "csrf": bili_jct},
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
    sessdata=SESSDATA,
    bili_jct=BILI_JCT,
    human_type2=1002,
    topic_detail={"from_topic_id": 1313687,"from_source": "arc.web.recommend"},
    topic_id: int = 1313687
) -> dict:
    """
    一步完成B站投稿流程，返回投稿结果。
    """
    if not all([video_path, cover_path, title, description, tags]):
        raise ValueError("缺少必要参数：视频路径/封面路径/标题/简介/标签")
    if not os.path.exists(video_path) or not os.path.exists(cover_path):
        raise FileNotFoundError("视频或封面文件不存在")

    sess = get_session(sessdata, bili_jct)
    cover_url = upload_cover(sess, cover_path, bili_jct=bili_jct)
    pre = preupload_video(sess, video_path)
    biz_id = pre["biz_id"]
    filename = os.path.splitext(os.path.basename(pre["upos_uri"]))[0]
    meta = post_video_meta(sess, pre, video_path)
    parts = upload_chunks(sess, video_path, pre, meta)
    finalize_upload(sess, pre, meta, parts)
    return submit_post(
        sess, cover_url, biz_id, filename,
        title, description, tags,
        copyright_type, tid, recreate, dynamic, no_reprint,bili_jct=bili_jct,human_type2=human_type2, topic_detail=topic_detail,topic_id=topic_id
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
