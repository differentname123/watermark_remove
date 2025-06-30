import requests
import time
import json
from hashlib import md5
import urllib.parse

from common_utils.common_utils import get_config

SESSDATA = get_config("nana_bilibili_sessdata_cookie")
BILI_JCT = get_config("nana_bilibili_csrf_token")
CONFIG = {
    "SESSDATA": SESSDATA,
    "BILI_JCT": BILI_JCT  # 文档中的 csrf
}

# 心跳间隔时间（秒），Web 端默认为 15 秒
HEARTBEAT_INTERVAL = 15


# =========================================================================
# 精确的 WBI 签名模块 (无需修改)
# =========================================================================
class WbiSigner:
    _MIXIN_KEY_ENC_TAB = [
        46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
        33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
        61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
        36, 20, 34, 44, 52
    ]

    def __init__(self):
        self.img_key = None
        self.sub_key = None
        self.key_expire_time = 0

    def _get_mixin_key(self, orig: str) -> str:
        return ''.join(orig[i] for i in self._MIXIN_KEY_ENC_TAB)[:32]

    def _get_wbi_keys(self):
        if time.time() < self.key_expire_time:
            return
        resp = requests.get(
            'https://api.bilibili.com/x/web-interface/nav',
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
        )
        resp.raise_for_status()
        data = resp.json()['data']['wbi_img']
        img_url = data['img_url']
        sub_url = data['sub_url']
        self.img_key = img_url.rsplit('/', 1)[1].split('.')[0]
        self.sub_key = sub_url.rsplit('/', 1)[1].split('.')[0]
        self.key_expire_time = time.time() + 600

    def sign_url_params(self, params: dict) -> dict:
        self._get_wbi_keys()
        mixin_key = self._get_mixin_key(self.img_key + self.sub_key)
        signed = params.copy()
        signed['wts'] = round(time.time())
        sorted_params = dict(sorted(signed.items()))
        query_str = urllib.parse.urlencode(sorted_params)
        signed['w_rid'] = md5((query_str + mixin_key).encode()).hexdigest()
        return signed


# =========================================================================
# 主功能函数
# =========================================================================
def get_video_info(session: requests.Session, bvid: str):
    url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    resp = session.get(url)
    resp.raise_for_status()
    data = resp.json()
    if data['code'] != 0:
        print(f"[错误] 获取视频信息失败: {data.get('message')}")
        return None
    d = data['data']
    return {
        "aid": d['aid'],
        "cid": d['cid'],
        "duration": d['duration'],
        "title": d['title'],
        "mid": d['owner']['mid']
    }


def simulate_watch_video(bvid: str):
    if not CONFIG["SESSDATA"] or "你的" in CONFIG["SESSDATA"]:
        print("[错误] 请先在脚本配置区域填入你的 SESSDATA 和 BILI_JCT！")
        return

    # 准备 Session
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Origin": "https://www.bilibili.com",
    })
    session.cookies.set("SESSDATA", CONFIG["SESSDATA"])
    session.cookies.set("bili_jct", CONFIG["BILI_JCT"])

    # 获取 viewer_mid（当前账号的 mid）
    nav = session.get("https://api.bilibili.com/x/web-interface/nav")
    nav.raise_for_status()
    viewer_mid = nav.json()['data']['mid']

    # 获取视频基本信息
    session.headers.update({"Referer": f"https://www.bilibili.com/video/{bvid}/"})
    info = get_video_info(session, bvid)
    if not info:
        return
    aid, cid, duration, title = (
        info["aid"], info["cid"], info["duration"], info["title"]
    )
    csrf = CONFIG["BILI_JCT"]

    print(f"[*] 视频《{title}》 (aid={aid}, cid={cid}, 时长={duration}s)")

    # 1/3: 模拟开始播放
    print("[1/3] 模拟开始播放 …")
    base_url = "https://api.bilibili.com/x/click-interface/click/web/h5"
    stime = int(time.time()) - 2

    url_params = {
        'w_aid': aid,
        'w_part': 1,
        'w_ftime': stime,
        'w_stime': stime,
        'w_type': 3,
        'web_location': 1315873,
    }

    body_params = {
        'mid': viewer_mid,    # 用 viewer_mid 而不是 owner_mid
        'aid': aid,
        'cid': cid,
        'part': 1,
        'lv': 2,
        'ftime': stime,
        'stime': stime,
        'type': 3,
        'sub_type': 0,
        'refer_url': 'https://member.bilibili.com/',
        'outer': 0,
        'statistics': json.dumps({
            "appId": 100,
            "platform": 5,
            "abtest": "",
            "version": ""
        }),
        'mobi_app': 'web',
        'device': 'web',
        'platform': 'web',
        'spmid': '333.788.0.0',
        'from_spmid': '333.788.0.0',
        'session': md5(str(time.time()).encode()).hexdigest(),
        'csrf': csrf,
    }

    session.headers.update({
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": f"https://www.bilibili.com/video/{bvid}/?vd_source=5d365f9cfcf15bb4bacfcd69a69fc4d4",
    })

    signer = WbiSigner()
    signed = signer.sign_url_params(url_params)
    final_url = f"{base_url}?{urllib.parse.urlencode(signed)}"
    res = session.post(final_url, data=body_params)
    res.raise_for_status()
    d = res.json()
    if d.get('code') == 0:
        print("[+] 开始播放请求成功")
    else:
        print(f"[-] 开始播放请求失败: {d}")

    # 2/3: 心跳
    print("\n[2/3] 模拟持续观看 (心跳)…")
    hb_url = "https://api.bilibili.com/x/click-interface/web/heartbeat"
    start_ts = int(time.time())
    played = 0
    while played < duration:
        payload = {
            'aid': aid, 'cid': cid, 'bvid': bvid, 'mid': viewer_mid,
            'csrf': csrf, 'played_time': played, 'realtime': played,
            'start_ts': start_ts, 'play_type': 0, 'type': 3,
            'sub_type': 0, 'dt': 2, 'last_play_progress_time': played,
        }
        hb = session.post(hb_url, data=payload).json()
        if hb.get('code') == 0:
            print(f"    - 心跳：{played}/{duration}s")
        else:
            print(f"    - 心跳失败: {hb}")
            if hb.get('code') == -101:
                break
        time.sleep(HEARTBEAT_INTERVAL)
        played = min(played + HEARTBEAT_INTERVAL, duration)

    # 3/3: 上报历史
    print("\n[3/3] 上报历史记录…")
    rpt = session.post(
        "https://api.bilibili.com/x/v2/history/report",
        data={"aid": aid, "cid": cid, "progress": duration, "csrf": csrf}
    ).json()
    if rpt.get('code') == 0:
        print("[+] 历史记录上报成功")
    else:
        print(f"[-] 上报失败: {rpt}")

    print("\n[*] 模拟观看完成！")


if __name__ == "__main__":
    simulate_watch_video("BV19G3FzeEHi")
