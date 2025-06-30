import requests
import time
import json
from hashlib import md5
import urllib.parse

from common_utils.common_utils import get_config

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
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
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
# 获取视频基本信息
# =========================================================================
def get_video_info(session: requests.Session, bvid: str):
    url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    resp = session.get(url)
    resp.raise_for_status()
    data = resp.json()
    if data.get('code') != 0:
        print(f"[错误] 获取视频信息失败: {data.get('message')}")
        return None
    d = data['data']
    return {
        "aid": d['aid'],
        "cid": d['cid'],
        "duration": d['duration'],
        "title": d['title'],
    }


# =========================================================================
# 模拟观看视频主函数
# =========================================================================
def simulate_watch_video(sessdata: str, bili_jct: str, bvid: str):
    """
    模拟在 B 站浏览器端完整观看一次视频。
    :param sessdata: 浏览器登录后的 SESSDATA
    :param bili_jct: CSRF Token（bili_jct）
    :param bvid: 视频的 BV 号，例如 "BV1Gw3Nz2ExL"
    """
    if not sessdata or "你的" in sessdata:
        print("[错误] 请传入有效的 SESSDATA 和 BILI_JCT！")
        return

    # 准备 Session
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Origin": "https://www.bilibili.com",
    })
    session.cookies.set("SESSDATA", sessdata)
    session.cookies.set("bili_jct", bili_jct)

    # 获取当前账号的 mid
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
        'mid': viewer_mid,
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
        'csrf': bili_jct,
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
            'csrf': bili_jct, 'played_time': played, 'realtime': played,
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
        data={"aid": aid, "cid": cid, "progress": duration, "csrf": bili_jct}
    ).json()
    if rpt.get('code') == 0:
        print("[+] 历史记录上报成功")
    else:
        print(f"[-] 上报失败: {rpt}")

    print("\n[*] 模拟观看完成！")


from concurrent.futures import ThreadPoolExecutor, as_completed


def simulate_watch_video_with_log(SESSDATA, BILI_JCT, BV_ID):
    if not SESSDATA or not BILI_JCT:
        print("[错误] 请确保配置文件中包含有效的 SESSDATA 和 BILI_JCT！")
        return
    print(f"\n正在模拟观看视频 {BV_ID}，使用 SESSDATA: {SESSDATA[:10]}... 和 BILI_JCT: {BILI_JCT[:10]}...")
    simulate_watch_video(SESSDATA, BILI_JCT, BV_ID)


def run_parallel_watch(config_list, BV_ID, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(simulate_watch_video_with_log, SESSDATA, BILI_JCT, BV_ID)
            for SESSDATA, BILI_JCT in config_list
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[异常] 某个线程执行出错: {e}")


if __name__ == "__main__":
    config_list = []
    mama_total_cookie = get_config("mama_bilibili_total_cookie")
    mama_csrf_token = get_config("mama_bilibili_csrf_token")
    config_list.append((mama_total_cookie, mama_csrf_token))
    nana_total_cookie = get_config("nana_bilibili_total_cookie")
    nana_csrf_token = get_config("nana_bilibili_csrf_token")
    config_list.append((nana_total_cookie, nana_csrf_token))
    ruru_total_cookie = get_config("ruru_bilibili_total_cookie")
    ruru_csrf_token = get_config("ruru_bilibili_csrf_token")
    config_list.append((ruru_total_cookie, ruru_csrf_token))
    total_cookie = get_config("bilibili_total_cookie")
    csrf_token = get_config("bilibili_csrf_token")
    config_list.append((total_cookie, csrf_token))

    BV_ID_list = ['BV1shK2zgEfb', 'BV1RdgrzHEmq', 'BV19BgkzXE23', 'BV1CZg6zwEP5', 'BV1sRg6zPEHo', 'BV1K53NzBECJ',
                  'BV1BG3KzZEAk', 'BV1xh3wzFEYd', 'BV19X3FzTEso', 'BV15S3FzuEEC', 'BV1X33FzNEoQ', 'BV1LYKmzZEF8',
                  'BV19G3FzeEHi', 'BV1fw3Fz3E22', 'BV1LFK6zTEp9', 'BV1GcKBzZEqa', 'BV1pLKozcEeQ', 'BV1XBKVzxEZb',
                  'BV1b1KVzDEUy', 'BV1qTKGzZEes', 'BV1sDK3zzEfe', 'BV1koK3zvEek', 'BV1V4g1zZEW6', 'BV1h8g1zdErz',
                  'BV138g1zdEZ2', 'BV1Qgg2ztE4T', 'BV1fog1zTE9G', 'BV1fog1zTEyq', 'BV1xqgCzVE6G', 'BV1HC3PzfEz3',
                  'BV1nC3PzfEvJ', 'BV1HC3PzfESh', 'BV1nC3PzfEfP', 'BV1t83PzBEWm', 'BV1Mq7pzMEW3']
    while True:
        for BV_ID in BV_ID_list:
            print(f"\n开始模拟观看视频 {BV_ID} 的任务...")
            run_parallel_watch(config_list, BV_ID)
            print(f"完成模拟观看视频 {BV_ID} 的任务。")

