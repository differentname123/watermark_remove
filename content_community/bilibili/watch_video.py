import os

import requests
import time
import json
import logging
from hashlib import md5
import urllib.parse

from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from common_utils.common_utils import get_config, init_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 心跳间隔时间（秒），Web 端默认为 15 秒
HEARTBEAT_INTERVAL = 15
REQUEST_TIMEOUT = 10  # seconds
config_map = init_config()
config_list = []
for uid, info in config_map.items():
    total = info.get("total_cookie")
    csrf = info.get("BILI_JCT")
    if total and csrf:
        config_list.append((total, csrf))



def create_session():
    """创建带重试和超时的 Session"""
    session = requests.Session()
    # 注意：urllib3.Retry 在新版本中用 allowed_methods 替代 method_whitelist
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    })
    return session


class WbiSigner:
    _MIXIN_KEY_ENC_TAB = [46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31,
                          58, 3, 45, 35, 27, 43, 5, 49, 33, 9, 42, 19, 29,
                          28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24,
                          55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25,
                          54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34,
                          44, 52]

    def __init__(self, session):
        self.session = session
        self.img_key = None
        self.sub_key = None
        self.key_expire_time = 0

    def _get_mixin_key(self, orig: str) -> str:
        return ''.join(orig[i] for i in self._MIXIN_KEY_ENC_TAB)[:32]

    def _get_wbi_keys(self):
        now = time.time()
        if now < self.key_expire_time:
            return
        try:
            resp = self.session.get(
                'https://api.bilibili.com/x/web-interface/nav',
                timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()['data']['wbi_img']
            img_url = data['img_url']
            sub_url = data['sub_url']
            self.img_key = img_url.rsplit('/', 1)[1].split('.')[0]
            self.sub_key = sub_url.rsplit('/', 1)[1].split('.')[0]
            self.key_expire_time = now + 600
        except Exception as e:
            logging.error(f"获取 WBI 密钥失败: {e}")

    def sign_url_params(self, params: dict) -> dict:
        self._get_wbi_keys()
        if not self.img_key or not self.sub_key:
            logging.warning("使用未初始化的 WBI 密钥，签名可能失败")
        mixin_key = self._get_mixin_key((self.img_key or '') + (self.sub_key or ''))
        signed = params.copy()
        signed['wts'] = round(time.time())
        sorted_params = dict(sorted(signed.items()))
        query_str = urllib.parse.urlencode(sorted_params)
        signed['w_rid'] = md5((query_str + mixin_key).encode()).hexdigest()
        return signed


def get_video_info(session: requests.Session, bvid: str):
    url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get('code') != 0:
            logging.error(f"获取视频信息失败: {data.get('message')}")
            return None
        d = data['data']
        return {
            "aid": d['aid'],
            "cid": d['cid'],
            "duration": d['duration'],
            "title": d['title'],
        }
    except Exception as e:
        logging.error(f"请求视频信息出错: {e}")
        return None


def simulate_watch_video(sessdata: str, bili_jct: str, bvid: str):
    session = create_session()
    session.headers.update({
        "Origin": "https://www.bilibili.com"
    })
    session.cookies.set("SESSDATA", sessdata)
    session.cookies.set("bili_jct", bili_jct)

    try:
        nav = session.get("https://api.bilibili.com/x/web-interface/nav", timeout=REQUEST_TIMEOUT)
        nav.raise_for_status()
        viewer_mid = nav.json()['data']['mid']
    except Exception as e:
        logging.error(f"获取账号 mid 失败: {e}")
        return

    info = get_video_info(session, bvid)
    if not info:
        return
    aid, cid, duration, title = info.values()
    logging.info(f"开始模拟观看: 《{title}》 (aid={aid}, cid={cid}, duration={duration}s)")

    signer = WbiSigner(session)

    # 1/3: 模拟开始播放
    try:
        stime = int(time.time()) - 2
        url_params = {
            'w_aid': aid, 'w_part': 1, 'w_ftime': stime,
            'w_stime': stime, 'w_type': 3, 'web_location': 1315873
        }
        signed = signer.sign_url_params(url_params)
        final_url = f"https://api.bilibili.com/x/click-interface/click/web/h5?{urllib.parse.urlencode(signed)}"
        res = session.post(final_url, data={
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
            'statistics': json.dumps({"appId": 100, "platform": 5, "abtest": "", "version": ""}),
            'mobi_app': 'web', 'device': 'web', 'platform': 'web',
            'spmid': '333.788.0.0', 'from_spmid': '333.788.0.0',
            'session': md5(str(time.time()).encode()).hexdigest(), 'csrf': bili_jct
        }, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        d = res.json()
        if d.get('code') == 0:
            logging.info("开始播放请求成功")
        else:
            logging.warning(f"开始播放请求失败: {d}")
    except Exception as e:
        logging.error(f"播发请求出错: {e}")
        return

    # 2/3: 心跳
    logging.info("开始心跳模拟...")
    start_ts = int(time.time())
    played = 0
    while played < duration and played < 150:
        try:
            payload = {
                'aid': aid, 'cid': cid, 'bvid': bvid, 'mid': viewer_mid,
                'csrf': bili_jct, 'played_time': played,
                'realtime': played, 'start_ts': start_ts,
                'play_type': 0, 'type': 3, 'sub_type': 0,
                'dt': 2, 'last_play_progress_time': played,
            }
            hb = session.post(
                "https://api.bilibili.com/x/click-interface/web/heartbeat",
                data=payload, timeout=REQUEST_TIMEOUT
            ).json()
            if hb.get('code') == 0:
                logging.info(f"心跳: {played}/{duration}s")
            else:
                logging.warning(f"心跳失败: {hb}")
                if hb.get('code') == -101:
                    break
        except Exception as e:
            logging.error(f"心跳异常: {e}")
            break
        time.sleep(HEARTBEAT_INTERVAL)
        played = min(played + HEARTBEAT_INTERVAL, duration)

    # 3/3: 上报历史
    try:
        rpt = session.post(
            "https://api.bilibili.com/x/v2/history/report",
            data={"aid": aid, "cid": cid, "progress": duration, "csrf": bili_jct},
            timeout=REQUEST_TIMEOUT
        ).json()
        if rpt.get('code') == 0:
            logging.info("历史记录上报成功")
        else:
            logging.warning(f"上报失败: {rpt}")
    except Exception as e:
        logging.error(f"上报历史异常: {e}")

    logging.info("模拟观看完成！")


def simulate_watch_video_with_log(sessdata, bili_jct, bvid):
    try:
        simulate_watch_video(sessdata, bili_jct, bvid)
    except Exception as e:
        logging.error(f"线程执行出错: {e}")


def run_parallel_watch(config_list, bvid, max_workers=15):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(simulate_watch_video_with_log, s, t, bvid)
            for s, t in config_list
        ]
        for future in as_completed(futures):
            pass


def load_processed_dict(filepath):
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def watch_video(bv_list):
    for bv in bv_list:
        logging.info(f"开始视频 {bv} 的模拟观看任务...")
        run_parallel_watch(config_list, bv)
        print(f"已完成视频 {bv} 的模拟观看任务。")


if __name__ == "__main__":
    # watch_video(['BV1Ab3uzyEtM'])
    bv_list = ['BV1Ab3uzyEtM', 'BV1ce3uz3Et8', 'BV1ne3uz3E9h', 'BV1Ab3uzyEPW', 'BV1Wh3uzVEor', 'BV1wh3uzVEMM', 'BV1Wh3uzVE4E', 'BV1w83TzKESH', 'BV1p43TzMEr2', 'BV1w43TzMEfn', 'BV1At3TzqEh4', 'BV1pe3azzEWP', 'BV1ri3azMEoq', 'BV1jC3azYEaW', 'BV1ki3azMEVP', 'BV1jC3azYEYx', 'BV1jC3azYEhT', 'BV1E63azvEF1', 'BV1R43hzsEyx', 'BV1PM3hzBExm', 'BV1mC3hzCExP', 'BV1hG3az9ESW', 'BV1zd3bzoEMe', 'BV1mh34zKEzv', 'BV1WH3tzwEYn', 'BV1pH3xzWE9e', 'BV1AA34zxEJ1', 'BV16n3WzjEyE', 'BV1Hn3WzjEQW', 'BV1A93Wz9ERi', 'BV1QE3sziE2J', 'BV1Vn3szKEGp', 'BV1qS3pzqEEX', 'BV1Xg3nzxEGT', 'BV1JpgQzREYN', 'BV12hgXzME9p', 'BV1b1gSzwEdH', 'BV1TSgSzbEPZ', 'BV1K2gSzhE1i', 'BV1KUgSzFEFr', 'BV1jtgSzCETR', 'BV1dugDzyECk', 'BV1ougDzyEjD', 'BV195gDz3En9', 'BV1GLgDzrERs', 'BV1PdgSzsEFu', 'BV1wdgSzsEo4', 'BV1AdgSzsE6W', 'BV1pogSz9EWb', 'BV1P9gSzdEiS', 'BV1W9gSzdE1Z', 'BV1nwgDznEWF', 'BV1nwgDznEb8', 'BV1SsgDzeEG9', 'BV1aogDzkEmz', 'BV18NgQzTE7i', 'BV18NgQzTEHg', 'BV1SNgQzTEN7', 'BV1bNgQzTEBb', 'BV1tNgQzTEdm', 'BV1vXgDzLEdg', 'BV1fmgDzUEAr', 'BV1qmgDzUExv', 'BV1qmgDzUEx9', 'BV1qmgDzUEpH', 'BV1qmgDzUEp6', 'BV1BmgDzUEEd', 'BV1PSgmzDEs9', 'BV1cdgmz2Ec7', 'BV1Hdgmz2EL2', 'BV1Fdgmz2Em6', 'BV1fZgmz5Ea7', 'BV1N8gUzWEEH', 'BV1y8gUzpELp', 'BV1N8gUzWE3z', 'BV1AhgUzXE1d', 'BV124gUzsEqg', 'BV124gUzsEtz', 'BV1PtgUzREfu', 'BV1YQgUzMEC3', 'BV1hdgUzvE9v', 'BV1eQgUzMEFd', 'BV1YQgUzMEo2', 'BV15QgUzuER8', 'BV1eQgUzMEB9', 'BV1eXgUzqEF3', 'BV1twgSzhEye', 'BV1VXgUzBEW5', 'BV1aXgUzqEoG', 'BV1aZgUzPEjN', 'BV1bwgSzhEyr', 'BV1bwgSzhE3q', 'BV1twgSzhEyB', 'BV1hfgUzYEh7', 'BV1hfgUzYEcC', 'BV18fgUzYEAu', 'BV1fkgUz5Exz', 'BV1qkgUz5EZS', 'BV1BkgUz5E42', 'BV1qCgUzCE13', 'BV1BCgUzCE6t', 'BV1fCgUzCEum', 'BV1ZCgUzCE3S', 'BV1ZCgUzCEwD', 'BV1vSgUzzEkL', 'BV1vSgUzzEaE', 'BV1HEgSzSEMQ', 'BV1JEgSzSEBU', 'BV1nEgSzSEPN', 'BV174gSz5EHR', 'BV1xMgSz4Eju', 'BV14MgSz4EuR', 'BV1RdgrzHEmq', 'BV19BgkzXE23', 'BV1CZg6zwEP5', 'BV1sRg6zPEHo', 'BV1K53NzBECJ', 'BV1BG3KzZEAk', 'BV1xh3wzFEYd', 'BV19X3FzTEso', 'BV15S3FzuEEC', 'BV1X33FzNEoQ', 'BV19G3FzeEHi', 'BV1fw3Fz3E22', 'BV1LYKmzZEF8', 'BV1shK2zgEfb', 'BV1NhK2zgEFV', 'BV1PaK2zzEqp', 'BV1v8KmzfEfL', 'BV1A4Kkz4E51', 'BV1fiKkzJE5A', 'BV1MjKkzPEvL', 'BV1pkKzzUEuQ', 'BV1LFK6zTEp9', 'BV1GcKBzZEqa', 'BV1pLKozcEeQ', 'BV1XBKVzxEZb', 'BV1b1KVzDEUy', 'BV1qTKGzZEes', 'BV1sDK3zzEfe', 'BV1koK3zvEek', 'BV19AN1z1EKn', 'BV1GNN1zLEMA', 'BV1u3N1z3Emm', 'BV1jEN2zsEHi', 'BV1tQNCzNEEP', 'BV1Mq7pzMEW3']
    # 打乱bv_list
    import random
    random.shuffle(bv_list)

    try:
        while True:
            bv_list = load_processed_dict('../../LLM/TikTokDownloader/bvid_list.json').get('bvid_list', bv_list)
            for bv in bv_list:
                logging.info(f"开始视频 {bv} 模拟任务...")
                run_parallel_watch(config_list, bv)
                logging.info(f"完成视频 {bv} 模拟任务。")
            # time.sleep(60 * 60)
    except KeyboardInterrupt:
        logging.info("已收到退出信号，程序终止。")
