# -- coding: utf-8 --
"""
:authors:
    zhuxiaohu
:create_date:
    2025/5/26 22:51
:last_date:
    2025/5/28 00:30 (修正 WBI 参数位置，放入 Body)
:description:
    Bilibili 评论和弹幕发送及点赞脚本。
    注意：dm_img_* 字段是硬编码的设备指纹，长期使用可能导致风控或失效。
          建议定期更新这些值或考虑使用自动化浏览器。
"""
import mimetypes
import os

import requests
import urllib.parse
import time
from hashlib import md5
import json

# 确保您的 common_utils.common_utils 模块和 get_config 函数可用
# 如果没有，请替换为实际获取配置的方法，例如从文件读取或直接硬编码(不推荐敏感信息)
try:
    from common_utils.common_utils import get_config, init_config
except ImportError:
    print("警告: 未找到 common_utils.common_utils 模块。请确保该模块存在或手动设置配置。")


    # 提供一个简单的模拟函数，实际使用请替换
    def get_config(key):
        configs = {
            "bilibili_csrf_token": "YOUR_CSRF_TOKEN",  # <-- 替换为你的 bili_jct 值
            "bilibili_total_cookie": "SESSDATA=YOUR_SESSDATA; bili_jct=YOUR_CSRF_TOKEN;"  # <-- 替换为你的完整Cookie字符串
        }
        return configs.get(key)


def upload_bilibili_image(image_path: str, cookies: dict, csrf_token: str):
    """
    模拟浏览器上传图片到Bilibili动态。

    :param image_path: 要上传的本地图片文件的路径。
    :param cookies: 用户登录后的 cookies，以字典形式提供。
    :param csrf_token: 用户的 CSRF token (通常与 bili_jct cookie 的值相同)。
    :return: requests 的 Response 对象，可以调用 .json() 查看返回结果。
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：文件 '{image_path}' 不存在。")
        return None

    # 目标 URL
    url = "https://api.bilibili.com/x/dynamic/feed/draw/upload_bfs"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Referer": "https://www.bilibili.com/",
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Origin": "https://t.bilibili.com",  # 有时 Origin 头也是必要的
    }

    data = {
        "biz": "new_dyn",
        "category": "daily",
        "csrf": csrf_token,
    }

    with open(image_path, 'rb') as f:
        # 猜测文件的MIME类型 (e.g., 'image/jpeg', 'image/png')
        mime_type = mimetypes.guess_type(image_path)[0] or 'application/octet-stream'

        files = {
            'file_up': (os.path.basename(image_path), f, mime_type)
        }

        try:
            # 发送 POST 请求
            print(f"正在上传图片: {image_path}...")
            response = requests.post(
                url,
                headers=headers,
                cookies=cookies,
                data=data,
                files=files
            )

            # 检查请求是否成功
            response.raise_for_status()  # 如果状态码是 4xx 或 5xx，将抛出异常

            print("上传成功！")
            return response

        except requests.exceptions.RequestException as e:
            print(f"上传失败: {e}")
            return None


class BilibiliCommenter:
    """
    用于发送 Bilibili 评论、弹幕并尝试点赞的类。
    封装了获取 AID/CID、WBI 签名生成和实际发送/点赞的逻辑。
    """

    # --- WBI 签名相关静态配置 ---
    _MIXIN_KEY_ENC_TAB = [
        46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
        33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
        61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
        36, 20, 34, 44, 52
    ]

    # --- 硬编码的 dm_img_* 字段（从您最近捕获的请求中复制）---
    # 警告：这些字段是动态的设备指纹，硬编码可能导致长期被风控或失效！
    # 它们通常由真实的浏览器环境生成。
    # 这些仅用于评论发送API
    _DM_IMG_LIST = [
        {"x": 3472, "y": 2797, "z": 0, "timestamp": 534786, "k": 118, "type": 0},
        {"x": 3473, "y": 2788, "z": 8, "timestamp": 534958, "k": 121, "type": 0},
        {"x": 3539, "y": 2837, "z": 79, "timestamp": 535069, "k": 112, "type": 0},
        {"x": 3764, "y": 2984, "z": 331, "timestamp": 535170, "k": 87, "type": 0},
        {"x": 3781, "y": 2978, "z": 371, "timestamp": 535449, "k": 126, "type": 0},
        {"x": 3194, "y": 2511, "z": 137, "timestamp": 551580, "k": 63, "type": 0},
        {"x": 3601, "y": 2972, "z": 474, "timestamp": 551681, "k": 62, "type": 0},
        {"x": 3785, "y": 3242, "z": 492, "timestamp": 551782, "k": 95, "type": 0},
        {"x": 3907, "y": 3849, "z": 332, "timestamp": 551883, "k": 87, "type": 0},
        {"x": 4566, "y": 4676, "z": 855, "timestamp": 551984, "k": 104, "type": 0},
        {"x": 5131, "y": 3722, "z": 986, "timestamp": 736197, "k": 114, "type": 0},
        {"x": 2755, "y": 2204, "z": 199, "timestamp": 736298, "k": 96, "type": 0},
        {"x": 2573, "y": 2918, "z": 1308, "timestamp": 736399, "k": 81, "type": 0},
        {"x": 4905, "y": 4426, "z": 1029, "timestamp": 1142346, "k": 87, "type": 0},
        {"x": 5066, "y": 4266, "z": 1256, "timestamp": 1142447, "k": 107, "type": 0},
        {"x": 4881, "y": 3786, "z": 1151, "timestamp": 1142547, "k": 83, "type": 0},
        {"x": 4457, "y": 3155, "z": 773, "timestamp": 1142648, "k": 86, "type": 0},
        {"x": 4256, "y": 2423, "z": 785, "timestamp": 1142749, "k": 106, "type": 0},
        {"x": 4080, "y": 2068, "z": 686, "timestamp": 1142849, "k": 92, "type": 0},
        {"x": 3576, "y": 1555, "z": 186, "timestamp": 1142955, "k": 108, "type": 0},
        {"x": 5316, "y": 3396, "z": 2014, "timestamp": 1143055, "k": 121, "type": 0},
        {"x": 5435, "y": 4186, "z": 2213, "timestamp": 1143155, "k": 105, "type": 0},
        {"x": 5413, "y": 4817, "z": 2118, "timestamp": 1143255, "k": 80, "type": 0},
        {"x": 6049, "y": 6067, "z": 2591, "timestamp": 1143355, "k": 102, "type": 0},
        {"x": 5872, "y": 6193, "z": 2333, "timestamp": 1143455, "k": 70, "type": 0},
        {"x": 5796, "y": 6154, "z": 2169, "timestamp": 1143555, "k": 110, "type": 0},
        {"x": 6286, "y": 5270, "z": 2296, "timestamp": 1146274, "k": 108, "type": 0},
        {"x": 6023, "y": 3331, "z": 2714, "timestamp": 1146374, "k": 66, "type": 0},
        {"x": 5415, "y": 1712, "z": 2747, "timestamp": 1146474, "k": 92, "type": 0},
        {"x": 4695, "y": 658, "z": 2178, "timestamp": 1146574, "k": 104, "type": 0},
        {"x": 2971, "y": -1256, "z": 518, "timestamp": 1146674, "k": 75, "type": 0},
        {"x": 4387, "y": 191, "z": 2094, "timestamp": 1146774, "k": 109, "type": 0},
        {"x": 2865, "y": -1292, "z": 639, "timestamp": 1146874, "k": 109, "type": 0},
        {"x": 4551, "y": 425, "z": 2393, "timestamp": 1146974, "k": 70, "type": 0},
        {"x": 3306, "y": -808, "z": 1158, "timestamp": 1147081, "k": 119, "type": 0},
        {"x": 4946, "y": 836, "z": 2809, "timestamp": 1147322, "k": 89, "type": 0},
        {"x": 2945, "y": -1165, "z": 808, "timestamp": 1149766, "k": 102, "type": 0},
        {"x": 2168, "y": -1942, "z": 31, "timestamp": 1149878, "k": 93, "type": 1},
        {"x": 5998, "y": 1886, "z": 3867, "timestamp": 1154218, "k": 106, "type": 0},
        {"x": 4096, "y": -152, "z": 1913, "timestamp": 1154319, "k": 124, "type": 0},
        {"x": 4449, "y": 711, "z": 1955, "timestamp": 1154419, "k": 61, "type": 0},
        {"x": 7350, "y": 3897, "z": 4668, "timestamp": 1154529, "k": 86, "type": 0},
        {"x": 6765, "y": 3376, "z": 4029, "timestamp": 1154630, "k": 121, "type": 0},
        {"x": 3712, "y": 345, "z": 910, "timestamp": 1154731, "k": 85, "type": 0},
        {"x": 3028, "y": -475, "z": 105, "timestamp": 1154831, "k": 114, "type": 0},
        {"x": 5220, "y": 1539, "z": 2187, "timestamp": 1154931, "k": 89, "type": 0},
        {"x": 3592, "y": -308, "z": 411, "timestamp": 1155031, "k": 66, "type": 0},
        {"x": 6407, "y": 2452, "z": 3138, "timestamp": 1155131, "k": 84, "type": 0},
        {"x": 4496, "y": 485, "z": 1165, "timestamp": 1155232, "k": 83, "type": 0},
        {"x": 5648, "y": 1637, "z": 2317, "timestamp": 1155401, "k": 63, "type": 0}
    ]
    _DM_IMG_STR = "V2ViR0wgMS4wIChPcGVuR0wgRVMgMi4wIENocm9taXVtKQ"
    _DM_COVER_IMG_STR = "QU5HTEUgKE5WSURJQSwgTlZJRElBIEdlRm9yY2UgUlRYIDIwODAgVGkgKDB4MDAwMDFFMDcpIERpcmVjdDNEMTEgdnNfNV8wIHBzXzVfMCwgRDNEMTEpR29vZ2xlIEluYy4gKE5WSURJQS"
    _DM_IMG_INTER = '{"ds":[{"t":0,"c":"","p":[1284,74,2003],"s":[374,3566,1044]}],"wh":[4332,3754,88],"of":[1239,1934,423]}'

    # --- API 端点 ---
    _COMMENT_ADD_API_URL = "https://api.bilibili.com/x/v2/reply/add"
    _COMMENT_ACTION_API_URL = "https://api.bilibili.com/x/v2/reply/action"
    _VIDEO_LIKE_API_URL = "https://api.bilibili.com/x/web-interface/archive/like"
    _DANMAKU_POST_API_URL = "https://api.bilibili.com/x/v2/dm/post"
    _NAV_API_URL = "https://api.bilibili.com/x/web-interface/nav"
    _VIEW_API_URL_TEMPLATE = "https://api.bilibili.com/x/web-interface/view?bvid={bvid_str}"
    _USER_VIDEOS_API_URL = "https://api.bilibili.com/x/space/wbi/arc/search"
    # --- 新增的API端点 ---
    _TRIPLE_LIKE_API_URL = "https://api.bilibili.com/x/web-interface/archive/like/triple"
    _SHARE_API_URL = "https://api.bilibili.com/x/web-interface/share/add"
    _PIN_COMMENT_API_URL = "https://api.bilibili.com/x/v2/reply/top" # <-- 新增此行

    def __init__(self, total_cookie: str, csrf_token: str, all_params={}):
        """
        初始化 BilibiliCommenter 实例。
        :param total_cookie: 包含 SESSDATA 和 bili_jct 的完整 Cookie 字符串。
        :param csrf_token: Bilibili 的 CSRF Token (即 bili_jct 的值)。
        """
        self.session = requests.Session()
        self.csrf_token = csrf_token
        self.total_cookie = total_cookie
        self.all_params = all_params

        # 设置会话的默认头
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            "accept": "*/*",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "content-type": "application/x-www-form-urlencoded",
            "priority": "u=1, i",
            "referrerPolicy": "no-referrer-when-downgrade",
            "sec-ch-ua": "\"Not/A)Brand\";v=\"99\", \"Google Chrome\";v=\"127\", \"Chromium\";v=\"127\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "credentials": "include"
        })

        # 解析 total_cookie 并设置到 session.cookies 中
        for pair in total_cookie.split(';'):
            if '=' in pair:
                key, value = pair.strip().split('=', 1)
                self.session.cookies.set(key, value)

        # 确保 bili_jct 也在 cookie 中，因为 CSRF 需要
        if 'bili_jct' not in self.session.cookies:
            self.session.cookies.set('bili_jct', csrf_token)

        self.img_key = None
        self.sub_key = None
        self._load_wbi_keys()  # 初始化时获取 WBI 密钥

    def _get_mixin_key(self, orig: str) -> str:
        """对 imgKey 和 subKey 进行字符顺序打乱编码"""
        return ''.join(orig[i] for i in self._MIXIN_KEY_ENC_TAB)[:32]

    def _filter_and_encode_param_value(self, value_str: str) -> str:
        """过滤 value 中的 "!'()*" 字符，并进行 URL 编码（大写编码，空格为 %20）"""
        value_str = str(value_str)
        filtered_value = ''.join(filter(lambda chr: chr not in "!'()*", value_str))
        return urllib.parse.quote(filtered_value, safe='').replace(' ', '%20')

    def pin_comment(self, bvid: str, rpid, action: int = 1, type_code: int = 1) -> bool:
        """
        置顶或取消置顶指定视频下的评论。
        需要UP主权限，且只能置顶一级评论。
        此操作参考了其他 v2/reply 接口，使用 WBI 签名。

        :param bvid: 视频的 BV 号。
        :param rpid: 目标评论的 rpid。
        :param action: 操作代码 (1: 设为置顶, 0: 取消置顶)。默认为 1。
        :param type_code: 评论区类型代码，1 通常代表视频。
        :return: True 如果操作成功，否则 False。
        """
        action_text = "置顶" if action == 1 else "取消置顶"
        print(f"准备对视频 {bvid} 下的评论 rpid={rpid} 进行'{action_text}'操作...")

        video_info = self._get_video_info(bvid)
        if not video_info:
            print(f"操作失败：无法获取视频信息。")
            return False
        oid = video_info['aid']  # oid 是视频的 aid

        # 与 like_comment 类似，此 /x/v2/reply/ 路径下的接口很可能需要 WBI 签名
        post_data_unsigned = {
            "oid": oid,  # 目标评论区id
            "rpid": rpid,  # 目标评论rpid
            "action": action,  # 操作代码
            "type": type_code,  # 评论区类型代码
            "csrf": self.csrf_token,  # CSRF Token
            "statistics": '{"appId":100,"platform":5}',  # 与 like_comment 保持一致
        }

        try:
            signed_post_data = self._sign_params_for_wbi(post_data_unsigned)
        except ValueError as e:
            print(f"操作失败：{e}")
            return False

        self.session.headers.update({
            "Referer": f"https://www.bilibili.com/video/{bvid}/"
        })
        proxies = self.all_params.get("proxies", {
                "http": None,
                "https": None
            })
        try:
            response = self.session.post(self._PIN_COMMENT_API_URL, data=signed_post_data, proxies=proxies)
            response.raise_for_status()
            result = response.json()

            if result.get("code") == 0:
                print(f"评论 rpid={rpid} {action_text}成功。")
                return True
            else:
                error_message = result.get('message', '未知错误')
                print(f"操作失败，错误码：{result.get('code')}, 信息：{error_message}")
                # 根据接口文件补充可能的错误原因
                if result.get("code") == 12029:  # 已经有置顶评论
                    print("原因：已经有置顶评论了。")
                elif result.get("code") == 12030:  # 不能置顶非一级评论
                    print("原因：不能置顶非一级评论。")
                elif result.get("code") == -403:  # 权限不足
                    print("原因：权限不足（您可能不是该视频的UP主）。")
                return False
        except requests.exceptions.RequestException as e:
            print(f"请求置顶评论时发生错误：{e}")
            return False
        except Exception as e:
            print(f"置顶评论时发生未知错误：{e}")
            return False

    def _sign_params_for_wbi(self, params: dict) -> dict:
        """
        为给定的参数字典生成 WBI 签名，并将 wts 和 w_rid 添加到字典中。
        返回修改后的字典，这个字典可以直接用于 POST 请求的 data 参数。
        """
        if not (self.img_key and self.sub_key):
            self._load_wbi_keys()
            if not (self.img_key and self.sub_key):
                raise ValueError("WBI Keys 不可用，无法生成签名。")

        mixin_key = self._get_mixin_key(self.img_key + self.sub_key)
        curr_time = round(time.time())

        params_with_wbi = params.copy()
        params_with_wbi['wts'] = curr_time

        sorted_params_for_md5 = dict(sorted(params_with_wbi.items()))

        encoded_parts_for_md5 = []
        for k, v in sorted_params_for_md5.items():
            encoded_key = urllib.parse.quote(str(k), safe='')
            encoded_value = self._filter_and_encode_param_value(v)
            encoded_parts_for_md5.append(f"{encoded_key}={encoded_value}")

        query_for_md5 = '&'.join(encoded_parts_for_md5)

        wbi_sign = md5((query_for_md5 + mixin_key).encode()).hexdigest()

        params_with_wbi['w_rid'] = wbi_sign
        return params_with_wbi

    def _get_video_info(self, bvid_str: str) -> dict | None:
        """根据 BV 号获取视频的 aid 和 cid"""
        url = self._VIEW_API_URL_TEMPLATE.format(bvid_str=bvid_str)
        temp_headers = {"Referer": f"https://www.bilibili.com/video/{bvid_str}/"}
        try:
            response = self.session.get(url, headers=temp_headers)
            response.raise_for_status()
            data = response.json()
            if data.get("code") == 0 and data.get("data"):
                return {"aid": data["data"]["aid"], "cid": data["data"]["cid"]}
            else:
                print(f"获取视频信息失败 (bvid: {bvid_str}): {data.get('message')}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"请求视频信息发生错误：{e}")
            return None

    def _load_wbi_keys(self) -> None:
        """获取最新的 img_key 和 sub_key 用于 WBI 签名"""
        try:
            response = self.session.get(self._NAV_API_URL)
            response.raise_for_status()
            json_content = response.json()
            if json_content.get('code') == 0:
                img_url: str = json_content['data']['wbi_img']['img_url']
                sub_url: str = json_content['data']['wbi_img']['sub_url']
                self.img_key = img_url.rsplit('/', 1)[1].split('.')[0]
                self.sub_key = sub_url.rsplit('/', 1)[1].split('.')[0]
            else:
                print(f"获取 WBI Keys 失败：{json_content.get('message')}")
                cookies = self.session.cookies.get_dict()
                print(f"当前 Cookies: {cookies}")
        except requests.exceptions.RequestException as e:
            print(f"请求 WBI Keys 发生错误：{e}")
            # 打印当前 session 的 cookies
            cookies = self.session.cookies.get_dict()
            print(f"当前 Cookies: {cookies}")

        if not (self.img_key and self.sub_key):
            print("警告：未能成功加载 WBI Keys，评论或点赞请求可能失败或被风控。")

    def send_danmaku(self, bvid: str, msg: str, progress: int, mode: int = 1, fontsize: int = 25, color: int = 16777215,
                     pool: int = 0, is_up: bool = False) -> bool:
        """
        发送视频弹幕。

        :param bvid: 视频的 BV 号。
        :param msg: 弹幕内容 (长度小于 100 字符)。
        :param progress: 弹幕出现在视频内的时间 (单位为毫秒)。
        :param mode: 弹幕类型 (1:普通滚动, 4:底部, 5:顶部)。默认为 1。
        :param fontsize: 字号 (12, 16, 18, 25, 36, 45, 64)。默认为 25。
        :param color: 弹幕颜色 (十进制 RGB888 值)。默认为 16777215 (白色)。
        :param pool: 弹幕池 (0:普通, 1:字幕, 2:特殊)。默认为 0。
        :return: True 如果发送成功，否则 False。
        """
        print(f"准备向视频 {bvid} 发送弹幕: '{msg}'")

        video_info = self._get_video_info(bvid)
        if not video_info:
            print("弹幕发送失败：无法获取视频信息 (aid, cid)。")
            return False
        cid = video_info['cid']
        aid = video_info['aid']

        unsigned_data = {
            'type': 1,
            'oid': cid,
            'msg': msg,
            'aid': aid,
            'progress': progress,
            'color': color,
            'fontsize': fontsize,
            'pool': pool,
            'mode': mode,
            'rnd': int(time.time() * 1000000),
            'csrf': self.csrf_token,
            'web_location': '1315873',
        }
        if is_up:
            unsigned_data['checkbox_type'] = 4

        try:
            signed_data = self._sign_params_for_wbi(unsigned_data)
        except ValueError as e:
            print(f"弹幕发送失败：{e}")
            return False

        self.session.headers.update({"Referer": f"https://www.bilibili.com/video/{bvid}/"})
        try:
            response = self.session.post(self._DANMAKU_POST_API_URL, data=signed_data)
            response.raise_for_status()
            result = response.json()

            if result.get("code") == 0:
                dmid = result.get("data", {}).get("dmid_str")
                print(f"弹幕发送成功！Dmid: {dmid}")
                return True
            else:
                print(f"弹幕发送失败，错误码：{result.get('code')}, 信息：{result.get('message')}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"请求弹幕发送接口时发生错误：{e}")
            return False
        except Exception as e:
            print(f"弹幕发送时发生未知错误：{e}")
            return False

    def like_video(self, bvid: str) -> bool:
        """
        对指定的视频进行点赞。此API不需要WBI签名。
        :param bvid: 视频的 BV 号。
        :return: 点赞是否成功（或已点赞）。
        """
        post_data = {
            "bvid": bvid,
            "like": 1,  # 1 为点赞, 2 为取消点赞
            "csrf": self.csrf_token,
        }

        self.session.headers.update({
            "Referer": f"https://www.bilibili.com/video/{bvid}/"
        })

        try:
            response = self.session.post(self._VIDEO_LIKE_API_URL, data=post_data)
            response.raise_for_status()
            result = response.json()

            if result.get("code") == 0:
                return True
            elif result.get("code") == 65006:  # 65006 代表已经点赞过
                return True
            else:
                print(f"视频点赞失败，错误码：{result.get('code')}, 错误信息：{result.get('message')} {self.all_params}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"请求视频点赞时发生错误：{e}")
            return False
        except Exception as e:
            print(f"视频点赞时发生未知错误：{e}")
            return False

    # =========================================================================
    # ===================== 新增的方法：一键三连 & 分享 ========================
    # =========================================================================

    def triple_like_video(self, bvid: str) -> bool:
        """
        对指定的视频进行一键三连（点赞、投币、收藏）。
        此操作会将视频收藏到默认收藏夹。此API不需要WBI签名。

        :param bvid: 视频的 BV 号。
        :return: True 如果三连操作中至少有一项成功，否则 False。
        """
        print(f"准备对视频 {bvid} 进行一键三连...")

        # 修正：根据实际浏览器捕获的请求，补充了必要的参数以避免 -401 错误。
        # 这些参数可能用于行为验证或风控。
        post_data = {
            "bvid": bvid,
            "csrf": self.csrf_token,
            "from_spmid": "333.1387.homepage.video_card.click", # 模拟来源，可使用通用值
            "spmid": "333.788.0.0",                              # 同上
            "statistics": '{"appId":100,"platform":5}',           # 统计信息，在其他API中也出现
            "eab_x": 2,                                           # 行为/测试相关参数
            "ramval": 0,                                          # 行为/测试相关参数 (值可以为0或正整数)
            "source": "web_normal",                               # 来源标识
            "ga": 1                                               # 可能与风控相关
        }

        self.session.headers.update({
            "Referer": f"https://www.bilibili.com/video/{bvid}/"
        })

        try:
            response = self.session.post(self._TRIPLE_LIKE_API_URL, data=post_data)
            response.raise_for_status()
            result = response.json()

            if result.get("code") == 0:
                data = result.get("data", {})
                like_status = "成功" if data.get('like') else "失败(可能已点赞)"
                coin_status = "成功" if data.get('coin') else "失败(硬币不足或已投币)"
                fav_status = "成功" if data.get('fav') else "失败(可能已收藏)"
                print(f"一键三连操作完成。状态 -> 点赞: {like_status}, 投币: {coin_status}, 收藏: {fav_status}")
                # 只要三连中有一项成功，就认为操作成功
                return data.get('like') or data.get('coin') or data.get('fav')
            else:
                print(f"一键三连失败，错误码：{result.get('code')}, 错误信息：{result.get('message')}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"请求一键三连时发生错误：{e}")
            return False
        except Exception as e:
            print(f"一键三连时发生未知错误：{e}")
            return False

    def share_video(self, bvid: str) -> bool:
        """
        分享指定的视频以增加分享数。此API不需要WBI签名。

        :param bvid: 视频的 BV 号。
        :return: True 如果分享成功或已分享过，否则 False。
        """
        print(f"准备分享视频 {bvid}...")

        post_data = {
            "bvid": bvid,
            "csrf": self.csrf_token,
        }

        self.session.headers.update({
            "Referer": f"https://www.bilibili.com/video/{bvid}/"
        })

        try:
            response = self.session.post(self._SHARE_API_URL, data=post_data)
            response.raise_for_status()
            result = response.json()

            if result.get("code") == 0:
                share_count = result.get("data")
                print(f"视频分享成功！当前分享数：{share_count}")
                return True
            elif result.get("code") == 71000:  # 重复分享
                print("视频分享成功（今日已分享过）。")
                return True
            else:
                print(f"视频分享失败，错误码：{result.get('code')}, 错误信息：{result.get('message')}{self.all_params}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"请求分享视频时发生错误：{e}")
            return False
        except Exception as e:
            print(f"分享视频时发生未知错误：{e}")
            return False

    def reply_to_comment(self, bvid: str, message_content: str, root_rpid: int, parent_rpid: int,
                         type_code: int = 1):
        """
        回复指定的 Bilibili 评论 (发送楼中楼评论)。
        在发送回复前，此方法会先尝试为被回复的评论（父评论）点赞。

        :param bvid: 视频 BV 号。
        :param message_content: 回复内容。
        :param root_rpid: 根评论的 ID (顶级评论的 rpid)。
        :param parent_rpid: 直接回复的评论 ID (父评论的 rpid)。
        :param type_code: 目标类型，1 通常代表视频。
        :param use_proxy: 是否开启代理，仅作用于本次 COMMENT_ADD_API_URL 请求。
        :return: 新回复的 rpid (评论ID) 如果成功，否则返回 None。
        """
        # 仅影响本次请求的代理设置，其他请求不受影响
        proxy_env_keys = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
        old_proxy_env = {k: os.environ.get(k) for k in proxy_env_keys}
        # 清除环境变量中的代理设置
        for k in proxy_env_keys:
            if k in os.environ:
                del os.environ[k]

        video_info = self._get_video_info(bvid)
        if not video_info:
            print("回复失败：无法获取有效的视频信息。")
            return None , "无法获取有效的视频信息"
        oid = video_info['aid']

        print(f"准备回复 rpid={parent_rpid} 的评论，先尝试为其点赞...")
        self.like_comment(oid=oid, rpid=parent_rpid, type_code=type_code)

        post_body_data_unsigned = {
            "plat": 1,
            "oid": oid,
            "type": type_code,
            "message": message_content,
            "root": root_rpid,
            "parent": parent_rpid,
            "at_name_to_mid": "{}",
            "gaia_source": "main_web",
            "csrf": self.csrf_token,
            "statistics": '{"appId":100,"platform":5}',
            "dm_img_list": json.dumps(self._DM_IMG_LIST),
            "dm_img_str": self._DM_IMG_STR,
            "dm_cover_img_str": self._DM_COVER_IMG_STR,
            "dm_img_inter": self._DM_IMG_INTER,
        }

        try:
            signed_post_body_data = self._sign_params_for_wbi(post_body_data_unsigned)
        except ValueError as e:
            print(f"回复失败：{e}")
            return None, str(e)

        full_url = self._COMMENT_ADD_API_URL
        self.session.headers.update({
            "Referer": f"https://www.bilibili.com/video/{bvid}/"
        })

        # 根据 use_proxy 决定此次请求是否走代理
        proxies = self.all_params.get("proxies", {
                "http": None,
                "https": None
            })
        try:
            response = self.session.post(full_url, data=signed_post_body_data, proxies=proxies)
            response.raise_for_status()
            result = response.json()

            if result.get("code") == 0:
                print(f"回复成功！内容：'{message_content}'")
                rpid = None
                if result.get("data") and result["data"].get("reply"):
                    rpid = result["data"]["reply"]["rpid"]
                    print(f"获取到新回复的 rpid: {rpid}")
                    time.sleep(5)
                    self.like_comment(oid=oid, rpid=rpid, type_code=1)
                return rpid, "回复成功"
            else:
                print(f"回复失败，错误码：{result.get('code')}, 错误信息：{result.get('message')} {message_content}")
                return None, result.get('message', '未知错误')
        except requests.exceptions.RequestException as e:
            print(f"请求发生错误：{e}")
            return None, str(e)
        except Exception as e:
            print(f"发生未知错误：{e}")
            return None, str(e)
        finally:
            # 恢复原有的代理环境变量，确保全局环境不变
            for k, v in old_proxy_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def post_comment(self,
                     bvid: str,
                     message_content: str,
                     type_code: int = 1,
                     forward_to_dynamic: bool = False,
                     like_video: bool = False,
                     image_path: str = "") -> int | None:
        """
        发送 Bilibili 评论，并可在内部上传图片。
        :param bvid: 视频 BV 号。
        :param message_content: 评论内容。
        :param type_code: 目标类型，1 通常代表视频。
        :param forward_to_dynamic: 是否同时转发到动态。
        :param like_video: 是否先为视频点赞。
        :param image_path: 本地图片路径，若非空则上传并附带到评论中。
        :param use_proxy: 是否开启代理，仅作用于本次 COMMENT_ADD_API_URL 请求。
        :return: 评论的 rpid (评论ID) 如果成功，否则返回 None。
        """
        # 记录是否要开启代理的状态
        # 注意：此处仅对 COMMENT_ADD_API_URL 的请求生效，其他请求会恢复原状
        proxy_env_keys = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
        old_proxy_env = {k: os.environ.get(k) for k in proxy_env_keys}

        # 暂时清除环境变量中的代理设置，确保本次请求不会被其他请求影响
        for k in proxy_env_keys:
            if k in os.environ:
                del os.environ[k]

        try:
            if like_video:
                print(f"准备评论视频 {bvid}，先尝试为该视频点赞...")
                self.like_video(bvid=bvid)

            video_info = self._get_video_info(bvid)
            if not video_info:
                print("评论失败：无法获取有效的视频信息。")
                return None
            oid = video_info['aid']

            pictures_data = None
            if image_path:
                print(f"检测到 image_path='{image_path}'，开始上传图片...")
                upload_resp = upload_bilibili_image(
                    image_path=image_path,
                    cookies={"bili_jct": self.csrf_token, "SESSDATA": self.session.cookies.get("SESSDATA")},
                    csrf_token=self.csrf_token
                )
                if not upload_resp or upload_resp.status_code != 200:
                    print("图片上传失败，评论将不包含图片。")
                else:
                    data = upload_resp.json().get("data", {})
                    data["img_src"] = data.get("image_url")
                    data["img_width"] = data.get("image_width")
                    data["img_height"] = data.get("image_height")
                    pictures_data = [data]
                    print("图片上传并组装完成，准备在评论中附带图片。")

            post_body_data = {
                "plat": 1,
                "oid": oid,
                "type": type_code,
                "message": message_content,
                "at_name_to_mid": "{}",
                "gaia_source": "main_web",
                "csrf": self.csrf_token,
                "statistics": '{"appId":100,"platform":5}',
                "dm_img_list": json.dumps(self._DM_IMG_LIST),
                "dm_img_str": self._DM_IMG_STR,
                "dm_cover_img_str": self._DM_COVER_IMG_STR,
                "dm_img_inter": self._DM_IMG_INTER,
            }
            if pictures_data:
                post_body_data["pictures"] = json.dumps(pictures_data)
            if forward_to_dynamic:
                post_body_data["sync_to_dynamic"] = 1

            try:
                signed_data = self._sign_params_for_wbi(post_body_data)
            except ValueError as e:
                print(f"评论失败：{e}")
                return None

            self.session.headers.update({"Referer": f"https://www.bilibili.com/video/{bvid}/"})

            # 再次确保最终请求使用明确的代理策略
            proxies = self.all_params.get("proxies", {
                "http": None,
                "https": None
            })
            try:
                resp = self.session.post(self._COMMENT_ADD_API_URL, data=signed_data, proxies=proxies)
                resp.raise_for_status()
                result = resp.json()
                if result.get("code") == 0:
                    print("评论发送成功！")
                    rpid = result["data"]["reply"]["rpid"]
                    time.sleep(5)
                    self.like_comment(oid=oid, rpid=rpid, type_code=type_code)
                    return rpid
                else:
                    print(f"评论失败，错误码：{result['code']}, 信息：{result['message']}")
            except Exception as e:
                print(f"请求出错：{e}")

        finally:
            # 恢复原有的代理环境变量
            for k, v in old_proxy_env.items():
                if v is None:
                    if k in os.environ:
                        del os.environ[k]
                else:
                    os.environ[k] = v

        return None

    def like_comment(self, oid: int, rpid: int, type_code: int = 1) -> bool:
        """
        对指定的评论进行点赞。
        :param oid: 对象 ID (视频的 AID)。
        :param rpid: 评论区评论 ID (评论的 rpid)。
        :param type_code: 目标类型，1 通常代表视频。
        :return: 点赞是否成功。
        """
        post_body_data_unsigned = {
            "oid": oid,
            "rpid": rpid,
            "action": 1,
            "type": type_code,
            "csrf": self.csrf_token,
            "statistics": '{"appId":100,"platform":5}',
        }

        try:
            signed_post_body_data = self._sign_params_for_wbi(post_body_data_unsigned)
        except ValueError as e:
            print(f"评论点赞失败：{e}")
            return False

        full_url = self._COMMENT_ACTION_API_URL
        self.session.headers.update({
            "Referer": f"https://www.bilibili.com/video/av{oid}/"
        })

        try:
            response = self.session.post(full_url, data=signed_post_body_data)
            response.raise_for_status()
            result = response.json()

            if result.get("code") == 0:
                print(f"评论 rpid={rpid} 点赞成功。")
                return True
            else:
                print(f"评论点赞失败，错误码：{result.get('code')}, 错误信息：{result.get('message')}")
                if result.get("code") == -653:
                    print("评论点赞失败原因：可能已经点赞过。")
                return False
        except requests.exceptions.RequestException as e:
            print(f"请求评论点赞时发生错误：{e}")
            return False
        except Exception as e:
            print(f"评论点赞时发生未知错误：{e}")
            return False

    def get_user_videos(self, mid: int, desired_count: int, order: str = 'pubdate', tid: int = 0, keyword: str = '') -> list[dict] | None:
        """
        查询指定用户的投稿视频明细，并自动分页直到满足期望的数量。

        :param mid: 目标用户的 mid。
        :param desired_count: 期望获取的视频数量。
        :param order: 排序方式 ('pubdate': 最新发布, 'click': 最多播放, 'stow': 最多收藏)。默认为 'pubdate'。
        :param tid: 筛选的分区 tid。默认为 0 (不筛选)。
        :param keyword: 用于搜索的关键词。默认为空。
        :return: 包含视频信息字典的列表，如果查询过程中发生严重错误则返回 None。
        """
        print(f"准备查询用户 mid={mid} 的投稿视频，目标数量: {desired_count}...")

        collected_videos = []
        current_page = 1
        page_size = 25

        while len(collected_videos) < desired_count:
            print(f"正在获取第 {current_page} 页...")

            unsigned_params = {
                'mid': mid, 'order': order, 'tid': tid,
                'keyword': keyword, 'pn': current_page, 'ps': page_size,
            }

            try:
                signed_params = self._sign_params_for_wbi(unsigned_params)
            except ValueError as e:
                print(f"查询投稿视频失败：WBI签名错误 - {e}")
                return None

            self.session.headers.update({"Referer": f"https://space.bilibili.com/{mid}/video"})
            try:
                response = self.session.get(self._USER_VIDEOS_API_URL, params=signed_params)
                response.raise_for_status()
                result = response.json()

                if result.get("code") == 0:
                    data = result.get("data")
                    if not data:
                        print("API返回成功但没有data字段，停止获取。")
                        break

                    new_videos = data.get('list', {}).get('vlist', [])
                    if not new_videos:
                        print("当前页没有更多视频，已获取全部内容。")
                        break

                    collected_videos.extend(new_videos)

                    total_server_count = data.get('page', {}).get('count', 0)
                    if len(collected_videos) >= total_server_count:
                        print("已获取该用户所有视频。")
                        break

                    current_page += 1
                else:
                    print(f"查询投稿视频失败，错误码：{result.get('code')}, 信息：{result.get('message')}")
                    break

            except requests.exceptions.RequestException as e:
                print(f"请求用户投稿视频接口时发生网络错误：{e}")
                return None
            except Exception as e:
                print(f"查询用户投稿视频时发生未知错误：{e}")
                return None

        print(f"获取完成，共收集到 {len(collected_videos)} 个视频。")
        return collected_videos[:desired_count]


# --- 主逻辑 ---
if __name__ == "__main__":
    config_map = init_config()
    user_name = 'hong'
    target_value = None
    for uid, value in config_map.items():
        if value.get('name') == user_name:
            target_value = value
            break


    target_bvid = "BV1c3t1zDEzw"
    comment_text = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]"
    comment_type = 1

    commenter = BilibiliCommenter(total_cookie=target_value['total_cookie'], csrf_token=target_value['BILI_JCT'],all_params=target_value['all_params'])
    # commenter.pin_comment(target_bvid, 271871684816)
    #
    # --- 步骤 1: 发送一条顶级评论 (现在会先点赞视频) ---
    print("-" * 30)
    print("步骤 1: 尝试发送一条顶级评论...")
    posted_rpid = commenter.post_comment(
        target_bvid, comment_text, comment_type,
        like_video=True    )

    # # --- 步骤 2: 如果顶级评论成功，回复这条评论 ---
    # if posted_rpid:
    #     print("-" * 30)
    #     print(f"步骤 2: 顶级评论发送成功，rpid 为 {posted_rpid}。现在尝试回复这条评论...")
    #     time.sleep(3)
    #     reply_text = f"这是对 rpid={posted_rpid} 的回复。[{time.strftime('%Y-%m-%d %H:%M:%S')}]"
    #     reply_rpid = commenter.reply_to_comment(
    #         bvid=target_bvid, message_content=reply_text,
    #         root_rpid=posted_rpid, parent_rpid=posted_rpid, type_code=comment_type,use_proxy=True
    #     )
    #     if reply_rpid:
    #         print("\n回复操作成功完成！")
    #     else:
    #         print("\n回复操作失败。")
    # else:
    #     print("-" * 30)
    #     print("顶级评论发送失败，无法进行回复操作。")
    # #
    # # # --- 步骤 3: 发送一条弹幕 ---
    # # print("-" * 30)
    # # print("步骤 3: 尝试发送一条弹幕...")
    # # danmaku_text = f"大家怎么样，心情都好"
    # # danmaku_time_ms = 2100
    # # danmaku_sent = commenter.send_danmaku(
    # #     bvid=target_bvid, msg=danmaku_text, progress=danmaku_time_ms, is_up=True
    # # )
    # # if danmaku_sent:
    # #     print("弹幕发送流程成功完成！")
    # # else:
    # #     print("弹幕发送流程失败。")
    # #
    # # # --- 步骤 4: 查询用户投稿视频 (修正了结果处理的BUG) ---
    # # print("-" * 30)
    # # print("步骤 4: 尝试查询用户投稿视频...")
    # # user_mid_to_query = 282994  # 以文档中的"warma"为例
    # # videos_list = commenter.get_user_videos(mid=user_mid_to_query, desired_count=5, order='click')
    # #
    # # if videos_list:
    # #     print(f"\n成功获取到用户 {user_mid_to_query} 的视频列表（按播放量前5）:")
    # #     if not videos_list:
    # #         print("该用户没有视频。")
    # #     else:
    # #         for video in videos_list:
    # #             print(f"  - 标题: {video.get('title')}")
    # #             print(f"    BVID: {video.get('bvid')}, 播放量: {video.get('play')}, 弹幕: {video.get('video_review')}")
    # # else:
    # #     print("查询用户投稿视频失败。")
    #
    # # --- 新增步骤 5: 分享视频 ---
    # print("-" * 30)
    # print("步骤 5: 尝试分享视频...")
    # share_success = commenter.share_video(bvid=target_bvid)
    # if share_success:
    #     print("分享操作流程成功完成！")
    # else:
    #     print("分享操作流程失败。")
    #
    # # --- 新增步骤 6: 一键三连视频 ---
    # print("-" * 30)
    # print("步骤 6: 尝试对视频进行一键三连...")
    # triple_like_success = commenter.triple_like_video(bvid=target_bvid)
    # if triple_like_success:
    #     print("一键三连操作流程成功完成！")
    # else:
    #     print("一键三连操作流程失败。")