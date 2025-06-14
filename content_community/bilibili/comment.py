# -- coding: utf-8 --
"""
:authors:
    zhuxiaohu
:create_date:
    2025/5/26 22:51
:last_date:
    2025/5/27 23:00 (重构为类结构，提高可读性和复用性)
:description:
    Bilibili 评论发送脚本。
    注意：dm_img_* 字段是硬编码的设备指纹，长期使用可能导致风控或失效。
          建议定期更新这些值或考虑使用自动化浏览器。
"""
import requests
import urllib.parse
import time
from hashlib import md5
import json
from common_utils.common_utils import get_config

class BilibiliCommenter:
    """
    用于发送 Bilibili 评论的类。
    封装了获取 AID、WBI 签名生成和实际评论发送的逻辑。
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
    _COMMENT_API_URL = "https://api.bilibili.com/x/v2/reply/add"
    _NAV_API_URL = "https://api.bilibili.com/x/web-interface/nav"
    _VIEW_API_URL_TEMPLATE = "https://api.bilibili.com/x/web-interface/view?bvid={bvid_str}"

    def __init__(self, total_cookie: str, csrf_token: str):
        """
        初始化 BilibiliCommenter 实例。
        :param total_cookie: 包含 SESSDATA 和 bili_jct 的完整 Cookie 字符串。
        :param csrf_token: Bilibili 的 CSRF Token (即 bili_jct 的值)。
        """
        self.session = requests.Session()
        self.csrf_token = csrf_token
        self.total_cookie = total_cookie

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
        # 注意: requests.Session 的 cookies 属性通常能自动处理 set-cookie 头
        # 但手动设置确保了初始状态
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
        filtered_value = ''.join(filter(lambda chr: chr not in "!'()*", str(value_str)))
        return urllib.parse.quote(filtered_value, safe='')

    def _get_aid_from_bvid(self, bvid_str: str) -> int | None:
        """根据 BV 号获取视频的 AID"""
        url = self._VIEW_API_URL_TEMPLATE.format(bvid_str=bvid_str)
        # 临时的 Referer 头，用于获取 AID，因为此时还没有视频的 Referer
        temp_headers = {"Referer": "https://www.bilibili.com/"}
        try:
            response = self.session.get(url, headers=temp_headers)
            response.raise_for_status()
            data = response.json()
            if data.get("code") == 0 and data.get("data"):
                return data["data"]["aid"]
            else:
                print(f"获取 AID 失败: {data.get('message')}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"请求 AID 发生错误：{e}")
            return None

    def _load_wbi_keys(self) -> None:
        """获取最新的 img_key 和 sub_key 用于 WBI 签名"""
        try:
            response = self.session.get(self._NAV_API_URL)  # Session 会自动带上Cookie
            response.raise_for_status()
            json_content = response.json()
            if json_content.get('code') == 0:
                img_url: str = json_content['data']['wbi_img']['img_url']
                sub_url: str = json_content['data']['wbi_img']['sub_url']
                self.img_key = img_url.rsplit('/', 1)[1].split('.')[0]
                self.sub_key = sub_url.rsplit('/', 1)[1].split('.')[0]
                print("WBI Keys 加载成功。")
            else:
                print(f"获取 WBI Keys 失败：{json_content.get('message')}")
        except requests.exceptions.RequestException as e:
            print(f"请求 WBI Keys 发生错误：{e}")

        if not (self.img_key and self.sub_key):
            print("警告：未能成功加载 WBI Keys，评论请求可能失败或被风控。")

    def _generate_wbi_signed_url_query(self, params: dict) -> str:
        """
        为给定的 URL 查询参数字典生成 WBI 签名，并返回完整的签名后的 URL 查询字符串。
        WBI 签名会将 `wts` 和 `w_rid` 添加到 params 中。
        """
        if not (self.img_key and self.sub_key):
            # 尝试重新加载一次WBI keys
            self._load_wbi_keys()
            if not (self.img_key and self.sub_key):
                raise ValueError("WBI Keys 不可用，无法生成签名。")

        mixin_key = self._get_mixin_key(self.img_key + self.sub_key)
        curr_time = round(time.time())

        # 复制一份，避免修改原始字典
        params_for_wbi = params.copy()
        params_for_wbi['wts'] = curr_time

        # 按照 key 重排参数 (排序是为了MD5计算的输入一致性)
        sorted_params_for_md5 = dict(sorted(params_for_wbi.items()))

        # 过滤 value 中的 "!'()*" 字符并进行 URL 编码
        encoded_parts_for_md5 = []
        for k, v in sorted_params_for_md5.items():
            encoded_key = urllib.parse.quote(str(k), safe='')
            encoded_value = self._filter_and_encode_param_value(v)
            encoded_parts_for_md5.append(f"{encoded_key}={encoded_value}")

        query_for_md5 = '&'.join(encoded_parts_for_md5)

        wbi_sign = md5((query_for_md5 + mixin_key).encode()).hexdigest()

        # 构造最终的 URL Query 字符串：原始参数（排序），然后添加 wts 和 w_rid
        # 原始参数（不含wts和w_rid）再次排序和编码，用于构建最终URL
        original_params_for_url = {k: v for k, v in params.items()}  # 这里的params已经是最初传入的了

        sorted_original_parts_for_url = []
        for k, v in sorted(original_params_for_url.items()):
            encoded_key = urllib.parse.quote(str(k), safe='')
            encoded_value = self._filter_and_encode_param_value(v)
            sorted_original_parts_for_url.append(f"{encoded_key}={encoded_value}")

        final_query = '&'.join(sorted_original_parts_for_url)
        final_query += f"&wts={curr_time}"
        final_query += f"&w_rid={wbi_sign}"

        return final_query

    def post_comment(self, bvid: str, message_content: str, type_code: int = 1) -> bool:
        """
        发送 Bilibili 评论。
        :param bvid: 视频 BV 号。
        :param message_content: 评论内容。
        :param type_code: 目标类型，1 通常代表视频。
        :return: 评论是否成功。
        """
        # print(f"尝试评论视频 BV 号：{bvid}，内容：'{message_content}'")

        # 1. 获取 AID
        oid = self._get_aid_from_bvid(bvid)
        if not oid:
            print("评论失败：无法获取有效的 AID。")
            return False

        # 2. 准备 URL 查询参数字典（参与 WBI 签名）
        url_query_params_for_wbi = {
            "dm_img_list": json.dumps(self._DM_IMG_LIST),
            "dm_img_str": self._DM_IMG_STR,
            "dm_cover_img_str": self._DM_COVER_IMG_STR,
            "dm_img_inter": self._DM_IMG_INTER,
            "csrf": self.csrf_token  # csrf也作为url参数的一部分参与WBI签名
        }

        # 3. 生成 WBI 签名后的 URL 查询字符串
        try:
            signed_url_query_string = self._generate_wbi_signed_url_query(url_query_params_for_wbi)
        except ValueError as e:
            print(f"评论失败：{e}")
            return False

        full_url = f"{self._COMMENT_API_URL}?{signed_url_query_string}"

        # 4. 准备 POST 请求的 Body 数据
        post_body_data = {
            "plat": 1,
            "oid": oid,
            "type": type_code,
            "message": message_content,
            "at_name_to_mid": "{}",
            "gaia_source": "main_web",
            "csrf": self.csrf_token,
            "statistics": '{"appId":100,"platform":5}'
        }

        # 5. 更新 Referer 头（针对当前评论的视频页面）
        self.session.headers.update({
            "Referer": f"https://www.bilibili.com/video/{bvid}/?spm_id_from=333.1387.homepage.video_card.click&vd_source=5d365f9cfcf15bb4bacfcd69a69fc4d4"
        })

        # 6. 发送 POST 请求
        try:
            response = self.session.post(full_url, data=post_body_data)
            response.raise_for_status()
            result = response.json()

            if result.get("code") == 0:
                # print("评论发送成功！")
                if result.get("data") and result["data"].get("reply"):
                    pass
                    # print(f"评论ID (rpid): {result['data']['reply']['rpid']}")
                # print(f"Bilibili 返回消息: {result.get('message')}")
                return True
            else:
                print(f"评论发送失败，错误码：{result.get('code')}, 错误信息：{result.get('message')}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"请求发生错误：{e}")
            return False
        except Exception as e:
            print(f"发生未知错误：{e}")
            return False


# --- 主逻辑 ---
if __name__ == "__main__":
    # --- 从配置中获取敏感信息 ---
    # 确保您的 common_utils.common_utils.get_config 函数能够正确返回这些值
    csrf_token = get_config("bilibili_csrf_token")
    # total_cookie 应该是一个包含 SESSDATA 和 bili_jct 的完整字符串
    # 例如："SESSDATA=xxxxxxxxxxxx; bili_jct=yyyyyyyyyyyy"
    total_cookie = get_config("bilibili_total_cookie")

    if not csrf_token or not total_cookie:
        print("错误：请在 common_utils.common_utils.get_config 中配置 csrf_token 和 total_cookie。")
        exit()

    # --- 实例化评论器 ---
    commenter = BilibiliCommenter(total_cookie=total_cookie, csrf_token=csrf_token)

    # --- 评论配置 ---
    target_bvid = "BV1Sz4y1S7bx"  # 视频 BV 号
    comment_text = "打得不错"  # 评论内容
    comment_type = 1  # 目标类型，1 一般代表视频

    # --- 执行评论 ---
    success = commenter.post_comment(target_bvid, comment_text, comment_type)

    if success:
        print("\n评论操作完成：成功。")
    else:
        print("\n评论操作完成：失败。")