# -- coding: utf-8 --
"""
:authors:
    zhuxiaohu
:create_date:
    2025/5/26 22:51
:last_date:
    2025/5/28 00:30 (修正 WBI 参数位置，放入 Body)
:description:
    Bilibili 评论发送及点赞脚本。
    注意：dm_img_* 字段是硬编码的设备指纹，长期使用可能导致风控或失效。
          建议定期更新这些值或考虑使用自动化浏览器。
"""
import requests
import urllib.parse
import time
from hashlib import md5
import json
# 确保您的 common_utils.common_utils 模块和 get_config 函数可用
# 如果没有，请替换为实际获取配置的方法，例如从文件读取或直接硬编码(不推荐敏感信息)
try:
    from common_utils.common_utils import get_config
except ImportError:
    print("警告: 未找到 common_utils.common_utils 模块。请确保该模块存在或手动设置配置。")
    # 提供一个简单的模拟函数，实际使用请替换
    def get_config(key):
        configs = {
            "bilibili_csrf_token": "YOUR_CSRF_TOKEN", # <-- 替换为你的 bili_jct 值
            "bilibili_total_cookie": "SESSDATA=YOUR_SESSDATA; bili_jct=YOUR_CSRF_TOKEN;" # <-- 替换为你的完整Cookie字符串
        }
        return configs.get(key)

class BilibiliCommenter:
    """
    用于发送 Bilibili 评论并尝试点赞的类。
    封装了获取 AID、WBI 签名生成和实际评论发送/点赞的逻辑。
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
    _COMMENT_ACTION_API_URL = "https://api.bilibili.com/x/v2/reply/action" # 用于点赞/点踩
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
            # We are sending form data in the body
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
        # Ensure the value is a string
        value_str = str(value_str)
        filtered_value = ''.join(filter(lambda chr: chr not in "!'()*", value_str))
        # Use quote_plus would encode space to '+', but WBI expects %20
        # Use quote, safe='', then replace + with %20
        quoted = urllib.parse.quote(filtered_value, safe='')
        # Handle potential '+' from quoting non-ASCII characters if needed,
        # though for typical parameters like numbers/strings it's less likely.
        # The filter handles the '!'()*' part.
        # Let's refine to handle spaces specifically as %20,
        # and let quote handle other special chars.
        return urllib.parse.quote(filtered_value, safe='').replace(' ', '%20')


    def _sign_params_for_wbi(self, params: dict) -> dict:
        """
        为给定的参数字典生成 WBI 签名，并将 wts 和 w_rid 添加到字典中。
        返回修改后的字典，这个字典可以直接用于 POST 请求的 data 参数。
        """
        if not (self.img_key and self.sub_key):
            # 尝试重新加载一次WBI keys
            self._load_wbi_keys()
            if not (self.img_key and self.sub_key):
                raise ValueError("WBI Keys 不可用，无法生成签名。")

        mixin_key = self._get_mixin_key(self.img_key + self.sub_key)
        curr_time = round(time.time())

        # 复制一份，避免修改原始字典
        params_with_wbi = params.copy()
        params_with_wbi['wts'] = curr_time

        # 按照 key 重排参数 (排序是为了MD5计算的输入一致性)
        sorted_params_for_md5 = dict(sorted(params_with_wbi.items()))

        # 过滤 value 中的 "!'()*" 字符并进行 URL 编码
        encoded_parts_for_md5 = []
        for k, v in sorted_params_for_md5.items():
            encoded_key = urllib.parse.quote(str(k), safe='')
            encoded_value = self._filter_and_encode_param_value(v)
            encoded_parts_for_md5.append(f"{encoded_key}={encoded_value}")

        query_for_md5 = '&'.join(encoded_parts_for_md5)

        wbi_sign = md5((query_for_md5 + mixin_key).encode()).hexdigest()

        # 将 w_rid 添加到参数字典中
        params_with_wbi['w_rid'] = wbi_sign

        # 返回包含所有参数（包括wts和w_rid）的字典
        return params_with_wbi


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
                # print("WBI Keys 加载成功。")
            else:
                print(f"获取 WBI Keys 失败：{json_content.get('message')}")
        except requests.exceptions.RequestException as e:
            print(f"请求 WBI Keys 发生错误：{e}")

        if not (self.img_key and self.sub_key):
            print("警告：未能成功加载 WBI Keys，评论或点赞请求可能失败或被风控。")

    def post_comment(self, bvid: str, message_content: str, type_code: int = 1) -> int | None:
        """
        发送 Bilibili 评论。
        :param bvid: 视频 BV 号。
        :param message_content: 评论内容。
        :param type_code: 目标类型，1 通常代表视频。
        :return: 评论的 rpid (评论ID) 如果成功，否则返回 None。
        """
        # print(f"尝试评论视频 BV 号：{bvid}，内容：'{message_content}'")

        # 1. 获取 AID (oid)
        oid = self._get_aid_from_bvid(bvid)
        if not oid:
            print("评论失败：无法获取有效的 AID。")
            return None

        # 2. 准备 POST 请求的 Body 数据 (这些是需要签名的参数)
        # 注意：所有这些参数都将参与 WBI 签名，并通过 POST body 发送
        post_body_data_unsigned = {
            "plat": 1,
            "oid": oid,
            "type": type_code,
            "message": message_content,
            "at_name_to_mid": "{}",
            "gaia_source": "main_web",
            "csrf": self.csrf_token,
            "statistics": '{"appId":100,"platform":5}',
            # dm_img_* fields are also part of the signature params for comment add
            "dm_img_list": json.dumps(self._DM_IMG_LIST),
            "dm_img_str": self._DM_IMG_STR,
            "dm_cover_img_str": self._DM_COVER_IMG_STR,
            "dm_img_inter": self._DM_IMG_INTER,
        }

        # 3. 生成 WBI 签名参数 (wts, w_rid) 并添加到 body 数据中
        try:
            # _sign_params_for_wbi now returns the complete body data including wts/w_rid
            signed_post_body_data = self._sign_params_for_wbi(post_body_data_unsigned)
        except ValueError as e:
            print(f"评论失败：{e}")
            return None

        # 4. 构造完整的 URL (POST 请求，WBI参数在Body中，URL无WBI参数)
        full_url = self._COMMENT_ADD_API_URL # WBI params are in the body

        # 5. 更新 Referer 头（针对当前评论的视频页面）
        self.session.headers.update({
            # Using BVid in referer is fine
            "Referer": f"https://www.bilibili.com/video/{bvid}/?spm_id_from=333.1387.homepage.video_card.click"
        })

        # 6. 发送 POST 请求
        try:
            # Pass the complete signed dictionary as data
            response = self.session.post(full_url, data=signed_post_body_data)
            response.raise_for_status()
            result = response.json()

            if result.get("code") == 0:
                # print("评论发送成功！")
                rpid = None
                if result.get("data") and result["data"].get("reply"):
                    rpid = result["data"]["reply"]["rpid"]
                    time.sleep(5)  # 延时10秒，避免过快提交评论
                    # 评论的type和视频的type通常一致
                    like_success = self.like_comment(oid=oid, rpid=rpid, type_code=1)
                    if like_success:
                        # print("点赞操作完成：成功。")
                        pass
                    else:
                        # print("点赞操作完成：失败。")
                        pass
                return rpid # 返回评论的 rpid
            else:
                print(f"评论发送失败，错误码：{result.get('code')}, 错误信息：{result.get('message')}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"请求发生错误：{e}")
            return None
        except Exception as e:
            print(f"发生未知错误：{e}")
            return None

    def like_comment(self, oid: int, rpid: int, type_code: int = 1) -> bool:
        """
        对指定的评论进行点赞。
        :param oid: 对象 ID (视频的 AID)。
        :param rpid: 评论区评论 ID (评论的 rpid)。
        :param type_code: 目标类型，1 通常代表视频。
        :return: 点赞是否成功。
        """
        # print(f"尝试点赞评论 rpid={rpid} (oid={oid})")

        # 1. 准备 POST 请求的 Body 数据 (这些是需要签名的参数)
        # 注意：所有这些参数都将参与 WBI 签名，并通过 POST body 发送
        post_body_data_unsigned = {
            "oid": oid,
            "rpid": rpid,
            "action": 1, # 1 为点赞
            "type": type_code,
            "csrf": self.csrf_token,
            "statistics": '{"appId":100,"platform":5}', # 点赞同样需要statistics
        }

        # 2. 生成 WBI 签名参数 (wts, w_rid) 并添加到 body 数据中
        try:
            # _sign_params_for_wbi now returns the complete body data including wts/w_rid
            signed_post_body_data = self._sign_params_for_wbi(post_body_data_unsigned)
        except ValueError as e:
            print(f"点赞失败：{e}")
            return False

        # 3. 构造完整的 URL (POST 请求，WBI参数在Body中，URL无WBI参数)
        full_url = self._COMMENT_ACTION_API_URL # WBI params are in the body

        # 4. 更新 Referer 头 (使用视频页面的 Referer)
        # Use av{oid} format or derive bvid if possible. av format is generally safer if only oid is available.
        # Let's use av format here as we have oid.
        self.session.headers.update({
             "Referer": f"https://www.bilibili.com/video/av{oid}/"
        })

        # 5. 发送 POST 请求
        try:
            # Pass the complete signed dictionary as data
            response = self.session.post(full_url, data=signed_post_body_data)
            response.raise_for_status()
            result = response.json()

            if result.get("code") == 0:
                return True
            else:
                print(f"评论点赞失败，错误码：{result.get('code')}, 错误信息：{result.get('message')}")
                # 常见的失败原因可能是已经点赞过 (错误码 -653)，这通常不算真正的失败
                if result.get("code") == -653:
                     print("评论点赞失败原因：可能已经点赞过 (错误码 -653)。")
                     # You might choose to return True here if -653 is acceptable as "already liked"
                     # For now, we return False as the *attempt* to like didn't change the state (it was already liked)
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
    # 或者直接在这里硬编码 (不推荐敏感信息)
    csrf_token = get_config("bilibili_csrf_token")
    # total_cookie 应该是一个包含 SESSDATA 和 bili_jct 的完整字符串
    # 例如："SESSDATA=xxxxxxxxxxxx; bili_jct=yyyyyyyyyyyy"
    total_cookie = get_config("bilibili_total_cookie")

    if not csrf_token or not total_cookie or "YOUR_CSRF_TOKEN" in csrf_token or "YOUR_SESSDATA" in total_cookie:
        print("\n!!!!!!!!! 配置错误 !!!!!!!!!")
        print("请编辑脚本，替换 common_utils.common_utils.get_config 返回的值，")
        print("或者直接修改这里的 csrf_token 和 total_cookie 变量，填写您的实际信息。")
        print("csrf_token 就是您的 bili_jct cookie 的值。")
        print("total_cookie 是完整的 Cookie 字符串，包含 SESSDATA 和 bili_jct 等。")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        exit()


    # --- 评论和点赞配置 ---
    target_bvid = "BV1Sz4y1S7bx"  # 视频 BV 号
    comment_text = f"操作很好" # 评论内容 (增加时间戳，方便观察新评论)
    comment_type = 1  # 目标类型，1 一般代表视频

    # 在评论前获取 AID (oid)，因为点赞也需要
    print(f"尝试获取视频 {target_bvid} 的 AID...")
    commenter = BilibiliCommenter(total_cookie=total_cookie, csrf_token=csrf_token)
    oid = commenter._get_aid_from_bvid(target_bvid) # 调用内部方法获取AID
    if not oid:
        print(f"无法获取视频 {target_bvid} 的 AID，无法继续评论和点赞。")
        exit()
    print(f"视频 {target_bvid} 的 AID 是 {oid}")


    # --- 执行评论 ---
    print("-" * 20)
    print("开始执行评论操作...")


    posted_rpid = commenter.post_comment(target_bvid, comment_text, comment_type)