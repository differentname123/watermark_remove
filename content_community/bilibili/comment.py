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
import random

import requests
import urllib.parse
import time
from hashlib import md5
import json

dm_img_inter_list = ['{"ds":[{"t":0,"c":"","p":[1826,102,2937],"s":[65,3275,2110]}],"wh":[4832,4299,50],"of":[1872,2744,372]}',
                        '{"ds":[{"t":0,"c":"","p":[849,21,1388],"s":[149,7249,-5502]}],"wh":[4682,4249,0],"of":[2200,3200,400]}',
                     '{"ds":[{"t":0,"c":"","p":[1341,67,132],"s":[456,3154,2196]}],"wh":[5032,7614,58],"of":[4324,5904,208]}',
{"ds":[{"t":0,"c":"","p":[1152,4,69],"s":[204,2902,1692]}],"wh":[5182,7664,108],"of":[3471,4946,477]}


                     ]
dm_img_list_list = [[{"x":4292,"y":126,"z":0,"timestamp":19145,"k":98,"type":0},{"x":4406,"y":280,"z":63,"timestamp":19245,"k":85,"type":0},{"x":4583,"y":543,"z":143,"timestamp":19347,"k":115,"type":0},{"x":4764,"y":896,"z":153,"timestamp":19448,"k":88,"type":0},{"x":4994,"y":1173,"z":311,"timestamp":19550,"k":101,"type":0},{"x":4856,"y":979,"z":42,"timestamp":19651,"k":86,"type":0},{"x":5415,"y":947,"z":258,"timestamp":19752,"k":77,"type":0},{"x":6040,"y":520,"z":658,"timestamp":19852,"k":121,"type":0},{"x":5791,"y":120,"z":425,"timestamp":19954,"k":78,"type":0},{"x":6006,"y":328,"z":661,"timestamp":20058,"k":67,"type":0},{"x":6145,"y":626,"z":829,"timestamp":20158,"k":100,"type":0},{"x":5958,"y":523,"z":620,"timestamp":20258,"k":88,"type":0},{"x":6495,"y":1077,"z":1152,"timestamp":20493,"k":120,"type":0},{"x":6013,"y":824,"z":627,"timestamp":20594,"k":71,"type":0},{"x":5924,"y":1307,"z":478,"timestamp":20694,"k":104,"type":0},{"x":6595,"y":2311,"z":1070,"timestamp":20795,"k":105,"type":0},{"x":7561,"y":1770,"z":1589,"timestamp":22150,"k":80,"type":0},{"x":7018,"y":1196,"z":1047,"timestamp":22262,"k":114,"type":0},{"x":6881,"y":1003,"z":1009,"timestamp":23221,"k":94,"type":0},{"x":6972,"y":2558,"z":1975,"timestamp":23321,"k":110,"type":0},{"x":3595,"y":1230,"z":1122,"timestamp":39814,"k":107,"type":0},{"x":3512,"y":-325,"z":2051,"timestamp":39922,"k":80,"type":0},{"x":2378,"y":-1978,"z":933,"timestamp":40024,"k":102,"type":0},{"x":1563,"y":-3090,"z":89,"timestamp":40124,"k":97,"type":0},{"x":2120,"y":-3317,"z":422,"timestamp":40224,"k":87,"type":0},{"x":2611,"y":-3701,"z":663,"timestamp":40324,"k":109,"type":0},{"x":3167,"y":-3157,"z":1071,"timestamp":40424,"k":113,"type":0},{"x":3140,"y":-2951,"z":759,"timestamp":40525,"k":90,"type":0},{"x":5930,"y":175,"z":2886,"timestamp":40628,"k":81,"type":0},{"x":6116,"y":469,"z":2702,"timestamp":40730,"k":62,"type":0},{"x":4812,"y":-752,"z":1264,"timestamp":40831,"k":94,"type":0},{"x":7155,"y":1753,"z":3512,"timestamp":40932,"k":67,"type":0},{"x":4967,"y":-387,"z":1318,"timestamp":41032,"k":110,"type":0},{"x":5791,"y":475,"z":2143,"timestamp":41137,"k":67,"type":0},{"x":3809,"y":-1507,"z":161,"timestamp":41286,"k":110,"type":0},{"x":7493,"y":2188,"z":3835,"timestamp":41389,"k":99,"type":0},{"x":7067,"y":2341,"z":3351,"timestamp":41489,"k":95,"type":0},{"x":5172,"y":2882,"z":772,"timestamp":41589,"k":88,"type":0},{"x":5330,"y":4261,"z":740,"timestamp":41690,"k":122,"type":0},{"x":8621,"y":8410,"z":4286,"timestamp":42830,"k":98,"type":0},{"x":5785,"y":3021,"z":1404,"timestamp":42929,"k":63,"type":0},{"x":5292,"y":790,"z":881,"timestamp":43029,"k":106,"type":0},{"x":8216,"y":3223,"z":3806,"timestamp":43132,"k":85,"type":0},{"x":5021,"y":-114,"z":623,"timestamp":43233,"k":61,"type":0},{"x":6706,"y":1455,"z":2334,"timestamp":43333,"k":85,"type":0},{"x":5889,"y":589,"z":1549,"timestamp":43434,"k":87,"type":0},{"x":9546,"y":4226,"z":5220,"timestamp":43535,"k":116,"type":0},{"x":7761,"y":2438,"z":3444,"timestamp":43636,"k":72,"type":0},{"x":6246,"y":917,"z":1947,"timestamp":43736,"k":87,"type":0},{"x":8836,"y":3507,"z":4537,"timestamp":44109,"k":103,"type":0}],
[{"x":3528,"y":-1207,"z":0,"timestamp":8465,"k":100,"type":0},{"x":3540,"y":-1226,"z":36,"timestamp":8566,"k":91,"type":0},{"x":3553,"y":-1229,"z":120,"timestamp":8667,"k":68,"type":0},{"x":3654,"y":-1131,"z":230,"timestamp":8767,"k":109,"type":0},{"x":3562,"y":-1234,"z":171,"timestamp":8867,"k":109,"type":0},{"x":3856,"y":-1004,"z":519,"timestamp":8969,"k":81,"type":0},{"x":3849,"y":-1024,"z":528,"timestamp":9072,"k":78,"type":0},{"x":3850,"y":-1032,"z":556,"timestamp":9173,"k":72,"type":0},{"x":3492,"y":-1416,"z":276,"timestamp":9275,"k":80,"type":0},{"x":4174,"y":-738,"z":970,"timestamp":9375,"k":87,"type":0},{"x":4031,"y":-884,"z":836,"timestamp":9527,"k":80,"type":0},{"x":3307,"y":-1611,"z":121,"timestamp":9855,"k":111,"type":0},{"x":3843,"y":-1109,"z":667,"timestamp":9955,"k":126,"type":0},{"x":3903,"y":-1081,"z":731,"timestamp":10303,"k":124,"type":0},{"x":3540,"y":-1462,"z":445,"timestamp":10404,"k":90,"type":0},{"x":2925,"y":-1911,"z":68,"timestamp":10505,"k":118,"type":0},{"x":3389,"y":-1211,"z":744,"timestamp":10606,"k":124,"type":0},{"x":2538,"y":-1688,"z":82,"timestamp":10706,"k":64,"type":0},{"x":2735,"y":-1277,"z":350,"timestamp":10807,"k":104,"type":0},{"x":4096,"y":126,"z":1723,"timestamp":10908,"k":122,"type":0},{"x":3220,"y":-666,"z":871,"timestamp":11008,"k":125,"type":0},{"x":4055,"y":240,"z":1723,"timestamp":11111,"k":74,"type":0},{"x":2457,"y":-1292,"z":134,"timestamp":11212,"k":69,"type":0},{"x":3620,"y":-105,"z":1294,"timestamp":11313,"k":115,"type":0},{"x":5004,"y":1323,"z":2684,"timestamp":12847,"k":88,"type":0},{"x":3220,"y":-400,"z":901,"timestamp":12951,"k":92,"type":0},{"x":2870,"y":-630,"z":559,"timestamp":13051,"k":83,"type":0},{"x":5102,"y":1741,"z":2811,"timestamp":13152,"k":69,"type":0},{"x":4075,"y":887,"z":1817,"timestamp":13254,"k":121,"type":0},{"x":4603,"y":1571,"z":2383,"timestamp":13354,"k":91,"type":0},{"x":4617,"y":1732,"z":2439,"timestamp":13454,"k":94,"type":0},{"x":2403,"y":-426,"z":241,"timestamp":13563,"k":126,"type":0},{"x":5290,"y":2460,"z":3131,"timestamp":13783,"k":112,"type":0},{"x":2723,"y":-214,"z":563,"timestamp":13883,"k":122,"type":0},{"x":5954,"y":2995,"z":3791,"timestamp":13990,"k":117,"type":0},{"x":6092,"y":3133,"z":3929,"timestamp":14191,"k":65,"type":1},{"x":2657,"y":-317,"z":493,"timestamp":14615,"k":124,"type":0},{"x":3941,"y":957,"z":1784,"timestamp":16991,"k":74,"type":0},{"x":3662,"y":674,"z":1494,"timestamp":17383,"k":81,"type":0},{"x":3470,"y":464,"z":1149,"timestamp":17485,"k":95,"type":0},{"x":4167,"y":1127,"z":1672,"timestamp":17587,"k":60,"type":0},{"x":3456,"y":327,"z":699,"timestamp":17689,"k":115,"type":0},{"x":4329,"y":1241,"z":1357,"timestamp":17789,"k":118,"type":0},{"x":3857,"y":869,"z":608,"timestamp":17889,"k":84,"type":0},{"x":8143,"y":5197,"z":4768,"timestamp":17990,"k":107,"type":0},{"x":3540,"y":633,"z":25,"timestamp":18091,"k":111,"type":0},{"x":7140,"y":4240,"z":3604,"timestamp":18194,"k":60,"type":0},{"x":3881,"y":983,"z":339,"timestamp":18305,"k":93,"type":0},{"x":4685,"y":1790,"z":1134,"timestamp":18422,"k":105,"type":0},{"x":5715,"y":2820,"z":2164,"timestamp":18569,"k":91,"type":0}],
[{"x":2342,"y":1114,"z":0,"timestamp":291,"k":90,"type":0},{"x":2316,"y":564,"z":5,"timestamp":396,"k":99,"type":0},{"x":2558,"y":527,"z":210,"timestamp":497,"k":119,"type":0},{"x":2629,"y":206,"z":169,"timestamp":604,"k":117,"type":0},{"x":2943,"y":359,"z":437,"timestamp":713,"k":121,"type":0},{"x":2640,"y":-68,"z":23,"timestamp":825,"k":120,"type":0},{"x":3099,"y":340,"z":336,"timestamp":934,"k":85,"type":0},{"x":2861,"y":119,"z":47,"timestamp":1112,"k":77,"type":0},{"x":3352,"y":709,"z":471,"timestamp":1413,"k":109,"type":1},{"x":3496,"y":855,"z":609,"timestamp":1586,"k":116,"type":0},{"x":3861,"y":603,"z":755,"timestamp":1688,"k":115,"type":0},{"x":4355,"y":892,"z":1105,"timestamp":1792,"k":113,"type":0},{"x":3710,"y":206,"z":261,"timestamp":1902,"k":108,"type":0},{"x":4688,"y":1188,"z":1135,"timestamp":2011,"k":67,"type":0},{"x":5086,"y":1594,"z":1532,"timestamp":2619,"k":75,"type":0},{"x":4952,"y":1474,"z":1402,"timestamp":2731,"k":91,"type":0},{"x":3891,"y":1515,"z":1244,"timestamp":3768,"k":73,"type":0},{"x":2707,"y":1598,"z":192,"timestamp":3870,"k":64,"type":0},{"x":4175,"y":3409,"z":1689,"timestamp":3981,"k":64,"type":0},{"x":2921,"y":2165,"z":428,"timestamp":4363,"k":111,"type":0},{"x":3046,"y":2291,"z":550,"timestamp":4710,"k":69,"type":0},{"x":3487,"y":2681,"z":983,"timestamp":4822,"k":95,"type":0},{"x":4722,"y":3582,"z":2162,"timestamp":4934,"k":87,"type":0},{"x":4163,"y":3014,"z":1607,"timestamp":6588,"k":125,"type":0},{"x":2728,"y":1564,"z":194,"timestamp":6697,"k":76,"type":0},{"x":3008,"y":1670,"z":582,"timestamp":6802,"k":83,"type":0},{"x":2741,"y":1358,"z":381,"timestamp":6903,"k":87,"type":0},{"x":2736,"y":1335,"z":430,"timestamp":7012,"k":117,"type":0},{"x":4248,"y":2826,"z":2005,"timestamp":7121,"k":60,"type":0},{"x":3997,"y":2558,"z":1759,"timestamp":7309,"k":103,"type":0},{"x":3053,"y":1515,"z":813,"timestamp":7414,"k":103,"type":0},{"x":5547,"y":3953,"z":3314,"timestamp":7527,"k":76,"type":0},{"x":3007,"y":1406,"z":772,"timestamp":8040,"k":115,"type":0},{"x":4596,"y":2973,"z":2335,"timestamp":8143,"k":118,"type":0},{"x":4305,"y":3040,"z":1982,"timestamp":8248,"k":116,"type":0},{"x":5111,"y":4313,"z":2767,"timestamp":8352,"k":119,"type":0},{"x":5750,"y":5107,"z":3424,"timestamp":8465,"k":114,"type":0},{"x":3333,"y":2711,"z":1013,"timestamp":8574,"k":68,"type":0},{"x":5071,"y":4449,"z":2751,"timestamp":8744,"k":92,"type":1},{"x":4973,"y":4351,"z":2653,"timestamp":8908,"k":91,"type":1},{"x":3065,"y":2459,"z":743,"timestamp":9082,"k":94,"type":0},{"x":5072,"y":4357,"z":2364,"timestamp":9182,"k":63,"type":0},{"x":7287,"y":6285,"z":4060,"timestamp":9293,"k":66,"type":0},{"x":7240,"y":6188,"z":3726,"timestamp":9394,"k":124,"type":0},{"x":5701,"y":4640,"z":2122,"timestamp":9502,"k":115,"type":0},{"x":4562,"y":3486,"z":959,"timestamp":9603,"k":90,"type":0},{"x":8227,"y":7141,"z":4539,"timestamp":9707,"k":72,"type":0},{"x":8157,"y":7069,"z":4429,"timestamp":9808,"k":103,"type":0},{"x":6117,"y":5042,"z":2281,"timestamp":9908,"k":67,"type":0},{"x":6145,"y":5043,"z":2252,"timestamp":10015,"k":122,"type":0}],
[{"x":2050,"y":1246,"z":0,"timestamp":12,"k":73,"type":0},{"x":2042,"y":1328,"z":44,"timestamp":128,"k":80,"type":0},{"x":2139,"y":1473,"z":112,"timestamp":114,"k":70,"type":0},{"x":2086,"y":1422,"z":53,"timestamp":267,"k":104,"type":0},{"x":2080,"y":1368,"z":30,"timestamp":375,"k":71,"type":0},{"x":2171,"y":1285,"z":68,"timestamp":1367,"k":93,"type":0},{"x":2650,"y":1285,"z":213,"timestamp":1520,"k":111,"type":0},{"x":3150,"y":1102,"z":416,"timestamp":1665,"k":123,"type":0},{"x":3011,"y":852,"z":219,"timestamp":1859,"k":65,"type":0},{"x":3087,"y":892,"z":173,"timestamp":1991,"k":74,"type":0},{"x":4059,"y":1831,"z":1083,"timestamp":2113,"k":84,"type":0},{"x":4073,"y":1940,"z":973,"timestamp":2224,"k":61,"type":0},{"x":3270,"y":1710,"z":153,"timestamp":2325,"k":76,"type":0},{"x":3886,"y":2665,"z":810,"timestamp":2428,"k":109,"type":0},{"x":4577,"y":3607,"z":1553,"timestamp":2541,"k":60,"type":0},{"x":3930,"y":3003,"z":915,"timestamp":2651,"k":119,"type":0},{"x":4487,"y":3560,"z":1472,"timestamp":2861,"k":94,"type":1},{"x":4761,"y":3851,"z":1764,"timestamp":2963,"k":126,"type":0},{"x":3160,"y":2210,"z":214,"timestamp":3075,"k":100,"type":0},{"x":3874,"y":2329,"z":827,"timestamp":3179,"k":112,"type":0},{"x":3903,"y":2292,"z":847,"timestamp":3304,"k":111,"type":0},{"x":3717,"y":2085,"z":655,"timestamp":3407,"k":83,"type":0},{"x":3903,"y":2264,"z":839,"timestamp":3583,"k":64,"type":0},{"x":4346,"y":2697,"z":1266,"timestamp":3689,"k":96,"type":0},{"x":4217,"y":2629,"z":1115,"timestamp":3800,"k":120,"type":0},{"x":3350,"y":1787,"z":242,"timestamp":3912,"k":88,"type":0},{"x":3514,"y":1983,"z":402,"timestamp":4013,"k":70,"type":0},{"x":3138,"y":1621,"z":30,"timestamp":5409,"k":92,"type":0},{"x":3837,"y":2363,"z":761,"timestamp":5515,"k":65,"type":0},{"x":6009,"y":4595,"z":3006,"timestamp":5622,"k":76,"type":0},{"x":5034,"y":3766,"z":2145,"timestamp":5724,"k":61,"type":0},{"x":3121,"y":1881,"z":263,"timestamp":5838,"k":92,"type":0},{"x":6074,"y":4700,"z":3250,"timestamp":5938,"k":85,"type":0},{"x":2843,"y":1406,"z":47,"timestamp":6042,"k":107,"type":0},{"x":3676,"y":2193,"z":880,"timestamp":6151,"k":78,"type":0},{"x":3222,"y":1736,"z":435,"timestamp":6500,"k":81,"type":0},{"x":6505,"y":5233,"z":3881,"timestamp":6605,"k":63,"type":0},{"x":5580,"y":4328,"z":2965,"timestamp":6715,"k":79,"type":0},{"x":4528,"y":3295,"z":1925,"timestamp":6824,"k":104,"type":0},{"x":6975,"y":5769,"z":4383,"timestamp":6935,"k":80,"type":0},{"x":6051,"y":4845,"z":3459,"timestamp":7092,"k":88,"type":1},{"x":4521,"y":3334,"z":1941,"timestamp":7751,"k":115,"type":0},{"x":3938,"y":2771,"z":1367,"timestamp":7939,"k":108,"type":0},{"x":5057,"y":3862,"z":2064,"timestamp":15730,"k":104,"type":0},{"x":5885,"y":4736,"z":2754,"timestamp":15833,"k":116,"type":0},{"x":3861,"y":2741,"z":459,"timestamp":15940,"k":115,"type":0},{"x":8943,"y":7818,"z":5165,"timestamp":16043,"k":125,"type":0},{"x":9050,"y":7893,"z":5230,"timestamp":16154,"k":89,"type":0}]
                    ]

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


def get_my_bilibili_tasks(cookie: str, csrf: str):
    """
    获取B站“我的任务”信息，模拟浏览器fetch请求。

    Args:
        cookie (str): 你的B站登录cookie字符串。
        csrf (str): 你的B站CSRF token (即bili_jct)。

    Returns:
        dict: 如果请求成功，返回API响应的JSON数据（字典格式）。
        None: 如果请求失败，返回None。
    """
    # 1. 目标URL
    url = "https://api.bilibili.com/x/up-activity-interface/arena/v2/my/tasks"

    # 2. URL查询参数 (Query Parameters)
    # fetch请求中的 "?csrf=...&web_location=0.0" 部分
    params = {
        'csrf': csrf,
        'web_location': '0.0'
    }

    # 3. 请求头 (Headers)
    # 从你的fetch请求中复制，并添加最重要的 Cookie
    headers = {
        "accept": "*/*",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        # 'User-Agent' 是一个非常常见的请求头，最好加上以模拟真实浏览器
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "sec-ch-ua": "\"Chromium\";v=\"142\", \"Microsoft Edge\";v=\"142\", \"Not_A Brand\";v=\"99\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "priority": "u=1, i",
        # 'Referer' 头也很重要，表明请求来源
        "Referer": "https://member.bilibili.com/",
        # 将传入的cookie字符串放入 'Cookie' 头中
        "Cookie": cookie
    }

    try:
        # 4. 发送GET请求
        # 使用 requests.get() 方法，传入url, headers, 和 params
        response = requests.get(url, headers=headers, params=params)

        # 5. 检查响应状态
        # raise_for_status() 会在遇到 4xx 或 5xx 的错误状态码时抛出异常
        response.raise_for_status()

        # 6. 解析并返回JSON数据
        # B站的API通常返回JSON格式的数据
        return response.json()

    except requests.exceptions.RequestException as e:
        # 捕获所有requests可能抛出的异常 (如网络问题、超时、HTTP错误等)
        print(f"请求发生错误: {e}")
        return None

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
    dm_img_inter_list_len = len(dm_img_inter_list)
    # 随机获取一个索引
    dm_img_inter_list_idx = random.randint(0, dm_img_inter_list_len - 1)

    _DM_IMG_LIST = dm_img_list_list[dm_img_inter_list_idx]
    _DM_IMG_STR = "V2ViR0wgMS4wIChPcGVuR0wgRVMgMi4wIENocm9taXVtKQ"
    _DM_COVER_IMG_STR = "QU5HTEUgKE5WSURJQSwgTlZJRElBIEdlRm9yY2UgUlRYIDIwODAgVGkgKDB4MDAwMDFFMDcpIERpcmVjdDNEMTEgdnNfNV8wIHBzXzVfMCwgRDNEMTEpR29vZ2xlIEluYy4gKE5WSURJQS"
    _DM_IMG_INTER = dm_img_inter_list[dm_img_inter_list_idx]

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
        self.session.timeout = (10, 20)

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
                print(f"获取 WBI Keys 失败：{json_content.get('message')} {self.all_params}")
                cookies = self.session.cookies.get_dict()
        except requests.exceptions.RequestException as e:
            print(f"请求 WBI Keys 发生错误：{e}")
            # 打印当前 session 的 cookies
            cookies = self.session.cookies.get_dict()
            print(f"当前 Cookies: {cookies}")

        if not (self.img_key and self.sub_key):
            print(f"警告：未能成功加载 WBI Keys，评论或点赞请求可能失败或被风控。{self.total_cookie} {self.all_params}")

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
                print(f"弹幕发送成功！Dmid: {dmid}  {self.all_params.get('name', '未知用户')}")
                return True
            else:
                print(f"弹幕发送失败，错误码：{result.get('code')}, 信息：{result.get('message')} {self.all_params}")
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
        # print(f"准备对视频 {bvid} 进行一键三连...")

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
                print(f"一键三连操作完成。状态 -> 点赞: {like_status}, 投币: {coin_status}, 收藏: {fav_status}  {self.all_params.get('name', '未知用户')}")
                # 只要三连中有一项成功，就认为操作成功
                return data.get('like') or data.get('coin') or data.get('fav')
            else:
                print(f"一键三连失败，错误码：{result.get('code')}, 错误信息：{result.get('message')}  {self.all_params.get('name', '未知用户')}")
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
        # print(f"准备分享视频 {bvid}...")

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
                print(f"{bvid} 视频分享成功！当前分享数：{share_count}")
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
                    # print("评论发送成功！")
                    rpid = result["data"]["reply"]["rpid"]
                    time.sleep(5)
                    self.like_comment(oid=oid, rpid=rpid, type_code=type_code)
                    return rpid
                else:
                    print(f"评论失败，错误码：{result['code']}, 信息：{result['message']} {self.all_params}")
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


danmu_praises_general_quality = [
    # --- 1. 极度通用型 (几乎适用于所有非劣质视频) ---
    "UP主用心了",
    "这个视频做得真好",
    "质量不错，支持一下",
    "观感很舒服",
    "好评！",
    "制作不易，给你点赞了",
    "感觉很流畅",
    "看得出来是认真做的",
    "这个质量可以的",
    "不错不错",

    # --- 2. 夸赞剪辑与节奏 ---
    "这剪辑，有点东西",
    "节奏很棒，不知不觉就看完了",
    "转场好自然",
    "BGM和画面配合得真好",
    "这个剪辑节奏爱了",
    "信息密度刚刚好，不拖沓",
    "神仙剪辑！",

    # --- 3. 夸赞画面与视听体验 (非特指高清) ---
    "画面很干净",
    "看着很清爽",
    "镜头很稳，好评",
    "这个构图学到了",
    "字幕好评，看得舒服多了",
    "收音很清晰，没有杂音",
    "字体和排版好评",
    "bgm好听，求bgm！",  # 侧面夸赞品味

    # --- 4. 夸赞整体质感与氛围 ---
    "质感拉满了",
    "有电影感了",  # 泛指，不一定是真的电影机
    "这视频有种高级感",
    "赏心悦目",
    "完成度好高啊",
    "是个宝藏UP主",

    # --- 5. 互动与鼓励型 ---
    "这质量，值得一个三连！",
    "果断三连了",
    "已关注，期待更多好作品",
    "好活，当赏！",  # 偏二次元/B站风格
    "你更新，我三连，就这么定了",
    "这不得狠狠点个赞",
    "码住，回头再看一遍",  # 表达对视频质量的认可
]

def send_one_comment_per_user(config_map, bvid, like_video=False, delay=2, retries=2):
    """
    简洁版：为 config_map 中每个用户发一条顶级评论。
    参数：
      - config_map: init_config() 返回的字典
      - bvid: 要评论的视频 BV 号
      - like_video: 是否先点赞视频
      - delay: 每个用户间的等待秒数
      - retries: 单用户失败时重试次数
    返回：{uid: {'ok': bool, 'rpid': rpid_or_None, 'error': err_msg_or_None}}
    """
    results = {}
    for uid, cfg in config_map.items():
        name = cfg.get('name', uid)
        if uid in  ['196823511', '3546972143225467', '3546717871934392']:
            continue
        if name not in ['shuijun2']:
            continue
        cookie = cfg.get('total_cookie')
        csrf = cfg.get('BILI_JCT') or cfg.get('csrf')
        if not cookie or not csrf:
            results[uid] = {'ok': False, 'rpid': None, 'error': 'missing cookie or csrf'}
            continue
        # 随机选择一条danmu_praises_general_quality
        comment = random.choice(danmu_praises_general_quality)
        text = f"来自 {name} {comment}"
        try:
            commenter = BilibiliCommenter(total_cookie=cookie, csrf_token=csrf, all_params=cfg.get('all_params', {}))
        except Exception as e:
            results[uid] = {'ok': False, 'rpid': None, 'error': f'create commenter error: {e}'}
            continue

        rpid = None
        err_msg = None
        for _ in range(retries):
            try:
                rpid = commenter.post_comment(bvid, text, 1, like_video=like_video)
                results[uid] = {'ok': True, 'rpid': rpid, 'error': None}
                break
            except Exception as e:
                err_msg = str(e)
                time.sleep(1)
        else:
            results[uid] = {'ok': False, 'rpid': None, 'error': err_msg}

        # time.sleep(delay)

    return results

def get_video_comment_user():
    """
    查询哪些用户有好片推荐这个任务
    """
    config_map = init_config()
    user_list = []
    for uid, target_value in config_map.items():
        user_name = target_value.get('name', uid)
        data = get_my_bilibili_tasks(target_value['total_cookie'], target_value['BILI_JCT'])
        if 'B站好片有奖种草' in str(data):
            print(f"用户 {user_name} ({uid}) 有好片推荐任务")
            user_list.append(user_name)
    print(f"总共有 {len(user_list)} 个用户有好片推荐任务：\n {user_list}")

# --- 主逻辑 ---
if __name__ == "__main__":
    # config_map = init_config()
    # target_bvid = "BV1mu43zFETB"
    # res = send_one_comment_per_user(config_map, target_bvid, like_video=False, delay=2, retries=2)
    # print(res)

    get_video_comment_user()


    # config_map = init_config()
    # user_name = 'yiyi'
    # target_value = None
    # for uid, value in config_map.items():
    #     if value.get('name') == user_name:
    #         target_value = value
    #         break
    #
    #
    # target_bvid = "BV1pe4kz5EiR"
    # comment_text = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]"
    # comment_type = 1
    #
    # # commenter = BilibiliCommenter(total_cookie=target_value['total_cookie'], csrf_token=target_value['BILI_JCT'],all_params=target_value['all_params'])
    # data = get_my_bilibili_tasks(target_value['total_cookie'], target_value['BILI_JCT'])
    # print(data)


    # commenter.pin_comment(target_bvid, 271871684816)
    # #
    # # --- 步骤 1: 发送一条顶级评论 (现在会先点赞视频) ---
    # # print("-" * 30)
    # # print("步骤 1: 尝试发送一条顶级评论...")
    # # posted_rpid = commenter.post_comment(
    # #     target_bvid, comment_text, comment_type,
    # #     like_video=True    )
    #
    # # # --- 步骤 2: 如果顶级评论成功，回复这条评论 ---
    # # if posted_rpid:
    # #     print("-" * 30)
    # #     print(f"步骤 2: 顶级评论发送成功，rpid 为 {posted_rpid}。现在尝试回复这条评论...")
    # #     time.sleep(3)
    # #     reply_text = f"这是对 rpid={posted_rpid} 的回复。[{time.strftime('%Y-%m-%d %H:%M:%S')}]"
    # #     reply_rpid = commenter.reply_to_comment(
    # #         bvid=target_bvid, message_content=reply_text,
    # #         root_rpid=posted_rpid, parent_rpid=posted_rpid, type_code=comment_type,use_proxy=True
    # #     )
    # #     if reply_rpid:
    # #         print("\n回复操作成功完成！")
    # #     else:
    # #         print("\n回复操作失败。")
    # # else:
    # #     print("-" * 30)
    # #     print("顶级评论发送失败，无法进行回复操作。")
    # # #
    # # --- 步骤 3: 发送一条弹幕 ---
    # # print("-" * 30)
    # # print("步骤 3: 尝试发送一条弹幕...")
    # # danmaku_text = f"大家怎么样，心情都好"
    # # danmaku_time_ms = 1000
    # # danmaku_sent = commenter.send_danmaku(
    # #     bvid=target_bvid, msg=danmaku_text, progress=danmaku_time_ms, is_up=False
    # # )
    # # if danmaku_sent:
    # #     print("弹幕发送流程成功完成！")
    # # else:
    # #     print("弹幕发送流程失败。")
    #
    # # # # --- 步骤 4: 查询用户投稿视频 (修正了结果处理的BUG) ---
    # # # print("-" * 30)
    # # # print("步骤 4: 尝试查询用户投稿视频...")
    # # # user_mid_to_query = 282994  # 以文档中的"warma"为例
    # # # videos_list = commenter.get_user_videos(mid=user_mid_to_query, desired_count=5, order='click')
    # # #
    # # # if videos_list:
    # # #     print(f"\n成功获取到用户 {user_mid_to_query} 的视频列表（按播放量前5）:")
    # # #     if not videos_list:
    # # #         print("该用户没有视频。")
    # # #     else:
    # # #         for video in videos_list:
    # # #             print(f"  - 标题: {video.get('title')}")
    # # #             print(f"    BVID: {video.get('bvid')}, 播放量: {video.get('play')}, 弹幕: {video.get('video_review')}")
    # # # else:
    # # #     print("查询用户投稿视频失败。")
    # #
    # # # --- 新增步骤 5: 分享视频 ---
    # # print("-" * 30)
    # # print("步骤 5: 尝试分享视频...")
    # # share_success = commenter.share_video(bvid=target_bvid)
    # # if share_success:
    # #     print("分享操作流程成功完成！")
    # # else:
    # #     print("分享操作流程失败。")
    # #
    # # --- 新增步骤 6: 一键三连视频 ---
    # print("-" * 30)
    # print("步骤 6: 尝试对视频进行一键三连...")
    # triple_like_success = commenter.triple_like_video(bvid=target_bvid)
    # if triple_like_success:
    #     print("一键三连操作流程成功完成！")
    # else:
    #     print("一键三连操作流程失败。")