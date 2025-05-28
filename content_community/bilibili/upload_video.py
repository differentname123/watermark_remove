import requests
import base64
import time
import mimetypes # 用于从文件名猜测MIME类型

from common_utils.common_utils import get_config

# --- 需要你填写的变量 ---
SESSDATA_COOKIE = get_config("bilibili_sessdata_cookie")  # 必需。你的B站登录会话 SESSDATA cookie 值。
BILI_JCT_COOKIE = get_config("bilibili_csrf_token")  # 必需。你的B站 bili_jct cookie 值，也用作 csrf 令牌。
IMAGE_FILE_PATH = "inpainted_image.jpg" # 必需。你要上传的本地封面图片的完整路径（例如：/path/to/your/cover.jpg 或 C:\\path\\to\\your\\cover.png）。
# --- 变量说明结束 ---

def upload_bilibili_cover(sessdata, bili_jct, image_path):
    """
    上传B站视频封面。

    参数:
    sessdata (str): SESSDATA cookie 值。
    bili_jct (str): bili_jct cookie 值 (CSRF token)。
    image_path (str): 本地图片文件的路径。

    返回:
    requests.Response: 服务器的响应对象，或在发生错误时返回 None。
    """
    try:
        # 1. 读取图片文件并进行 Base64 编码
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # 2. 猜测图片的MIME类型
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not (mime_type.startswith("image/jpeg") or mime_type.startswith("image/png")):
            print(f"无法识别的图片类型或不支持的图片格式: {mime_type}。请确保是 JPEG 或 PNG 格式。")
            # 默认使用 image/jpeg，如果无法识别，B站服务器可能会拒绝
            mime_type = "image/jpeg"


        # 3. 构建 cover 参数
        cover_param = f"data:{mime_type};base64,{encoded_string}"

        # 4. 构建请求 URL 和 Headers
        timestamp = int(time.time() * 1000)
        upload_url = f"https://member.bilibili.com/x/vu/web/cover/up?ts={timestamp}"

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Cookie": f"SESSDATA={sessdata}; bili_jct={bili_jct}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" # 模拟浏览器User-Agent
        }

        # 5. 构建 POST 数据
        payload = {
            "cover": cover_param,
            "csrf": bili_jct
        }

        # 6. 发送 POST 请求
        print(f"正在上传封面到: {upload_url}")
        response = requests.post(upload_url, headers=headers, data=payload)

        # 7. 检查响应
        response.raise_for_status() # 如果请求失败 (状态码 4xx 或 5xx), 会抛出 HTTPError 异常
        print("封面上传请求已发送。")
        return response

    except FileNotFoundError:
        print(f"错误: 图片文件未找到 - {image_path}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"服务器响应内容: {e.response.text}")
        return None
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None

if __name__ == "__main__":
    # 确保你已经填写了上面的变量
    if SESSDATA_COOKIE == "你的SESSDATA" or BILI_JCT_COOKIE == "你的bili_jct" or IMAGE_FILE_PATH == "你的图片文件路径.jpg":
        print("请先在脚本中填写 SESSDATA_COOKIE, BILI_JCT_COOKIE, 和 IMAGE_FILE_PATH 变量。")
    else:
        response = upload_bilibili_cover(SESSDATA_COOKIE, BILI_JCT_COOKIE, IMAGE_FILE_PATH)

        if response:
            print("\n服务器响应状态码:", response.status_code)
            try:
                response_data = response.json()
                print("服务器响应内容 (JSON):", response_data)
                if response_data.get("code") == 0 and response_data.get("data", {}).get("url"):
                    print(f"\n封面上传成功! 封面URL: {response_data['data']['url']}")
                else:
                    print("\n封面上传可能失败，请检查响应内容。")
                    print(f"错误代码: {response_data.get('code')}, 错误信息: {response_data.get('message')}")
            except ValueError: # 处理非JSON响应
                print("服务器响应内容 (非JSON):", response.text)