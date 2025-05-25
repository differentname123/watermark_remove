import base64
import json
import os
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import cv2
from volcenginesdkarkruntime import Ark

from common_utils.common_utils import get_config

ARK_API_KEY = get_config("doubao_api_key")


def encode_image(img: Image.Image, fmt="jpeg") -> str:
    """将PIL Image转换为Base64字符串"""
    bio = BytesIO()
    img.save(bio, format=fmt)
    return base64.b64encode(bio.getvalue()).decode("utf-8")


def detect_watermarks_api(img: Image.Image, client: Ark) -> dict:
    """
    通过调用API检测图像中水印，返回结果为一个字典，
    格式形如：{"文本或图标": [x_min, y_min, x_max, y_max]}
    坐标归一化到0-1（保留6位小数），无检测时返回 {}。
    """
    b64 = encode_image(img)
    data_uri = f"data:image/jpeg;base64,{b64}"
    prompt = f"""
    请分析输入图像，检测所有水印（包括文字和图标）。
    要求：
    1. 仅返回一个 JSON 对象，禁止输出任何额外说明。
    2. JSON 对象的键为水印标识：文字水印直接用检测到的文本（UTF-8，双引号包裹）；图标水印请生成唯一标识（如 "icon_watermark"，若有多个依次命名为 "icon_watermark_1"，"icon_watermark_2" 等）。
    3. 每个键对应的值为一个浮点数组 [x_min, y_min, x_max, y_max]，表示水印框边界，坐标已根据图像尺寸归一化到 0–1，保留 3 位小数。
    4. 无检测时返回 {{}}。
    示例：
    {{
      "Company © 2024": [0.134, 0.245, 0.356, 0.469],
      "icon_watermark": [0.500, 0.600, 0.700, 0.500]
    }}
    """
    try:
        resp = client.chat.completions.create(
            model="doubao-1-5-thinking-vision-pro-250428",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt}
                ]
            }],
            extra_headers={'x-is-encrypted': 'true'},
            temperature=0,
            top_p=0.7,
            max_tokens=4096,
        )
        res_content = resp.choices[0].message.content
    except Exception as e:
        print("调用API出错:", e)
        return {}

    try:
        wm_data = json.loads(res_content)
    except Exception:
        print("JSON解析失败:", res_content)
        wm_data = {}
    return wm_data


def annotate_image(img: Image.Image, wm_data: dict) -> Image.Image:
    """
    根据wm_data在图像上绘制检测到的水印边界框，
    坐标为归一化数据，根据图像实际尺寸还原到像素位置。
    """
    draw = ImageDraw.Draw(img)
    width, height = img.size
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    for label, coords in wm_data.items():
        if isinstance(coords, list) and len(coords) == 4:
            # 坐标归一化，这里转换为实际像素
            x0, y0, x1, y1 = [int(c * s) for c, s in zip(coords, (width, height, width, height))]
            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
            text_pos = (x0, y0 - 20 if y0 - 20 > 0 else y0 + 5)
            draw.text(text_pos, label, fill="red", font=font)
            print(f"检测到水印: {label} at [{x0}, {y0}, {x1}, {y1}]")
    return img


def extract_frames(video_path: str):
    """
    从视频中提取第一帧、中间帧和最后一帧，
    返回列表，每个元素为 (帧位置, PIL.Image)。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        raise ValueError("视频中无帧可处理")
    positions = [0, count // 2, count - 1]
    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((pos, Image.fromarray(frame_rgb)))
        else:
            print(f"读取帧 {pos} 失败")
    cap.release()
    return frames


if __name__ == "__main__":
    video_file = "../inpainting/test.mp4"  # 输入视频文件路径
    base_name = os.path.basename(video_file)
    output_dir = "output_frames"  # 输出目录，既保存结果图片，也保存坐标JSON文件
    os.makedirs(output_dir, exist_ok=True)

    # 初始化Ark客户端
    ark_client = Ark(api_key=ARK_API_KEY)

    try:
        frames = extract_frames(video_file)
        labels = [f"{base_name}_first_frame", f"{base_name}_middle_frame", f"{base_name}_last_frame"]
        for i, (pos, img) in enumerate(frames):
            # 坐标数据存储文件，例如 first_frame.json
            json_filename = os.path.join(output_dir, f"{labels[i]}.json")

            if os.path.exists(json_filename):
                with open(json_filename, "r", encoding="utf-8") as f:
                    wm_data = json.load(f)
                print(f"{labels[i]} 水印数据已加载")
            else:
                wm_data = detect_watermarks_api(img, ark_client)
                with open(json_filename, "w", encoding="utf-8") as f:
                    json.dump(wm_data, f, ensure_ascii=False, indent=2)
                print(f"{labels[i]} 水印数据已保存至 {json_filename}")

            # 使用加载或新检测到的水印数据进行图像标注
            annotated_img = annotate_image(img.copy(), wm_data)
            output_path = os.path.join(output_dir, f"{labels[i]}.png")
            annotated_img.save(output_path)
            print("保存处理结果到:", output_path)
    except Exception as e:
        print("处理过程中出错:", e)