# -- coding: utf-8 --
import base64
import json
import os
import time
from io import BytesIO

import cv2
from PIL import Image, ImageDraw, ImageFont

from volcenginesdkarkruntime import Ark
from common_utils.common_utils import get_config
from common_utils.image_utils import count_box_occurrences

# 全局配置，获取 Ark API Key
ARK_API_KEY = get_config("doubao_api_key")


def encode_image_to_base64(img: Image.Image, fmt: str = "jpeg") -> str:
    """
    将 PIL Image 转换为 Base64 编码的字符串
    """
    bio = BytesIO()
    img.save(bio, format=fmt)
    return base64.b64encode(bio.getvalue()).decode("utf-8")


def detect_watermarks(image: Image.Image, ark_client: Ark) -> dict:
    """
    调用 Ark API 检测图像中的水印，返回一个字典，例如：
    {
        "Company © 2024": [0.134, 0.245, 0.356, 0.469],
        "icon_watermark": [0.500, 0.600, 0.700, 0.800]
    }
    坐标均为归一化数值（0–1，保留 3 位小数），无检测时返回 {}.
    """
    b64 = encode_image_to_base64(image)
    data_uri = f"data:image/jpeg;base64,{b64}"
    prompt = """
    请分析输入图像，检测所有水印（包括文字和图标）。
    要求：
    1. 仅返回一个 JSON 对象，禁止输出任何额外说明。
    2. JSON 对象的键为水印标识：文字水印直接用检测到的文本（UTF-8，双引号包裹）；图标水印请生成唯一标识（如 "icon_watermark"，若有多个依次命名为 "icon_watermark_1"，"icon_watermark_2" 等）。
    3. 每个键对应的值为一个浮点数组 [x_min, y_min, x_max, y_max]，表示水印框边界，坐标已根据图像尺寸归一化到 0–1，保留 3 位小数。
    4. 无检测时返回 {}.
    示例：
    {
      "Company © 2024": [0.134, 0.245, 0.356, 0.469],
      "icon_watermark": [0.500, 0.600, 0.700, 0.800]
    }
    """
    try:
        response = ark_client.chat.completions.create(
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
        res_content = response.choices[0].message.content
    except Exception as e:
        print("调用 API 出错:", e)
        return {}

    try:
        watermark_data = json.loads(res_content)
    except Exception:
        print("JSON 解析失败:", res_content)
        watermark_data = {}
    return watermark_data


def annotate_watermarks_on_image(image: Image.Image, watermark_data: dict) -> Image.Image:
    """
    根据 watermark_data 在图像上绘制检测到的水印框（红色），标注水印文本
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    for label, coords in watermark_data.items():
        if isinstance(coords, list) and len(coords) == 4:
            # 将归一化坐标转换为实际像素
            x0, y0, x1, y1 = [int(c * s) for c, s in zip(coords, (width, height, width, height))]
            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
            text_y = y0 - 20 if y0 - 20 > 0 else y0 + 5
            draw.text((x0, text_y), label, fill="red", font=font)
            # print(f"检测到水印: {label} at [{x0}, {y0}, {x1}, {y1}]")
    return image


def extract_frames_from_video(video_path: str, num_frames: int = 3) -> list:
    """
    从视频中均匀提取指定数量的帧。
    返回值为列表，每个元素为 (帧索引, PIL.Image)。
    帧的命名中将使用绝对帧数。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError("视频中无帧可处理")

    # 均匀选取帧索引，当 num_frames==1 时，直接选第一帧
    if num_frames == 1:
        frame_indices = [0]
    else:
        frame_indices = [round(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((idx, Image.fromarray(frame_rgb)))
        else:
            print(f"读取帧 {idx} 失败")
    cap.release()
    return frames


def annotate_clusters_on_image(image: Image.Image, clusters: list) -> Image.Image:
    """
    在图像上绘制聚类统计结果。
    对每个聚类，绘制最小包围框（蓝色）以及统计文本（水印文本及出现次数）。
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    for cluster in clusters:
        text = cluster.get("text", "")
        count = cluster.get("count", 0)
        enclosing_box = cluster.get("enclosing_box", None)
        if enclosing_box and isinstance(enclosing_box, list) and len(enclosing_box) == 4:
            # 将归一化坐标转换为像素数值
            x0, y0, x1, y1 = [int(coord * dim) for coord, dim in zip(enclosing_box, (width, height, width, height))]
            draw.rectangle([x0, y0, x1, y1], outline="blue", width=2)
            annotation_text = f"{text} ({count})"
            text_y = y0 - 20 if y0 - 20 > 0 else y0 + 5
            draw.text((x0, text_y), annotation_text, fill="blue", font=font)
            # print(f"聚类统计: {annotation_text} - 包围框: [{x0}, {y0}, {x1}, {y1}]")
    return image


def process_video(video_path: str, output_dir: str, ark_client: Ark, num_frames: int = 3):
    """
    处理视频：
    1. 均匀提取指定数量的帧。
    2. 对每帧调用 API 检测水印，并保存 JSON 文件。
    3. 对每帧图像标注水印检测结果，并保存图片。
    4. 统计所有帧的水印数据聚类信息，并在第一帧上绘制聚类结果，保存图片。
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(video_path)
    wm_data_list = []

    frames = extract_frames_from_video(video_path, num_frames)
    for frame_index, image in frames:
        # 使用视频名和帧的绝对帧数构造唯一标识
        frame_label = f"{base_name}_{frame_index}"
        json_path = os.path.join(output_dir, f"{frame_label}.json")

        # 如果 JSON 文件已存在，则加载，否则调用 API 检测水印
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                watermark_data = json.load(f)
        else:
            watermark_data = detect_watermarks(image, ark_client)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(watermark_data, f, ensure_ascii=False, indent=2)
        wm_data_list.append(watermark_data)

        # 在图像上标注检测到的水印并保存结果图
        annotated_image = annotate_watermarks_on_image(image.copy(), watermark_data)
        output_image_path = os.path.join(output_dir, f"{frame_label}.png")
        annotated_image.save(output_image_path)

    # 统计所有帧的水印检测数据的聚类情况
    clusters = count_box_occurrences(wm_data_list)
    # 过滤出count大于1的聚类
    clusters = [cluster for cluster in clusters if cluster.get("count", 0) > 2]
    if frames:
        first_frame_index, first_image = frames[0]
        annotated_clusters_image = annotate_clusters_on_image(first_image.copy(), clusters)
        cluster_output_path = os.path.join(output_dir, f"{base_name}_{first_frame_index}_clusters.png")
        annotated_clusters_image.save(cluster_output_path)
    clusters_text = [cluster["text"] for cluster in clusters]
    print(f"{video_path} 提取 {len(frames)} 帧，处理时间: {time.time() - start_time:.2f} 秒 水印个数: {len(clusters)} 水印内容为: {clusters_text}")
    return clusters


def detection_watermark(video_file="../inpainting/test2.mp4",output_dir="output_frames", num_frames=10):
    ark_client = Ark(api_key=ARK_API_KEY)
    try:
        return process_video(video_file, output_dir, ark_client, num_frames=num_frames)
    except Exception as e:
        print("处理过程中出错:", e)
    return None


if __name__ == "__main__":
    detection_watermark()