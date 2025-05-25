# -- coding: utf-8 --
import base64
import json
import os
import time
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from PIL import Image, ImageDraw, ImageFont

from volcenginesdkarkruntime import Ark
from common_utils.common_utils import get_config
from common_utils.image_utils import count_box_occurrences

# 全局配置，获取 Ark API Key
ARK_API_KEY = get_config("doubao_api_key")


def encode_image_to_base64(img: Image.Image, fmt: str = "jpeg") -> str:
    bio = BytesIO()
    img.save(bio, format=fmt)
    return base64.b64encode(bio.getvalue()).decode("utf-8")


def detect_watermarks(image: Image.Image, ark_client: Ark) -> dict:
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
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    for label, coords in watermark_data.items():
        if isinstance(coords, list) and len(coords) == 4:
            x0, y0, x1, y1 = [int(c * s) for c, s in zip(coords, (width, height, width, height))]
            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
            text_y = y0 - 20 if y0 - 20 > 0 else y0 + 5
            draw.text((x0, text_y), label, fill="red", font=font)
    return image


def extract_frames_from_video(video_path: str, num_frames: int = 3) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError("视频中无帧可处理")

    frame_indices = [round(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)] if num_frames > 1 else [0]

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
            x0, y0, x1, y1 = [int(coord * dim) for coord, dim in zip(enclosing_box, (width, height, width, height))]
            draw.rectangle([x0, y0, x1, y1], outline="blue", width=2)
            annotation_text = f"{text} ({count})"
            text_y = y0 - 20 if y0 - 20 > 0 else y0 + 5
            draw.text((x0, text_y), annotation_text, fill="blue", font=font)
    return image


def _detect_task(args):
    """
    单线程任务封装：创建独立 Ark 客户端并调用 detect_watermarks
    返回 (frame_label, watermark_data)
    """
    frame_label, image = args
    ark_client = Ark(api_key=ARK_API_KEY)   # 每个线程各自实例化
    return frame_label, detect_watermarks(image, ark_client)


def process_video(video_path: str, output_dir: str, num_frames: int = 3, max_workers: int = 30):
    """
    处理视频：先提取帧，再用多线程并行检测水印。
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(video_path)

    frames = extract_frames_from_video(video_path, num_frames)

    # 1. 先加载已存在的 JSON，剩余的帧异步检测
    wm_data_dict = {}
    tasks_to_run = []   # [(frame_label, Image), ...]

    for frame_index, image in frames:
        frame_label = f"{base_name}_{frame_index}"
        json_path = os.path.join(output_dir, f"{frame_label}.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                wm_data_dict[frame_label] = json.load(f)
        else:
            tasks_to_run.append((frame_label, image))

    # 2. 并行调用 detect_watermarks
    if tasks_to_run:
        workers = min(max_workers, len(tasks_to_run))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_label = {executor.submit(_detect_task, task): task[0] for task in tasks_to_run}
            for future in as_completed(future_to_label):
                frame_label = future_to_label[future]
                try:
                    frame_label_ret, wm_data = future.result()
                    assert frame_label_ret == frame_label
                except Exception as exc:
                    print(f"{frame_label} 处理失败: {exc}")
                    wm_data = {}
                wm_data_dict[frame_label] = wm_data
                # 落盘 JSON
                json_path = os.path.join(output_dir, f"{frame_label}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(wm_data, f, ensure_ascii=False, indent=2)

    # 3. 标注图片并组装 wm_data_list（保持与 frames 顺序一致）
    wm_data_list = []
    for frame_index, image in frames:
        frame_label = f"{base_name}_{frame_index}"
        watermark_data = wm_data_dict.get(frame_label, {})
        wm_data_list.append(watermark_data)

        annotated_image = annotate_watermarks_on_image(image.copy(), watermark_data)
        annotated_image.save(os.path.join(output_dir, f"{frame_label}.png"))

    # 4. 聚类统计
    clusters = count_box_occurrences(wm_data_list)
    clusters = [c for c in clusters if c.get("count", 0) > 2]

    if frames:
        first_frame_index, first_image = frames[0]
        cluster_img = annotate_clusters_on_image(first_image.copy(), clusters)
        cluster_img.save(os.path.join(output_dir, f"{base_name}_{first_frame_index}_clusters.png"))

    clusters_text = [c["text"] for c in clusters]
    print(f"{video_path} 提取 {len(frames)} 帧，处理时间: {time.time() - start_time:.2f}s；"
          f"检测到水印种类: {len(clusters)}；内容: {clusters_text}")
    return clusters


def detection_watermark(video_file="../inpainting/test2.mp4",
                        output_dir="output_frames",
                        num_frames=10):
    try:
        return process_video(video_file, output_dir, num_frames=num_frames)
    except Exception as e:
        print("处理过程中出错:", e)
    return None


if __name__ == "__main__":
    detection_watermark()