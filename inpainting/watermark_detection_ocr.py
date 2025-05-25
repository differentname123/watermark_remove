import json
import time
import cv2
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont
import concurrent.futures
import os

from common_utils.image_utils import count_box_occurrences, annotate_clusters_on_image, extract_frames_from_video


def process_frame(frame, reader):
    """
    对单帧图像进行 OCR 识别、绘制检测框和中文文本注释。

    参数:
        frame: 输入图像，现为 PIL Image 对象（RGB 格式）
        reader: easyocr.Reader 对象

    返回:
        processed_frame: 绘制好检测框和文本的图像 (BGR 格式)，以 np.array 表示
        results: OCR 识别结果，格式为 [(bbox, text, confidence), ...]
                 其中 bbox 为识别区域的四个顶点坐标
    """
    # 如果输入图像为 PIL Image，则先转换为 NumPy 数组，并转换为 BGR 格式（OpenCV 使用 BGR）
    if isinstance(frame, Image.Image):
        frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    else:
        frame_cv = frame

    # 用 OCR 识别文本，注意 easyocr.readtext 接受的是 numpy 数组格式
    results = reader.readtext(frame_cv, detail=1)

    # 绘制检测框（使用 OpenCV 绘制多边形）
    for bbox, text, confidence in results:
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame_cv, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 转换为 PIL 图像以便于绘制中文文本
    pil_im = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)
    try:
        # 尝试加载中文字体，如加载失败则使用默认字体
        font = ImageFont.truetype("simsun.ttc", size=18)
    except Exception as e:
        print("无法加载指定中文字体, 使用默认字体, 错误信息:", e)
        font = ImageFont.load_default()

    # 在每个检测框左上角绘制相应的中文文字
    for bbox, text, confidence in results:
        top_left = bbox[0]
        text_position = (int(top_left[0]), max(0, int(top_left[1]) - 20))
        draw.text(text_position, text, font=font, fill=(255, 0, 0))

    # 将 PIL 图像转换回 OpenCV 格式（BGR）
    processed_frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    return processed_frame, results


def process_and_save_frame(frame_tuple, video_name, output_dir, reader):
    """
    对单帧进行 OCR 识别，绘制结果后将图像和 OCR 结果分别保存为图片和 JSON 文件。

    参数:
        frame_tuple: (frame_index, frame)，其中的 frame 为 PIL Image（RGB 格式）
        video_name: 视频的基本名称（不带扩展名）
        output_dir: 输出目录
        reader: easyocr.Reader 对象

    返回:
        result_grouped: OCR 识别结果，格式为 {文本: [xmin, ymin, xmax, ymax], ...}
                        坐标为归一化后的相对值（归一到图像宽度和高度）
    """
    frame_no, frame = frame_tuple
    json_path = os.path.join(output_dir, f"{video_name}_{frame_no}.json")
    image_path = os.path.join(output_dir, f"{video_name}_{frame_no}.png")

    # 如果 JSON 文件已存在，则直接加载
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                result_grouped = json.load(f)
            return result_grouped
        except Exception as e:
            print(f"加载 JSON 文件失败: {json_path}, 错误: {e}。将重新进行 OCR 识别。")

    # 处理帧，得到 OCR 结果和绘制后的图像（OpenCV 格式）
    processed_frame, ocr_results = process_frame(frame, reader)

    # 获取图像尺寸，用于归一化
    width, height = frame.size

    # 将 OCR 结果转换为 JSON 格式：键为识别文本，值为对应的归一化矩形框 [xmin, ymin, xmax, ymax]
    result_grouped = {}
    for bbox, text, confidence in ocr_results:
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        min_x = min(x_coords)
        min_y = min(y_coords)
        max_x = max(x_coords)
        max_y = max(y_coords)
        # 归一化处理：除以图像宽和高
        normalized_box = [min_x / width, min_y / height, max_x / width, max_y / height]
        # 如果同一文本出现多次，仅保存第一个检测到的结果
        if text not in result_grouped:
            result_grouped[text] = normalized_box

    # 保存 JSON 文件
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_grouped, f, ensure_ascii=False, indent=4, default=json_converter)
    except Exception as e:
        print(f"写入 JSON 文件失败: {json_path}, 错误: {e}")

    # 保存绘制了 OCR 结果的图片
    cv2.imwrite(image_path, processed_frame)
    print(f"已保存: {json_path} 和 {image_path}")

    return result_grouped


def json_converter(o):
    """
    辅助函数，将 NumPy 类型转换为 JSON 可序列化的类型。
    """
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"类型 {type(o)} 无法序列化")


def detection_watermark(input_video_path='test1.mp4',
                        output_dir="output",
                        num_sample_frames=10):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    print(
        f"正在处理视频: {input_video_path} 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    # 采集视频帧（这里假设 extract_frames_from_video 已返回 (frame_index, PIL Image) 格式的帧）
    sampled_frames = extract_frames_from_video(input_video_path, num_sample_frames)
    if not sampled_frames:
        print("未能采集到视频帧。")
        return []

    # 初始化 OCR 识别器（支持中文和英文，如 GPU 可用则自动使用）
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)

    # 使用多线程并行处理每一帧
    result_grouped_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for frame_tuple in sampled_frames:
            future = executor.submit(process_and_save_frame, frame_tuple, video_name, output_dir, reader)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                result_grouped_list.append(result)
            except Exception as exc:
                print(f"处理帧时产生异常: {exc}")

    # 统计所有帧中的检测框结果
    clusters = count_box_occurrences(result_grouped_list)
    clusters = [c for c in clusters if c.get("count", 0) > 2]

    if sampled_frames:
        first_frame_index, first_image = sampled_frames[0]
        cluster_img = annotate_clusters_on_image(first_image, clusters)
        cluster_img.save(os.path.join(output_dir, f"{video_name}_{first_frame_index}_clusters.png"))

    clusters_text = [c["text"] for c in clusters]
    print(f"{input_video_path} 提取 {len(sampled_frames)} 帧，处理时间: {time.time() - start_time:.2f}s；"
          f"检测到水印种类: {len(clusters)}；内容: {clusters_text}")
    return clusters


if __name__ == '__main__':
    detection_watermark()