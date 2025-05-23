import json
import time

import cv2
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont
import concurrent.futures
import difflib


def process_frame(frame, reader):
    """
    对单帧图像进行 OCR 识别、绘制检测框和中文文本注释。

    参数:
        frame: 输入图像 (BGR 格式)
        reader: easyocr.Reader 对象

    返回:
        processed_frame: 绘制好检测框和文本的图像 (BGR 格式)
        results: OCR 识别结果，格式为 [(bbox, text, confidence), ...]
    """
    # 进行 OCR 识别
    results = reader.readtext(frame, detail=1)

    # 利用 OpenCV 绘制检测框
    for bbox, text, confidence in results:
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 转换图像为 PIL 格式以便绘制中文文本
    pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)

    # 尝试加载中文字体，失败时使用默认字体
    try:
        font = ImageFont.truetype("simsun.ttc", size=18)
    except Exception as e:
        print("无法加载指定中文字体, 使用默认字体, 错误信息:", e)
        font = ImageFont.load_default()

    # 绘制 OCR 检测文本（位于检测框左上方位置）
    for bbox, text, confidence in results:
        top_left = bbox[0]
        text_position = (int(top_left[0]), max(0, int(top_left[1]) - 20))
        draw.text(text_position, text, font=font, fill=(255, 0, 0))

    # 转换回 OpenCV 格式返回处理后的图像
    processed_frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    return processed_frame, results


def sample_frames(input_video, max_frames=1000, sample_interval=30):
    """
    从视频中采样指定的帧。

    参数:
        input_video: 输入视频文件路径
        max_frames: 最大处理帧数（包括跳过的帧）
        sample_interval: 采样间隔（除了第一帧，之后每隔 sample_interval 帧采样一次）

    返回:
        first_frame: 视频中的第一帧
        sampled_frames: 列表，每个元素为 (frame_no, frame) 的元组
    """
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"无法打开视频文件: {input_video}")
        return None, []

    first_frame = None
    sampled_frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count == 1:
            first_frame = frame.copy()
            sampled_frames.append((frame_count, frame.copy()))
        elif frame_count % sample_interval == 0:
            sampled_frames.append((frame_count, frame.copy()))

        if frame_count >= max_frames:
            break

    cap.release()
    print(f"采样完成，总采样帧数: {len(sampled_frames)}，总处理帧数: {frame_count}")
    return first_frame, sampled_frames


def batch_predict(sampled_frames, reader, max_workers=4):
    """
    对采样帧批量执行 OCR 预测，支持多线程加速。

    参数:
        sampled_frames: 采样帧列表，每项为 (frame_no, frame)
        reader: easyocr.Reader 对象
        max_workers: 线程池中最大工作线程数

    返回:
        ocr_results: OCR 识别结果列表，每项为字典，包含帧号、bbox、识别文本和置信度
    """
    ocr_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for frame_no, frame in sampled_frames:
            # 使用 frame.copy() 避免多线程中数据共享冲突
            future = executor.submit(process_frame, frame.copy(), reader)
            futures.append((frame_no, future))

        for frame_no, future in futures:
            _, results = future.result()
            for bbox, text, confidence in results:
                ocr_results.append({
                    "frame_no": frame_no,
                    "bbox": bbox,
                    "text": text,
                    "confidence": confidence
                })
    return ocr_results


def are_similar_detections(det1, det2, pos_thresh=0.2, text_thresh=0.7):
    """
    判断两个检测结果是否相似，依据检测框位置尺寸和文本相似性。

    参数:
        det1, det2: 检测结果字典，每个包含 "bbox" 和 "text" 字段
        pos_thresh: 检测框位置和大小容差（相对于平均宽高的比例）
        text_thresh: 文本相似度阈值（0~1）

    返回:
        True：检测结果相似；False：不相似。
    """

    def get_bbox_params(bbox):
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x = min(xs)
        y = min(ys)
        w = max(xs) - x
        h = max(ys) - y
        return x, y, w, h

    x1, y1, w1, h1 = get_bbox_params(det1["bbox"])
    x2, y2, w2, h2 = get_bbox_params(det2["bbox"])
    avg_w = (w1 + w2) / 2.0 if (w1 + w2) != 0 else 1
    avg_h = (h1 + h2) / 2.0 if (h1 + h2) != 0 else 1

    if abs(x1 - x2) > pos_thresh * avg_w or abs(y1 - y2) > pos_thresh * avg_h:
        return False
    if abs(w1 - w2) > pos_thresh * avg_w or abs(h1 - h2) > pos_thresh * avg_h:
        return False

    similarity = difflib.SequenceMatcher(
        None,
        det1["text"].strip().lower(),
        det2["text"].strip().lower()
    ).ratio()
    return similarity >= text_thresh


def get_top5_detections(ocr_results, top_k=5):
    """
    对 OCR 检测结果进行分组，只对文本长度不少于 3 的结果进行归类，
    要求检测框位置尺寸相近且识别文本相似的结果归为同一组。
    返回出现次数最多的前 top_k 个组。

    参数:
        ocr_results: OCR 检测结果列表

    返回:
        top_groups: 列表，每项为字典，包含：
            "detection": 该组中最早出现（代表）的检测结果
            "count": 该组的出现次数
    """
    groups = []
    filtered = [det for det in ocr_results if len(det["text"].strip()) >= 3]

    for detection in filtered:
        matched = False
        for group in groups:
            rep = group["detection"]
            if are_similar_detections(rep, detection):
                group["count"] += 1
                if detection["frame_no"] < rep["frame_no"]:
                    group["detection"] = detection
                matched = True
                break
        if not matched:
            groups.append({
                "detection": detection,
                "count": 1
            })

    groups.sort(key=lambda g: g["count"], reverse=True)
    return groups[:top_k]


def draw_top5_on_frame(frame, top5_groups):
    """
    在给定帧上绘制 Top5 检测（组）的检测框和标签（支持中文显示）。

    参数:
        frame: 待绘制图像 (BGR 格式)
        top5_groups: 前5名检测组，每组包含 "detection" 和 "count" 字段，
                     其中 detection 包含 "bbox"（检测框坐标列表）和 "text"（检测文本）

    返回:
        result_frame: 绘制后的图像 (BGR 格式)
    """
    result_frame = frame.copy()
    colors = [
        (255, 0, 0),    # 蓝色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 红色
        (0, 255, 255),  # 黄色
        (255, 0, 255)   # 洋红
    ]

    # 绘制检测框
    for index, group in enumerate(top5_groups):
        detection = group["detection"]
        bbox = np.array(detection["bbox"], dtype=np.int32).reshape((-1, 1, 2))
        color = colors[index % len(colors)]
        cv2.polylines(result_frame, [bbox], isClosed=True, color=color, thickness=3)

    # 转换为 PIL 图像以绘制中文标签
    pil_img = Image.fromarray(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("simhei.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for index, group in enumerate(top5_groups):
        detection = group["detection"]
        count = group["count"]
        top_left = detection["bbox"][0]
        text = f"{detection['text']}:{count}"
        color = colors[index % len(colors)]
        # PIL 使用 RGB 格式颜色，因此转换 BGR -> RGB
        color_rgb = (color[2], color[1], color[0])
        pos = (top_left[0], max(top_left[1] - 30, 0))
        draw.text(pos, text, font=font, fill=color_rgb)

    result_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return result_frame


def json_converter(o):
    """
    辅助函数，用于将 NumPy 数据类型转换为 JSON 可序列化的数据类型。
    """
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Type {type(o)} not serializable")


def main():
    input_video_path_list = ['test.mp4', 'test1.mp4', 'test2.mp4']
    for input_video_path in input_video_path_list:
        start_time = time.time()
        print(f"正在处理视频: {input_video_path} 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        # input_video_path = 'test.mp4'
        max_frames = 1000000  # 最大处理帧数
        sample_interval = 30  # 每隔30帧采样一次
        base_name = input_video_path.split('.')[0]
        # 阶段1：采样
        first_frame, sampled_frames = sample_frames(input_video_path, max_frames, sample_interval)
        if first_frame is None:
            print("未能采集到视频帧。")
            return

        # 初始化 OCR 识别器（支持中文和英文，若 GPU 可用则优先使用）
        reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)

        # 阶段2：批量预测（利用线程池加速）
        ocr_results = batch_predict(sampled_frames, reader, max_workers=4)
        with open(f"{base_name}_ocr_results.json", "w", encoding="utf-8") as f:
            json.dump(ocr_results, f, ensure_ascii=False, indent=4, default=json_converter)

        # 读取保存的 OCR 结果
        with open(f"{base_name}_ocr_results.json", "r", encoding="utf-8") as f:
            ocr_results = json.load(f)

        # 阶段3：统一处理 OCR 结果：分组并选出出现次数最多的 Top 检测组
        top5_groups = get_top5_detections(ocr_results, top_k=10)
        for idx, group in enumerate(top5_groups):
            print(f"组 {idx + 1}: 文本：{group['detection']['text']} 次数：{group['count']}")

        # 在第一帧上绘制 Top 检测框及标签，并保存结果图像
        annotated_frame = draw_top5_on_frame(first_frame, top5_groups)
        cv2.imwrite(f"{base_name}_top5_start_frame.jpg", annotated_frame)
        print("已保存含有 Top 检测框的起始帧到 'top5_start_frame.jpg'")
        print("处理完成，耗时:", time.time() - start_time)


if __name__ == '__main__':
    main()