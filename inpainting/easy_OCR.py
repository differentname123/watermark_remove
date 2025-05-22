import json

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
    # OCR 识别文本
    results = reader.readtext(frame, detail=1)

    # 在图像上绘制检测框（OpenCV 绘制）
    for bbox, text, confidence in results:
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 将图像从 OpenCV 格式转换到 PIL 格式便于绘制中文
    pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)

    # 尝试加载中文字体，若失败则使用默认字体
    try:
        font = ImageFont.truetype("simsun.ttc", size=18)
    except Exception as e:
        print("无法加载指定中文字体, 使用默认字体, 错误信息:", str(e))
        font = ImageFont.load_default()

    # 在 PIL 图像上绘制文本
    for bbox, text, confidence in results:
        top_left = bbox[0]
        text_position = (int(top_left[0]), max(0, int(top_left[1]) - 20))
        draw.text(text_position, text, font=font, fill=(255, 0, 0))

    # 转换回 OpenCV 格式
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

        # 第一帧始终采样
        if frame_count == 1:
            first_frame = frame.copy()
            sampled_frames.append((frame_count, frame.copy()))
        # 每隔 sample_interval 采样一帧
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
            # 传入 frame.copy() 避免多线程中数据共享冲突
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
    判断两个检测结果是否“相似”，依据是检测框的位置尺寸和文本是否相似。

    参数:
        det1, det2: 检测结果字典，每个包含 "bbox" 和 "text" 字段
        pos_thresh: 检测框位置和大小的容差（相对于平均宽高的比例）
        text_thresh: 文本相似度阈值（0~1之间）

    返回:
        True 表示两检测结果相似，否则返回 False
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
    # 判断文字相似度
    similarity = difflib.SequenceMatcher(None,
                                         det1["text"].strip().lower(),
                                         det2["text"].strip().lower()).ratio()
    if similarity < text_thresh:
        return False
    return True


def get_top5_detections(ocr_results, top_k=5):
    """
    对 OCR 检测结果进行分组，只对文本长度不少于 3 的结果进行归类，
    同时要求检测框位置尺寸相近且识别文本相似的结果归为同一组。
    返回出现次数最多的前 5 个组。

    参数:
        ocr_results: OCR 检测结果列表

    返回:
        top5_groups: 一个列表，每项为字典，包含：
            "detection": 该组中最早出现（代表）的检测结果
            "count": 该组的出现次数
    """
    groups = []
    # 过滤掉识别文本少于3字符（去除空白）的检测结果
    filtered = [det for det in ocr_results if len(det["text"].strip()) >= 3]

    for detection in filtered:
        matched = False
        for group in groups:
            rep = group["detection"]
            if are_similar_detections(rep, detection):
                group["count"] += 1
                # 若该检测出现更早，则更新代表检测
                if detection["frame_no"] < rep["frame_no"]:
                    group["detection"] = detection
                matched = True
                break
        if not matched:
            groups.append({
                "detection": detection,
                "count": 1
            })

    # 按出现次数从高到低排序，选出前5组
    groups.sort(key=lambda g: g["count"], reverse=True)
    top5_groups = groups[:top_k]
    return top5_groups


def draw_top5_on_frame(frame, top5_groups):
    """
    在给定帧上绘制 Top5 检测（组）对应的检测框和标签（支持中文显示）。

    参数:
        frame: 待绘制图像 (BGR 格式)
        top5_groups: 前5名检测组，每个组包含 "detection" 和 "count" 字段，
                     detection 中应包含 "bbox"（检测框的坐标列表）和 "text"（检测文本）

    返回:
        result_frame: 绘制后的图像 (BGR 格式)
    """
    # 复制输入图像
    result_frame = frame.copy()

    # 定义几个不同的颜色（BGR格式）
    colors = [
        (255, 0, 0),    # 蓝色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 红色
        (0, 255, 255),  # 黄色
        (255, 0, 255)   # 洋红
    ]

    # 绘制多边形检测框（使用 cv2.polylines）
    for index, group in enumerate(top5_groups):
        detection = group["detection"]
        count = group["count"]
        bbox = np.array(detection["bbox"], dtype=np.int32).reshape((-1, 1, 2))
        color = colors[index % len(colors)]
        cv2.polylines(result_frame, [bbox], isClosed=True, color=color, thickness=3)

    # 转换为 PIL 图像以便绘制中文文本
    result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(result_frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    # 指定支持中文的字体（请确保当前目录下有 simhei.ttf 文件，或替换为其他字体路径）
    try:
        font = ImageFont.truetype("simhei.ttf", 20)
    except IOError:
        # 若未找到指定的中文字体，则回退为默认字体（可能无法显示中文）
        font = ImageFont.load_default()

    # 绘制文本：文本内容为检测到的文字和出现次数，文本颜色与对应的框颜色保持一致
    for index, group in enumerate(top5_groups):
        detection = group["detection"]
        count = group["count"]
        # 检测框左上角作为文本起始位置（可根据需要略作偏移）
        top_left = detection["bbox"][0]
        text = f"{detection['text']}:{count}"
        color = colors[index % len(colors)]
        # 将 BGR 转换为 RGB 颜色，用于 PIL 绘制
        color_rgb = (color[2], color[1], color[0])
        # 调整文本位置，避免文本超出图像边界
        pos = (top_left[0], max(top_left[1] - 30, 0))
        draw.text(pos, text, font=font, fill=color_rgb)

    # 转换回 OpenCV 图像（BGR格式）
    result_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return result_frame


def json_converter(o):
    # 如果是 NumPy 标量，直接转成 Python 标量
    if isinstance(o, np.generic):
        return o.item()
    # 如果是 NumPy 数组，转成 list
    if isinstance(o, np.ndarray):
        return o.tolist()
    # 如果是 Torch 张量，也可以加一个判断
    # import torch
    # if isinstance(o, torch.Tensor):
    #     return o.cpu().numpy().tolist()
    # 其它类型交给默认的 TypeError
    raise TypeError(f"Type {type(o)} not serializable")


def main():
    input_video_path = 'test.mp4'
    max_frames = 100  # 最大处理帧数
    sample_interval = 30  # 每隔 30 帧采样一次

    # 阶段1：采样
    first_frame, sampled_frames = sample_frames(input_video_path, max_frames, sample_interval)
    if first_frame is None:
        print("未能采集到视频帧。")
        return

    # # 初始化 OCR 识别器（支持中文和英文）
    # reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
    #
    # # 阶段2：批量预测（利用线程池加速）
    # ocr_results = batch_predict(sampled_frames, reader, max_workers=4)
    # with open("ocr_results.json", "w", encoding="utf-8") as f:
    #     json.dump(ocr_results,
    #               f,
    #               ensure_ascii=False,
    #               indent=4,
    #               default=json_converter)

    # 加载保存的ocr_results.json
    with open("ocr_results.json", "r", encoding="utf-8") as f:
        ocr_results = json.load(f)

    # 阶段3：统一处理OCR结果：分组并选出 Top5 检测
    top5_groups = get_top5_detections(ocr_results,top_k=10)
    # 输出分组统计信息
    for idx, group in enumerate(top5_groups):
        print(f"组 {idx + 1}: 文本：{group['detection']['text']} 次数：{group['count']}")

    # 在第一帧上绘制 Top5 检测框及标签，并保存图片
    annotated_frame = draw_top5_on_frame(first_frame, top5_groups)
    cv2.imwrite("top5_start_frame.jpg", annotated_frame)
    print("已保存含有 Top5 检测框的起始帧到 'top5_start_frame.jpg'")


if __name__ == '__main__':
    main()
