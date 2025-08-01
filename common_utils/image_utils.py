import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import difflib

from common_utils.common_utils import time_to_ms


def compute_iou(box1, box2):
    """
    计算两个框之间的交并比 (IoU)。
    框格式：[left, top, right, bottom]
    """
    # 计算交集区域
    left_inter = max(box1[0], box2[0])
    top_inter = max(box1[1], box2[1])
    right_inter = min(box1[2], box2[2])
    bottom_inter = min(box1[3], box2[3])

    if right_inter < left_inter or bottom_inter < top_inter:
        return 0.0  # 无交集
    area_inter = (right_inter - left_inter) * (bottom_inter - top_inter)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # IoU = 交集面积 / (两框并集面积)
    iou = area_inter / (area_box1 + area_box2 - area_inter)
    return iou


def compute_text_similarity(text1, text2):
    """
    利用 difflib 计算两个文本的相似度，返回值范围 0.0～1.0
    """
    return difflib.SequenceMatcher(None, text1, text2).ratio()


def is_same_box(box1, text1, box2, text2, overlap_threshold=0.8, content_threshold=0.8):
    """
    判断两个框是否为同一框：
      1. IoU 大于 overlap_threshold
      2. 文本相似度 大于 content_threshold
    """
    iou = compute_iou(box1, box2)
    sim = compute_text_similarity(text1, text2)
    # 调试信息（可选）
    # print(f"IoU: {iou:.2f}, 文本相似度: {sim:.2f}")
    return iou >= overlap_threshold and sim >= content_threshold


def compute_enclosing_box(boxes):
    """
    给定多个框，计算能框住所有框的最小边界框
    """
    min_left = min(box[0] for box in boxes)
    min_top = min(box[1] for box in boxes)
    max_right = max(box[2] for box in boxes)
    max_bottom = max(box[3] for box in boxes)
    return [min_left, min_top, max_right, max_bottom]

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



def save_frames_around_timestamp(
    video_path: str,
    timestamp,
    num_frames: int,
    output_dir: str
) -> None:
    """
    从视频中在给定时间戳前后各截取 num_frames 帧并保存为图片。
    输出文件命名格式：frame_{idx}.png，其中 idx 为帧在视频中的索引。
    """
    ts_sec = time_to_ms(timestamp) / 1000

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError("无法获取视频帧率或总帧数")

    # 目标帧序号
    target_idx = int(round(ts_sec * fps))
    # 计算要截取的帧索引区间
    start_idx = max(0, target_idx - num_frames)
    end_idx   = min(total_frames - 1, target_idx + num_frames)

    os.makedirs(output_dir, exist_ok=True)

    for idx in range(start_idx, end_idx + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"警告：读取帧 {idx} 失败，跳过")
            continue

        # BGR 转 RGB 并保存
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        out_path = os.path.join(output_dir, f"frame_{idx}.png")
        img.save(out_path)
        # print(f"已保存帧 {idx} -> {out_path}")

    cap.release()

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

def compress_jpeg_with_pillow(input_path, output_path, quality=90, optimize=True, progressive=True):
    """
    使用 Pillow 压缩 JPEG 图片。
    如果输入是 PNG 等带透明通道的格式，会先转换为 RGB。

    :param input_path: 输入图片路径 (可以是 JPEG, PNG 等)
    :param output_path: 输出压缩后的 JPEG 图片路径
    :param quality: 压缩质量 (1-95)。Squoosh 的 MozJPEG 默认75是一个不错的起点。
                    值越低，文件越小，质量损失越大。
    :param optimize: 是否优化霍夫曼表 (True/False)。True 可以稍微减小文件大小，但会增加压缩时间。
    :param progressive: 是否生成渐进式 JPEG (True/False)。渐进式 JPEG 通常文件稍小，且在网络加载时体验更好。
    """
    try:
        img = Image.open(input_path)
        original_size = os.path.getsize(input_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if img.mode in ("RGBA", "P"):
            print(f"信息: 输入图片模式为 {img.mode}，将转换为 RGB 模式以保存为 JPEG。")
            img = img.convert("RGB")
        elif img.mode != "RGB" and img.mode != "L": # L 是灰度模式，JPEG 支持
            print(f"警告: 输入图片模式为 {img.mode}，可能不是JPEG直接支持的。尝试直接保存...")
        img.save(output_path,
                 "JPEG",
                 quality=quality,
                 optimize=optimize,
                 progressive=progressive)
        compressed_size = os.path.getsize(output_path)
        reduction = original_size - compressed_size
        percentage = (reduction / original_size) * 100 if original_size > 0 else 0

        print(f"Pillow JPEG 压缩完成: {input_path} -> {output_path}")
        print(f"  参数: quality={quality}, optimize={optimize}, progressive={progressive}")
        print(f"原始大小: {original_size / 1024:.2f} KB")
        print(f"压缩后大小: {compressed_size / 1024:.2f} KB")
        print(f"减少体积: {reduction / 1024:.2f} KB ({percentage:.2f}%)")

    except FileNotFoundError:
        print(f"错误: 输入图片 '{input_path}' 未找到。")
    except Exception as e:
        print(f"Pillow 压缩 JPEG 时发生错误: {e}")

def count_box_occurrences(images, overlap_threshold=0.5, content_threshold=0.5):
    """
    统计不同图片中“同一框”出现的次数，并收集所有匹配的框信息以及计算最小包围框
    输入:
      images: 列表，每个元素代表一张图片的检测结果，格式为字典
              键为文本，值为框的坐标 [left, top, right, bottom]
    输出:
      clusters: 每一项为一个聚类结果，包含：
                - text: 代表文本信息（可作为聚类标识）
                - boxes: 列表，包含该聚类下所有的框信息
                - count: 出现次数
                - enclosing_box: 能框住所有框的最小边界框信息
    """
    clusters = []

    for image in images:
        for text, box in image.items():
            matched = False
            for cluster in clusters:
                if is_same_box(cluster["box"], cluster["text"], box, text, overlap_threshold, content_threshold):
                    cluster["count"] += 1
                    cluster["boxes"].append(box)
                    matched = True
                    break
            if not matched:
                # 创建新一条聚类记录，保存文本、初始化一个 boxes 列表
                clusters.append({
                    "text": text,
                    "box": box,  # 保存第一个出现的框，可用于快速比较
                    "boxes": [box],  # 保存所有匹配到的框
                    "count": 1
                })

    # 对每个聚类计算能够包住所有框的最小边界框
    for cluster in clusters:
        cluster["enclosing_box"] = compute_enclosing_box(cluster["boxes"])

    return clusters


def denormalize_bbox(bbox_norm, image_width, image_height, to_int=True):
    """
    将归一化的边界框坐标转换为实际图像坐标。

    参数：
        bbox_norm: list or tuple, 归一化坐标 [x_min, y_min, x_max, y_max]，取值范围在 0~1 之间
        image_width: int, 图像的实际宽度
        image_height: int, 图像的实际高度
        to_int: bool, 是否将输出坐标转换为整数，默认 True

    返回：
        list: 实际图像坐标 [x_min, y_min, x_max, y_max]
    """
    x_min = bbox_norm[0] * image_width
    y_min = bbox_norm[1] * image_height
    x_max = bbox_norm[2] * image_width
    y_max = bbox_norm[3] * image_height

    if to_int:
        return [int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max))]
    else:
        return [x_min, y_min, x_max, y_max]


def select_region_and_create_mask(image_path, window_width=800, window_height=600):
    """
    加载图片，若图片尺寸过大，则生成一个缩略图用于交互显示，而不改变原图；
    用户在缩略图上选择两个点（左上角和右下角），程序将缩略图中的坐标
    映射回原图坐标，从而生成与原图尺寸一致的 mask 图片（选中区域像素为 255，其余为 0）。

    参数:
        image_path: 图片的文件路径。
        window_width: 交互窗口期望的宽度（单位像素）。
        window_height: 交互窗口期望的高度（单位像素）。

    返回:
        mask: 与原图尺寸一致的 mask 图像（灰度图），若选择操作未完成则返回 None。
    """
    # 加载原始图片
    img = cv2.imread(image_path)
    if img is None:
        print("错误：无法加载图片！")
        return None

    orig_h, orig_w = img.shape[:2]
    print(f"原图尺寸: {orig_w}x{orig_h}")

    # 计算缩放因子，确保整张图片都能显示在指定窗口范围内（不放大）
    scale = min(window_width / orig_w, window_height / orig_h, 1.0)
    if scale < 1.0:
        # 使用最近邻插值，避免缩小后出现模糊现象
        disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
        display_img = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        print(f"生成了缩略图，缩放因子: {scale:.2f}，显示尺寸: {disp_w}x{disp_h}")
    else:
        scale = 1.0  # 图片尺寸本身就在窗口尺寸内
        display_img = img.copy()
        print("图片尺寸未超过窗口设定，无需缩放。")

    clone = display_img.copy()
    points = []

    def click_event(event, x, y, flags, param):
        nonlocal points, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # 在点击处绘制一个小圆点提示用户当前的点击位置
            cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Image", clone)
            # 如果已点击两次，则画出矩形
            if len(points) == 2:
                pt1 = points[0]
                pt2 = points[1]
                x1_disp = min(pt1[0], pt2[0])
                y1_disp = min(pt1[1], pt2[1])
                x2_disp = max(pt1[0], pt2[0])
                y2_disp = max(pt1[1], pt2[1])
                cv2.rectangle(clone, (x1_disp, y1_disp), (x2_disp, y2_disp), (0, 0, 255), 2)
                cv2.imshow("Image", clone)

    # 创建窗口（WINDOW_AUTOSIZE 确保窗口大小按照图片设置，且不允许手动拖动改变）
    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Image", clone)
    cv2.setMouseCallback("Image", click_event)

    print("请依次点击图片中的左上角和右下角位置，或按 'q' / Esc 退出。")

    # 等待用户操作
    while True:
        key = cv2.waitKey(1) & 0xFF
        # 按 q 或 Esc 退出程序
        if key == ord('q') or key == 27:
            break
        # 如果已选择两个点，则退出
        if len(points) >= 2:
            break

    cv2.destroyWindow("Image")

    if len(points) < 2:
        print("错误：未正确选择两个点。")
        return None

    # 将缩略图的坐标映射回原图坐标
    pt1_disp, pt2_disp = points[0], points[1]
    x1_disp, y1_disp = pt1_disp
    x2_disp, y2_disp = pt2_disp
    x1 = int(min(x1_disp, x2_disp) / scale)
    y1 = int(min(y1_disp, y2_disp) / scale)
    x2 = int(max(x1_disp, x2_disp) / scale)
    y2 = int(max(y1_disp, y2_disp) / scale)

    bbox_norm = [0.525000, 0.025000, 0.962000, 0.080000]
    width = orig_w
    height = orig_h

    bbox_pixel = denormalize_bbox(bbox_norm, width, height)
    x1,y1,x2,y2 = bbox_pixel



    print(f"映射回原图的ROI坐标: ({x1}, {y1}) 到 ({x2}, {y2})")

    # 生成与原图尺寸一致的 mask，选择区域像素设置为 255，其余为 0
    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    return mask


if __name__ == "__main__":
    image_path = "../inpainting/output1.png"  # 请替换为你的图片文件路径
    compress_jpeg_with_pillow(image_path, "../inpainting/optimized_a.png")

    # mask_image = select_region_and_create_mask(image_path, window_width=800, window_height=600)
    #
    # if mask_image is not None:
    #     # 显示并保存mask图
    #     cv2.imshow("Mask", mask_image)
    #     cv2.imwrite("../inpainting/mask_image.jpg", mask_image)
    #     print("mask_image.jpg 已保存。")
    #
    #     # 生成原图上掩码位置涂白后的图片
    #     original_img = cv2.imread(image_path)
    #     if original_img is None:
    #         print("错误：无法加载原始图片！")
    #     else:
    #         white_masked_image = original_img.copy()
    #         # 将掩码区域的像素全部设置为白色（BGR格式下白色为 [255, 255, 255]）
    #         white_masked_image[mask_image == 255] = [255, 255, 255]
    #         cv2.imshow("White Masked Image", white_masked_image)
    #         cv2.imwrite("../inpainting/white_masked_image.jpg", white_masked_image)
    #         print("white_masked_image.jpg 已保存。")
    #
    #     print("按任意键退出。")
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()