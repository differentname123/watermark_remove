import cv2
import os
import numpy as np
import shutil
from typing import List, Optional

import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import List, Tuple, Dict, Any, Optional

from common_utils.common_utils import time_to_ms


def init_ocr_model() -> PaddleOCR:
    """
    初始化并返回 PaddleOCR 模型实例。
    """
    print("正在初始化 PaddleOCR 模型...")
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='ch'
    )
    print("PaddleOCR 模型初始化完成。")
    return ocr


# --- 2. 图像处理模块 (无变化) ---
def load_image(image_path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[int]]:
    """
    从指定路径加载图片，并返回图片对象、高度和宽度。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法从路径加载图片: {image_path}")
        return None, None, None
    height, width, _ = image.shape
    print(f"图片加载成功: {image_path}, 尺寸: {width}x{height}")
    return image, height, width


# --- 3. OCR预测模块 (无变化) ---
def predict_text(ocr_model: PaddleOCR, image: np.ndarray) -> List[Dict[str, Any]]:
    """
    使用OCR模型对图像进行文本预测。
    """
    print("正在进行OCR识别...")
    result = ocr_model.predict(input=image)
    print("OCR识别完成。")
    return result


def find_subtitle(
    ocr_result: List[Dict[str, Any]],
    image_height: int,
    image_width: int,
    bottom_ratio: float = 0.5,
    rect_ang_thresh: float = 10.0,      # 最大旋转角度阈值（°）
    rect_ratio_thresh: float = 0.8,    # 最小矩形度阈值
    aspect_ratio_thresh: float = 2.0,   # 最小宽高比阈值（可选）
    width_ratio_thresh: float = 0.1     # 最小宽度占比（相对于 image_width）
) -> Optional[np.ndarray]:
    """
    从OCR结果中定位字幕框，并剔除不符合要求的框：
      1) 只保留位于画面底部的多边形
      2) 只保留四点多边形
      3) 最小外接矩形旋转角度小于 rect_ang_thresh
      4) 矩形度 (多边形面积 / minAreaRect面积) > rect_ratio_thresh
      5) 可选：宽高比 > aspect_ratio_thresh
      6) 剔除宽度 < image_width * width_ratio_thresh 的框
      7) **新增**：框的水平投影必须跨过画面中心线
      8) 在剩余候选框里，用 Y 位置和 X 居中打分，选出最高者

    返回最终框的 4×2 numpy 数组，或 None。
    """

    # 1. 底部候选筛选
    if not ocr_result or 'rec_polys' not in ocr_result[0] or ocr_result[0]['rec_polys'] is None:
        return None
    all_boxes = ocr_result[0]['rec_polys']
    bottom_y = image_height * bottom_ratio
    cand = [box for box in all_boxes if np.min(box[:, 1]) > bottom_y]
    if not cand:
        return None
    if len(cand) == 1:
        return cand[0]

    # 2. 形状 & 大小 & 中心线 跨越 筛选
    filtered = []
    min_width = image_width * width_ratio_thresh
    center_x = image_width / 2  # 中心线横坐标

    for box in cand:
        # 必须是四边形
        if box.shape[0] != 4:
            continue

        # 计算包围盒宽度
        xs = box[:, 0]
        width = xs.max() - xs.min()
        if width < min_width:
            # 宽度太小，剔除
            continue

        # 新增条件：水平投影必须跨过中心线
        if not (xs.min() < center_x < xs.max()):
            # 没有跨过中线，剔除
            continue

        # 计算最小外接矩形
        rect = cv2.minAreaRect(box.astype(np.float32))
        angle = abs(rect[2])
        if angle > rect_ang_thresh:
            continue

        # 矩形度（多边形面积 / minAreaRect 面积）
        area_poly = cv2.contourArea(box.astype(np.float32))
        box_w, box_h = rect[1]
        area_rect = box_w * box_h
        if area_rect <= 0 or (area_poly / area_rect) < rect_ratio_thresh:
            continue

        # 可选：宽高比过滤
        if aspect_ratio_thresh is not None:
            ar = max(box_w, box_h) / (min(box_w, box_h) + 1e-6)
            if ar < aspect_ratio_thresh:
                continue

        filtered.append(box)

    if not filtered:
        return None
    if len(filtered) == 1:
        return filtered[0]

    # 3. 打分选最佳
    Y_WEIGHT = 1.0
    X_WEIGHT = 10
    CENTER_X = center_x

    def score(box: np.ndarray) -> float:
        cy = np.mean(box[:, 1])
        y_score = (cy / image_height) * Y_WEIGHT
        cx = np.mean(box[:, 0])
        x_pen = abs(cx - CENTER_X) / image_width * X_WEIGHT
        return y_score - x_pen

    best = max(filtered, key=score)
    return best




# --- 5. 可视化模块 (无变化) ---
def draw_box_on_image(image: np.ndarray, box: Optional[np.ndarray], output_path: str):
    """
    在图片上绘制给定的框，并保存到文件。
    """
    if box is None:
        print("没有定位到字幕框，不进行绘制。")
        return

    points = box.astype(np.int32)
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imwrite(output_path, image)
    print(f"结果已绘制并保存到: {output_path}")


# --- 6. 主流程控制 (调用方式简化) ---
def main_find_subtitle(image_path: str, output_path='temp.jpg'):
    """
    主函数，执行完整的字幕定位和可视化流程。

    :param image_path: 输入图片的路径。
    :param output_path: 输出图片的保存路径。
    """
    # 步骤1: 初始化OCR模型
    ocr_model = init_ocr_model()

    # 步骤2: 加载图片
    image, height, width = load_image(image_path)
    if image is None:
        return

    # 步骤3: 执行OCR预测
    ocr_result = predict_text(ocr_model, image)

    # --- 关键变化：一步到位寻找字幕 ---
    # 步骤4: 从OCR结果中寻找字幕
    subtitle_box = find_subtitle(ocr_result, height, width)

    # 步骤5: 打印结果并进行可视化
    print("\n" + "=" * 20 + " 定位结果 " + "=" * 20)
    if subtitle_box is not None:
        print(f"✅ 成功定位到字幕！高度为: {subtitle_box[2][1] - subtitle_box[0][1]} 像素")
        print("字幕坐标为:")
        print(subtitle_box.tolist())  # 打印时转为list
        draw_box_on_image(image.copy(), subtitle_box, image_path)
    else:
        print("❌ 未能在图片中定位到字幕。")
    print("=" * 54)
    # 将subtitle_box转换为列表
    if subtitle_box is not None:
        subtitle_box = subtitle_box.tolist()
    else:
        subtitle_box = None
    return subtitle_box


# ==============================================================================
# 核心功能实现
# ==============================================================================

def analyze_and_filter_boxes(
        boxes: List[List[List[int]]],
        height_tolerance_ratio: float = 0.3,
        y_pos_tolerance_ratio: float = 0.5,
        z_score_threshold: float = 2.0
) -> List[List[List[int]]]:
    """
    【三阶段优化版】分析并过滤字幕框，增加下底位置筛选。

    Args:
        boxes (List[List[List[int]]]): 所有检测到的字幕框。
        height_tolerance_ratio (float): 第一阶段过滤中，框高与中位数的最大允许偏差比例。
        y_pos_tolerance_ratio (float): 第二阶段过滤中，框下底与中位下底的最大允许偏差比例
                                       (以中位数高度为基准)。
        z_score_threshold (float): 第三阶段过滤中，用于判断是否为异常值的Z分数阈值。

    Returns:
        List[List[List[int]]]: 过滤后的高质量字幕框列表。
    """
    if len(boxes) < 5:  # 数据太少，统计无意义，直接返回
        print("[过滤] 检测到的框数量过少，跳过复杂过滤。")
        return boxes

    # --- 阶段 1: 基于高度中位数的粗筛 ---
    heights = []
    for box in boxes:
        y_coords = [p[1] for p in box]
        heights.append(max(y_coords) - min(y_coords))

    median_height = np.median(heights)
    if median_height == 0: return []  # 避免后续除零错误

    height_diff_threshold = median_height * height_tolerance_ratio

    height_filtered_boxes = []
    print(f"[过滤-阶段1: 高度] 以中位高度 {median_height:.2f} 为基准 (容忍度: {height_diff_threshold:.2f}px)。")
    for i, box in enumerate(boxes):
        if abs(heights[i] - median_height) <= height_diff_threshold:
            height_filtered_boxes.append(box)
        else:
            print(f"  - 剔除高度异常框: 高度为 {heights[i]}, 与中位数差异过大。")

    print(f"[过滤-阶段1: 高度] 完成，剩余 {len(height_filtered_boxes)} 个框进入下一阶段。")
    if len(height_filtered_boxes) < 3:
        return height_filtered_boxes

    # --- 阶段 2: 基于下底位置的中筛 (新增筛选逻辑) ---
    bottom_ys = [max(p[1] for p in box) for box in height_filtered_boxes]
    median_bottom_y = np.median(bottom_ys)

    # 位置容忍度以中位高度为基准，更具自适应性
    y_pos_diff_threshold = median_height * y_pos_tolerance_ratio

    position_filtered_boxes = []
    print(f"[过滤-阶段2: 位置] 以中位下底 {median_bottom_y:.2f} 为基准 (容忍度: {y_pos_diff_threshold:.2f}px)。")
    for i, box in enumerate(height_filtered_boxes):
        if abs(bottom_ys[i] - median_bottom_y) <= y_pos_diff_threshold:
            position_filtered_boxes.append(box)
        else:
            print(f"  - 剔除位置异常框: 下底为 {bottom_ys[i]} (期望: {median_bottom_y:.2f})，差异过大。")

    print(f"[过滤-阶段2: 位置] 完成，剩余 {len(position_filtered_boxes)} 个框进入精筛。")
    if len(position_filtered_boxes) < 3:
        return position_filtered_boxes

    # --- 阶段 3: 对精筛后的框进行Z-score统计分析 ---
    properties = []
    for box in position_filtered_boxes:
        y_coords = [p[1] for p in box]
        min_y, max_y = min(y_coords), max(y_coords)
        properties.append({'height': max_y - min_y, 'center_y': min_y + (max_y - min_y) / 2})

    clean_heights = np.array([p['height'] for p in properties])
    clean_center_ys = np.array([p['center_y'] for p in properties])

    mean_height, std_height = np.mean(clean_heights), np.std(clean_heights)
    mean_center_y, std_center_y = np.mean(clean_center_ys), np.std(clean_center_ys)

    final_good_boxes = []
    print(f"[过滤-阶段3: 精筛] 以稳定数据为基准进行Z-score分析。")
    for i, box in enumerate(position_filtered_boxes):
        prop = properties[i]
        # 计算 Z-score, 避免标准差为0时除零
        height_z = abs(prop['height'] - mean_height) / std_height if std_height > 0 else 0
        center_y_z = abs(prop['center_y'] - mean_center_y) / std_center_y if std_center_y > 0 else 0

        if height_z < z_score_threshold and center_y_z < z_score_threshold:
            final_good_boxes.append(box)
        else:
            print(f"  - 剔除Z-score异常框 (高度Z: {height_z:.2f}, 中心Y Z: {center_y_z:.2f})")

    print(f"[过滤] 所有流程完成，最终保留 {len(final_good_boxes)} 个高质量字幕框。")
    return final_good_boxes

def draw_box_on_images(image_paths: List[str], box_coords: List[List[int]]):
    """
    将一个指定的包围框绘制到一组图片上并覆盖保存。
    """
    # 从[[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]中获取左上角和右下角点
    pt1 = tuple(box_coords[0])  # (min_x, min_y)
    pt2 = tuple(box_coords[2])  # (max_x, max_y)
    color = (0, 255, 0)  # 绿色 (BGR)
    thickness = 2

    for path in image_paths:
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            cv2.rectangle(img, pt1, pt2, color, thickness)
            cv2.imwrite(path, img)  # 覆盖保存
        except Exception as e:
            print(f"绘制 '{os.path.basename(path)}' 时出错: {e}")


def find_overall_subtitle_box(video_path: str, time_interval_seconds: int = 20):
    """
    主函数，找到包围视频字幕的最小框，并将框绘制到所有抽帧图片上。
    """
    output_dir = 'temp_dir'

    # --- 准备工作：创建或清空输出目录 ---
    if os.path.exists(output_dir):
        print(f"目录 '{output_dir}' 已存在，正在清空...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"已创建空的输出目录: '{output_dir}'")

    # --- 检查视频文件 ---
    if not os.path.exists(video_path):
        print(f"错误: 视频文件未找到 '{video_path}'")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 '{video_path}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("错误: 无法获取视频的FPS，将使用默认抽帧间隔（每600帧）。")
        frame_interval = 600
    else:
        frame_interval = int(fps * time_interval_seconds)

    # --- 阶段 1: 抽帧并保存到 'temp_dir' ---
    print(f"\n[阶段 1] 抽帧已开始 (每 {time_interval_seconds} 秒 / {frame_interval} 帧)。")

    saved_frame_paths = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            image_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(image_path, frame)
            saved_frame_paths.append(image_path)

        frame_count += 1

    cap.release()
    print(f"[阶段 1] 抽帧完成。共提取 {len(saved_frame_paths)} 帧图片。")

    if not saved_frame_paths:
        print("未能从视频中提取任何帧。")
        return

    # --- 阶段 2: 调用函数进行字幕检测 ---
    print(f"\n[阶段 2] 开始对 {len(saved_frame_paths)} 张图片进行字幕检测...")
    detected_boxes = [box for path in saved_frame_paths if (box := main_find_subtitle(path)) is not None]
    print(f"[阶段 2] 检测完成。共在 {len(detected_boxes)} 帧中找到字幕。")

    if not detected_boxes:
        print("\n[结果] 未找到任何字幕框。图片已保存在 'temp_dir' 目录中，但未做任何修改。")
        return

    # --- 阶段 3: 分析并计算最终包围框 ---
    print("\n[阶段 3] 开始分析字幕框并计算最终包围区域...")
    good_boxes = analyze_and_filter_boxes(detected_boxes)

    if not good_boxes:
        print("\n[结果] 所有检测到的框都被过滤为异常值。图片已保存在 'temp_dir' 目录中，但未做任何修改。")
        return

    print(f"[阶段 3] 过滤后剩余 {len(good_boxes)} 个有效字幕框。")

    all_points = np.array([point for box in good_boxes for point in box])
    final_box = [
        [int(np.min(all_points[:, 0])), int(np.min(all_points[:, 1]))],  # min_x, min_y
        [int(np.max(all_points[:, 0])), int(np.min(all_points[:, 1]))],  # max_x, min_y
        [int(np.max(all_points[:, 0])), int(np.max(all_points[:, 1]))],  # max_x, max_y
        [int(np.min(all_points[:, 0])), int(np.max(all_points[:, 1]))]  # min_x, max_y
    ]
    print(f"[阶段 3] 计算出的最终包围框: {final_box}")
    return final_box

    # --- 阶段 4: 将最终包围框绘制到所有抽取的帧上 ---
    print(f"\n[阶段 4] 正在将最终包围框绘制到 '{output_dir}' 目录下的 {len(saved_frame_paths)} 张图片上...")
    draw_box_on_images(saved_frame_paths, final_box)
    print("[阶段 4] 绘制完成。")

    # --- 任务结束 ---
    print("\n" + "=" * 60)
    print("任务成功！")
    print(f"最终的字幕包围框为: {final_box}")
    print(f"带有包围框的图片已全部保存在 '{output_dir}' 目录中。")
    print("=" * 60)


import os
import shutil
import cv2
import numpy as np

def find_overall_subtitle_box_target_number(
    video_path: str,
    merged_timerange_list: list[dict],
    num_samples: int = 10
):
    """
    主函数，找到包围视频字幕的最小框，并将框绘制到所有抽帧图片上。
    且保证抽取到 num_samples 帧（或所有符合区间的帧）。
    :param video_path: 视频文件路径
    :param merged_timerange_list: [{ "startTime": "00:00:00.205", "endTime": "00:00:09.060" }, ...]
    :param num_samples: 希望抽取的帧数
    :return: 最终包围框顶点列表 [[x1, y1], ..., [x4, y4]]
    """
    output_dir = 'temp_dir'

    # --- 准备工作：创建或清空输出目录 ---
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- 检查视频文件 ---
    if not os.path.exists(video_path):
        print(f"错误: 视频文件未找到 '{video_path}'")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 '{video_path}'")
        return

    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0 or fps <= 0:
        print("错误: 无法获取视频信息，抽帧失败。")
        cap.release()
        return

    # --- 预处理：将所有时间区间转为毫秒，并对应到帧索引 ---
    valid_frames = set()
    for tr in merged_timerange_list:
        start_ms = time_to_ms(tr['startTime'])
        end_ms   = time_to_ms(tr['endTime'])
        if end_ms <= start_ms:
            continue
        # 对应的帧索引区间
        start_idx = int(np.ceil(start_ms / 1000 * fps))
        end_idx   = int(np.floor(end_ms   / 1000 * fps))
        # 限制在 [0, total_frames-1]
        start_idx = max(0, start_idx)
        end_idx   = min(total_frames - 1, end_idx)
        valid_frames.update(range(start_idx, end_idx + 1))

    valid_frames = sorted(valid_frames)
    if not valid_frames:
        print("错误: 没有任何帧落在指定的时间区间内。")
        cap.release()
        return

    # --- 从 valid_frames 中均匀抽取 num_samples 帧 ---
    num_to_pick = min(num_samples, len(valid_frames))
    step = len(valid_frames) / num_to_pick
    sample_indices = [valid_frames[int(i * step)] for i in range(num_to_pick)]

    # --- 阶段 1: 按 sample_indices 抽帧并保存 ---
    print(f"\n[阶段 1] 从 {len(valid_frames)} 个符合区间的帧中抽取 {len(sample_indices)} 帧...")
    saved_frame_paths = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"警告: 无法读取第 {idx} 帧。")
            continue
        path = os.path.join(output_dir, f"frame_{idx}.jpg")
        cv2.imwrite(path, frame)
        saved_frame_paths.append(path)
    cap.release()
    print(f"[阶段 1] 抽帧完成。共保存 {len(saved_frame_paths)} 帧图片。")

    if not saved_frame_paths:
        print("未能提取任何帧。")
        return

    # --- 阶段 2: 对抽出的帧进行字幕检测 ---
    print(f"\n[阶段 2] 开始检测 {len(saved_frame_paths)} 张图片的字幕框...")
    detected_boxes = []
    for p in saved_frame_paths:
        box = main_find_subtitle(p)
        if box is not None:
            detected_boxes.append(box)
    print(f"[阶段 2] 检测完成。共检测到 {len(detected_boxes)} 个字幕框。")

    if not detected_boxes:
        print("未找到任何字幕框。")
        return

    # --- 阶段 3: 分析并计算最终包围框 ---
    print("\n[阶段 3] 分析并计算最终包围区域...")
    good_boxes = analyze_and_filter_boxes(detected_boxes)
    if not good_boxes:
        print("所有字幕框均被过滤。")
        return

    all_pts = np.array([pt for box in good_boxes for pt in box])
    final_box = [
        [int(all_pts[:,0].min()), int(all_pts[:,1].min())],
        [int(all_pts[:,0].max()), int(all_pts[:,1].min())],
        [int(all_pts[:,0].max()), int(all_pts[:,1].max())],
        [int(all_pts[:,0].min()), int(all_pts[:,1].max())],
    ]
    print(f"[阶段 3] 最终包围框: {final_box}")

    # --- 阶段 4: 绘制到所有抽帧图片上 ---
    print(f"\n[阶段 4] 将包围框绘制到 {len(saved_frame_paths)} 张图片上...")
    draw_box_on_images(saved_frame_paths, final_box)
    print("[阶段 4] 绘制完成。")

    print("\n" + "="*60)
    print("任务成功！")
    return final_box


def find_overall_subtitle_box_target_number_old(video_path: str, num_samples: int = 10):
    """
    主函数，找到包围视频字幕的最小框，并将框绘制到所有抽帧图片上。
    :param video_path: 视频文件路径
    :param num_samples: 希望平均抽取的帧数
    :return: 最终包围框顶点列表 [[x1, y1], ..., [x4, y4]]
    """
    output_dir = 'temp_dir'

    # --- 准备工作：创建或清空输出目录 ---
    if os.path.exists(output_dir):
        print(f"目录 '{output_dir}' 已存在，正在清空...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"已创建空的输出目录: '{output_dir}'")

    # --- 检查视频文件 ---
    if not os.path.exists(video_path):
        print(f"错误: 视频文件未找到 '{video_path}'")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 '{video_path}'")
        return

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("错误: 无法获取视频总帧数，抽帧失败。")
        cap.release()
        return

    # 计算采样帧索引
    num_samples = min(num_samples, total_frames)
    interval = total_frames / num_samples
    sample_indices = [int(i * interval) for i in range(num_samples)]

    # --- 阶段 1: 抽帧并保存到 'temp_dir' ---
    print(f"\n[阶段 1] 抽帧已开始 (共 {num_samples} 帧)。")
    saved_frame_paths = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"警告: 无法读取第 {idx} 帧。")
            continue

        image_path = os.path.join(output_dir, f"frame_{idx}.jpg")
        cv2.imwrite(image_path, frame)
        saved_frame_paths.append(image_path)

    cap.release()
    print(f"[阶段 1] 抽帧完成。共提取 {len(saved_frame_paths)} 帧图片。")

    if not saved_frame_paths:
        print("未能从视频中提取任何帧。")
        return

    # --- 阶段 2: 调用函数进行字幕检测 ---
    print(f"\n[阶段 2] 开始对 {len(saved_frame_paths)} 张图片进行字幕检测...")
    detected_boxes = []
    for path in saved_frame_paths:
        box = main_find_subtitle(path)
        if box is not None:
            detected_boxes.append(box)
    print(f"[阶段 2] 检测完成。共在 {len(detected_boxes)} 帧中找到字幕。")

    if not detected_boxes:
        print("\n[结果] 未找到任何字幕框。图片已保存在 'temp_dir' 目录中，但未做任何修改。")
        return

    # --- 阶段 3: 分析并计算最终包围框 ---
    print("\n[阶段 3] 开始分析字幕框并计算最终包围区域...")
    good_boxes = analyze_and_filter_boxes(detected_boxes)

    if not good_boxes:
        print("\n[结果] 所有检测到的框都被过滤为异常值。图片已保存在 'temp_dir' 目录中，但未做任何修改。")
        return

    print(f"[阶段 3] 过滤后剩余 {len(good_boxes)} 个有效字幕框。")

    all_points = np.array([pt for box in good_boxes for pt in box])
    final_box = [
        [int(np.min(all_points[:, 0])), int(np.min(all_points[:, 1]))],  # min_x, min_y
        [int(np.max(all_points[:, 0])), int(np.min(all_points[:, 1]))],  # max_x, min_y
        [int(np.max(all_points[:, 0])), int(np.max(all_points[:, 1]))],  # max_x, max_y
        [int(np.min(all_points[:, 0])), int(np.max(all_points[:, 1]))]   # min_x, max_y
    ]
    print(f"[阶段 3] 计算出的最终包围框: {final_box}")

    # --- 阶段 4: 将最终包围框绘制到所有抽取的帧上 ---
    print(f"\n[阶段 4] 正在将最终包围框绘制到 '{output_dir}' 目录下的 {len(saved_frame_paths)} 张图片上...")
    draw_box_on_images(saved_frame_paths, final_box)
    print("[阶段 4] 绘制完成。")

    # --- 任务结束 ---
    print("\n" + "=" * 60)
    print("任务成功！")
    print(f"最终的字幕包围框为: {final_box}")
    print(f"带有包围框的图片已全部保存在 '{output_dir}' 目录中。")
    print("=" * 60)

    return final_box


# ==============================================================================
# 使用示例
# ==============================================================================
if __name__ == '__main__':
    # 1. 将此路径替换为您的视频文件路径
    video_file_path = "../test3.mp4"

    # 2. 确保您已在上面的 main_find_subtitle 函数中实现了自己的逻辑

    print("=" * 60)
    print("              开始执行视频字幕区域查找任务")
    print("=" * 60)

    # 调用主函数
    find_overall_subtitle_box_target_number(
        video_path=video_file_path,
        num_samples=10
    )