import cv2
import os
import numpy as np
import shutil
from typing import List, Optional

from common_utils.ocr.paddle_ocr_utils import main_find_subtitle


# ==============================================================================
# 核心功能实现
# ==============================================================================

def analyze_and_filter_boxes(boxes: List[List[List[int]]], z_score_threshold: float = 2.0) -> List[List[List[int]]]:
    """
    分析所有检测到的字幕框，并基于统计数据（垂直中心和高度）过滤掉异常值。
    """
    if len(boxes) < 3:
        return boxes

    properties = []
    for box in boxes:
        y_coords = [p[1] for p in box]
        min_y, max_y = min(y_coords), max(y_coords)
        properties.append({'height': max_y - min_y, 'center_y': min_y + (max_y - min_y) / 2})

    heights = np.array([p['height'] for p in properties])
    center_ys = np.array([p['center_y'] for p in properties])

    mean_height, std_height = np.mean(heights), np.std(heights)
    mean_center_y, std_center_y = np.mean(center_ys), np.std(center_ys)

    good_boxes = []
    for i, box in enumerate(boxes):
        prop = properties[i]
        height_z = abs(prop['height'] - mean_height) / std_height if std_height > 0 else 0
        center_y_z = abs(prop['center_y'] - mean_center_y) / std_center_y if std_center_y > 0 else 0

        if height_z < z_score_threshold and center_y_z < z_score_threshold:
            good_boxes.append(box)
        else:
            print(f"[过滤] 剔除异常框: {box}")

    return good_boxes


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


# ==============================================================================
# 使用示例
# ==============================================================================
if __name__ == '__main__':
    # 1. 将此路径替换为您的视频文件路径
    video_file_path = "../test.mp4"

    # 2. 确保您已在上面的 main_find_subtitle 函数中实现了自己的逻辑

    print("=" * 60)
    print("              开始执行视频字幕区域查找任务")
    print("=" * 60)

    # 调用主函数
    find_overall_subtitle_box(
        video_path=video_file_path,
        time_interval_seconds=20
    )