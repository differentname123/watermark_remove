import cv2
import torch
import numpy as np
import time
import math
from typing import List
from ultralytics import YOLO

# -------------------------------------------------
# 全局变量：用于收集所有检测到的框及其图像信息（附带 frame_id）
# -------------------------------------------------
collected_boxes = []  # 每个元素为字典：{"frame_id": int, "bbox": (x1, y1, x2, y2), "crop": 图像区域}
first_frame = None  # 用于保存视频第一帧（作为后续绘制的背景）

# -------------------------------------------------
# 1. 设备与模型初始化
# -------------------------------------------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[{time.strftime('%H:%M:%S')}] [INFO] Using device: {DEVICE}")

yolo_model = YOLO("yolo11x-train28-best.pt")
if DEVICE.startswith("cuda"):
    yolo_model.to(DEVICE)
    yolo_model.fuse()
    yolo_model.half()
print(f"[{time.strftime('%H:%M:%S')}] [INFO] YOLO model loaded. Classes: {len(yolo_model.names)}")


# -------------------------------------------------
# 2. 批量帧处理函数（检测、绘制框，并收集每个框及其对应图像，同时记录 frame_id）
# -------------------------------------------------
@torch.inference_mode()
def yolo_batch_process(frames: List[np.ndarray], frame_ids: List[int]) -> List[np.ndarray]:
    """
    对一批 BGR 帧做 YOLO 检测：
      - 收集每个检测框的位置信息、对应的区域图像以及 frame_id 到 collected_boxes；
      - 返回绘制了检测结果（利用 Ultralytics 的 r.plot()）的帧列表。
    """
    global collected_boxes
    results = yolo_model(
        frames,
        imgsz=1024,
        augment=True,
        iou=0.5,
        verbose=False
    )
    processed_frames = []
    # 遍历每个帧及对应的 frame_id 与检测结果
    for (frame, fid), r in zip(zip(frames, frame_ids), results):
        # 如果检测到框，则进行处理
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            # 得到所有框的坐标：[x1, y1, x2, y2]
            xyxys = r.boxes.xyxy.cpu().numpy()
            for box in xyxys:
                x1, y1, x2, y2 = box.astype(int)
                # 限制坐标不超出图像边界
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, frame.shape[1] - 1)
                y2 = min(y2, frame.shape[0] - 1)
                # 裁剪出检测框内图像（复制一份以防后续修改）
                crop = frame[y1:y2, x1:x2].copy()
                collected_boxes.append({
                    "frame_id": fid,
                    "bbox": (x1, y1, x2, y2),
                    "crop": crop
                })
        # 使用 Ultralytics 的 r.plot() 绘制检测框
        annotated_frame = r.plot()
        processed_frames.append(annotated_frame)
    return processed_frames


# -------------------------------------------------
# 3. 视频处理函数（读取视频、批量调用检测、写入输出视频，同时记录 frame_id）
# -------------------------------------------------
def process_video(input_path: str,
                  output_path: str,
                  func,
                  batch_size: int = 16):
    """
    读取视频，分批调用检测函数，同时记录每帧的全局 frame_id，
    输出带检测框的视频，并保存首帧用于后续绘制 summary。
    """
    global first_frame
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_batches = math.ceil(total_frames / batch_size)

    print(f"[{time.strftime('%H:%M:%S')}] [INFO] Video info:")
    print(f"    Resolution: {width}x{height}")
    print(f"    FPS:        {fps:.2f}")
    print(f"    Total frames: {total_frames}")
    print(f"    Batch size: {batch_size}")
    print(f"    Estimated batches: {total_batches}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    buffer = []
    frame_ids = []
    frame_idx = 0  # 全局帧计数
    batch_idx = 0

    def flush_buffer():
        nonlocal frame_idx, batch_idx, buffer, frame_ids
        batch_idx += 1
        t0 = time.time()
        processed = func(buffer, frame_ids)
        t1 = time.time()
        print(f"[Batch {batch_idx}/{total_batches}] Processed {len(buffer)} frames in {t1 - t0:.2f}s")
        for img in processed:
            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height))
            if img.ndim == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            out.write(img)
        frame_idx += len(buffer)
        buffer.clear()
        frame_ids.clear()

    print(f"[{time.strftime('%H:%M:%S')}] [INFO] Start processing video...")
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 保存第一帧，用于后续绘制 summary
        if first_frame is None:
            first_frame = frame.copy()
        buffer.append(frame)
        frame_ids.append(frame_idx)
        frame_idx += 1
        if len(buffer) >= batch_size:
            flush_buffer()
            count += 1
            # 调试时仅处理部分帧，处理完整视频时请删除下面这行
            if count > 10:
                break

    if buffer:
        flush_buffer()

    cap.release()
    out.release()
    print(f"[{time.strftime('%H:%M:%S')}] [INFO] Done. Total frames written: {frame_idx}. Output: {output_path}")


# -------------------------------------------------
# 4. 后续处理：对收集到的框基于直方图相似度进行聚类（不依赖 skimage）
# -------------------------------------------------
def compare_histograms(img1, img2, bins=256):
    """
    计算两幅灰度图像的直方图相似度：
      - 使用 cv2.calcHist 计算直方图，再归一化，
      - 利用 cv2.compareHist (HISTCMP_CORREL) 进行比较，
      - 返回值范围为 -1 至 1, 1 表示完全一致。
    """
    hist1 = cv2.calcHist([img1], [0], None, [bins], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [bins], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    similarity = cv2.compareHist(hist1, hist2, method=cv2.HISTCMP_CORREL)
    return similarity


def cluster_boxes(boxes, threshold=0.8):
    """
    根据每个检测框（crop 区域）的直方图相似度进行聚类：
      - 若新框与某聚类中代表框的直方图相似度大于 threshold，则认为为同一类；
      - 聚类中记录的信息包括：
            "rep_img": 用于比较的固定尺寸灰度图（不再更改，用于相似度比较）,
            "original": 代表图（原始尺寸，用于展示，始终保存最早出现的检测）,
            "bbox": 最早出现时的框位置,
            "min_frame_id": 该聚类中最早的 frame_id,
            "count": 出现次数,
            "sim_total": 累计相似度（第一项自比为 1）
    """
    clusters = []
    for item in boxes:
        crop = item["crop"]
        if crop.size == 0:
            continue
        try:
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        except Exception:
            continue
        try:
            crop_resized = cv2.resize(crop_gray, (64, 64))
        except Exception:
            continue

        matched = False
        for cluster in clusters:
            sim = compare_histograms(crop_resized, cluster["rep_img"])
            if sim > threshold:
                cluster["count"] += 1
                cluster["sim_total"] += sim
                # 若该检测出现在更早的帧中，则更新代表用于绘制框
                if item["frame_id"] < cluster["min_frame_id"]:
                    cluster["min_frame_id"] = item["frame_id"]
                    cluster["bbox"] = item["bbox"]
                    cluster["original"] = crop
                matched = True
                break
        if not matched:
            clusters.append({
                "rep_img": crop_resized,  # 固定用于比较的代表图
                "original": crop,  # 用于展示的原始尺寸代表图
                "bbox": item["bbox"],  # 检测框位置信息
                "min_frame_id": item["frame_id"],
                "count": 1,
                "sim_total": 1.0  # 第一个加入视为与自身相似度为1
            })
    return clusters


def create_summary_image(base_frame, clusters, output_path="summary.png"):
    """
    以视频第一帧为背景，直接在对应位置绘制各聚类的代表检测框，
    同时标注出现次数和平均相似度（仅取出现次数最多的前 5 个聚类）。
    """
    if base_frame is None:
        print("[ERROR] No base frame available for summary image.")
        return
    # 复制首帧用于绘制
    summary_image = base_frame.copy()
    # 按出现次数降序排序，并只取前 5 个聚类
    clusters_sorted = sorted(clusters, key=lambda c: c["count"], reverse=True)
    top_clusters = clusters_sorted[:5]

    for cluster in top_clusters:
        # 计算平均相似度
        avg_sim = cluster["sim_total"] / cluster["count"]
        x1, y1, x2, y2 = cluster["bbox"]
        # 在首帧上画出检测框（绿色，线宽 2）
        cv2.rectangle(summary_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 标注出现次数和平均相似度（红色字体）
        text = f"Count: {cluster['count']}, Avg: {avg_sim:.2f}"
        cv2.putText(summary_image, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(output_path, summary_image)
    print(f"[INFO] Summary image saved as {output_path}")


# -------------------------------------------------
# 5. 主程序
# -------------------------------------------------
if __name__ == "__main__":
    # 修改下面的视频路径和输出视频路径
    input_video = "test2.mp4"
    output_video = "test_out.mp4"

    # 处理视频：检测、绘制框并生成输出视频
    process_video(input_video, output_video, yolo_batch_process, batch_size=32)

    # 检测结束后，对收集到的框做后续聚类
    print("[INFO] Post-processing collected boxes...")
    clusters = cluster_boxes(collected_boxes, threshold=0.8)
    if not clusters:
        print("[WARN] No clusters found. Summary image will not be generated.")
    else:
        print(f"[INFO] {len(clusters)} unique box types found (threshold=0.8).")
        # 直接在第一帧上绘制前 5 个出现次数最多的聚类结果，标注出现次数及平均相似度
        create_summary_image(first_frame, clusters, output_path="summary.png")