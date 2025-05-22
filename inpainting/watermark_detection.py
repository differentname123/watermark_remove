import cv2
import torch
import numpy as np
import time
import math
from typing import List
from ultralytics import YOLO

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
# 2. 批量帧处理函数（屏蔽内置日志）
# -------------------------------------------------
@torch.inference_mode()
def yolo_batch_process(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    对一批 BGR 帧做 YOLO 检测并画框，返回 BGR 帧列表。
    verbose=False 屏蔽 Ultralytics 的内置打印。
    """
    results = yolo_model(
        frames,
        imgsz=1024,
        augment=True,
        iou=0.5,
        verbose=False
    )
    return [r.plot() for r in results]

# -------------------------------------------------
# 3. 视频处理函数（关键日志：视频信息、总帧、预计批次、每批耗时）
# -------------------------------------------------
def process_video(input_path: str,
                  output_path: str,
                  func,
                  batch_size: int = 16):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_batches = math.ceil(total_frames / batch_size)

    print(f"[{time.strftime('%H:%M:%S')}] [INFO] Video info:")
    print(f"    Resolution: {width}x{height}")
    print(f"    FPS:        {fps:.2f}")
    print(f"    Total frames:   {total_frames}")
    print(f"    Batch size:     {batch_size}")
    print(f"    Estimated batches: {total_batches}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    buffer     = []
    frame_idx  = 0
    batch_idx  = 0

    def flush_buffer():
        nonlocal frame_idx, batch_idx, buffer
        batch_idx += 1
        t0 = time.time()
        processed = func(buffer)
        t1 = time.time()
        print(f"[Batch {batch_idx}/{total_batches}] processed {len(buffer)} frames in {t1-t0:.2f}s")
        for img in processed:
            if img.shape[1]!=width or img.shape[0]!=height:
                img = cv2.resize(img, (width, height))
            if img.ndim==2 or img.shape[2]==1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            out.write(img)
        frame_idx += len(buffer)
        buffer = []

    print(f"[{time.strftime('%H:%M:%S')}] [INFO] Start processing video...")
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buffer.append(frame)

        if len(buffer) >= batch_size:
            flush_buffer()
            count += 1
            if count > 10:
                break

    if buffer:
        flush_buffer()

    cap.release()
    out.release()
    print(f"[{time.strftime('%H:%M:%S')}] [INFO] Done. Total frames written: {frame_idx}. Output: {output_path}")

# -------------------------------------------------
# 4. 主程序
# -------------------------------------------------
if __name__ == "__main__":
    process_video("test1.mp4", "test_out.mp4", yolo_batch_process, batch_size=32)