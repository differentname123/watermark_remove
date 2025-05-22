# -*- coding: utf-8 -*-
import time
import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def process_frame(frame, reader):
    """
    对单帧图像进行 OCR 识别、绘制框、中文文本注释以及置信度，并返回处理后的帧和 OCR 检测结果
    """
    # OCR
    results = reader.readtext(frame, detail=1)
    # 先用 OpenCV 画框
    for bbox, text, conf in results:
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    # 转为 PIL 画字
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("simsun.ttc", 18)
    except:
        font = ImageFont.load_default()
    for bbox, text, conf in results:
        x, y = int(bbox[0][0]), int(bbox[0][1]) - 20
        y = max(0, y)
        draw.text((x, y), f"{text}: {conf:.2f}", font=font, fill=(255, 0, 0))
    frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return frame, results

def realtime_ocr(input_source=0, save_frames=False, max_frames=None):
    """
    实时读取摄像头或视频文件，处理并显示每一帧。
    参数:
      input_source: 摄像头索引或视频文件路径
      save_frames: 是否把每帧处理后的图像保存为 JPEG
      max_frames: 最多处理多少帧（None 为不限）
    """
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"无法打开输入：{input_source}")
        return

    # 初始化 OCR（中文+英文）
    reader = easyocr.Reader(['ch_sim','en'], gpu=True)

    frame_no = 0
    start = time.time()
    print("开始实时 OCR 处理，按 ESC 键退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        proc_frame, _ = process_frame(frame, reader)

        # 实时显示
        cv2.imshow("OCR 实时结果", proc_frame)

        # 可选：保存每帧
        if save_frames:
            fn = f"frame_{frame_no:04d}.jpg"
            cv2.imwrite(fn, proc_frame)

        # 按 ESC 退出，或者达到 max_frames
        if cv2.waitKey(1) & 0xFF == 27:
            break
        if max_frames and frame_no >= max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"处理完毕，总帧数={frame_no}, 耗时={time.time()-start:.2f}s")

if __name__ == "__main__":
    # 示例1：从摄像头实时 OCR，不保存帧
    # realtime_ocr(0, save_frames=False)

    # 示例2：从视频文件实时 OCR，把每帧保存为 JPEG，最多处理 200 帧
    realtime_ocr("test2.mp4", save_frames=False, max_frames=20000)