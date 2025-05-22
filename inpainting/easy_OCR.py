import time
import cv2
import easyocr
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont  # 导入 Pillow 相关模块


def process_frame(frame, reader):
    """
    对单帧图像进行 OCR 识别、绘制框和中文文本注释
    """
    # 利用 easyocr 识别图像中的文本
    results = reader.readtext(frame, detail=1)

    # 先用 OpenCV 绘制识别框
    for bbox, text, confidence in results:
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 将 OpenCV 图像从 BGR 格式转换为 RGB 格式后转成 PIL Image 对象
    pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)

    # 加载中文字体（路径需根据实际情况修改）
    try:
        # 此处使用宋体 (simsun.ttc)，如果没有可替换为其他字体，例如 "msyh.ttf"（微软雅黑）等
        font = ImageFont.truetype("simsun.ttc", size=18)
    except Exception as e:
        print("无法加载指定中文字体, 使用默认字体, 错误信息:", str(e))
        font = ImageFont.load_default()

    # 在 PIL Image 上绘制文本
    for bbox, text, confidence in results:
        top_left = bbox[0]
        # 设置文本位置，若检测区域较靠顶则避免负值坐标
        text_position = (int(top_left[0]), max(0, int(top_left[1]) - 20))
        # 绘制文本，fill 参数使用 RGB 颜色，此处设置为红色 (255, 0, 0)
        draw.text(text_position, text, font=font, fill=(255, 0, 0))

    # 将 PIL Image 转回 OpenCV 图像（RGB->BGR）
    frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    return frame


def process_video(input_video, output_video, batch_size=10, max_workers=4, max_frames=100):
    """
    从输入视频读取帧，并批量调用 OCR 识别，再将结果合成新视频输出。
    """
    # 打开视频文件
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"无法打开视频文件: {input_video}")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息 --> 帧率: {fps}, 分辨率: {width}x{height}, 总帧数: {total_frames}")

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 初始化 OCR 识别器（languages 参数中包含中文）
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)

    # 创建线程池，以复用线程处理多帧
    executor = ThreadPoolExecutor(max_workers=max_workers)

    frame_count = 0
    batch_frames = []
    batch_counter = 1
    start_time = time.time()
    print("开始视频处理...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        batch_frames.append(frame)
        frame_count += 1

        # 可视需求添加帧数限制
        if frame_count > max_frames:
            break

        if len(batch_frames) >= batch_size:
            print(f"处理第 {batch_counter} 批: 本批次 {len(batch_frames)} 帧，累计处理帧数: {frame_count}")
            futures = {executor.submit(process_frame, frame, reader): idx for idx, frame in enumerate(batch_frames)}
            # 为了保证帧序，可以按照索引存放处理结果
            processed_frames = [None] * len(batch_frames)
            for future in as_completed(futures):
                idx = futures[future]
                processed_frames[idx] = future.result()
            for pf in processed_frames:
                out.write(pf)
            batch_frames = []
            batch_counter += 1

    # 处理最后一批不足 batch_size 的帧
    if batch_frames:
        print(f"处理最后一批: {len(batch_frames)} 帧，累计处理帧数: {frame_count}")
        futures = {executor.submit(process_frame, frame, reader): idx for idx, frame in enumerate(batch_frames)}
        processed_frames = [None] * len(batch_frames)
        for future in as_completed(futures):
            idx = futures[future]
            processed_frames[idx] = future.result()
        for pf in processed_frames:
            out.write(pf)

    executor.shutdown()
    cap.release()
    out.release()

    print(f"视频处理完成，总共处理帧数: {frame_count}")
    print(f"生成视频存储路径: {output_video}")
    print(f"处理时间: {time.time() - start_time:.2f} 秒")


def main():
    input_video_path = 'test.mp4'
    output_video_path = 'output_video.mp4'
    batch_size = 50
    max_workers = 4
    process_video(input_video_path, output_video_path, batch_size, max_workers)


if __name__ == '__main__':
    main()