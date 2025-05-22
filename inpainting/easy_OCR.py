import time
import cv2
import easyocr
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont  # 导入 Pillow 相关模块

def process_frame(frame, reader):
    """
    对单帧图像进行 OCR 识别、绘制框和中文文本注释，并返回处理后的帧和 OCR 检测结果

    参数:
        frame: 输入图像 (BGR格式)
        reader: easyocr.Reader 对象

    返回:
        processed_frame: 绘制好框和文本的图像
        results: OCR 识别结果，格式为 [(bbox, text, confidence), ...]
    """
    # 利用 easyocr 识别图像中的文本
    results = reader.readtext(frame, detail=1)

    # 先用 OpenCV 绘制识别框
    for bbox, text, confidence in results:
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 将 OpenCV 图像（BGR格式）转换为 RGB 格式，再转成 PIL Image 对象
    pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)

    # 加载中文字体（路径根据你的环境进行修改）
    try:
        font = ImageFont.truetype("simsun.ttc", size=18)
    except Exception as e:
        print("无法加载指定中文字体, 使用默认字体, 错误信息:", str(e))
        font = ImageFont.load_default()

    # 在 PIL Image 上绘制文本信息
    for bbox, text, confidence in results:
        top_left = bbox[0]
        text_position = (int(top_left[0]), max(0, int(top_left[1]) - 20))
        draw.text(text_position, text, font=font, fill=(255, 0, 0))

    # 将 PIL Image 转回 OpenCV 图像 (RGB->BGR)
    processed_frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    return processed_frame, results


def process_video(input_video, output_video, batch_size=10, max_workers=4, max_frames=100):
    """
    从输入视频文件读取帧，批量调用 OCR 识别（每隔 30 帧进行 OCR 处理），同时存储 OCR 检测结果；
    最后将出现最多的前 5 个检测结果的框绘制到起始帧上并保存图片，
    最终生成带有 OCR 标注的视频文件。

    参数:
        input_video: 输入视频文件路径
        output_video: 输出视频文件路径
        batch_size: 每个批次处理的帧数
        max_workers: 线程池中最大的工作线程数
        max_frames: 最大处理帧数（包括不处理的帧）
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

    # 初始化 OCR 识别器（支持中文与英文）
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)

    # 创建线程池以处理多帧
    executor = ThreadPoolExecutor(max_workers=max_workers)

    frame_count = 0
    # batch_frames 列表中每个元素为 (frame, global_frame_no, do_process)
    batch_frames = []
    batch_counter = 1
    start_time = time.time()
    print("开始视频处理...")

    # 用于存储所有经过 OCR 处理帧的检测结果，格式为：
    # {"frame_no": <帧号>, "bbox": <检测框>, "text": <识别文本>, "confidence": <置信度>}
    ocr_results_all = []

    first_frame = None  # 用于保存第一帧（起始帧），后续用于绘制 Top5 检测框

    def process_batch(batch_frames):
        """
        处理当前批次的帧：
          - 对标记为需要 OCR 处理的帧，调用线程池进行异步 OCR 处理；
          - 不需要处理的帧直接保留原帧；
        最后按原有顺序依次写入视频，并更新 OCR 检测结果列表。
        """
        futures = []
        for idx, (frm, frm_no, do_process) in enumerate(batch_frames):
            if do_process:
                future = executor.submit(process_frame, frm, reader)
                futures.append((future, frm_no, idx, True))
            else:
                futures.append((None, frm_no, idx, False))
        processed_frames = [None] * len(batch_frames)
        for future, frm_no, idx, do_process in futures:
            if do_process:
                processed_frame, results = future.result()
                processed_frames[idx] = processed_frame
                for item in results:
                    detection = {
                        "frame_no": frm_no,
                        "bbox": item[0],
                        "text": item[1],
                        "confidence": item[2]
                    }
                    ocr_results_all.append(detection)
            else:
                processed_frames[idx] = batch_frames[idx][0]
        # 按原有顺序写入视频文件
        for pf in processed_frames:
            out.write(pf)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 保存首帧（原始帧）用于后续绘制 Top5 检测框
        if frame_count == 1:
            first_frame = frame.copy()

        # 判断当前帧是否需要 OCR 处理：第一帧或每隔 30 帧进行处理
        if frame_count == 1 or frame_count % 30 == 0:
            do_process = True
        else:
            do_process = False

        batch_frames.append((frame, frame_count, do_process))

        # 达到最大处理帧数则退出（包括跳过的帧）
        if frame_count >= max_frames:
            break

        # 当累计一批（batch_size）帧时，进行批量处理
        if len(batch_frames) >= batch_size:
            print(f"处理第 {batch_counter} 批: 本批次 {len(batch_frames)} 帧，累计处理帧数: {frame_count}")
            process_batch(batch_frames)
            batch_frames = []
            batch_counter += 1

    # 处理最后一批不足 batch_size 的帧
    if batch_frames:
        print(f"处理最后一批: {len(batch_frames)} 帧，累计处理帧数: {frame_count}")
        process_batch(batch_frames)

    executor.shutdown()
    cap.release()
    out.release()

    print(f"视频处理完成，总共处理帧数: {frame_count}")
    print(f"生成视频存储路径: {output_video}")
    print(f"处理时间: {time.time() - start_time:.2f} 秒")

    # -----------------------------------------------------------
    # 统计所有 OCR 检测结果：按照识别文本分组，
    # 并记录首次出现（帧号最小）的检测结果；随后选出出现最多的前 5 个文本。
    # -----------------------------------------------------------
    text_counts = {}
    detection_by_text = {}
    for detection in ocr_results_all:
        text = detection["text"]
        if text not in text_counts:
            text_counts[text] = 0
            detection_by_text[text] = detection  # 保存首次出现的检测结果
        text_counts[text] += 1
        if detection["frame_no"] < detection_by_text[text]["frame_no"]:
            detection_by_text[text] = detection

    sorted_texts = sorted(text_counts.items(), key=lambda x: x[1], reverse=True)
    top5 = sorted_texts[:5]
    print("Top 5识别文本及其出现次数:", top5)

    # 在首帧上绘制 Top5 检测框
    if first_frame is not None:
        result_frame = first_frame.copy()
        for text, count in top5:
            detection = detection_by_text[text]
            bbox = np.array(detection["bbox"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(result_frame, [bbox], isClosed=True, color=(0, 255, 255), thickness=3)
            top_left = detection["bbox"][0]
            cv2.putText(result_frame, f"{text}:{count}", (int(top_left[0]), int(top_left[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite("top5_start_frame.jpg", result_frame)
        print("已保存含有 Top5 检测框的起始帧到 'top5_start_frame.jpg'")


def main():
    input_video_path = 'test.mp4'
    output_video_path = 'output_video.mp4'
    batch_size = 50
    max_workers = 4
    # 根据需要调整最大帧数，例如 100 或更大的值
    process_video(input_video_path, output_video_path, batch_size, max_workers, max_frames=1000000)


if __name__ == '__main__':
    main()