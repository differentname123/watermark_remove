import cv2
import sys


def func(frame):
    """
    用户自定义的帧处理函数。
    输入：frame（numpy.ndarray，BGR 格式）
    输出：processed_frame（numpy.ndarray，BGR 格式，大小须与输入相同）

    这里示例：将帧转换为灰度并再转换回 BGR（灰度效果）。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return processed


def process_video(input_path: str, output_path: str, func):
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {input_path}")

    # 读取视频的基本信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # fourcc：这里用 mp4v 可以输出 .mp4；如需 .avi 可改成 'XVID'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 创建 VideoWriter，注意输出尺寸必须与处理后帧保持一致
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 读完了
        # 调用用户自定义处理函数
        processed = func(frame)

        # 如果 func 返回的尺寸、通道数与原始不符，可在此处做调整或抛错
        if processed.shape[1] != width or processed.shape[0] != height:
            processed = cv2.resize(processed, (width, height))
        # 写入新视频
        out.write(processed)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"已处理 {frame_idx} 帧...")

    # 释放资源
    cap.release()
    out.release()
    print(f"处理完成，共写入 {frame_idx} 帧，新视频保存在：{output_path}")


if __name__ == '__main__':
    input_video = "test.mp4"
    output_video = "test_out.mp4"
    process_video(input_video, output_video, func)