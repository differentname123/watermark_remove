import cv2
from typing import List


def func(frames: List[cv2.Mat]) -> List[cv2.Mat]:
    """
    用户自定义的批量帧处理函数。
    输入：frames（List[numpy.ndarray]，每帧 BGR 格式）
    输出：processed_frames（List[numpy.ndarray]，每帧 BGR 格式，大小须与对应输入相同）

    这里示例：把每帧转换为灰度再转回 BGR（保持三通道，得到灰度效果）。
    """
    processed = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        processed.append(bgr)
    return processed


def process_video(input_path: str,
                  output_path: str,
                  func,
                  batch_size: int = 16):
    """
    以 batch_size 为单位读取视频帧，调用 func(list_of_frames) 进行批量处理。
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    buffer: List[cv2.Mat] = []
    frame_idx = 0

    def flush_buffer(buf: List[cv2.Mat]):
        """
        调用 func 处理一批帧并写回视频。
        """
        processed_list = func(buf)
        if not isinstance(processed_list, list) or len(processed_list) != len(buf):
            raise ValueError("func 必须返回与输入列表等长的 List[np.ndarray]")
        for i, proc in enumerate(processed_list):
            # 尺寸或通道数对不上时，做自动调整
            if proc.shape[1] != width or proc.shape[0] != height:
                proc = cv2.resize(proc, (width, height))
            # 确保是三通道 BGR
            if len(proc.shape) == 2 or proc.shape[2] == 1:
                proc = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
            out.write(proc)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buffer.append(frame)
        # 达到一个 batch，处理并写入
        if len(buffer) >= batch_size:
            flush_buffer(buffer)
            frame_idx += len(buffer)
            print(f"已处理 {frame_idx} 帧...")
            buffer.clear()

    # 处理剩余帧
    if buffer:
        flush_buffer(buffer)
        frame_idx += len(buffer)
        print(f"已处理 {frame_idx} 帧（含尾帧）")

    cap.release()
    out.release()
    print(f"全部完成，共写入 {frame_idx} 帧，新视频保存在：{output_path}")


if __name__ == '__main__':
    input_video = "test.mp4"
    output_video = "test_out.mp4"
    # 注意传入批处理函数 func 和你想要的 batch_size
    process_video(input_video, output_video, func, batch_size=32)