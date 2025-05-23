# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/5/22 19:41
:last_date:
    2025/5/22 19:41
:description:
    
"""
import cv2

# Global list to store the two corner points
pts = []

def click_event(event, x, y, flags, param):
    """
    Mouse callback function to record clicks.
    On each left-button down event, save the point and draw a circle.
    """
    global pts, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts) < 2:
            pts.append((x, y))
            cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select ROI - press ESC when done", img_display)

def main():
    global img_display

    # 1. 打开视频并读取第一帧
    cap = cv2.VideoCapture('test.mp4')
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("无法读取第一帧")
        return

    # 保留一份原图用于后续裁剪
    img_original = frame.copy()
    img_display = frame.copy()

    # 2. 显示窗口并设置鼠标回调
    cv2.namedWindow("Select ROI - press ESC when done")
    cv2.imshow("Select ROI - press ESC when done", img_display)
    cv2.setMouseCallback("Select ROI - press ESC when done", click_event)

    # 3. 等待用户点击两次，或按 ESC 退出
    while True:
        key = cv2.waitKey(1) & 0xFF
        # 当用户按下 ESC 键，退出循环
        if key == 27:
            break
        # 如果已经记录了两点，也可以直接退出
        if len(pts) == 2:
            break

    cv2.destroyAllWindows()

    # 4. 确保用户选了两点
    if len(pts) != 2:
        print("未选择完整的 ROI 点。")
        return

    # 5. 计算裁剪区域的左上角和右下角
    (x1, y1), (x2, y2) = pts
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    # 6. 裁剪并保存
    roi = img_original[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        print("裁剪区域为空，请检查所选坐标。")
        return

    out_path = 'watermark.jpg'
    cv2.imwrite(out_path, roi)
    print(f"水印已保存到 {out_path}")


def trim_video_opencv(input_path: str, output_path: str, duration: float = 10.0) -> None:
    """
    使用 OpenCV 截取视频前 duration 秒，并保存到 output_path。
    会重新编码，速度和效率不如 ffmpeg 直接 copy 快。
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 计算要写入的最大帧数
    max_frames = min(total_frames, int(fps * duration))

    # fourcc 编码器（这里用常见的 XVID，也可根据需求改成 'mp4v'、'H264' 等）
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"无法创建输出视频: {output_path}")

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break  # 提前读完了
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"已成功截取前 {duration} 秒（{frame_idx} 帧），输出到：{output_path}")

if __name__ == "__main__":
    try:
        trim_video_opencv("../inpainting/test.mp4", "output_10s.mp4", duration=2)
    except Exception as e:
        print("截取失败：", e)