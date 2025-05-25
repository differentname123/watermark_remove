import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from simple_lama_inpainting import SimpleLama  # 确保这个包已安装，如果有GPU可以修改为 SimpleLama(device=torch.device("cuda"))

def inpating_video(video_path, box_list, output_video_path=None, batch_size=10, max_frames=100):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件:", video_path)
        return
    if output_video_path is None:
        output_video_path = video_path.replace(".mp4", "_inpainted.mp4")
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频属性: 宽={width}, 高={height}, fps={fps}, 总帧数={total_frames}")

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 初始化 Lama inpainting 模块
    lama = SimpleLama()  # 如有GPU支持，可改为 SimpleLama(device=torch.device("cuda"))

    images_batch = []
    masks_batch = []
    frame_count = 0

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count > max_frames:
            print(f"达到最大帧数{max_frames}限制，停止处理")
            break

        # 将帧从BGR转换为RGB，并转换为 PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        images_batch.append(pil_frame)

        # 创建与帧同尺寸的纯黑 mask (灰度图)，然后依据 box_list 将区域涂白
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for box in box_list:
            # box 格式：[x_min, y_min, x_max, y_max]，归一化坐标(0~1)
            x_min = int(box[0] * width)
            y_min = int(box[1] * height)
            x_max = int(box[2] * width)
            y_max = int(box[3] * height)
            draw.rectangle([x_min, y_min, x_max, y_max], fill=255)
        masks_batch.append(mask)

        frame_count += 1

        # 达到一个批次的帧数时进行处理
        if len(images_batch) == batch_size:
            result_images = lama.inpaint_batch(images_batch, masks_batch)
            for res_img in result_images:
                # 将 PIL 图像转换为 OpenCV BGR 格式并写入视频
                res_img_cv = cv2.cvtColor(np.array(res_img), cv2.COLOR_RGB2BGR)
                out.write(res_img_cv)
            print(f"已处理 {frame_count} 帧")
            images_batch.clear()
            masks_batch.clear()

    # 处理剩余不足一批的帧
    if images_batch:
        result_images = lama.inpaint_batch(images_batch, masks_batch)
        for res_img in result_images:
            res_img_cv = cv2.cvtColor(np.array(res_img), cv2.COLOR_RGB2BGR)
            out.write(res_img_cv)
        print(f"已处理 {frame_count} 帧 (最后一批)")

    cap.release()
    out.release()
    print("总耗时: {:.2f} 秒".format(time.time() - start_time))
    print("处理后的视频已保存为:", output_video_path)

if __name__ == "__main__":
    # 输入视频文件路径
    video_path = "input_video.mp4"
    # 输出视频文件路径
    output_video_path = "output_video.mp4"
    # 示例 box_list：每个 box 坐标均为归一化值 [x_min, y_min, x_max, y_max]
    box_list = [
        [0.1, 0.1, 0.4, 0.4],  # 框1
        [0.6, 0.5, 0.9, 0.8]   # 框2
    ]
    inpating_video(video_path, box_list, output_video_path)