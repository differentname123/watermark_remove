import subprocess
import os
from PIL import Image


def create_video_from_image_smooth(
        image_path: str,
        output_path: str,
        duration: int = 5,
        resolution: tuple = (1920, 1080),
        fps: int = 25,
        zoom_factor: float = 1.1
):
    """
    【优化版】使用 FFmpeg 将单张图片转换为带有平滑动态效果的视频。
    此版本使用 scale+crop 替代 zoompan 以解决抖动问题。

    参数:
    - image_path (str): 输入图片的路径。
    - output_path (str): 输出视频的保存路径。
    - duration (int): 视频时长（秒）。
    - resolution (tuple): 视频分辨率 (宽, 高)，默认为 1920x1080。
    - fps (int): 视频帧率。
    - zoom_factor (float): 最终缩放倍数，1.1 表示放大10%。
    """
    if not os.path.exists(image_path):
        print(f"错误：找不到输入图片 '{image_path}'")
        return

    width, height = resolution

    # --- 构建 FFmpeg 滤镜链 (filter_complex) ---
    # [前半部分与之前相同：创建模糊背景并叠加前景]
    filter_complex_base = (
        "[0:v]split=2[bg][fg];"
        f"[bg]scale={width}:-1,gblur=sigma=20,crop={width}:{height}[bg_pp];"
        f"[fg]scale=w='if(gte(iw/ih,{width}/{height}),{width},-1)':h='if(gte(iw/ih,{width}/{height}),-1,{height})'[fg_pp];"
        "[bg_pp][fg_pp]overlay=(W-w)/2:(H-h)/2[overlay_out];"
    )

    # --- 【核心优化】使用 scale 和 crop 实现平滑缩放 ---
    # 定义一个基于时间't'的缩放表达式，实现线性放大
    # 从 1 倍 (t=0) 线性增长到 zoom_factor 倍 (t=duration)
    zoom_expr = f"1+({zoom_factor}-1)*t/{duration}"

    # 构建动画滤镜部分
    filter_complex_animation = (
        # 1. scale: 基于上面的表达式，动态放大整个画布
        #    w='iw*({zoom_expr})':h='ih*({zoom_expr})' -> 将输入宽度(iw)和高度(ih)乘以当前的缩放系数
        #    eval=frame -> 确保每一帧都重新计算表达式
        f"[overlay_out]scale=w='iw*({zoom_expr})':h='ih*({zoom_expr})':eval=frame,"

        # 2. crop: 将放大的画布从中心裁切回目标分辨率
        #    w={width}:h={height} -> 裁切后的尺寸是我们的目标视频尺寸
        #    x='(iw-{width})/2':y='(ih-{height})/2' -> 裁切的起始点，确保中心对齐
        f"crop=w={width}:h={height}:x='(iw-{width})/2':y='(ih-{height})/2',"

        # 3. format: 设置像素格式，确保最佳兼容性
        "format=yuv420p"
    )

    # 组合完整的滤镜链
    final_filter_complex = filter_complex_base + filter_complex_animation

    # --- 构建完整的 FFmpeg 命令 ---
    command = [
        'ffmpeg',
        '-y',
        '-loop', '1',  # 让图片作为无限循环的输入流
        '-i', image_path,
        '-filter_complex', final_filter_complex,
        '-c:v', 'libx264',
        '-preset', 'slow',  # 使用稍慢的预设可以获得更好的压缩质量
        '-crf', '18',  # 恒定质量因子，数值越低质量越好，18是很好的平衡点
        '-t', str(duration),
        '-r', str(fps),  # 明确指定输出帧率
        output_path
    ]

    # --- 执行命令 ---
    print("正在生成平滑动画视频，请稍候...")
    print(f"执行的 FFmpeg 命令: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"\n视频 '{output_path}' 生成成功！动画效果应该非常平滑。")
    except subprocess.CalledProcessError as e:
        print("\n视频生成失败！")
        print("FFmpeg 错误信息:")
        print(e.stderr)

def scroll_image_vertically(
    image_path,
    output_path,
    scroll_speed=50,  # 每秒滚动多少像素
    output_width=1920,
    output_height=1080,
    fps=30
):
    # 1. 获取图片高度
    img = Image.open(image_path)
    img_width, img_height = img.size
    scroll_distance = max(0, img_height - output_height)  # 实际可滚动距离

    # 2. 计算时长（以滚完整张图为准）
    duration = scroll_distance / scroll_speed if scroll_distance > 0 else 3

    # 3. 每帧滚动速度（像素/帧）
    speed_per_frame = scroll_speed / fps

    # 4. FFmpeg 命令构造
    cmd = [
        "ffmpeg",
        "-y",
        "-loop", "1",
        "-t", str(duration),
        "-i", image_path,
        "-filter_complex",
        f"[0:v]scale={output_width}:-1,boxblur=luma_radius=20:luma_power=1,crop={output_width}:{output_height}[bg];"
        f"[0:v]scale={output_width}:-1,format=rgba,setsar=1,"
        f"crop={output_width}:{output_height}:x=0:y='min({scroll_distance},max(0,n*{speed_per_frame}))'[fg];"
        f"[bg][fg]overlay=0:0,format=yuv420p",
        "-c:v", "libx264",
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        output_path
    ]

    print("Running FFmpeg:\n", " ".join(cmd))
    subprocess.run(cmd, check=True)


# --- 如何使用 ---
if __name__ == "__main__":
    # 1. 设置你的图片文件名
    input_image = "test5.jpg"  # <--- 修改这里为你的图片文件名

    # 2. 设置输出视频的文件名
    output_video = "output_video_smooth.mp4"

    # 3. 执行新的转换函数
    create_video_from_image_smooth(input_image, output_video, duration=5, fps=30)

    # 为了获得最流畅的效果，可以尝试将fps提高到30或60
    # create_video_from_image_smooth(input_image, "output_60fps.mp4", duration=5, fps=60)

    scroll_image_vertically("test5.jpg", "scroll_output.mp4", scroll_speed=20)
