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
        scroll_speed=30,  # 每秒滚动多少像素（决定视觉速度）
        output_width=1920,
        output_height=1080,
        fps=30,
        target_duration=None  # 新增参数：期望的输出视频时长（秒）
):
    """
    将一张高图从上到下滚动生成视频。

    Args:
        image_path (str): 输入图片路径。
        output_path (str): 输出视频路径。
        scroll_speed (int): 滚动的视觉速度，单位：像素/秒。
        output_width (int): 输出视频的宽度。
        output_height (int): 输出视频的高度。
        fps (int): 输出视频的帧率。
        target_duration (float, optional): 期望的视频总时长（秒）。
            - 如果为 None (默认): 视频时长会根据图片高度和滚动速度自动计算。
            - 如果设置了值:
                - 若自动计算的时长 > target_duration, 视频会被截断到 target_duration。
                - 若自动计算的时长 < target_duration, 视频滚动结束后会保持最后一帧直到 target_duration。
    """
    try:
        # 1. 获取图片高度
        img = Image.open(image_path)
        img_width, img_height = img.size
    except FileNotFoundError:
        print(f"错误: 图片文件未找到于 {image_path}")
        return
    except Exception as e:
        print(f"错误: 读取图片时发生意外 {e}")
        return

    # 2. 计算滚动的核心参数
    # 图像可以滚动的总像素距离
    scroll_distance = max(0, img_height - output_height)

    # 根据固定的视觉速度，计算每帧应该滚动的像素值
    # 这个值决定了滚动的快慢，与最终视频时长无关
    speed_per_frame = scroll_speed / fps

    # 3. 决定最终的视频时长
    if target_duration is not None:
        # 如果用户指定了时长，就使用该时长
        final_duration = float(target_duration)
    else:
        # 如果用户未指定时长，则按老方法计算
        # 计算完整滚完一遍需要的时间
        if scroll_distance > 0:
            # 完整滚动时长 = 总距离 / 速度
            final_duration = scroll_distance / scroll_speed
        else:
            # 如果图片本身不够高，无法滚动，则默认给一个3秒的静止时长
            final_duration = 3

    # 4. FFmpeg 命令构造
    # 关键点:
    # - '-t' 参数使用我们最终确定的 `final_duration`。
    # - crop 滤镜中的 'y' 表达式 `min({scroll_distance}, n*{speed_per_frame})` 保持不变。
    #   - 当滚动到达底部 (n*speed_per_frame > scroll_distance) 时，`min`函数会确保y值停在 `scroll_distance`，从而实现“保持最后一帧”的效果。
    #   - FFmpeg 会在 `final_duration` 秒后自动停止生成，从而实现“截断”或“延长”的效果。
    cmd = [
        "ffmpeg",
        "-y",  # 覆盖已存在的文件
        "-loop", "1",  # 无限循环输入图片
        "-t", str(final_duration),  # 【修改点】使用最终计算出的时长
        "-i", image_path,
        "-filter_complex",
        # 背景层：缩放、模糊、裁剪成最终尺寸
        f"[0:v]scale={output_width}:-1,boxblur=luma_radius=20:luma_power=1,crop={output_width}:{output_height}[bg];"
        # 前景层：缩放、并根据帧数(n)和每帧速度来动态裁剪y轴位置
        f"[0:v]scale={output_width}:-1,format=rgba,setsar=1,"
        f"crop={output_width}:{output_height}:x=0:y='min({scroll_distance},max(0,n*{speed_per_frame}))'[fg];"
        # 将前景叠加到背景上
        f"[bg][fg]overlay=0:0,format=yuv420p",
        "-c:v", "libx264",
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        output_path
    ]

    print("Running FFmpeg with command:\n", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"\n视频成功保存到: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nFFmpeg 执行失败: {e}")
    except FileNotFoundError:
        print("\n错误: 未找到 ffmpeg 命令。请确保 FFmpeg 已安装并配置在系统路径中。")


# --- 如何使用 ---
if __name__ == '__main__':
    test_image_path = 'test5.jpg'
    # --- 场景 1: 不设置 target_duration, 自动计算时长 (约78.4秒) ---
    print("\n--- 场景 1: 自动计算时长 ---")
    scroll_image_vertically(
        image_path=test_image_path,
        output_path="output_auto_duration.mp4",
        target_duration=None
    )

    # --- 场景 2: 设置 target_duration < 自动时长，视频被截断 ---
    print("\n--- 场景 2: 截断为10秒 ---")
    scroll_image_vertically(
        image_path=test_image_path,
        output_path="output_truncated_10s.mp4",
        target_duration=2  # 视频总长只有10秒
    )

    # --- 场景 3: 设置 target_duration > 自动时长，保持最后一帧 ---
    print("\n--- 场景 3: 延长到90秒（滚动结束后保持最后一帧） ---")
    scroll_image_vertically(
        image_path=test_image_path,
        output_path="output_extended_90s.mp4",
        target_duration=10 # 视频总长90秒
    )