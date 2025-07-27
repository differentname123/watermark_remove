import subprocess
import os
from PIL import Image
import math  # 需要导入 math 模块以使用 PI


# ==============================================================================
# 平滑缩放（已修正）
# ==============================================================================
def create_video_from_image_smooth(
        image_path: str,
        output_path: str,
        duration: int = 5,
        resolution: tuple = (1920, 1080),
        fps: int = 30,
        zoom_factor: float = 1.01,
        use_background_fill: bool = True
):
    if not os.path.exists(image_path):
        print(f"错误：找不到输入图片 '{image_path}'")
        return

    width, height = resolution
    final_filter_complex = ""

    if use_background_fill:
        # 方案A: 使用模糊背景填充
        filter_complex_base = (
            ### <<< 修正：为 split 滤镜明确指定输入流 [0:v]
            f"[0:v]split[bg][fg];"
            f"[bg]scale=w='if(gte(iw/ih,{width}/{height}),-1,{width})':h='if(gte(iw/ih,{width}/{height}),{height},-1)',"
            f"gblur=sigma=20,crop={width}:{height}[bg_pp];"
            f"[fg]scale=w='if(gte(iw/ih,{width}/{height}),{width},-1)':h='if(gte(iw/ih,{width}/{height}),-1,{height})'[fg_pp];"
            "[bg_pp][fg_pp]overlay=(W-w)/2:(H-h)/2[overlay_out];"
        )
    else:
        # 方案B: 使用黑边
        filter_complex_base = (
            ### <<< 优化：为 color 滤镜添加时长，使其与视频总长一致
            f"color=c=black:s={width}x{height}:d={duration}[black_bg];"
            f"[0:v]scale=w='if(gte(iw/ih,{width}/{height}),{width},-2)':h='if(gte(iw/ih,{width}/{height}),-2,{height})'[fg_scaled];"
            f"[black_bg][fg_scaled]overlay=(W-w)/2:(H-h)/2[overlay_out];"
        )

    # 动画滤镜部分作用于 [overlay_out]
    zoom_expr = f"1+({zoom_factor}-1)*t/{duration}"
    filter_complex_animation = (
        f"[overlay_out]scale=w='iw*({zoom_expr})':h='ih*({zoom_expr})':eval=frame,"
        f"crop=w={width}:h={height}:x='(iw-{width})/2':y='(ih-{height})/2',"
        "format=yuv420p"
    )
    final_filter_complex = filter_complex_base + filter_complex_animation

    command = [
        'ffmpeg', '-y',
        '-loglevel', 'error',
        '-loop', '1', '-i', image_path,
        '-filter_complex', final_filter_complex,
        '-c:v', 'libx264',
        '-preset', 'slow', '-crf', '18',
        '-t', str(duration), '-r', str(fps),
        output_path
    ]

    print("正在生成平滑动画视频，请稍候...")
    print(f"执行的 FFmpeg 命令: {' '.join(command)}")

    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"\n视频 '{output_path}' 生成成功！")
    except subprocess.CalledProcessError as e:
        print("\n视频生成失败！")
        print(f"FFmpeg 错误信息:\n{e.stderr}")



# ==============================================================================
# 水平滚动（已修正）
# ==============================================================================
def scroll_image_horizontally(
        image_path,
        output_path,
        scroll_speed=30,
        output_width=1920,
        output_height=1080,
        fps=30,
        target_duration=None,
        use_background_fill: bool = True
):
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
    except Exception as e:
        print(f"读取图片失败: {e}")
        return

    if img_height == 0: return
    scaled_width = img_width * (output_height / img_height)
    scroll_distance = max(0, scaled_width - output_width)

    # 决定最终视频时长
    if scroll_distance <= 0:
        # 如果图片不够宽，无法滚动，则生成一个静止视频
        final_duration = target_duration if target_duration is not None else 3
        scroll_distance = 0  # 确保滚动距离为0
    else:
        # 如果指定了时长，就用指定的；否则根据滚动速度计算
        calculated_duration = scroll_distance / scroll_speed
        final_duration = target_duration if target_duration is not None else calculated_duration

    speed_per_frame = scroll_speed / fps

    filter_complex = ""
    if use_background_fill:
        filter_complex = (
            f"[0:v]split[original][bg_src];"
            f"[bg_src]scale=-1:{output_height},boxblur=luma_radius=20:luma_power=1,crop={output_width}:{output_height}[bg];"
            f"[original]scale=-1:{output_height},format=rgba,setsar=1,"
            f"crop={output_width}:{output_height}:x='min({scroll_distance},max(0,n*{speed_per_frame}))':y=0[fg];"
            f"[bg][fg]overlay=0:0,format=yuv420p"
        )
    else:
        # 方案B: 黑色背景
        filter_complex = (
            ### <<< 修正：为 color 滤镜添加 d={final_duration} 参数
            f"color=c=black:s={output_width}x{output_height}:d={final_duration}[bg];"
            f"[0:v]scale=-1:{output_height},format=rgba,setsar=1,"
            f"crop={output_width}:{output_height}:x='min({scroll_distance},max(0,n*{speed_per_frame}))':y=0[fg];"
            f"[bg][fg]overlay=0:0,format=yuv420p"
        )

    ### <<< 优化：采用与垂直滚动相同的、更健壮的命令结构
    cmd = [
        "ffmpeg", "-y", '-loglevel', 'error',
        "-loop", "1", "-i", image_path,
        "-filter_complex", filter_complex,
        "-c:v", "libx264",
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        # 将 -t 作为输出选项放在最后，确保视频总长
        "-t", str(final_duration),
        output_path
    ]

    print("正在生成水平滚动视频...\n", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"\n视频成功保存到: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nFFmpeg 执行失败:\n{e.stderr}")


# ==============================================================================
# 垂直滚动（已优化）
# ==============================================================================
def scroll_image_vertically(
        image_path,
        output_path,
        scroll_speed=30,
        output_width=1920,
        output_height=1080,
        fps=30,
        target_duration=None,
        use_background_fill: bool = True
):
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"读取图片失败: {e}")
        return

    if img_width == 0: return
    scaled_height = img_height * (output_width / img_width)
    scroll_distance = max(0, scaled_height - output_height)

    if scroll_distance <= 0:
        final_duration = target_duration if target_duration is not None else 3
        scroll_distance = 0
    else:
        calculated_duration = scroll_distance / scroll_speed
        final_duration = target_duration if target_duration is not None else calculated_duration

    speed_per_frame = scroll_speed / fps

    filter_complex = ""
    if use_background_fill:
        filter_complex = (
            f"[0:v]split[original][bg_src];"
            f"[bg_src]scale={output_width}:-1,boxblur=luma_radius=20:luma_power=1,crop={output_width}:{output_height}[bg];"
            f"[original]scale={output_width}:-1,format=rgba,setsar=1,"
            f"crop={output_width}:{output_height}:x=0:y='min({scroll_distance},max(0,n*{speed_per_frame}))'[fg];"
            f"[bg][fg]overlay=0:0,format=yuv420p"
        )
    else:
        filter_complex = (
            # 你的代码中这里已经正确添加了 d={final_duration}，这里保持
            f"color=c=black:s={output_width}x{output_height}:d={final_duration}[bg];"
            f"[0:v]scale={output_width}:-1,format=rgba,setsar=1,"
            f"crop={output_width}:{output_height}:x=0:y='min({scroll_distance},max(0,n*{speed_per_frame}))'[fg];"
            f"[bg][fg]overlay=0:0,format=yuv420p"
        )

    # ### <<< 优化：清理了你代码中被注释掉的旧命令，只保留最终的、最正确的版本
    cmd = [
        "ffmpeg", "-y", '-loglevel', 'error',
        "-loop", "1", "-i", image_path,
        "-filter_complex", filter_complex,
        "-c:v", "libx264",
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        "-t", str(final_duration),
        output_path
    ]

    print("正在生成垂直滚动视频...\n", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"\n视频成功保存到: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nFFmpeg 执行失败:\n{e.stderr}")


# ==============================================================================
# 自动选择函数（已优化）
# ==============================================================================
def create_video_from_image_auto_select(
        image_path: str,
        output_path: str,
        duration: int = 5,
        resolution: tuple = (1920, 1080),
        fps: int = 30,
        zoom_factor: float = 1.0,
        scroll_speed: int = 30,
        use_background_fill: bool = True
):
    if not os.path.exists(image_path):
        print(f"错误：找不到输入图片 '{image_path}'")
        return

    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"错误: 无法读取图片 '{image_path}'。 错误信息: {e}")
        return

    output_width, output_height = resolution

    # 决策逻辑保持不变，但调用时简化参数传递
    if img_height > 3 * img_width:
        print(f"检测到高图 -> 【垂直滚动】")
        ### <<< 优化：不再需要在此处计算 final_duration，交由子函数处理
        scroll_image_vertically(
            image_path=image_path, output_path=output_path,
            scroll_speed=scroll_speed, output_width=output_width,
            output_height=output_height, fps=fps, target_duration=duration,  # 直接传递 duration
            use_background_fill=use_background_fill
        )
    elif img_width > 3 * img_height:
        print(f"检测到宽图 -> 【水平滚动】")
        scroll_image_horizontally(
            image_path=image_path, output_path=output_path,
            scroll_speed=scroll_speed, output_width=output_width,
            output_height=output_height, fps=fps, target_duration=duration,  # 直接传递 duration
            use_background_fill=use_background_fill
        )
    else:
        print(f"检测到常规图 -> 【平滑缩放】")
        create_video_from_image_smooth(
            image_path=image_path, output_path=output_path,
            duration=duration, resolution=resolution, fps=fps,
            zoom_factor=zoom_factor, use_background_fill=use_background_fill
        )


# ... (示例使用部分保持不变) ...
if __name__ == '__main__':
    # 假设你有一张非16:9的图片，比如一张竖屏图 test_portrait.jpg
    # 你可以自己创建或下载一张，例如 1080x1920 尺寸
    test_portrait_image = 'test4.jpg'


    if os.path.exists(test_portrait_image):
        print("\n--- 1. 测试常规缩放 (使用模糊背景填充) ---")
        create_video_from_image_auto_select(
            image_path=test_portrait_image,
            output_path='video_smooth_with_fill.mp4',
            use_background_fill=True  # 明确指定使用填充
        )

        print("\n--- 2. 测试常规缩放 (使用黑边) ---")
        create_video_from_image_auto_select(
            image_path=test_portrait_image,
            output_path='video_smooth_with_black_bars.mp4',
            use_background_fill=False  # 禁用填充
        )

    # # 假设你有一张很高的长图 test_tall.jpg
    # test_tall_image = 'test6.jpg'
    # if os.path.exists(test_tall_image):
    #     print("\n--- 3. 测试自动选择 (高图，使用黑边) ---")
    #     create_video_from_image_auto_select(
    #         image_path=test_tall_image,
    #         output_path='video_auto_tall_with_black_bars.mp4',
    #         duration=10,
    #         use_background_fill=False  # 测试在自动模式下禁用填充
    #     )