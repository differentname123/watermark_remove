import pathlib
import random
import shutil
import subprocess
import os
from PIL import Image
import math  # 需要导入 math 模块以使用 PI

from common_utils.common_utils import ms_to_time
from common_utils.tts.edge_tts_utils import generate_audio_and_get_duration_sync
from common_utils.video_utils import add_subtitles_to_video, probe_duration
from common_utils.video_utils1 import redub_video_with_ffmpeg
from common_utils.video_utils2 import add_bgm_to_video


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
    # 随机设置use_background_fill
    use_background_fill = random.choice([True, False])  # 随机选择是否使用背景填充
    zoom_factor = random.choice([1.1, 1.1, 1.1])
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


def text_image_to_video_with_subtitles(
    text: str,
    image_path: str,
    output_path: str,
    short_text: str = "",
    voice_name: str = "",
    bgm_path: str = "",
    cleanup: bool = True,
    resolution: tuple = (1920, 1080)
) -> str:
    """
    根据文本和图片生成带字幕的视频，并可选添加背景音乐（bgm），并自动清理中间视频文件。

    参数:
        text: 完整文案
        image_path: 图片路径
        output_path: 输出视频路径
        short_text: 简略文案（可选）
        voice_name: 语音合成声音
        bgm_path: 背景音乐文件路径（可选，若存在则在生成最终视频后添加）
        cleanup: 是否在生成最终视频后清理中间视频文件

    返回:
        最终视频路径（若提供了 bgm_path，返回带 bgm 的视频路径；否则返回无 bgm 的视频路径）
    """
    if not voice_name:
        voice_name = random.choice([
            "zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural","zh-CN-YunjianNeural","zh-CN-YunxiNeural",
            "zh-CN-YunxiaNeural", "zh-CN-YunyangNeural"
        ])
    if not bgm_path:
        bgm_id = random.choice(['1212a7cf29e09ef63e689cb23b1b6fed.wav', '0671d099e221faf1b77922fa08ade356.wav', '428eaba81088bd92cbc5a6a273dbf873.wav', '8dfb680196265fcafe4cc19ce6e75ffe.wav', '9d34a87ec50e5bf577f1405f1475ec7f.wav'])
        bgm_path = r"W:\project\python_project\watermark_remove\content_community\app\bgm_audio" + os.sep + bgm_id

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 文本转语音
    audio_path = output_path.with_suffix(".mp3")
    duration = generate_audio_and_get_duration_sync(
        text=text,
        output_filename=str(audio_path),
        voice_name=voice_name,
        trim_silence=False,
        # rate="+15%",
        # pitch='+10Hz',
    )

    # 2. 图片转视频
    image_video_path = output_path.with_name(output_path.stem + "_img.mp4")
    create_video_from_image_auto_select(
        image_path=image_path,
        output_path=str(image_video_path),
        duration=duration,
        resolution=resolution
    )

    # 3. 合成语音
    audio_video_path = output_path.with_name(output_path.stem + "_audio.mp4")
    segments_info = [{
        'startTime': "00:00:00.000",
        'endTime': ms_to_time(duration * 1000),
        'outputPath': str(audio_path),
        'trimmedDuration': duration,
    }]
    redub_video_with_ffmpeg(str(image_video_path), segments_info, output_path=str(audio_video_path))

    # 4. 添加字幕
    subtitle_data = [{
        'startTime': "00:00:00.000",
        'endTime': ms_to_time(duration * 1000),
        'optimizedText': text
    }]
    subtitle_video_path = output_path.with_name(output_path.stem + "_sub.mp4")
    add_subtitles_to_video(
        video_path=str(audio_video_path),
        subtitles_info=subtitle_data,
        output_path=str(subtitle_video_path),
        font_size=70,
        bottom_margin=30
    )

    # 5. 如果有简略文案，加第二层字幕
    if short_text and len(text) > 30:
        subtitle_data = [{
            'startTime': ms_to_time(duration * 500),
            'endTime': ms_to_time(duration * 1000),
            'optimizedText': short_text
        }]
        add_subtitles_to_video(
            video_path=str(subtitle_video_path),
            subtitles_info=subtitle_data,
            output_path=str(output_path),
            font_color='#FFD700',
            font_size=80,
            bottom_margin=1000
        )
    else:
        shutil.copy(str(subtitle_video_path), str(output_path))

    final_video_path = str(output_path.resolve())

    # 6. 可选：为最终视频添加背景音乐
    final_with_bgm_path = None
    if bgm_path and os.path.exists(bgm_path):
        # print(f"正在为视频添加背景音乐: {bgm_path}")
        final_with_bgm_path = output_path.parent / f"{output_path.stem}_bgm.mp4"
        add_bgm_to_video(final_video_path, bgm_path, str(final_with_bgm_path))
        # print(f"背景音乐已添加，输出视频: {final_with_bgm_path.resolve()}")

    # 7. 清理中间视频文件
    if cleanup:
        # 确定需要保留哪一个最终文件
        kept_final_paths = set()
        if final_with_bgm_path:
            kept_final_paths.add(str(final_with_bgm_path.resolve()))
            # 如果存在无 bgm 的最终视频，也可以选择删掉
        else:
            kept_final_paths.add(final_video_path)

        # 需要清理的中间视频路径
        intermediates = [
            str(audio_path),
            str(image_video_path),
            str(audio_video_path),
            str(subtitle_video_path),
        ]

        # 如果存在无 bgm 的最终视频且仍然存在，且不是要保留的最终视频，则删除它
        if final_with_bgm_path:
            # 删除无 bgm 的最终视频（因为已经有带 bgm 的最终版本）
            if os.path.exists(final_video_path) and final_video_path not in kept_final_paths:
                intermediates.append(final_video_path)

        # 删除中间视频文件
        for p in intermediates:
            if p and os.path.exists(p) and p not in kept_final_paths:
                try:
                    os.remove(p)
                    # print(f"已清理中间视频：{p}")
                except Exception as e:
                    print(f"警告：无法清理中间视频 {p}，原因: {e}")

        # 如果最终带 bgm，则删除未保留的最终无 bgm 视频
        if final_with_bgm_path:
            if os.path.exists(final_video_path) and final_video_path not in kept_final_paths:
                try:
                    os.remove(final_video_path)
                    # print(f"已清理无 BGm 的最终视频：{final_video_path}")
                except Exception as e:
                    print(f"警告：无法清理无 BGm 的最终视频 {final_video_path}，原因: {e}")

        # 如果希望在清理后仅保留最终版本，可以确保最终版本路径被返回
        if final_with_bgm_path:
            return str(final_with_bgm_path.resolve())
        else:
            return final_video_path

    # 未开启清理，返回最终视频路径
    return final_with_bgm_path.resolve() if final_with_bgm_path else final_video_path


def gen_ending_video(text, output_path, origin_ending_video_path):
    """
    生成结尾视频（测试用），结尾语为txt
    """
    voice_name = random.choice([
        "zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural", "zh-CN-YunjianNeural", "zh-CN-YunxiNeural",
        "zh-CN-YunxiaNeural", "zh-CN-YunyangNeural"
    ])
    output_path = pathlib.Path(output_path)
    audio_path = output_path.with_suffix(".mp3")
    duration = generate_audio_and_get_duration_sync(
        text=text,
        output_filename=str(audio_path),
        voice_name=voice_name,
        trim_silence=False,
        # rate="+15%",
        # pitch='+10Hz',
    )
    video_duration = probe_duration(origin_ending_video_path)
    segments_info = [{
        'startTime': "00:00:00.000",
        'endTime': ms_to_time(video_duration * 1000),
        'outputPath': str(audio_path),
        'trimmedDuration': duration,
    }]
    with_audio_path = output_path.with_name(output_path.stem + "_with_audio.mp4")
    redub_video_with_ffmpeg(video_path=origin_ending_video_path, segments_info=segments_info, output_path=str(with_audio_path), keep_original_audio=True)

    # 4. 添加字幕
    subtitle_data = [{
        'startTime': "00:00:00.000",
        'endTime': ms_to_time(duration * 1000),
        'optimizedText': text
    }]
    add_subtitles_to_video(
        video_path=str(with_audio_path),
        subtitles_info=subtitle_data,
        output_path=str(output_path),
        font_size=70,
        bottom_margin=30
    )


    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists(with_audio_path):
        os.remove(with_audio_path)
    return str(output_path.resolve())

def gen_video(text, output_path, origin_video_path, voice_name="zh-CN-YunjianNeural",keep_original_audio=False):
    """
    生成结尾视频（测试用），结尾语为txt
    """
    if voice_name is None:
        voice_name = random.choice([
            "zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural", "zh-CN-YunjianNeural", "zh-CN-YunxiNeural",
            "zh-CN-YunxiaNeural", "zh-CN-YunyangNeural"
        ])
    output_path = pathlib.Path(output_path)
    audio_path = output_path.with_suffix(".mp3")
    duration = generate_audio_and_get_duration_sync(
        text=text,
        output_filename=str(audio_path),
        voice_name=voice_name,
        trim_silence=False,
        rate="+20%",
        # pitch='+10Hz',
    )
    video_duration = probe_duration(origin_video_path)
    segments_info = [{
        'startTime': "00:00:00.000",
        'endTime': ms_to_time(video_duration * 1000),
        'outputPath': str(audio_path),
        'trimmedDuration': duration,
    }]
    with_audio_path = output_path.with_name(output_path.stem + "_with_audio.mp4")
    redub_video_with_ffmpeg(video_path=origin_video_path, segments_info=segments_info, output_path=str(with_audio_path),keep_original_audio=keep_original_audio)

    # 4. 添加字幕
    subtitle_data = [{
        'startTime': "00:00:00.000",
        'endTime': ms_to_time(duration * 1000),
        'optimizedText': text
    }]
    add_subtitles_to_video(
        video_path=str(with_audio_path),
        subtitles_info=subtitle_data,
        output_path=str(output_path),
        font_size=70,
        bottom_margin=30
    )


    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists(with_audio_path):
        os.remove(with_audio_path)
    return str(output_path.resolve())

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