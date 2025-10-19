import pathlib
import platform
import random
import shlex
import shutil
import subprocess
import os
import sys
import os
import random
import shlex
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import numpy as np
from PIL import Image
import math  # 需要导入 math 模块以使用 PI

from common_utils.common_utils import ms_to_time, is_valid_target_file_simple
from common_utils.tts.edge_tts_utils import generate_audio_and_get_duration_sync
from common_utils.video_utils import add_subtitles_to_video, probe_duration, probe_video_new
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
        font_size=60,
        bottom_margin=30
    )


    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists(with_audio_path):
        os.remove(with_audio_path)
    return str(output_path.resolve())

def gen_video(text, output_path, origin_video_path, voice_name="zh-CN-XiaoxiaoNeural",keep_original_audio=False, fixed_rect=None):
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
        rate="+30%",
        pitch='+30Hz',
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
        bottom_margin=30,
        fixed_rect=fixed_rect
    )


    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists(with_audio_path):
        os.remove(with_audio_path)
    return str(output_path.resolve())


def find_motion_bbox(video_path, start_frame=60, end_frame_offset=60, num_samples=20, motion_threshold=30, padding=10):
    """
    分析视频指定片段，通过均匀采样固定数量的帧来找到运动区域的边界框。

    :param video_path: 视频文件路径
    :param start_frame: 开始分析的绝对帧号 (默认为0)
    :param end_frame_offset: 从视频末尾向前偏移的帧数。0表示分析到最后一帧。(默认为0)
    :param num_samples: 在指定范围内均匀采样的帧数 (默认为20)
    :param motion_threshold: 像素差异多大时算作运动
    :param padding: 在计算出的边界框外围增加的像素边距
    :return: (x, y, w, h) 的边界框元组，如果失败则返回 None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}", file=sys.stderr)
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"视频信息: {frame_width}x{frame_height}, {total_frames} 帧, {fps:.2f} FPS")
    if total_frames < 2:
        print("错误: 视频文件帧数不足，无法进行分析。", file=sys.stderr)
        cap.release()
        return None

    # --- 新逻辑: 计算实际的分析范围 ---
    actual_end_frame = total_frames - end_frame_offset

    # --- 参数有效性检查 ---
    if start_frame < 0 or start_frame >= total_frames:
        print(f"错误: 起始帧 {start_frame} 超出范围 (0-{total_frames - 1})。", file=sys.stderr)
        return None
    if actual_end_frame <= start_frame:
        print(f"错误: 计算出的结束帧({actual_end_frame})必须大于起始帧({start_frame})。", file=sys.stderr)
        return None
    if num_samples < 2:
        print(f"错误: 采样帧数 {num_samples} 必须至少为2。", file=sys.stderr)
        return None

    # --- 新逻辑: 使用 linspace 生成均匀分布的采样帧索引 ---
    # np.linspace 包含端点，所以我们从 start_frame 到 actual_end_frame - 1
    sample_indices = np.linspace(start_frame, actual_end_frame - 1, num=num_samples, dtype=int)
    print(f"将在第 {start_frame} 帧到第 {actual_end_frame} 帧之间，均匀采样 {len(sample_indices)} 帧进行分析。")

    motion_accumulator = np.zeros((frame_height, frame_width), dtype=np.uint8)

    # --- 新逻辑: 处理第一个采样帧来初始化 prev_gray ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, sample_indices[0])
    ret, prev_frame = cap.read()
    if not ret:
        print(f"错误: 无法读取帧 {sample_indices[0]}。", file=sys.stderr)
        cap.release()
        return None
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    # --- 新逻辑: 循环遍历剩余的采样帧 ---
    for i, frame_index in enumerate(sample_indices[1:]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"\n警告: 无法读取帧 {frame_index}，跳过此帧。")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, motion_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        motion_accumulator = cv2.bitwise_or(motion_accumulator, thresh)
        prev_gray = gray  # 更新 prev_gray 以便下次比较

        # 进度条基于已处理的采样帧数
        progress = ((i + 2) / len(sample_indices)) * 100
        sys.stdout.write(f"\r正在分析... {progress:.2f}% (已处理 {i + 2}/{len(sample_indices)} 帧)")
        sys.stdout.flush()

    print("\n分析完成！")
    cap.release()

    points = cv2.findNonZero(motion_accumulator)
    if points is None:
        print("警告：在指定片段内未检测到任何运动。将返回整个视频区域。")
        return 0, 0, frame_width, frame_height

    # 后续处理与之前相同
    x, y, w, h = cv2.boundingRect(points)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame_width - x, w + 2 * padding)
    h = min(frame_height - y, h + 2 * padding)
    w = w + (w % 2)
    h = h + (h % 2)
    if x + w > frame_width: w = frame_width - x
    if y + h > frame_height: h = frame_height - y

    return ((x, y, w, h), frame_width, frame_height)


def crop_video(input_path, output_path, bbox, crf=23):
    """
    使用 FFmpeg 调用来裁剪视频，并使用 CRF 控制输出质量和文件大小。

    :param input_path: 输入视频路径
    :param output_path: 输出视频路径
    :param bbox: (x, y, w, h) 的边界框
    :param crf: Constant Rate Factor (CRF)。范围 0-51，默认 23。
                数值越小，质量越高，文件越大。
    """
    x, y, w, h = bbox
    print(f"\n检测到的活动区域 (x, y, w, h): ({x}, {y}, {w}, {h})")

    # 构建 FFmpeg 命令列表
    command = [
        'ffmpeg',
        '-y',  # 自动覆盖输出文件
        '-i', input_path,
        '-vf', f'crop={w}:{h}:{x}:{y}',
        '-c:v', 'libx264',  # 指定视频编码器为 H.264
        '-crf', str(crf),   # 指定质量因子，23 是一个很好的平衡值
        '-preset', 'ultrafast',# 预设，影响编码速度和压缩率的平衡。'medium' 是默认值，通常无需更改。
        '-c:a', 'copy',     # 直接复制音频流，不做重新编码
        output_path
    ]

    print("\n将要执行的 FFmpeg 命令:")
    # 为了清晰地打印命令
    command_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in command)
    print(command_str)
    print("-" * 50)

    try:
        # 执行命令
        print("正在执行裁剪，请稍候...")
        # 使用 PIPE 捕获输出，可以在出错时提供更详细的信息
        process = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True # Python 3.7+
        )
        print(f"\n裁剪成功！输出文件已保存至: {output_path}")

    except FileNotFoundError:
        print("错误: 'ffmpeg' 命令未找到。", file=sys.stderr)
        print("请确保 FFmpeg 已安装并配置在系统的 PATH 环境变量中。", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"错误: FFmpeg 执行失败，返回码 {e.returncode}", file=sys.stderr)
        print("\n--- FFmpeg 标准输出 ---", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("\n--- FFmpeg 错误输出 ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)

def process_and_crop_video(video_path, area_threshold_ratio=0.9, **kwargs):
    """
    分析视频中的运动区域，如果运动区域显著小于整个画面，则进行裁剪。

    :param video_path: 待处理的视频文件路径
    :param area_threshold_ratio: 面积阈值比例。当运动区域面积小于原面积的该比例时，触发裁剪。
                                 例如, 0.8 表示小于80%。
    :param kwargs: 传递给 find_motion_bbox 的其他参数，如 start_frame, num_samples 等。
    :return: 元组 (was_cropped, final_path)。
             was_cropped: 布尔值，True表示已裁剪，False表示未裁剪。
             final_path: 最终视频文件的路径（可能是裁剪后的新路径，也可能是原始路径）。
    """
    print(f"--- 开始处理视频: {video_path} ---")

    # 1. 查找运动边界框
    analysis_result = find_motion_bbox(video_path, **kwargs)

    if analysis_result is None:
        print("分析失败，无法获取边界框。")
        return (False, video_path)

    bbox, original_w, original_h = analysis_result
    x, y, w, h = bbox

    # 2. 判断是否需要裁剪
    original_area = original_w * original_h
    crop_area = w * h

    # 避免除以零的错误
    if original_area == 0:
        print("视频原始面积为0，无法计算比例。")
        return (False, video_path)

    current_ratio = crop_area / original_area
    print(f"运动区域面积占总面积的 {current_ratio:.2%}")

    # 条件：当前比例小于阈值，并且裁剪区域不等于整个视频（这是 find_motion_bbox 的回退情况）
    if current_ratio < area_threshold_ratio and (w, h) != (original_w, original_h):
        print(f"面积比例 ({current_ratio:.2%}) 小于阈值 ({area_threshold_ratio:.2%})，将执行裁剪。")

        # 构造输出文件名，例如 a.mp4 -> a_crop.mp4
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_crop{ext}"

        # 3. 执行裁剪
        crop_video(video_path, output_path, bbox)

        return (True, output_path)
    else:
        print(f"面积比例不小于阈值或与原尺寸相同，无需裁剪。")
        return (False, video_path)


def _escape_ffmpeg_path(path):
    """为 FFmpeg 滤镜中的文件路径进行转义，特别处理 Windows 路径。"""
    if platform.system() == 'Windows':
        return path.replace('\\', '\\\\').replace(':', '\\:')
    return path


def _escape_ffmpeg_text(text):
    """为 FFmpeg 的 drawtext 滤镜中的文本内容进行转义。"""
    escaped_text = text.replace('\\', '\\\\')
    escaped_text = escaped_text.replace("'", "'\\\\\\''")
    escaped_text = escaped_text.replace('%', '\\%')
    escaped_text = escaped_text.replace(':', '\\:')
    return escaped_text


def get_coordinate_offset(original_w: int, original_h: int, padding_ratio: float = 0.1) -> tuple[int, int]:
    """
    根据视频的原始尺寸和边框比例，计算原始视频画面在新画布上的坐标偏移量。

    这个函数模拟了 `add_text_adaptive_padding` 函数中用于定位视频的逻辑。
    返回的偏移量可以直接用于旧坐标到新坐标的转换：
    new_x = old_x + x_offset
    new_y = old_y + y_offset

    Args:
        original_w (int): 视频的原始宽度。
        original_h (int): 视频的原始高度。
        padding_ratio (float, optional): 添加的边框高度占原始视频高度的比例。默认为 0.25。

    Returns:
        tuple[int, int]: 一个包含 (x_offset, y_offset) 的元组。
                         在这个逻辑中，x_offset 总是 0。
    """
    if original_h <= 0 or original_w <= 0:
        return 0, 0

    # 水平偏移量始终为 0
    x_offset = 0

    # --- 计算垂直偏移量 (video_y_start) ---
    top_padding = int(original_h * padding_ratio)
    video_y_start = top_padding  # 默认的垂直偏移量

    # 针对宽屏视频的特殊处理逻辑
    if original_w / original_h > 1.5:
        bottom_padding = top_padding // 2
        # 重新计算顶部 padding，这会成为新的垂直偏移量
        new_top_padding = top_padding + bottom_padding
        video_y_start = new_top_padding

    y_offset = video_y_start

    return x_offset, y_offset

def add_text_adaptive_padding(input_video_path, output_video_path, text_events, font_path=None,
                                    padding_ratio=0.1):
    """
    自适应地为视频添加边框和文字，实现文字靠近视频上边界的“底部对-齐”效果。

    Args:
        input_video_path (str): 输入视频的文件路径。
        output_video_path (str): 输出视频的文件路径。
        text_events (list): 包含文字事件的列表。
        font_path (str, optional): 字体文件路径。
        padding_ratio (float, optional): 添加的边框高度占原始视频高度的比例。
    """
    # --- 1. 参数校验和准备 (不变) ---
    if not os.path.exists(input_video_path):
        print(f"错误：输入视频文件不存在 -> {input_video_path}")
        return
    if font_path is None:
        font_path = 'C:/Windows/Fonts/msyhbd.ttc'
        print(f"提示：未使用指定字体，将尝试使用默认字体 -> {font_path}")
    if not os.path.exists(font_path):
        print(f"错误：字体文件不存在 -> {font_path}")
        return
    original_w, original_h, _, _ = probe_video_new(input_video_path)
    if not all([original_w, original_h]):
        print("错误：无法获取有效的视频尺寸。")
        return

    # --- 2. 计算新画布尺寸和视频位置 (不变) ---
    top_padding = int(original_h * padding_ratio)
    output_w = original_w
    output_h = original_h + top_padding
    video_y_start = top_padding
    if original_w / original_h > 1.5:
        bottom_padding = top_padding // 2
        top_padding = top_padding + bottom_padding
        output_h = original_h + top_padding + bottom_padding
        video_y_start = top_padding

    # --- 3. 构建滤镜链 ---
    base_filter = f"pad={output_w}:{output_h}:0:{video_y_start}:color=black"
    drawtext_filters = []
    escaped_font_path = _escape_ffmpeg_path(font_path)
    for event in text_events:
        text_list = event.get('text_list', [])
        if not text_list: continue

        start_time = event.get('start_time', 0)
        end_time = event.get('end_time', 99999)
        if start_time >= end_time: continue

        colors = event.get('color_config', {})
        PALETTE = ['#FFFFFF', '#FF4C4C', '#FFD700']  # 白 / 黑 / 金
        fontcolor = colors.get('fontcolor', random.choice(PALETTE))
        # fontcolor = colors.get('fontcolor', '#FFD700')
        shadowcolor = colors.get('shadowcolor', 'black@0.8')

        # --- 字体大小计算逻辑 (保持不变，依然健壮) ---
        margin_ratio = 0.0
        line_spacing_ratio = 0.1
        available_width = output_w * 0.9
        available_height = top_padding * (1.0 - margin_ratio * 2)
        longest_text = max(text_list, key=len) if any(text_list) else ''
        fontsize_w = (available_width / len(longest_text)) if longest_text else 9999
        num_lines = len(text_list)
        if num_lines > 1:
            denominator = num_lines + (num_lines - 1) * line_spacing_ratio
            fontsize_h = available_height / denominator
        else:
            fontsize_h = available_height
        fontsize = min(fontsize_w, fontsize_h, 100)

        # === NEW: 重新计算文本块起始位置以实现“底部对齐” ===

        # 1. 计算单行高度（字体+行间距）
        line_height = fontsize * (1 + line_spacing_ratio)

        # 2. 计算整个文本块的总高度
        # 总高度 = (行数 - 1) * 行高 + 最后一行的字体高度
        total_text_block_height = (num_lines - 1) * line_height + fontsize

        # 3. 计算文本块的起始Y坐标（反推法）
        # 底部锚点 = 视频上边界 - 安全边距
        bottom_anchor = video_y_start - (top_padding * margin_ratio)
        # 起始Y坐标 = 底部锚点 - 文本块总高度
        text_block_y_start = bottom_anchor - total_text_block_height

        # 确保起始点不为负（在文本极多的情况下）
        text_block_y_start = max(0, text_block_y_start)
        # ======================================================

        for i, text in enumerate(text_list):
            if not text: continue

            # 每行的Y坐标计算方式不变，因为它依赖于起始点
            current_y = text_block_y_start + i * line_height
            escaped_text = _escape_ffmpeg_text(text)

            filter_str = (
                f"drawtext="
                f"fontfile='{escaped_font_path}':"
                f"text='{escaped_text}':"
                f"fontsize={fontsize}:"
                f"fontcolor='{fontcolor}':"
                f"shadowcolor='{shadowcolor}':shadowx=2:shadowy=2:"
                f"x=(w-text_w)/2:"
                f"y={current_y}:"
                f"enable='between(t,{start_time},{end_time})'"
            )
            drawtext_filters.append(filter_str)

    # --- 4. 组合最终滤镜并构建命令 (不变) ---
    all_filters = [base_filter] + drawtext_filters
    full_filter_chain = ",".join(all_filters)
    command = [
        'ffmpeg', '-i', input_video_path,
        '-filter_complex', f"[0:v]{full_filter_chain}[outv]",
        '-map', '[outv]', '-map', '0:a?',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
        '-c:a', 'aac', '-y', output_video_path
    ]
    print("即将执行的 FFmpeg 命令:")
    print(shlex.join(command))

    # --- 5. 执行命令 (不变) ---
    try:
        process = subprocess.run(
            command, check=True, capture_output=True, text=True, encoding='utf-8'
        )
        print(f"\n视频处理成功！输出文件位于: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print("\n--- FFmpeg 处理失败! ---")
        print("FFmpeg 返回码:", e.returncode)
        print("FFmpeg 错误信息:\n" + e.stderr)

def add_text_to_video_robust(input_video_path, output_video_path, text_events, font_path=None,
                             output_width=1080):
    """
    将视频转换为带文字的竖屏格式，输出宽度固定。

    Args:
        input_video_path (str): 输入视频的文件路径。
        output_video_path (str): 输出视频的文件路径。
        text_events (list): 包含文字事件的列表。
        probe_video_new (function): 获取视频信息的函数，返回 (width, height, fps, sar)。
        font_path (str, optional): 字体文件路径。默认为 'C:/Windows/Fonts/msyhbd.ttc'。
        output_width (int, optional): 输出竖屏视频的宽度。默认为 1080。
    """
    # --- 1. 参数校验和准备 ---
    if not os.path.exists(input_video_path):
        print(f"错误：输入视频文件不存在 -> {input_video_path}")
        return

    if font_path is None:
        font_path = 'C:/Windows/Fonts/msyhbd.ttc'
        print(f"提示：未使用指定字体，将尝试使用默认字体 -> {font_path}")

    if not os.path.exists(font_path):
        print(f"错误：字体文件不存在 -> {font_path}")
        print("请通过 'font_path' 参数指定一个有效的字体文件路径。")
        return

    try:
        original_w, original_h, _, _ = probe_video_new(input_video_path)
        if not all([original_w, original_h]):
            raise ValueError("获取到的视频尺寸无效")
    except Exception as e:
        print(f"错误：使用 probe_video_new 获取视频信息失败: {e}")
        return

    # --- 2. 计算尺寸 (已更新) ---
    output_w = output_width
    output_h = int(output_w * 16 / 9)

    # 计算视频缩放后的高度，以保持原始宽高比
    scaled_video_h = int(original_h * (output_w / original_w))

    # 计算视频在竖屏画布中的Y轴起始位置
    video_y_start = (output_h - scaled_video_h) / 2

    # --- 3. 构建滤镜链 (已更新) ---
    # 重新引入 scale 滤镜，先缩放视频，再用 pad 填充
    # scale={output_w}:-1 表示宽度固定为 output_w，高度按比例自动计算
    base_filter = f"scale={output_w}:-1,pad={output_w}:{output_h}:(ow-iw)/2:(oh-ih)/2:color=black"

    drawtext_filters = []
    escaped_font_path = _escape_ffmpeg_path(font_path)

    for event in text_events:
        text_list = event.get('text_list', [])
        start_time = event.get('start_time', 0)
        end_time = event.get('end_time', 99999)

        if start_time >= end_time:
            print(f"警告：跳过一个无效的时间段事件 (start >= end): {event}")
            continue

        colors = event.get('color_config', {})
        fontcolor = colors.get('fontcolor', '#FFD700')
        shadowcolor = colors.get('shadowcolor', 'black@0.8')

        # --- 字体大小计算 ---
        longest_text = max(text_list, key=len) if any(text_list) else ''
        available_width = output_w * 0.9
        max_fontsize = available_width / 10

        if not longest_text:
            calculated_fontsize = max_fontsize
        else:
            calculated_fontsize = available_width / len(longest_text)

        fontsize = min(max_fontsize, calculated_fontsize)

        # --- Y轴位置计算 ---
        line_spacing = fontsize * 0.25
        bottom_gap = fontsize * 0.5
        last_line_y = video_y_start - bottom_gap - fontsize

        for i, text in enumerate(reversed(text_list)):
            if not text:
                continue

            current_y = last_line_y - i * (fontsize + line_spacing)
            escaped_text = _escape_ffmpeg_text(text)

            filter_str = (
                f"drawtext="
                f"fontfile='{escaped_font_path}':"
                f"text='{escaped_text}':"
                f"fontsize={fontsize}:"
                f"fontcolor='{fontcolor}':"
                f"shadowcolor='{shadowcolor}':shadowx=2:shadowy=2:"
                f"x=(w-text_w)/2:"
                f"y={current_y}:"
                f"enable='between(t,{start_time},{end_time})'"
            )
            drawtext_filters.append(filter_str)

    # --- 4. 组合最终滤镜并构建命令 ---
    if drawtext_filters:
        full_filter_chain = f"{base_filter},{','.join(drawtext_filters)}"
    else:
        full_filter_chain = base_filter

    command = [
        'ffmpeg', '-i', input_video_path,
        '-vf', full_filter_chain,
        '-map', '0:v:0', '-map', '0:a:0?',
        '-c:a', 'copy',
        '-y', output_video_path
    ]

    print("即将执行的 FFmpeg 命令:")
    print(shlex.join(command))

    # --- 5. 执行命令 ---
    try:
        process = subprocess.run(
            command, check=True, capture_output=True, text=True, encoding='utf-8'
        )
        print(f"\n视频处理成功！输出文件位于: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print("\n--- FFmpeg 处理失败! ---")
        print("FFmpeg 返回码:", e.returncode)
        print("FFmpeg 错误信息:\n" + e.stderr)


def create_variety_text(text: str, font_size: int, output_image_path: str, text_type: str = "正式"):
    """
    一个为不同场景优化的、自动化的风格化文字生成函数。

    你只需要关心：【文字内容、字体大小、输出路径、文字类型】。
    颜色会根据类型从内置颜色池随机选择，描边宽度会根据字体大小和类型自动适配。

    Args:
        text (str): 要生成的文字内容。
        font_size (int): 字体大小。
        output_image_path (str): 生成图片的保存路径。
        text_type (str, optional): 文字类型，可选值为 "综艺" 或 "正式"。默认为 "综艺"。
    """
    DEFAULT_FONT_PATH = r'C:\Users\zxh\AppData\Local\Microsoft\Windows\Fonts\AaFengKuangYuanShiRen-2.ttf'

    # --- 1. 样式配置库 ---
    # 将不同类型的配置集中管理，方便扩展
    STYLE_CONFIG = {
        "综艺": {
            "colors": [
                (255, 204, 0),    # 醒目柠檬黄 (经典高对比色)
                (0, 230, 230),    # 能量青色 (科技感、未来感)
                (255, 87, 34),    # 活力亮橙 (温暖、引人注目)
                (236, 64, 122),   # 魅力洋红 (时尚、大胆)
                (124, 252, 0),    # 荧光绿 (赛博朋克、年轻)
                (173, 216, 230),  # 天空浅蓝 (清新、宁静)
                (255, 218, 185),  # 蜜桃粉橙 (温柔、有亲和力)
                (181, 230, 194),  # 薄荷绿 (自然、舒适)
                (220, 190, 240),  # 薰衣草紫 (优雅、梦幻)
                (226, 192, 112),  # 高光香槟金 (比之前的金色更亮，质感更好)
                (205, 127, 50),  # 古铜色 (沉稳、有历史感)
                (255, 225, 1),  # 亮黄色
                (255, 120, 177),  # 甜粉色
                (0, 225, 233),  # 天青色
                (138, 88, 255),  # 潮紫色
                (255, 108, 0),  # 活力橙
                (124, 252, 0),  # 荧光绿
                (173, 216, 230),  # 浅天蓝
                (255, 20, 147),  # 深粉色
                (255, 140, 0),  # 深橙色
                (34, 139, 34),  # 森林绿
                (75, 0, 130),  # 靛蓝色
                (199, 21, 133),  # 深洋红色
                (255, 215, 0),  # 金色
                (255, 225, 1),  # 亮黄色
                (255, 120, 177),  # 甜粉色
                (0, 225, 233),  # 天青色
                (138, 88, 255),  # 潮紫色
                (255, 108, 0),  # 活力橙
                (124, 252, 0),  # 荧光绿
            ],
            "inner_stroke_ratio": 0.12,  # 内层白色描边，占字号的12%，较粗
            "outer_stroke_ratio": 0.05,  # 最外层深色描边，占字号的5%，较粗
            'font_path': r'C:\Users\zxh\AppData\Local\Microsoft\Windows\Fonts\AaFengKuangYuanShiRen-2.ttf'

        },
        "正式": {
            "colors": [
                (19, 41, 75),  # 深海军蓝
                (218, 165, 32),  # 高级金色
                (139, 0, 0),  # 暗红色
                (0, 100, 0),  # 深绿色
                (255, 204, 0),  # 醒目柠檬黄 (经典高对比色)
                (0, 230, 230),  # 能量青色 (科技感、未来感)
                (255, 87, 34),  # 活力亮橙 (温暖、引人注目)
                (236, 64, 122),  # 魅力洋红 (时尚、大胆)
                (124, 252, 0),  # 荧光绿 (赛博朋克、年轻)
                (173, 216, 230),  # 天空浅蓝 (清新、宁静)
                (255, 218, 185),  # 蜜桃粉橙 (温柔、有亲和力)
                (181, 230, 194),  # 薄荷绿 (自然、舒适)
                (220, 190, 240),  # 薰衣草紫 (优雅、梦幻)
                (226, 192, 112),  # 高光香槟金 (比之前的金色更亮，质感更好)
                (205, 127, 50),  # 古铜色 (沉稳、有历史感)
                (255, 225, 1),  # 亮黄色
                (255, 120, 177),  # 甜粉色
                (0, 225, 233),  # 天青色
                (138, 88, 255),  # 潮紫色
                (255, 108, 0),  # 活力橙
                (124, 252, 0),  # 荧光绿
                (173, 216, 230),  # 浅天蓝
                (255, 20, 147),  # 深粉色
                (255, 140, 0),  # 深橙色
                (34, 139, 34),  # 森林绿
                (75, 0, 130),  # 靛蓝色
                (199, 21, 133),  # 深洋红色
                (255, 215, 0),  # 金色
                (255, 225, 1),  # 亮黄色
                (255, 120, 177),  # 甜粉色
                (0, 225, 233),  # 天青色
                (138, 88, 255),  # 潮紫色
                (255, 108, 0),  # 活力橙
                (124, 252, 0),  # 荧光绿
            ],
            "inner_stroke_ratio": 0.06,  # 内层白色描边，占字号的6%，更精致
            "outer_stroke_ratio": 0.03,  # 最外层深色描边，占字号的3%，更纤细
            'font_path': 'C:/Windows/Fonts/msyhbd.ttc'
        }
    }

    # --- 2. 自动化参数计算 ---
    # 检查传入的 text_type 是否有效，无效则报错
    if text_type not in STYLE_CONFIG:
        print(f"错误：无效的文字类型 '{text_type}'。可用类型为: {list(STYLE_CONFIG.keys())}")
        return False

    # 根据类型选择配置
    config = STYLE_CONFIG[text_type]
    color_pool = config["colors"]
    font_path = config.get('font_path', DEFAULT_FONT_PATH)
    inner_stroke_ratio = config["inner_stroke_ratio"]
    outer_stroke_ratio = config["outer_stroke_ratio"]

    # 根据字体大小和选择的比例，自动计算描边的最佳宽度
    stroke_width = max(1, int(font_size * inner_stroke_ratio))
    outer_stroke_width = max(1, int(font_size * outer_stroke_ratio))

    # 自动选择颜色
    fill_color = random.choice(color_pool)
    stroke_color = (255, 255, 255)  # 内描边固定为白色
    outer_stroke_color = (40, 40, 40)  # 外描边固定为深灰色/黑色

    # --- 3. 加载字体并准备画布 ---
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"错误：无法加载字体 -> {font_path}")
        print("请检查函数中的 DEFAULT_FONT_PATH 变量是否设置正确！")
        return False

    # 计算画布大小，需要给描边留出足够的“扩张”空间
    padding = (stroke_width + outer_stroke_width) * 2
    # 使用 getbbox 来精确计算文字边界
    text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    canvas_width = text_width + padding
    canvas_height = text_height + padding

    # --- 4. 绘制、扩张、合成图层 (核心逻辑) ---

    # [底层] 绘制最顶层的文字，用于提取形状
    text_layer = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    # text_bbox[1] 是文字顶部的偏移量，减去它来让文字从画布的(padding/2, 0)位置开始绘制
    draw.text((padding // 2, padding // 2 - text_bbox[1]), text, font=font, fill=fill_color)

    # 提取文字形状的Alpha通道作为蒙版
    alpha_mask = text_layer.getchannel('A')

    # [中层] 创建白色描边
    white_stroke_mask = alpha_mask.filter(ImageFilter.MaxFilter(stroke_width * 2 + 1))
    white_stroke_layer = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    white_stroke_layer.paste(Image.new('RGB', (canvas_width, canvas_height), stroke_color), mask=white_stroke_mask)

    # [顶层] 创建深色外框
    black_stroke_mask = white_stroke_mask.filter(ImageFilter.MaxFilter(outer_stroke_width * 2 + 1))
    black_stroke_layer = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    black_stroke_layer.paste(Image.new('RGB', (canvas_width, canvas_height), outer_stroke_color),
                             mask=black_stroke_mask)

    # 从后往前，完美地合成所有图层 (背景 -> 外描边 -> 内描边 -> 文字)
    final_image = Image.alpha_composite(black_stroke_layer, white_stroke_layer)
    final_image = Image.alpha_composite(final_image, text_layer)

    # --- 5. 裁剪并保存 ---
    bbox = final_image.getbbox()
    if bbox:
        final_image = final_image.crop(bbox)

    final_image.save(output_image_path)
    # print(f"-> 类型 '{text_type}' 的文字已生成：{output_image_path} (颜色: {fill_color})")
    return True


def add_text_overlays_to_video(
        video_path: str,
        text_info_list: list,
        output_video_path: str,
        image_dir_path: str,
        is_fun=False
):
    """
    为视频叠加多个综艺花字图片，并将生成的图片保存到指定目录。

    Args:
        video_path (str): 输入视频的路径。
        text_info_list (list): 花字信息列表。
        output_video_path (str): 输出视频的路径。
        image_dir_path (str): 用于存放生成的所有花字图片的目录路径。
    """
    print("开始处理视频...")

    # --- 步骤 1: 获取视频信息 ---
    try:
        video_w, video_h, _, _ = probe_video_new(video_path)
        print(f"视频分辨率: {video_w}x{video_h}")
    except (TypeError, FileNotFoundError) as e:
        print(f"无法继续处理，因为获取视频信息失败: {e}")
        return

    min_video_size = min(video_w, video_h)
    auto_font_size = int(min_video_size / 15)
    margin = int(min_video_size * 0.15)
    print(f"自动计算字体大小为: {auto_font_size}px, 边距为: {margin}px")

    # --- 步骤 2: 创建指定目录并生成所有花字图片 ---
    # 【改动】确保指定的图片输出目录存在
    os.makedirs(image_dir_path, exist_ok=True)
    print(f"花字图片将保存到目录: {image_dir_path}")

    generated_images = []
    # 【改动】为图片文件名添加视频文件名前缀，避免混淆
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    for i, info in enumerate(text_info_list):
        # 【改动】使用指定的目录和新的命名规则
        image_filename = f"{video_basename}_text_{i}.png"
        image_path = os.path.join(image_dir_path, image_filename)
        text_type = "综艺" if is_fun else "正式"
        # print(f"正在生成图片: {image_filename} ...")
        success = create_variety_text(
            text=info['text'],
            font_size=auto_font_size,
            output_image_path=image_path,
            text_type=text_type
        )
        # if is_valid_target_file_simple(image_path):
        #     success = True
        if success:
            generated_images.append({**info, 'path': image_path})
        else:
            print(f"警告: 生成文字 '{info['text']}' 的图片失败，将跳过。")

    if not generated_images:
        print("没有成功生成任何花字图片，处理终止。")
        return

    # --- 步骤 3: 构建并执行 FFmpeg 命令 (逻辑无变化) ---
    position_map = {
        'TL': f"x={margin}:y={margin}", 'TC': f"x=(W-overlay_w)/2:y={margin}",
        'TR': f"x=W-overlay_w-{margin}:y={margin}",
        'ML': f"x={margin}:y=(H-overlay_h)/2", 'MC': f"x=(W-overlay_w)/2:y=(H-overlay_h)/2",
        'MR': f"x=W-overlay_w-{margin}:y=(H-overlay_h)/2",
        'BL': f"x={margin}:y=H-overlay_h-{margin}", 'BC': f"x=(W-overlay_w)/2:y=H-overlay_h-{margin}",
        'BR': f"x=W-overlay_w-{margin}:y=H-overlay_h-{margin}",
    }
    base_cmd = ['ffmpeg', '-y', '-i', video_path]
    for img_info in generated_images:
        base_cmd.extend(['-i', img_info['path']])

    filter_complex = []
    last_video_stream = "[0:v]"

    for i, img_info in enumerate(generated_images):
        image_stream = f"[{i + 1}:v]"
        output_stream = f"[v{i + 1}]"
        start, end = img_info['start'], img_info['start'] + img_info['duration']
        position = img_info['position'].upper()
        overlay_coords = position_map.get(position, position_map['MC'])
        filter_str = f"{last_video_stream}{image_stream}overlay={overlay_coords}:enable='between(t,{start},{end})'{output_stream}"
        filter_complex.append(filter_str)
        last_video_stream = output_stream

    full_cmd = base_cmd + [
        '-filter_complex', ";".join(filter_complex),
        '-map', last_video_stream, '-map', '0:a?',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-c:a', 'aac',
        output_video_path
    ]

    print("\n即将执行 FFmpeg 命令:")
    print(shlex.join(full_cmd))

    try:
        subprocess.run(full_cmd, check=True, capture_output=True, text=True)
        print(f"\n✅ 视频处理成功！输出文件: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print("\n❌ FFmpeg 执行失败! 错误信息:\n", e.stderr)



if __name__ == '__main__':

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