import pathlib
import random
import shutil
import subprocess
import os
import sys

import cv2
import numpy as np
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
        '-preset', 'medium',# 预设，影响编码速度和压缩率的平衡。'medium' 是默认值，通常无需更改。
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

def process_and_crop_video(video_path, area_threshold_ratio=0.8, **kwargs):
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


# ... (示例使用部分保持不变) ...
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