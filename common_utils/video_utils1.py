import os
import subprocess
import tempfile
import shutil
import time
from datetime import datetime, timedelta


def _time_str_to_seconds(time_str: str) -> float:
    """
    将多种格式的时间字符串（如 HH:MM:SS.f, MM:SS.f, SS.f）转换为总秒数。

    Args:
        time_str: 时间字符串。

    Returns:
        代表总秒数的浮点数。

    Raises:
        ValueError: 如果时间字符串格式不正确。

    Examples:
        >>> time_str_to_seconds('01:02:03.456')
        3723.456
        >>> time_str_to_seconds('03:45.123')
        225.123
        >>> time_str_to_seconds('59.9')
        59.9
        >>> time_str_to_seconds('01:10:05')
        4205.0
    """
    if not isinstance(time_str, str) or not time_str.strip():
        raise ValueError("输入必须是非空字符串")

    parts = time_str.split(':')
    seconds = 0.0

    try:
        # 根据冒号分割后的部分数量来确定时间单位
        if len(parts) == 3:  # HH:MM:SS.f
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:  # MM:SS.f
            seconds = int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1:  # SS.f
            seconds = float(parts[0])
        else:
            raise ValueError(f"时间格式不正确: '{time_str}'")
    except (ValueError, IndexError):
        # 捕获 float()或int() 转换失败，或 parts 索引错误
        raise ValueError(f"无法解析时间字符串: '{time_str}'")

    return seconds

# def redub_video_with_ffmpeg(video_path: str,
#                             segments_info: list,
#                             output_path: str = "final_video_ffmpeg.mp4") -> str:
#     """
#     使用 FFmpeg 直接为视频重新配音。
#     如果新音频比对应的视频片段长，则慢放视频以匹配音频时长。
#
#     :param video_path: 原始视频文件的路径。
#     :param segments_info: 一个包含片段信息的列表，每个元素至少包括：
#                           - 'startTime' (如 "00:00:05.000")
#                           - 'endTime'   (如 "00:00:10.000")
#                           - 'outputPath' (对应音频文件路径)
#                           - 'trimmedDuration' (新音频时长，秒)
#     :param output_path: 输出的最终视频文件路径。
#     :return: 输出视频的路径。
#     """
#     if not shutil.which("ffmpeg"):
#         raise FileNotFoundError("FFmpeg not found. Please install FFmpeg and ensure it is in your system's PATH.")
#
#     if not os.path.exists(video_path):
#         raise FileNotFoundError(f"视频文件未找到: {video_path}")
#
#     with tempfile.TemporaryDirectory() as temp_dir:
#         temp_files_list = []
#         concat_file_path = os.path.join(temp_dir, "file_list.txt")
#
#         print("开始处理视频片段...")
#         for i, segment in enumerate(segments_info):
#             segment_id = segment.get('id', i + 1)
#             start_time_str = segment['startTime']
#             end_time_str = segment['endTime']
#             audio_path = segment['outputPath']
#
#             print(f"\n--- 正在处理片段 {segment_id} ---")
#             if not os.path.exists(audio_path):
#                 print(f"警告: 音频文件未找到 {audio_path}，跳过此片段。")
#                 continue
#
#             original_duration = _time_str_to_seconds(end_time_str) - _time_str_to_seconds(start_time_str)
#             new_audio_duration = segment['trimmedDuration']
#
#             temp_output_path = os.path.join(temp_dir, f"temp_segment_{segment_id}.mp4")
#             temp_files_list.append(temp_output_path)
#
#             speed_multiplier = 1.0
#             if new_audio_duration > original_duration > 0:
#                 speed_multiplier = new_audio_duration / original_duration
#
#             print(f"原片段时长: {original_duration:.3f}s, 新音频时长: {new_audio_duration:.3f}s")
#             print(f"视频速度调整为: {1/speed_multiplier:.2f}x (setpts 乘数: {speed_multiplier:.2f})")
#             print("标准化音频参数为: 采样率 44100 Hz, 声道数 2")
#
#             cmd = [
#                 "ffmpeg", "-y", "-loglevel", "error",
#                 "-ss", start_time_str, "-to", end_time_str,
#                 "-i", video_path, "-i", audio_path,
#                 "-filter_complex", f"[0:v]setpts={speed_multiplier:.4f}*PTS[v]",
#                 "-map", "[v]", "-map", "1:a",
#                 "-c:v", "libx264", "-preset", "veryfast",
#                 "-c:a", "aac", "-b:a", "256k", "-ar", "44100", "-ac", "2",
#                 "-shortest", temp_output_path
#             ]
#
#             try:
#                 subprocess.run(
#                     cmd,
#                     check=True,
#                     capture_output=True,
#                     text=True,            # 文本模式
#                     encoding='utf-8',     # 强制 UTF-8 解码
#                     errors='ignore',      # 忽略非法字节
#                 )
#             except subprocess.CalledProcessError as e:
#                 print(f"处理片段 {segment_id} 时 FFmpeg 发生错误：")
#                 print(f"FFmpeg Stderr:\n{e.stderr}")
#                 raise
#
#         if not temp_files_list:
#             print("没有可处理的片段，无法生成最终视频。")
#             return ""
#
#         print("\n所有片段处理完毕，正在拼接成最终视频...")
#         with open(concat_file_path, 'w', encoding='utf-8') as f:
#             for file_path in temp_files_list:
#                 safe_path = file_path.replace('\\', '/')
#                 f.write(f"file '{safe_path}'\n")
#
#         concat_cmd = [
#             "ffmpeg", "-y", "-f", "concat", "-safe", "0",
#             "-i", concat_file_path, "-c", "copy", output_path
#         ]
#
#         try:
#             subprocess.run(
#                 concat_cmd,
#                 check=True,
#                 capture_output=True,
#                 text=True,
#                 encoding='utf-8',
#                 errors='ignore',
#             )
#         except subprocess.CalledProcessError as e:
#             print("拼接视频时 FFmpeg 发生错误：")
#             print(f"FFmpeg Stderr:\n{e.stderr}")
#             raise
#
#     print(f"成功！最终视频已保存至: {output_path}")
#     return output_path

def replace_video_audio(video_path, start_ms, end_ms, audio_path, output_path, audio_volume=1):
    """
    将指定音频片段替换到视频中，并可调节音量大小

    Args:
        video_path (str): 输入视频文件路径
        start_ms (int): 音频开始时间（毫秒）
        end_ms (int): 音频结束时间（毫秒）
        audio_path (str): 输入音频文件路径
        output_path (str): 输出文件路径
        audio_volume (float): 音频音量比例，1.0 为原始音量，默认 0.5（50%）
    """
    if audio_volume < 0:
        raise ValueError("audio_volume must be non-negative (e.g., 0.5 for 50%)")

    # 验证输入文件存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 转换毫秒为秒（ffmpeg使用秒为单位）
    start_sec = start_ms / 1000.0
    duration_sec = (end_ms - start_ms) / 1000.0

    if duration_sec <= 0:
        raise ValueError("End time must be greater than start time")

    # 创建临时文件用于存储裁剪后的音频
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name

    try:
        # 第一步：从音频文件中提取指定时间段的音频片段
        audio_extract_cmd = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-ss', str(start_sec),
            '-i', audio_path,
            '-t', str(duration_sec),
            '-c', 'copy',  # 尝试直接复制以提高效率
            '-avoid_negative_ts', 'make_zero',
            temp_audio_path
        ]

        try:
            subprocess.run(audio_extract_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            # 如果直接复制失败（比如格式不支持），使用重新编码
            audio_extract_cmd = [
                'ffmpeg',
                '-y',
                '-ss', str(start_sec),
                '-i', audio_path,
                '-t', str(duration_sec),
                '-acodec', 'pcm_s16le',  # 显式指定编码器
                '-f', 'wav',
                '-avoid_negative_ts', 'make_zero',
                temp_audio_path
            ]
            subprocess.run(audio_extract_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 第二步：将裁剪后的音频与原视频合并（替换音频），并应用音量调整
        merge_cmd = [
            'ffmpeg',
            '-y',
            '-i', video_path,
            '-i', temp_audio_path,
            '-c:v', 'copy',               # 视频流直接复制
            '-c:a', 'aac',                # 音频重新编码为AAC
            '-af', f'volume={audio_volume}',  # 应用音量滤镜
            '-map', '0:v:0',              # 使用第一个输入的视频流
            '-map', '1:a:0',              # 使用第二个输入的音频流
            '-shortest',                  # 以较短的流为准
            output_path
        ]

        subprocess.run(merge_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    finally:
        # 清理临时文件
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

def redub_video_with_ffmpeg(video_path: str,
                            segments_info: list,
                            output_path: str = "final_video_ffmpeg.mp4",
                            keep_original_audio: bool = False) -> str:
    """
    使用 FFmpeg 直接为视频重新配音。
    如果新音频比对应的视频片段长，则慢放视频以匹配音频时长。
    修复点：避免混音后的音量自然变小（禁用 amix 归一化 + 限幅防削波）。

    :param video_path: 原始视频文件的路径。
    :param segments_info: 一个包含片段信息的列表，每个元素至少包括：
                          - 'startTime' (如 "00:00:05.000")
                          - 'endTime'   (如 "00:00:10.000")
                          - 'outputPath' (对应音频文件路径)
                          - 'trimmedDuration' (新音频时长，秒) 可选；缺失时将尝试用 ffprobe 推断
    :param output_path: 输出的最终视频文件路径。
    :param keep_original_audio: 是否保留原始音频并与新音频混合。
                                False (默认) - 替换原始音频。
                                True - 混合原始音频和新音频（不归一化，不降音量）。
    :return: 输出视频的路径。
    """
    start_time = time.time()
    # ---------- 内部工具函数 ----------
    def _time_str_to_seconds(ts: str) -> float:
        """
        支持 "HH:MM:SS", "HH:MM:SS.mmm" 等格式。
        """
        if not ts:
            return 0.0
        ts = ts.strip()
        neg = ts.startswith("-")
        if neg:
            ts = ts[1:]
        parts = ts.split(":")
        parts = [float(p) for p in parts]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        h, m, s = parts[-3], parts[-2], parts[-1]
        val = h * 3600 + m * 60 + s
        return -val if neg else val

    def _probe_has_audio(path: str) -> bool:
        """
        通过 ffprobe 判断是否存在音频流。
        """
        if not shutil.which("ffprobe"):
            return True
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                path
            ]
            out = subprocess.run(cmd, check=False, capture_output=True, text=True).stdout.strip()
            return len(out) > 0
        except Exception:
            return True

    def _probe_duration_seconds(path: str) -> float:
        """
        用 ffprobe 获取媒体总时长（秒），失败则返回 0。
        """
        if not shutil.which("ffprobe"):
            return 0.0
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path
            ]
            out = subprocess.run(cmd, check=False, capture_output=True, text=True).stdout.strip()
            return float(out) if out else 0.0
        except Exception:
            return 0.0

    def build_atempo_filter(tempo: float) -> str:
        """
        构建安全的 atempo 滤镜链，支持任意正 tempo 值。
        FFmpeg 的 atempo 范围是 [0.5, 2.0]，超出需链式组合。
        """
        if abs(tempo - 1.0) < 1e-6:
            return "anull"
        filters = []
        t = tempo
        # 处理小于 0.5 的情况
        while t < 0.5:
            filters.append("atempo=0.5")
            t *= 2.0
        # 处理大于 2.0 的情况
        while t > 2.0:
            filters.append("atempo=2.0")
            t /= 2.0
        # 剩余部分
        if abs(t - 1.0) > 1e-6:
            filters.append(f"atempo={t:.6f}")
        return ",".join(filters)

    # ---------- 依赖与输入检查 ----------
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("FFmpeg not found. Please install FFmpeg and ensure it is in your system's PATH.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件未找到: {video_path}")

    source_has_audio = _probe_has_audio(video_path)

    # ---------- 处理每个片段 ----------
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_files_list = []
        concat_file_path = os.path.join(temp_dir, "file_list.txt")

        for i, segment in enumerate(segments_info):
            segment_id = segment.get('id', i + 1)
            start_time_str = segment['startTime']
            end_time_str = segment['endTime']
            audio_path = segment['outputPath']

            if not os.path.exists(audio_path):
                print(f"警告: 音频文件未找到 {audio_path}，跳过此片段。")
                continue

            original_duration = max(0.000001, _time_str_to_seconds(end_time_str) - _time_str_to_seconds(start_time_str))
            new_audio_duration = float(segment.get('trimmedDuration') or 0.0)
            if new_audio_duration <= 0.0:
                new_audio_duration = _probe_duration_seconds(audio_path) or original_duration

            temp_output_path = os.path.join(temp_dir, f"temp_segment_{segment_id}.mp4")
            temp_files_list.append(temp_output_path)

            speed_multiplier = 1.0
            if new_audio_duration > original_duration and original_duration > 0:
                speed_multiplier = new_audio_duration / original_duration


            # ---------- 构建滤镜与映射 ----------
            if keep_original_audio and source_has_audio:
                # 需要混合原始音频和新音频
                audio_tempo = 1.0 / speed_multiplier
                atempo_filter = build_atempo_filter(audio_tempo)
                filter_complex = (
                    f"[0:v]setpts={speed_multiplier:.6f}*PTS[v];"
                    f"[0:a]{atempo_filter},aformat=sample_fmts=fltp:channel_layouts=stereo[a0];"
                    f"[1:a]aformat=sample_fmts=fltp:channel_layouts=stereo[a1];"
                    f"[a0][a1]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.97[a]"
                )
                map_args = ["-map", "[v]", "-map", "[a]"]
                print("模式: 混合新旧音频（原始音频已同步变速）")
            else:
                # 替换模式：直接使用新音频
                filter_complex = (
                    f"[0:v]setpts={speed_multiplier:.6f}*PTS[v];"
                    f"[1:a]aformat=sample_fmts=fltp:channel_layouts=stereo,alimiter=limit=0.97[a]"
                )
                map_args = ["-map", "[v]", "-map", "[a]"]
                if keep_original_audio and not source_has_audio:
                    print("模式: 源视频无音轨，使用新音频替换。")

            base_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", start_time_str, "-to", end_time_str,
                "-i", video_path, "-i", audio_path,
                "-filter_complex", filter_complex,
            ]

            encoding_cmd = [
                "-c:v", "libx264", "-preset", "veryfast",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", temp_output_path
            ]

            cmd = base_cmd + map_args + encoding_cmd

            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                )
            except subprocess.CalledProcessError as e:
                print(f"处理片段 {segment_id} 时 FFmpeg 发生错误：")
                print(f"FFmpeg Stderr:\n{e.stderr}")
                raise

        if not temp_files_list:
            print("没有可处理的片段，无法生成最终视频。")
            return ""

        # ---------- 拼接片段 ----------
        with open(concat_file_path, 'w', encoding='utf-8') as f:
            for file_path in temp_files_list:
                safe_path = file_path.replace('\\', '/')
                f.write(f"file '{safe_path}'\n")

        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file_path, "-c", "copy", output_path
        ]

        try:
            subprocess.run(
                concat_cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
            )
        except subprocess.CalledProcessError as e:
            print("拼接视频时 FFmpeg 发生错误：")
            print(f"FFmpeg Stderr:\n{e.stderr}")
            raise
    print(f"进行音频匹配画面：原片段时长: {original_duration:.3f}s, 新音频时长: {new_audio_duration:.3f}s 视频速度调整为: {1/speed_multiplier:.3f}x 耗时: {time.time() - start_time:.2f}s")

    return output_path

segments_data =[
    {
        "id": 1,
        "startTime": "00:00:00.376",
        "endTime": "00:00:03.812",
        "text": "AG让一追三击败KSG，实现跨赛季大场22连胜。",
        "optimizedText": "AG以三比一逆转KSG，达成跨赛季大场22连胜。",
        "old_startTime": "00:00:00.752",
        "old_endTime": "00:03.482",
        "forward_shift_ms": 376,
        "backward_shift_ms": 330,
        "duration": 3.436,
        "outputPath": "output_audio\\output_1.wav",
        "trimmedDuration": 5.776,
        "currentSpeed": 1.5000000000000004
    }]
# 你的原始视频路径
source_video_path = "output_with_subtitles.mp4"

if __name__ == "__main__":
    try:
        # 正式调用函数
        redub_video_with_ffmpeg(source_video_path, segments_data)

    except Exception as e:
        print(f"处理过程中发生严重错误: {e}")