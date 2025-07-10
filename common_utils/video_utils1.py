import os
import subprocess
import tempfile
import shutil
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

def redub_video_with_ffmpeg(video_path: str, segments_info: list, output_path="final_video_ffmpeg.mp4") -> str:
    """
    使用 FFmpeg 直接为视频重新配音。
    如果新音频比对应的视频片段长，则慢放视频以匹配音频时长。

    :param video_path: 原始视频文件的路径。
    :param segments_info: 一个包含片段信息的列表。
    :param output_path: 输出的最终视频文件路径。
    :return: 输出视频的路径。
    """
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("FFmpeg not found. Please install FFmpeg and ensure it is in your system's PATH.")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件未找到: {video_path}")

    # 使用临时目录来存放中间文件，更安全整洁
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_files_list = []
        concat_file_path = os.path.join(temp_dir, "file_list.txt")

        print("开始处理视频片段...")
        # 1. 处理阶段：生成所有临时片段文件
        for i, segment in enumerate(segments_info):
            segment_id = segment.get('id', i + 1)
            start_time_str = segment['startTime']
            end_time_str = segment['endTime']
            audio_path = segment['outputPath']

            print(f"\n--- 正在处理片段 {segment_id} ---")

            if not os.path.exists(audio_path):
                print(f"警告: 音频文件未找到 {audio_path}，跳过此片段。")
                continue

            original_duration = _time_str_to_seconds(end_time_str) - _time_str_to_seconds(start_time_str)
            new_audio_duration = segment['trimmedDuration']

            # 临时视频片段的输出路径
            temp_output_path = os.path.join(temp_dir, f"temp_segment_{segment_id}.mp4")
            temp_files_list.append(temp_output_path)

            # 计算速度调整因子
            # setpts滤镜需要的是时间戳的乘数。速度变为0.5倍(慢一倍)，时间戳就要变为2倍。
            speed_multiplier = 1.0
            if new_audio_duration > original_duration and original_duration > 0:
                speed_multiplier = new_audio_duration / original_duration

            print(f"原片段时长: {original_duration:.3f}s, 新音频时长: {new_audio_duration:.3f}s")
            print(f"视频速度调整为: {1 / speed_multiplier:.2f}x (setpts乘数: {speed_multiplier:.2f})")

            # 构建FFmpeg命令
            cmd = [
                "ffmpeg",
                "-y",  # 覆盖输出文件
                "-loglevel", "error",  # 只显示错误信息
                "-ss", start_time_str,
                "-to", end_time_str,
                "-i", video_path,
                "-i", audio_path,
                "-filter_complex", f"[0:v]setpts={speed_multiplier:.4f}*PTS[v]",  # 变速
                "-map", "[v]",  # 映射处理后的视频
                "-map", "1:a",  # 映射新音频
                "-c:v", "libx264",  # 使用标准视频编码器
                "-preset", "veryfast",  # 快速编码
                "-c:a", "aac",  # 使用标准音频编码器
                "-shortest",  # 以最短的流为准结束
                temp_output_path
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"处理片段 {segment_id} 时 FFmpeg 发生错误:")
                print(f"FFmpeg Stderr:\n{e.stderr}")
                raise  # 重新抛出异常，中断执行

        if not temp_files_list:
            print("没有可处理的片段，无法生成最终视频。")
            return ""

        # 2. 拼接阶段：创建列表文件并执行拼接
        print("\n所有片段处理完毕，正在拼接成最终视频...")
        with open(concat_file_path, 'w', encoding='utf-8') as f:
            for file_path in temp_files_list:
                # FFmpeg concat demuxer 需要特定的格式和转义
                # 使用正斜杠，即使在Windows上
                safe_path = file_path.replace('\\', '/')
                f.write(f"file '{safe_path}'\n")

        # 构建拼接命令
        concat_cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file_path,
            "-c", "copy",  # 直接复制流，速度极快
            output_path
        ]

        try:
            subprocess.run(concat_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("拼接视频时 FFmpeg 发生错误:")
            print(f"FFmpeg Stderr:\n{e.stderr}")
            raise

    print(f"成功！最终视频已保存至: {output_path}")
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