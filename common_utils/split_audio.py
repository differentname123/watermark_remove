import subprocess
import os
from pathlib import Path

from pydub import AudioSegment


def separate_with_cli(input_path: str, output_dir: str, two_stems: bool = True):
    """
    使用 Demucs 分离音轨。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 用 Path 处理路径可防止跨平台路径问题
    input_path = str(Path(input_path))
    output_dir = str(Path(output_dir))

    cmd = ["demucs"]
    if two_stems:
        cmd.append("--two-stems=vocals")
    cmd += [input_path, "-o", output_dir]

    print("Running command:", ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd))  # 为显示目的添加引号
    subprocess.run(cmd, check=True)
    print("分离完成，结果在:", output_dir)

def trim_audio(input_path: str, output_path: str, start_time: str, end_time: str):
    """
    使用 ffmpeg 截取输入音频的指定时间段。

    参数:
    input_path (str): 输入音频文件的路径。
    output_path (str): 输出音频文件的路径。
    start_time (str): 截取的开始时间，格式为 HH:MM:SS 或秒数。
    end_time (str): 截取的结束时间，格式为 HH:MM:SS 或秒数。
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",  # 新增：如果输出文件已存在，则直接覆盖

        "-i", input_path,      # 输入文件
        "-ss", start_time,     # 开始时间
        "-to", end_time,       # 结束时间
        "-c", "copy",          # 直接复制流，不重新编码，速度快且无损
        output_path            # 输出文件
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"音频截取完成，结果保存为: {output_path}")


def merge_overlapping_intervals(intervals):
    """
    一个辅助函数，用于合并重叠或相邻的时间区间。
    """
    if not intervals:
        return []

    # 按起始时间对区间进行排序
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]
    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]

        # 如果当前区间与前一个区间重叠或相邻，则合并
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))

    return merged


def get_average_volume(media_path: str) -> float | None:
    """
    计算给定媒体文件（音频或视频）的平均音量。

    该函数会加载媒体文件，并计算其音量的分贝值（dBFS）。
    如果文件是视频，它会自动提取音频部分进行计算。

    :param media_path: 媒体文件的路径（可以是音频或视频）。
    :return: 以dBFS为单位的平均音量。如果文件无法加载或为完全静音，
             对于无法加载的情况返回 None，对于完全静音的情况返回 -inf。
    """
    if not os.path.exists(media_path):
        print(f"错误：文件不存在于路径: {media_path}")
        return None

    try:
        print(f"正在加载媒体文件: {media_path}...")
        # AudioSegment.from_file 可以智能处理音频和视频文件（提取音频）
        audio = AudioSegment.from_file(media_path)
    except Exception as e:
        print(f"错误：无法加载媒体文件。请确保文件路径正确且FFmpeg已正确安装。错误信息: {e}")
        return None

    # .dBFS 属性可以计算出平均音量
    average_dbfs = audio.dBFS

    # -float('inf') 表示完全的静音
    if average_dbfs == -float('inf'):
        print("警告：输入媒体文件似乎是完全静音的。")

    print(f"计算出的平均音量为: {average_dbfs:.2f} dBFS")
    return average_dbfs

def process_media_by_volume(original_media_path: str, processed_audio_path: str):
    """
    处理媒体文件（音频或视频），将低于特定音量阈值的部分静音。

    该函数会：
    1. 计算整个媒体文件的平均音量。
    2. 将平均音量增加10dB作为阈值。
    3. 以10毫秒为单位检测超过阈值的音频片段。
    4. 将检测到的片段前后各扩展50毫秒。
    5. 合并所有重叠的时间段，生成最终需要保留的音频范围列表。
    6. 根据这个列表，生成一个新的音频文件，其中只有指定范围内的声音被保留，其余部分为静音。

    :param original_media_path: 原始媒体文件（音频或视频）的路径。
    :param processed_audio_path: 处理后要保存的音频文件的路径。
    """
    # 1. 计算平均音量并设定阈值
    average_dbfs = get_average_volume(original_media_path)

    # 如果无法计算音量（例如文件问题），则终止
    if average_dbfs is None:
        return

    # 为后续处理加载音频（如果刚才已经加载过，这里会再次加载，但为了函数独立性可接受）
    try:
        audio = AudioSegment.from_file(original_media_path)
    except Exception as e:
        print(f"错误：在处理阶段无法重新加载音频文件。错误: {e}")
        return

    # 如果音频是完全的静音
    if average_dbfs == -float('inf'):
        silent_audio = AudioSegment.silent(duration=len(audio))
        silent_audio.export(processed_audio_path, format=processed_audio_path.split('.')[-1])
        print(f"已生成一个完全静音的文件到: {processed_audio_path}")
        return

    volume_threshold = average_dbfs + 10.0
    print(f"设定保留音量的阈值为: {volume_threshold:.2f} dBFS")

    # 2. 以10ms为单位检测满足阈值的音频块
    chunk_size_ms = 10
    loud_chunks_times = []
    print(f"正在以 {chunk_size_ms}ms 为单位检测高音量片段...")

    for i in range(0, len(audio), chunk_size_ms):
        chunk = audio[i:i + chunk_size_ms]
        if chunk.dBFS > volume_threshold:
            loud_chunks_times.append((i, i + chunk_size_ms))

    if not loud_chunks_times:
        print("未检测到任何超过阈值的音频片段。将生成一个完全静音的文件。")
        silent_audio = AudioSegment.silent(duration=len(audio))
        silent_audio.export(processed_audio_path, format=processed_audio_path.split('.')[-1])
        print(f"已生成一个完全静音的文件到: {processed_audio_path}")
        return

    # 3. 对每个时间范围前后拓展50ms
    expansion_ms = 50
    expanded_times = [(max(0, s - expansion_ms), min(len(audio), e + expansion_ms)) for s, e in loud_chunks_times]

    # 4. 合并重叠的时间范围
    print("正在合并需要保留的音频时间段...")
    # 假设 merge_overlapping_intervals 函数已定义
    # final_keep_ranges = merge_overlapping_intervals(expanded_times)
    # 为了代码可直接运行，这里提供一个简单的合并实现
    if not expanded_times:
        final_keep_ranges = []
    else:
        # 按起始时间排序
        expanded_times.sort(key=lambda interval: interval[0])
        merged = [expanded_times[0]]
        for current_start, current_end in expanded_times[1:]:
            last_start, last_end = merged[-1]
            if current_start <= last_end: # 有重叠
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        final_keep_ranges = merged


    print(f"最终需要保留的时间段数量: {len(final_keep_ranges)} {final_keep_ranges}")

    # 5. 生成最终音频
    print("正在生成处理后的音频文件...")
    output_audio = AudioSegment.silent(duration=len(audio))

    for start_ms, end_ms in final_keep_ranges:
        original_segment = audio[start_ms:end_ms]
        output_audio = output_audio.overlay(original_segment, position=start_ms)

    # 6. 导出处理后的音频
    output_format = processed_audio_path.split('.')[-1]
    output_audio.export(processed_audio_path, format=output_format)
    print(f"处理完成！音频已保存到: {processed_audio_path}")


if __name__ == "__main__":
    process_media_by_volume("test4_no_vocals.wav", "denoised_no_vocals.wav")

    # --- 示例1: 调用原有的音轨分离函数 ---
    # print("--- 开始音轨分离 ---")
    # separate_with_cli(
    #     r"mix.mp3",
    #     r"videos",
    #     two_stems=True
    # )
    # print("\n" + "="*30 + "\n")


    # # --- 示例2: 调用新增的音频截取函数 ---
    # print("--- 开始音频截取 ---")
    # # 假设我们要从 "mix.mp3" 中截取从第10秒到第30秒的片段
    # trim_audio(
    #     input_path=r"ruru.m4a",
    #     output_path=r"trimmed_audio/ruru.m4a",
    #     start_time="00:00:1",  # 从第10秒开始
    #     end_time="00:00:10"      # 到第30秒结束
    # )

    # 另一个例子：使用秒数作为时间
    # print("\n--- 使用秒数截取 ---")
    # trim_audio(
    #     input_path=r"mix.mp3",
    #     output_path=r"trimmed_audio/cut_by_seconds.mp3",
    #     start_time="45",  # 从第45秒开始
    #     end_time="55"     # 到第55秒结束
    # )