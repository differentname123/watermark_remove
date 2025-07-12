import subprocess
import os
from pathlib import Path


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


if __name__ == "__main__":
    # --- 示例1: 调用原有的音轨分离函数 ---
    # print("--- 开始音轨分离 ---")
    # separate_with_cli(
    #     r"mix.mp3",
    #     r"videos",
    #     two_stems=True
    # )
    # print("\n" + "="*30 + "\n")


    # --- 示例2: 调用新增的音频截取函数 ---
    print("--- 开始音频截取 ---")
    # 假设我们要从 "mix.mp3" 中截取从第10秒到第30秒的片段
    trim_audio(
        input_path=r"ruru.m4a",
        output_path=r"trimmed_audio/ruru.m4a",
        start_time="00:00:1",  # 从第10秒开始
        end_time="00:00:10"      # 到第30秒结束
    )

    # 另一个例子：使用秒数作为时间
    # print("\n--- 使用秒数截取 ---")
    # trim_audio(
    #     input_path=r"mix.mp3",
    #     output_path=r"trimmed_audio/cut_by_seconds.mp3",
    #     start_time="45",  # 从第45秒开始
    #     end_time="55"     # 到第55秒结束
    # )