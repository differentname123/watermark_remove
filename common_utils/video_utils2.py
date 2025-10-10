import subprocess
import json
import os
import shutil

from common_utils.split_audio import get_average_volume


# --- 您提供的原始函数，稍作修改以提高健壮性 ---
def probe_video(path):
    """
    使用 ffprobe 获取视频的基本信息。
    返回一个字典: {"width", "height", "fps", "duration"}
    如果文件不存在或不是有效的视频文件，则抛出异常。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件未找到: {path}")

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "json",
        path
    ]
    try:
        # 使用 subprocess.run 提供更详细的错误信息
        result = subprocess.run(
            cmd,
            check=True,  # 如果 ffprobe 返回非零退出码，则抛出 CalledProcessError
            capture_output=True,  # 捕获 stdout 和 stderr
            text=True  # 将输出解码为文本
        )
        info = json.loads(result.stdout)["streams"][0]

        # 解析帧率
        num, den = map(int, info["r_frame_rate"].split("/"))

        return {
            "width": info["width"],
            "height": info["height"],
            "fps": num / den,
            "duration": float(info["duration"])
        }
    except subprocess.CalledProcessError as e:
        print("ffprobe 执行失败!")
        print(f"命令: {' '.join(e.cmd)}")
        print(f"错误输出:\n{e.stderr}")
        raise
    except (KeyError, IndexError):
        raise ValueError(f"无法从 {path} 解析视频流信息。文件可能已损坏或不包含视频轨道。")


# --- 核心功能函数：为视频添加BGM ---

def _probe_has_audio(path):
    """
    一个内部辅助函数，用于检查媒体文件是否包含音频流。
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",  # 只选择第一个音频流
        "-show_entries", "stream=codec_type",
        "-of", "json",
        path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # 如果有音频流，输出将包含 "audio"
    return "audio" in result.stdout


def add_bgm_to_video(video_path: str, bgm_path: str, output_path: str, volume_percentage: int = 20, auto_compute=False, rate=1):
    """
    为视频添加背景音乐(BGM)。

    功能:
    - 如果 BGM 时长小于视频，则循环播放 BGM。
    - 将 BGM 音量调整为指定百分比（例如 20 表示 20% 的原始音量）。
    - 保留视频原声，与 BGM 进行混合。如果视频无原声，则仅添加 BGM。
    - 视频流直接复制，不重新编码，以保证速度和画质。
    - 输出视频的时长与原视频完全一致。

    参数:
    - video_path (str): 输入视频的文件路径。
    - bgm_path (str): BGM 音频文件的路径。
    - output_path (str): 输出视频的文件路径。
    - volume_percentage (int, optional): BGM 的音量百分比 (0-100)。默认为 20。

    返回:
    - bool: 如果成功，返回 True。

    抛出异常:
    - FileNotFoundError: 如果输入文件或 ffmpeg/ffprobe 命令不存在。
    - ValueError: 如果音量百分比无效。
    - subprocess.CalledProcessError: 如果 ffmpeg 命令执行失败。
    """
    # 1. 检查依赖和输入参数
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("错误: ffmpeg 命令未找到。请确保已安装 ffmpeg 并将其添加至系统 PATH。")
    if not shutil.which("ffprobe"):
        raise FileNotFoundError(
            "错误: ffprobe 命令未找到。请确保已安装 ffmpeg (通常包含 ffprobe) 并将其添加至系统 PATH。")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"输入视频文件未找到: {video_path}")
    if not os.path.exists(bgm_path):
        raise FileNotFoundError(f"BGM 文件未找到: {bgm_path}")
    if not 0 <= volume_percentage <= 100:
        raise ValueError("音量百分比必须在 0 到 100 之间。")

    if auto_compute:
        bgm_volume = get_average_volume(bgm_path)
        video_volume = get_average_volume(video_path)
        volume_percentage = bgm_volume / video_volume * 100
        volume_percentage *= 0.6
        volume_percentage *= rate
        if volume_percentage > 100:
            volume_percentage = 100

        print(f"自动计算的 BGM 音量百分比: {volume_percentage:.2f}% bgm平均音量: {bgm_volume:.2f}dBFS, 视频平均音量: {video_volume:.2f}dBFS")

    # 2. 构建 ffmpeg 命令
    # 将百分比转换为 ffmpeg 的音量因子（例如 20 -> 0.2）
    volume_factor = volume_percentage / 100.0

    # 检查视频是否已有音轨
    video_has_audio = _probe_has_audio(video_path)

    # -i video.mp4          -> 输入视频 (流 0)
    # -stream_loop -1       -> 无限循环下一个输入
    # -i bgm.mp3            -> 输入BGM (流 1)
    # -filter_complex       -> 定义复杂的滤镜图
    #   "[1:a]volume=...[bgm]" -> 将BGM(流1的音频)调整音量，并标记为[bgm]
    #   "[0:a][bgm]amix=..."   -> 如果视频有原声(流0的音频)，则将其与[bgm]混合
    # -map 0:v              -> 映射视频流
    # -map "[a_out]"        -> 映射处理后的音频流
    # -c:v copy             -> 复制视频流，不重新编码
    # -shortest             -> 使输出文件的时长与最短的输入流（即原视频）一致
    # -y                    -> 覆盖输出文件

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",  # 只输出错误信息
        "-hide_banner",  # 隐藏启动横幅
        "-i", video_path,
        "-stream_loop", "-1",
        "-i", bgm_path,
    ]

    if video_has_audio:
        # 混合原声和BGM
        filter_complex = f"[0:a]volume=1.0[orig_a]; [1:a]volume={volume_factor}[bgm]; [orig_a][bgm]amix=inputs=2:duration=first:normalize=0[a_out]"
        map_audio = "[a_out]"
    else:
        # 视频无原声，仅处理BGM
        filter_complex = f"[1:a]volume={volume_factor}[a_out]"
        map_audio = "[a_out]"

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", f"{map_audio}",
        "-c:v", "copy",  # 直接复制视频流，速度快
        "-c:a", "aac",  # 使用高质量的 AAC 音频编码器
        "-b:a", "192k",  # 设置音频比特率为 192k
        "-shortest",
        output_path
    ])

    # 3. 执行命令
    print("--------------------------------------------------")
    print("开始为视频添加背景音乐...")
    print(f"  输入视频: {video_path}")
    print(f"  BGM: {bgm_path}")
    print(f"  输出视频: {output_path}")
    print(f"  BGM 音量: {volume_percentage}%")
    # 使用 ' '.join 打印一个易于阅读和复制的命令版本
    print(f"  执行的 FFmpeg 命令:\n  {' '.join(cmd)}")
    print("--------------------------------------------------")

    try:
        # 使用 Popen 以便实时看到 ffmpeg 的输出，对于长时间任务更友好
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in process.stdout:
            # 你可以在这里解析ffmpeg的进度，但为了简单起见，我们只打印它
            print(line.strip())

        process.wait()  # 等待命令执行完成

        if process.returncode != 0:
            # 如果ffmpeg返回了错误码
            raise subprocess.CalledProcessError(process.returncode, cmd)

        print(f"\n处理完成！带BGM的视频已保存至: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        print("\nFFmpeg 执行失败!")
        print(f"错误码: {e.returncode}")
        # 由于我们重定向了stderr，错误信息会在上面的循环中打印出来
        return False
    except Exception as e:
        print(f"\n发生未知错误: {e}")
        return False


# --- 使用示例 ---
if __name__ == "__main__":
    video_file = "final_video_ffmpeg.mp4"
    bgm_file = "background_music.mp3"
    output_file = "output_with_bgm.mp4"

    # 调用函数
    try:
        # 示例1: 基本调用，使用默认音量 20%
        success = add_bgm_to_video(video_file, bgm_file, output_file)

        # 示例2: 自定义音量为 50%
        # output_file_loud = "output_with_bgm_50_percent.mp4"
        # success = add_bgm_to_video(video_file, bgm_file, output_file_loud, volume_percentage=50)

        if success:
            print("\n你可以播放输出文件来验证效果。")
            # 简单验证一下输出文件信息
            print("\n--- 输出文件信息 ---")
            info = probe_video(output_file)
            print(json.dumps(info, indent=2))

    except (FileNotFoundError, ValueError) as e:
        print(f"发生错误: {e}")