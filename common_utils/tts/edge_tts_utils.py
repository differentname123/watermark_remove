import asyncio
import edge_tts
import os
import subprocess
import re
import shlex
import os
import tempfile

# --- 新增依赖：使用 librosa 和 soundfile 进行音频处理 ---
try:
    import librosa
    import soundfile as sf
    import numpy as np

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ 警告: `librosa` 或 `soundfile` 未安装。静音切除功能 (`trim_silence=True`) 将不可用。")
    print("   请运行 `pip install librosa soundfile`。")


import os
import asyncio

import edge_tts
import soundfile as sf

# 如果可选安装了 librosa
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

def _get_volume_info(file_path: str) -> dict:
    """
    调用 ffmpeg + volumedetect，获取音频的 mean_volume 和 max_volume（单位 dB）。
    """
    if not os.path.exists(file_path):
        print(f"警告: 文件 '{file_path}' 不存在，无法获取音量信息。")
        return {"mean_volume": None, "max_volume": None}

    null_device = os.devnull
    cmd = f'ffmpeg -hide_banner -nostats -i "{file_path}" -af volumedetect -f null {null_device}'

    proc = subprocess.run(
        shlex.split(cmd),
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        encoding='utf-8'
    )
    stderr = proc.stderr

    mean_v, max_v = None, None
    for line in stderr.splitlines():
        if m := re.search(r"mean_volume:\s*([-+\d\.]+)\s*dB", line):
            mean_v = float(m.group(1))
        if m := re.search(r"max_volume:\s*([-+\d\.]+)\s*dB", line):
            max_v = float(m.group(1))

    return {"mean_volume": mean_v, "max_volume": max_v}


def maximize_volume(
        input_path: str,
        output_path: str = "output_maximized.mp3"  # 默认输出名，可被覆盖
) -> None:
    """
    通过一个函数调用，实现“先压缩，后标准化”的音量最大化处理。
    日志输出精简为三行核心信息。
    """
    # 1. 分析原始文件
    before_info = _get_volume_info(input_path)
    if before_info['mean_volume'] is None:
        print(f"错误: 无法读取输入文件 '{input_path}'，处理中止。")
        return
    print(f"处理前: mean_volume={before_info['mean_volume']:.2f} dB, max_volume={before_info['max_volume']:.2f} dB")

    # 创建临时文件
    temp_fd, temp_path = tempfile.mkstemp(suffix='.mp3')
    os.close(temp_fd)

    try:
        # 2. 压缩并存入临时文件
        compress_filter = "acompressor=threshold=-20dB:ratio=4:attack=20:release=250"
        cmd_compress = f'ffmpeg -y -hide_banner -nostats -i "{input_path}" -af "{compress_filter}" "{temp_path}"'
        subprocess.run(shlex.split(cmd_compress), check=True, capture_output=True)

        # 3. 分析压缩后文件
        after_compress_info = _get_volume_info(temp_path)
        if after_compress_info['max_volume'] is None:
            print("错误: 压缩步骤失败，处理中止。")
            return
        print(
            f"压缩后: mean_volume={after_compress_info['mean_volume']:.2f} dB, max_volume={after_compress_info['max_volume']:.2f} dB")

        # 4. 标准化至 0dB 并生成最终文件
        gain_db = 0.0 - after_compress_info['max_volume']
        normalize_filter = f"volume={gain_db:.2f}dB"
        cmd_normalize = f'ffmpeg -y -hide_banner -nostats -i "{temp_path}" -af "{normalize_filter}" -c:a libmp3lame -q:a 2 "{output_path}"'
        subprocess.run(shlex.split(cmd_normalize), check=True, capture_output=True)

    finally:
        # 5. 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

    final_info = _get_volume_info(output_path)
    print(f"处理后: mean_volume={final_info['mean_volume']:.2f} dB, max_volume={final_info['max_volume']:.2f} dB")
    print(f"处理完成！文件已保存到: {output_path}")

def generate_audio_and_get_duration_sync(
        text: str,
        output_filename: str,
        voice_name: str = "zh-CN-XiaoxiaoNeural",
        trim_silence: bool = True
) -> float | None:
    """
    【同步版本】使用指定文本和语音合成音频，保存后返回该音频的时长。
    此版本使用 Librosa 进行静音切除，并兼容多种音频格式（如 .mp3, .wav）。

    Args:
        text (str): 需要转换为语音的文本。
        output_filename (str): 保存音频文件的路径和名称。
        voice_name (str, optional): 使用的语音名称。默认为 "zh-CN-XiaoxiaoNeural"。
        trim_silence (bool, optional): 如果为 True，则使用 librosa 切除音频的首尾静音部分。
                                      默认为 False。
    Returns:
        float | None: 成功则返回音频时长（秒），否则返回 None。
    """
    if trim_silence and not LIBROSA_AVAILABLE:
        print("❌ 错误: 请求了静音切除 (`trim_silence=True`)，但 `librosa` 或 `soundfile` 不可用。")
        print("--- 任务失败 ---\n")
        return 0.0

    temp_mp3_filename = os.path.splitext(output_filename)[0] + ".temp.mp3"

    async def _generate_task():
        communicate = edge_tts.Communicate(text, voice_name)
        await communicate.save(temp_mp3_filename)

    try:
        # 1. 生成原始 mp3
        asyncio.run(_generate_task())

        # 2. 读入音频
        y, sr = librosa.load(temp_mp3_filename, sr=None)

        if trim_silence:
            # 3. 切除首尾静音
            y_trimmed, index = librosa.effects.trim(y, top_db=25)
            original_duration = librosa.get_duration(y=y, sr=sr)
            trimmed_duration = librosa.get_duration(y=y_trimmed, sr=sr)

            if original_duration - trimmed_duration > 0.1:
                print(f"  - 成功: 静音已切除。原时长 {original_duration:.2f}s -> 新时长 {trimmed_duration:.2f}s")
                y = y_trimmed

                # 4. 在结尾增加 0.1s 的静音缓冲
                pad_length = int(sr * 0.2)  # 0.1 秒对应的样本数
                y = np.concatenate([y, np.zeros(pad_length)])
                print(f"  - 信息: 在末尾追加 0.1s 缓冲静音。")
            else:
                print("  - 信息: 未检测到明显的首尾静音，无需切除。")

        # 5. 写出最终文件
        sf.write(output_filename, y, sr)
        maximize_volume(output_filename, output_filename)
        final_duration = librosa.get_duration(y=y, sr=sr)
        return final_duration

    except Exception as e:
        print(f"❌ 在处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("--- 任务失败 ---\n")
        return 0.0

    finally:
        # 清理临时文件
        if os.path.exists(temp_mp3_filename):
            os.remove(temp_mp3_filename)



# ================================================================
# 下面的演示代码无需任何修改，可以直接运行
# ================================================================
if __name__ == "__main__":

    print("演示如何直接调用一个标准的同步函数，无需 async/await。\n")

    text_with_silence = "我们再次测试，这次换一个稳重的男声，并且直接调用函数。"
    filename_no_trim = "test_librosa_no_trim2.mp3"
    filename_trimmed = "test_librosa_trimmed.mp3"

    print("--- 首先，生成一个带有人为静音但不切除的版本 ---")
    duration_no_trim = generate_audio_and_get_duration_sync(
        text=text_with_silence,
        output_filename=filename_no_trim,
        voice_name= "zh-CN-YunxiNeural",
        trim_silence=True
    )
    if duration_no_trim is not None:
        print(f"✅ 结果: '{filename_no_trim}' (未切除) 的时长是 {duration_no_trim:.2f} 秒。")