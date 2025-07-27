import asyncio
import os
import re
import subprocess
import shlex
import tempfile
from pathlib import Path

# --- 依赖：librosa, soundfile, numpy, edge_tts ---
try:
    import librosa
    import soundfile as sf
    import numpy as np

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ 警告: `librosa` 或 `soundfile` 未安装。静音切除功能将不可用。")
    print("   请运行 `pip install librosa soundfile numpy`。")

import edge_tts

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

# --- 核心音频处理函数 (使用 loudnorm) ---

def process_audio_with_loudnorm(
        input_path: str,
        output_path: str,
        target_loudness: int = -16
) -> bool:
    """
    使用 ffmpeg 的 loudnorm 滤镜对音频进行专业响度归一化。
    这是实现洪亮、饱满且音量一致的推荐方法。

    Args:
        input_path (str): 输入音频文件路径。
        output_path (str): 输出音频文件路径。
        target_loudness (int): 目标响度，单位为 LUFS。-16 是播客/流媒体的常用值。

    Returns:
        bool: 成功返回 True，失败返回 False。
    """
    if not Path(input_path).exists():
        print(f"❌ 错误: 输入文件 '{input_path}' 不存在。")
        return False

    # loudnorm 滤镜有两个阶段，但我们可以用一条命令让 ffmpeg 自动处理
    # I: Integrated Loudness (目标综合响度)
    # LRA: Loudness Range (响度范围)
    # TP: True Peak (真实峰值，防止削波)
    cmd = (
        f'ffmpeg -y -hide_banner -i "{input_path}" '
        f'-af "loudnorm=I={target_loudness}:LRA=7:TP=-1.5" '
        f'"{output_path}"'
    )

    try:
        # 使用 DEVNULL 来隐藏 ffmpeg 的大量输出，只在出错时打印
        result = subprocess.run(
            shlex.split(cmd),
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        return True
    except subprocess.CalledProcessError as e:
        print("❌ ffmpeg 处理失败！")
        print(f"   命令: {e.cmd}")
        print(f"   错误输出:\n{e.stderr}")
        return False


# --- 重构后的主生成函数 ---

def generate_audio_and_get_duration_sync(
        text: str,
        output_filename: str,
        voice_name: str = "zh-CN-XiaoxiaoNeural",
        trim_silence: bool = True,
        target_loudness: int = -14
) -> float | None:
    """
    【重构版本】生成、处理并保存高质量音频。

    流程:
    1. 使用 edge-tts 生成原始 MP3。
    2. 使用 librosa 加载并切除首尾静音（如果启用）。
    3. 将处理后的音频保存为临时的 WAV 文件（无损格式，适合处理）。
    4. 使用 ffmpeg 的 loudnorm 对 WAV 文件进行响度归一化。
    5. 返回最终音频的时长。
    """
    if trim_silence and not LIBROSA_AVAILABLE:
        print("❌ 错误: 请求了静音切除，但 `librosa` 不可用。任务中止。")
        return None

    output_path = Path(output_filename)
    # 使用临时文件来处理中间步骤，避免格式混乱
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        raw_mp3 = temp_dir / "raw.mp3"
        trimmed_wav = temp_dir / "trimmed.wav"

        # 1. 生成原始音频
        async def _generate_task():
            communicate = edge_tts.Communicate(text, voice_name, volume='+100%')
            await communicate.save(str(raw_mp3))

        try:
            asyncio.run(_generate_task())

            # 2. 加载音频并进行静音切除
            y, sr = librosa.load(str(raw_mp3), sr=None)

            if trim_silence:
                y_trimmed, index = librosa.effects.trim(y, top_db=25)
                if len(y) - len(y_trimmed) > sr * 0.1:  # 如果切除超过0.1秒
                    y = y_trimmed
                else:
                    print("ⓘ 未检测到明显静音，跳过切除。")

            # 在结尾增加一点静音缓冲，防止声音戛然而止
            pad_samples = int(sr * 0.2)
            y = np.concatenate([y, np.zeros(pad_samples)])

            # 3. 保存为临时的 WAV 文件
            sf.write(str(trimmed_wav), y, sr)
            # 4. 进行响度归一化
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            success = process_audio_with_loudnorm(str(trimmed_wav), str(output_path), target_loudness)
            print(_get_volume_info(output_path), text)

            if success:
                # 重新加载最终文件以获取准确时长
                y_final, sr_final = librosa.load(str(output_path), sr=None)
                return librosa.get_duration(y=y_final, sr=sr_final)
            else:
                return None

        except Exception as e:
            import traceback
            print(f"❌ 在主流程中发生严重错误: {e}")
            traceback.print_exc()
            return None


# ================================================================
# 演示代码
# ================================================================
if __name__ == "__main__":
    print("🚀 演示使用专业响度归一化 (`loudnorm`) 生成高质量语音。\n")

    text_list = [
        "你好，这是一个测试。",
        "现在我们来听听，经过专业处理后，声音是不是非常洪亮，而且不同句子之间的音量保持了一致？",
        "即使是比较轻柔的句子，听起来也很有力。",
    ]

    for i, text in enumerate(text_list):
        output_file = f"output_processed_{i + 1}.mp3"
        print(f"--- 正在生成第 {i + 1}/{len(text_list)} 个文件: {output_file} ---")

        duration = generate_audio_and_get_duration_sync(
            text=text,
            output_filename=output_file,
            voice_name="zh-CN-XiaoxiaoNeural",  # 可以换成你喜欢的语音
            trim_silence=True,
            target_loudness=-14  # 这是关键参数，可以调整，-14更响，-18更轻
        )

        if duration:
            print(f"🎉 文件 '{output_file}' 生成成功，时长: {duration:.2f} 秒。\n")
        else:
            print(f"🔥 文件 '{output_file}' 生成失败。\n")

    print("所有文件生成完毕。请试听 `output_processed_*.mp3` 文件，对比效果。")