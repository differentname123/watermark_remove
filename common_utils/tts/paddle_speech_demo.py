import os
import wave
import subprocess
import shutil
from paddlespeech.cli.tts.infer import TTSExecutor



def synthesize_and_get_duration(
        tts_executor: TTSExecutor,
        text: str,
        output_path: str,
        speed: float = 1.5
) -> float:
    """
    使用指定的 TTS 执行器合成语音，并可选择性地调整语速。

    Args:
        tts_executor (TTSExecutor): PaddleSpeech TTS 的执行器实例。
        text (str): 需要合成的文本。
        output_path (str): 最终音频的保存路径。
        lang (str, optional): 语言。默认为 'mix'。
        spk_id (int, optional): 音色ID。默认为 174。
        speed (float, optional): 语音的倍速。默认为 2.0 (即2倍速)。
                                 设置为 1.0 表示原始速度。

    Returns:
        float: 最终生成音频文件的时长（秒）。

    Raises:
        RuntimeError: 当 ffprobe 或 ffmpeg 执行失败时抛出。
    """
    # 1. 定义一个临时文件路径，用于存放原始速度的音频
    temp_output_path = output_path.replace('.wav', '_temp.wav')
    if temp_output_path == output_path:  # 防止文件名不含.wav导致路径相同
        temp_output_path = output_path + ".temp"

    try:
        # 2. 执行原始语音合成，生成到临时文件
        tts_executor(
            text=text,
            output=temp_output_path
        )
        print(f"✅ 原始语音合成成功！{text} -> {temp_output_path}")

        if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) == 0:
            print(f"❌ 错误：临时文件 '{temp_output_path}' 未能成功生成或文件大小为0。")
            return 0.0

        # 3. 如果需要变速，则调用 ffmpeg 处理
        if speed != 1.0:
            print(f"🚀 正在进行 {speed}x 倍速处理...")
            cmd = [
                "ffmpeg",
                "-i", temp_output_path,
                "-filter:a", f"atempo={speed}",
                "-y",
                "-v", "error",
                output_path
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg 倍速处理失败：{proc.stderr.strip()}")
            print(f"✅ 倍速处理成功！-> {output_path}")
        else:
            # 如果是1倍速，直接移动文件，效率更高
            print("ℹ️ 速度为 1.0x，无需处理，直接移动文件。")
            shutil.move(temp_output_path, output_path)

        # 4. 计算并返回最终音频文件的时长
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print(f"❌ 错误：最终文件 '{output_path}' 未能成功生成或文件大小为0。")
            return 0.0

        with wave.open(output_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration

    except Exception as e:
        print(f"❌ 在合成或处理过程中发生错误: {e}")
        return 0.0

    finally:
        # 5. 无论成功与否，都尝试删除临时文件
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            # print(f"🧹 已清理临时文件: {temp_output_path}")


# --- 使用示例 ---
if __name__ == '__main__':
    # 确保 ffmpeg 在系统 PATH 中
    # 模拟一个 TTS 执行器
    mock_tts_executor = TTSExecutor()

    # 创建一个输出目录
    output_dir = "tts_output"
    os.makedirs(output_dir, exist_ok=True)

    text_to_synthesize = "你好，welcome to 使用飞桨语音合成工具，这是一个测试句子。"

    # 示例1：使用默认的2倍速
    print("\n--- 示例1: 默认2倍速 ---")
    output_file_2x = os.path.join(output_dir, "test_audio_2x.wav")
    duration_2x = synthesize_and_get_duration(
        mock_tts_executor,
        text_to_synthesize,
        output_file_2x
    )
    print(f"最终音频 '{output_file_2x}' 时长: {duration_2x:.2f} 秒")