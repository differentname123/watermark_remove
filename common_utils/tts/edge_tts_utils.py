import asyncio
import edge_tts
import os

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

    # edge-tts 本身可以指定输出格式，但我们保持它默认输出 mp3，
    # 然后由 librosa/soundfile 决定最终保存的格式，这样更灵活。
    temp_mp3_filename = os.path.splitext(output_filename)[0] + ".temp.mp3"

    async def _generate_task():
        communicate = edge_tts.Communicate(text, voice_name)
        # 先保存到一个临时的 mp3 文件
        await communicate.save(temp_mp3_filename)

    try:
        asyncio.run(_generate_task())

        y, sr = librosa.load(temp_mp3_filename, sr=None)

        if trim_silence:
            y_trimmed, index = librosa.effects.trim(y, top_db=25)

            original_duration = librosa.get_duration(y=y, sr=sr)
            trimmed_duration = librosa.get_duration(y=y_trimmed, sr=sr)

            if original_duration - trimmed_duration > 0.1:
                print(f"  - 成功: 静音已切除。原时长 {original_duration:.2f}s -> 新时长 {trimmed_duration:.2f}s")
                # 更新音频数据为切除后的
                y = y_trimmed
            else:
                print("  - 信息: 未检测到明显的首尾静音。")
        sf.write(output_filename, y, sr)

        final_duration = librosa.get_duration(y=y, sr=sr)
        return final_duration

    except Exception as e:
        print(f"❌ 在处理过程中发生错误: {e}")
        # 增加对错误的详细回溯
        import traceback
        traceback.print_exc()
        print("--- 任务失败 ---\n")
        return 0.0

    finally:
        # 确保删除临时文件
        if os.path.exists(temp_mp3_filename):
            os.remove(temp_mp3_filename)


# ================================================================
# 下面的演示代码无需任何修改，可以直接运行
# ================================================================
if __name__ == "__main__":

    print("演示如何直接调用一个标准的同步函数，无需 async/await。\n")

    text_with_silence = "我们再次测试，这次换一个稳重的男声，并且直接调用函数。"
    filename_no_trim = "test_librosa_no_trim.mp3"
    filename_trimmed = "test_librosa_trimmed.mp3"

    print("--- 首先，生成一个带有人为静音但不切除的版本 ---")
    duration_no_trim = generate_audio_and_get_duration_sync(
        text=text_with_silence,
        output_filename=filename_no_trim,
        voice_name="zh-CN-YunjianNeural",
        trim_silence=True
    )
    if duration_no_trim is not None:
        print(f"✅ 结果: '{filename_no_trim}' (未切除) 的时长是 {duration_no_trim:.2f} 秒。")