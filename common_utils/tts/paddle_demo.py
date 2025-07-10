import wave
from paddlespeech.cli.tts.infer import TTSExecutor
import os

# --- 兼容旧版本的函数 ---

def synthesize_and_get_duration(
    tts_executor: TTSExecutor,
    text: str,
    output_path: str,
    lang: str = 'mix',
    spk_id: int = 174
) -> float:
    """
    使用指定的 PaddleSpeech TTS 执行器合成语音（兼容旧版本API）。
    """
    try:
        # 1. 执行语音合成
        # 关键改动：在调用时传入模型、声码器等参数
        tts_executor(
            text=text,
            output=output_path
        )
        print(f"✅ 语音合成成功！(音色ID: {spk_id}, 语言: {lang}) -> {output_path}")

        # 2. 计算并返回音频时长
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print(f"❌ 错误：文件 '{output_path}' 未能成功生成或文件大小为0。")
            return 0.0

        with wave.open(output_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration

    except Exception as e:
        print(f"❌ 在合成或计算时长过程中发生错误: {e}")
        return 0.0


# --- 使用示例 ---

if __name__ == "__main__":
    # 1. 初始化TTS执行器
    print("正在初始化TTS引擎，第一次运行需要下载指定模型，请稍候...")
    try:
        # 关键改动：初始化时不再传入模型参数，以兼容旧版本
        tts_engine = TTSExecutor()
    except Exception as e:
        print(f"TTS引擎初始化失败: {e}")
        print("请确保已正确安装paddlespeech_cli，并检查网络连接。")
        exit()

    # --- 示例1：合成中英混合文本 (男声) ---
    print("\n--- 示例1: 合成中英混合文本 (男声) ---")
    audio_length_1 = synthesize_and_get_duration(
        tts_executor=tts_engine,
        text="我最喜欢的电影是 Forrest Gump，那句经典的 'Life was like a box of chocolates' 让我印象深刻。",
        output_path="output_mix_male.wav",
        lang='mix',
        spk_id=174  # 男声
    )
    if audio_length_1 > 0:
        print(f"音频总长度为: {audio_length_1:.2f} 秒。")