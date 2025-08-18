import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HF_TOKEN'] = "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"

from faster_whisper import WhisperModel
import time

# ========== 配置 ==========
MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
AUDIO_FILE = "test.wav"
# AUDIO_FILE = "mix.mp3"


# 确保音频文件存在
if not os.path.exists(AUDIO_FILE):
    print(f"音频文件不存在: {AUDIO_FILE}")
    exit()

# ========== 1. 加载模型 ==========
print("正在加载模型...")
model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)
print("模型加载完成。")

# ========== 2. 执行语音识别 ==========
print(f"开始识别音频: {AUDIO_FILE}")
start_time = time.time()

segments, info = model.transcribe(AUDIO_FILE, beam_size=5, word_timestamps=True

                                  )

# ！！！关键修复：将生成器转换为列表，以便多次使用 ！！！
segments = list(segments)

print(f"识别完成，耗时: {time.time() - start_time:.2f} 秒")
print(f"识别到的语言: '{info.language}' (置信度: {info.language_probability:.2f})")
print("-" * 50)

# ========== 3. 处理并打印结果 ==========
full_text = []
# 现在可以安全地遍历列表了
for segment in segments:
    print(f"Segment: [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    full_text.append(segment.text.strip())

    if segment.words:
        for word in segment.words:
            print(f"  - [{word.start:.2f}s -> {word.end:.2f}s] '{word.word}' (p={word.probability:.2f})")
    else:
        print("  - (该片段没有词级别时间戳)")


print("-" * 50)
print("完整识别结果:")
print("".join(full_text))


# ========== 4. (可选) 保存为 SRT 字幕文件 ==========
def format_time(seconds):
    """将秒转换为 SRT 时间格式 (HH:MM:SS,ms)"""
    millis = int((seconds - int(seconds)) * 1000)
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


srt_filename = AUDIO_FILE.replace(".wav", ".srt").replace(".mp3", ".srt")
with open(srt_filename, "w", encoding="utf-8") as f:
    subtitle_index = 1
    # 再次遍历同一个列表，这次是写入文件
    for segment in segments:
        if segment.words:
            for word in segment.words:
                start_srt = format_time(word.start)
                end_srt = format_time(word.end)
                f.write(f"{subtitle_index}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{word.word.strip()}\n\n")
                subtitle_index += 1

print(f"词级别时间戳已保存到 SRT 文件: {srt_filename}")