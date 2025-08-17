import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
import os
os.environ['HF_TOKEN'] = "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"   # 临时测试用（生产别这样写）

from faster_whisper import WhisperModel
import time

# ========== 配置 ==========
MODEL_SIZE = "large-v3"  # 可选: "tiny", "base", "small", "medium", "large-v2", "large-v3"
# 对于高质量中文识别，推荐 "large-v3"
DEVICE = "cuda"  # 如果有NVIDIA GPU, 使用 "cuda"; 否则 "cpu"
COMPUTE_TYPE = "float16"  # 在CUDA上使用 "float16" 加速; CPU上使用 "int8"
AUDIO_FILE = "mix_vocal.wav"  # 你的输入音频文件 (确保是 16k Hz)

# 确保音频文件存在
if not os.path.exists(AUDIO_FILE):
    print(f"音频文件不存在: {AUDIO_FILE}")
    # 这里可以添加你之前的 mp3 -> wav 转换代码
    # from pydub import AudioSegment
    # input_mp3 = "mix.mp3"
    # audio = AudioSegment.from_file(input_mp3, format="mp3")
    # audio = audio.set_frame_rate(16000).set_channels(1)
    # audio.export(AUDIO_FILE, format="wav")
    exit()

# ========== 1. 加载模型 ==========
model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

# ========== 2. 执行语音识别 ==========
print(f"开始识别音频: {AUDIO_FILE}")
start_time = time.time()

# word_timestamps=True 是获取词级别时间戳的关键
segments, info = model.transcribe(AUDIO_FILE, beam_size=5, word_timestamps=True)

print(f"识别完成，耗时: {time.time() - start_time:.2f} 秒")
print(f"识别到的语言: '{info.language}' (置信度: {info.language_probability:.2f})")
print("-" * 50)

# ========== 3. 处理并打印结果 ==========
full_text = []
for segment in segments:
    print(f"Segment: [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    full_text.append(segment.text.strip())

    # 打印每个词的时间戳
    for word in segment.words:
        # word 是一个 namedtuple, 包含 word, start, end, probability
        print(f"  - [{word.start:.2f}s -> {word.end:.2f}s] '{word.word}' (p={word.probability:.2f})")

print("-" * 50)
print("完整识别结果:")
print("".join(full_text))


# ========== 4. (可选) 保存为 SRT 字幕文件 ==========
def format_time(seconds):
    """将秒转换为 SRT 时间格式 (HH:MM:SS,ms)"""
    millis = int((seconds - int(seconds)) * 1000)
    # 使用 divmod 进行转换
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


srt_filename = "output.srt"
with open(srt_filename, "w", encoding="utf-8") as f:
    word_count = 1
    for segment in segments:
        for word in segment.words:
            start_srt = format_time(word.start)
            end_srt = format_time(word.end)
            f.write(f"{word_count}\n")
            f.write(f"{start_srt} --> {end_srt}\n")
            f.write(f"{word.word.strip()}\n\n")
            word_count += 1
print(f"词级别时间戳已保存到 SRT 文件: {srt_filename}")