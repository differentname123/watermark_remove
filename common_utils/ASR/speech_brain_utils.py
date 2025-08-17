import os
import time
from faster_whisper import WhisperModel

# ========== 配置 ==========
MODEL_SIZE = "large-v3"
DEVICE = "cuda"          # 无 GPU 用 "cpu"
# 准确率优先：显存允许建议 float32；速度优先用 float16
COMPUTE_TYPE = "float16"
AUDIO_FILE = "mix.mp3"

# ========== 可选：快速音频健检 ==========
if not os.path.exists(AUDIO_FILE):
    raise FileNotFoundError(f"音频文件不存在: {AUDIO_FILE}")

# ========== 1. 加载模型 ==========
model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

# ========== 2. 执行语音识别（准确率导向参数） ==========
print(f"开始识别音频: {AUDIO_FILE}")
start_time = time.time()

segments, info = model.transcribe(
    AUDIO_FILE,
    task="transcribe",
    language="zh",                 # 已知为中文时强制指定
    beam_size=10,                  # 5–10 之间权衡
    patience=1.0,                  # beam search 容忍度
    temperature=(0.0, 0.2, 0.4),   # 失败回退到更高温度，减少幻觉
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
    condition_on_previous_text=True,   # 长段保持连贯；话题跳变大时可阶段性置 False
    initial_prompt=(
        "这是一次中文访谈录音，包含大量人名与地名。请使用简体中文标点。"
        "专有名词示例：OpenAI、字节跳动、华为、上海、北京、深圳。"
    ),
    word_timestamps=True,          # 中文下相当于“字级”时间戳
    vad_filter=True,               # 过滤静音/非语音段
    vad_parameters=dict(
        min_silence_duration_ms=200,  # 根据语速/环境微调
        speech_pad_ms=100
    ),
)

print(f"识别完成，耗时: {time.time() - start_time:.2f} 秒")
print(f"识别到的语言: '{info.language}' (置信度: {info.language_probability:.2f})")
print("-" * 50)

# ========== 3. 打印结果 ==========
full_text = []
for segment in segments:
    print(f"Segment: [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    full_text.append(segment.text.strip())

    # 字/词级时间戳（中文多为单字切分），过滤掉极低置信度或缺失时间戳的条目
    for w in segment.words or []:
        if (w.start is None) or (w.end is None):
            continue
        if (w.probability is not None) and (w.probability < 0.2):
            continue
        print(f"  - [{w.start:.2f}s -> {w.end:.2f}s] '{w.word}' (p={w.probability:.2f})")

print("-" * 50)
print("完整识别结果:")
print("".join(full_text))

# ========== 4. (可选) 保存为 SRT 字幕（字/词级） ==========
def format_time(seconds: float) -> str:
    millis = int(round((seconds - int(seconds)) * 1000))
    minutes, seconds_int = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds_int:02},{millis:03}"

srt_filename = "output.srt"
with open(srt_filename, "w", encoding="utf-8") as f:
    idx = 1
    for segment in segments:
        for w in segment.words or []:
            if (w.start is None) or (w.end is None):
                continue
            # 可选：对低置信度字/词不写入 SRT，避免错误字幕误导
            if (w.probability is not None) and (w.probability < 0.2):
                continue
            f.write(f"{idx}\n")
            f.write(f"{format_time(w.start)} --> {format_time(w.end)}\n")
            f.write(f"{w.word.strip()}\n\n")
            idx += 1
print(f"词/字级时间戳已保存到 SRT 文件: {srt_filename}")