# 导入所需的库
import os
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment  # 导入 pydub

# --- 配置区 ---
# 将你的Hugging Face访问令牌放在这里
# 你可以在这里获取: https://hf.co/settings/tokens
AUTH_TOKEN = "HUGGINGFACE_ACCESS_TOKEN_GOES_HERE"
# 输入的音频文件
INPUT_AUDIO = "test.wav"
# 输出剪辑的文件夹
OUTPUT_DIR = "speaker_clips"

# 设置代理（如果需要）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 1. 自动检测并设置设备
# 检查是否有可用的 CUDA (NVIDIA GPU) 设备，否则回退到 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# 2. 实例化 pipeline
# 使用 from_pretrained 加载预训练模型
print("正在加载说话人日志分析模型...")
try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=AUTH_TOKEN)
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    print("请确保你的 AUTH_TOKEN 是正确的，并且已经接受了模型的使用条款。")
    exit()

# 3. 将 pipeline 移动到 GPU 设备
pipeline.to(device)

# 4. 在音频文件上运行 pipeline
print(f"开始对 '{INPUT_AUDIO}' 进行说话人日志分析...")
diarization = pipeline(INPUT_AUDIO)
print("分析完成！")

# 5. 将结果以 RTTM 格式写入磁盘 (可选，但有助于调试)
rttm_path = os.path.join(OUTPUT_DIR, "audio.rttm")
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 确保输出目录存在
with open(rttm_path, "w") as rttm:
    diarization.write_rttm(rttm)
print(f"分析结果已保存到 {rttm_path} 文件。")

# ==================================================================
# 新增功能：根据分析结果切割并合并同一个说话人的音频
# ==================================================================

print("\n开始加载原始音频并准备切片...")
# 1. 使用 pydub 加载原始音频文件
try:
    audio = AudioSegment.from_file(INPUT_AUDIO)
except FileNotFoundError:
    print(f"错误：找不到音频文件 '{INPUT_AUDIO}'。请检查文件名和路径。")
    exit()

# 2. 创建一个字典来按说话人存储合并后的音频
speaker_segments = {}

print("正在处理和合并每个说话人的音频片段...")
# 3. 遍历 diarization 结果
# diarization.itertracks(yield_label=True) 会产出 (segment, track, speaker_label)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    # turn.start 和 turn.end 是以秒为单位的浮点数
    # pydub 使用毫秒进行切片，所以需要转换
    start_ms = int(turn.start * 1000)
    end_ms = int(turn.end * 1000)

    # 4. 从原始音频中切出片段
    segment = audio[start_ms:end_ms]

    # 5. 将切片添加到对应说话人的集合中
    if speaker not in speaker_segments:
        # 如果是第一次遇到这个说话人，创建一个空的 AudioSegment
        speaker_segments[speaker] = AudioSegment.empty()

    # 将当前片段拼接到这个说话人的音频末尾
    speaker_segments[speaker] += segment

# 6. 导出每个说话人合并后的音频
print(f"处理完成！正在将合并后的音频导出到 '{OUTPUT_DIR}' 文件夹...")
for speaker, combined_segment in speaker_segments.items():
    # 创建一个安全的文件名，例如 'SPEAKER_00.wav'
    output_filename = f"{speaker.replace(' ', '_')}.wav"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"  -> 正在导出 {output_path}...")
    # 以 wav 格式导出文件
    combined_segment.export(output_path, format="wav")

print("\n所有说话人的音频已成功分离并合并！")