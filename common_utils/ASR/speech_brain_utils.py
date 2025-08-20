# 导入所需的库
import os
import json
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment  # 导入 pydub

from common_utils.common_utils import save_json

# --- 函数封装 ---
def perform_speaker_diarization(input_audio_path: str):
    """
    对输入的音频文件执行说话人日志分析，并将结果保存为 JSON 文件。

    Args:
        input_audio_path (str): 要进行分析的音频文件路径。

    注意:
        - Hugging Face 访问令牌 (AUTH_TOKEN) 和输出目录 (OUTPUT_DIR) 已在函数内部固定。
          如果你需要修改它们，请编辑此函数内部的配置部分。
        - 代理设置 (HTTP_PROXY, HTTPS_PROXY) 也已在函数内部固定。
        - 如果发生严重错误（如模型加载失败、文件未找到），脚本将终止。
    """

    # --- 配置区 (函数内部固定) ---
    # 将你的Hugging Face访问令牌放在这里
    # 你可以在这里获取: https://hf.co/settings/tokens
    # !!! 请将 "HUGGINGFACE_ACCESS_TOKEN_GOES_HERE" 替换为你自己的令牌 !!!
    AUTH_TOKEN = "HUGGINGFACE_ACCESS_TOKEN_GOES_HERE"

    # 输出剪辑的文件夹
    OUTPUT_DIR = "output"

    # 设置代理（如果需要）
    # 注意：此处代理配置是硬编码的，如果不需要或需要其他代理，请修改此部分
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    # ---------------------------

    # 1. 自动检测并设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # 2. 实例化 pipeline
    print("正在加载说话人日志分析模型...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=AUTH_TOKEN # 使用内部固定的 auth_token
        )
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保你的 AUTH_TOKEN 是正确的，并且已经接受了模型的使用条款。")
        # 保持 exit() 以遵循“不擅自修改代码”的指令，这会导致整个脚本终止
        exit()

    # 3. 将 pipeline 移动到设备
    pipeline.to(device)

    # 4. 在音频文件上运行 pipeline
    print(f"开始对 '{input_audio_path}' 进行说话人日志分析...")
    try:
        diarization = pipeline(input_audio_path)
        print("分析完成！")
    except Exception as e:
        print(f"音频文件 '{input_audio_path}' 分析时出错: {e}")
        # 保持 exit()
        exit()

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 5.1 新增：将结果导出为 JSON（毫秒级）
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_ms = int(round(turn.start * 1000))
        end_ms = int(round(turn.end * 1000))
        segments.append({
            "start": start_ms,
            "end": end_ms,
            "speaker": speaker
        })

    json_path = os.path.join(OUTPUT_DIR, "segments_speech.json")
    # 确保 save_json 函数可用
    try:
        # 假设 save_json 函数从 common_utils.common_utils 导入成功
        save_json(json_path, segments)
        print(f"JSON 段信息已保存到 {json_path}")
        return json_path
    except NameError:
        print("错误: 'save_json' 函数未找到。请确保 'common_utils.common_utils' 已正确导入或定义了 'save_json' 函数。")
        # 保持 exit()
        exit()


    # ==================================================================
    # 根据分析结果切割并合并同一个说话人的音频
    # ==================================================================
    print("\n开始加载原始音频并准备切片...")
    try:
        audio = AudioSegment.from_file(input_audio_path)
    except FileNotFoundError:
        print(f"错误：找不到音频文件 '{input_audio_path}'。请检查文件名和路径。")
        # 保持 exit()
        exit()
    except Exception as e: # Catch other potential pydub errors
        print(f"加载音频文件 '{input_audio_path}' 时出错: {e}")
        # 保持 exit()
        exit()


    speaker_segments = {}
    print("正在处理和合并每个说话人的音频片段...")
    for seg in segments:
        start_ms = seg["start"]
        end_ms = seg["end"]
        speaker = seg["speaker"]

        # 确保切片不会超出音频的实际长度，并且 start_ms < end_ms
        if end_ms > len(audio):
            end_ms = len(audio)
        if start_ms >= end_ms: # Skip invalid segments
            continue

        snippet = audio[start_ms:end_ms]
        if speaker not in speaker_segments:
            speaker_segments[speaker] = AudioSegment.empty()
        speaker_segments[speaker] += snippet

    # 原始代码中这部分被注释掉了，我将保持注释状态，遵循“不擅自修改代码”的指令。
    # print(f"处理完成！正在将合并后的音频导出到 '{OUTPUT_DIR}' 文件夹...")
    # for speaker, combined_segment in speaker_segments.items():
    #     output_filename = f"{speaker.replace(' ', '_')}.wav"
    #     output_path = os.path.join(OUTPUT_DIR, output_filename)
    #     print(f"  -> 正在导出 {output_path}...")
    #     combined_segment.export(output_path, format="wav")

    print("\n所有说话人的音频已成功分离并合并（未导出为文件，因原始代码该部分被注释）！")
    print(f"JSON 结果已保存到 {json_path}")
    # print(f"每个说话人的合并音频片段已在内存中生成，并存储在 'speaker_segments' 字典中。")
    # 原代码没有返回值，函数也不需要返回值。

# --- 如何使用函数 ---
if __name__ == "__main__":
    audio_file_to_process = "test.wav" # <<< 请在这里填写你的音频文件路径

    print("--- 开始执行说话人日志分析 ---")
    # 调用封装好的函数
    # 注意：如果函数内部发生错误，如令牌无效，脚本会直接退出。
    perform_speaker_diarization(
        input_audio_path=audio_file_to_process
    )
    print("\n--- 说话人日志分析执行完毕 ---")