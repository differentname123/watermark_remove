import asyncio
import os
import re
import subprocess
import shlex
import tempfile
import time
import traceback
from pathlib import Path
import random

from common_utils.common_utils import read_json

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
all_voice_name_list = [
        "zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural", "zh-CN-YunjianNeural", "zh-CN-YunxiNeural",
        "zh-CN-YunxiaNeural", "zh-CN-YunyangNeural", "zh-CN-liaoning-XiaobeiNeural", "zh-CN-shaanxi-XiaoniNeural"
    ]



def parse_tts_filename(output_file, voice_name_list):
    """
    解析 TTS 文件名。
    如果解析成功，返回 (原始voice_name, pitch, rate)。
    如果文件名格式不对，或者找不到对应的 voice_name，voice_name 将返回 None，不抛出错误。
    """

    # 1. 建立 [简称 -> 完整 voice_name] 的映射字典
    voice_map = {}
    if voice_name_list:
        for original_name in voice_name_list:
            # 你的生成逻辑：取横杠最后一段，去掉 Neural
            short_name = original_name.split('-')[-1].replace('Neural', '')
            voice_map[short_name] = original_name

    # 2. 获取文件名（去除路径）
    filename = os.path.basename(output_file)

    # 3. 正则提取
    # 匹配：说话人(内容)_音调(内容)_语速(内容)_句子
    pattern = r"说话人(?P<short_name>[^_]+)_音调(?P<pitch>[^_]+)_语速(?P<rate>[^_]+)_句子"

    match = re.search(pattern, filename)

    if match:
        short_name = match.group("short_name")
        pitch = match.group("pitch")
        rate = match.group("rate")

        # 4. 查表 (如果找不到，get 方法默认返回 None)
        original_voice_name = voice_map.get(short_name)

        # 返回结果：(可能为None的name, pitch, rate)
        return original_voice_name, pitch, rate
    else:
        # 文件名格式完全不匹配，全部返回 None
        return None, None, None


def get_voice_info(tags):
    # 1. 把输入的 tags 字典里的所有列表展平，变成一个集合
    user_tags = {t for sublist in tags.values() for t in sublist}

    # 2. 读取数据
    voice_info = read_json(r"W:\project\python_project\watermark_remove\content_community\app\voice_info.json")

    # 3. 一行代码计算所有音色的匹配分
    # 逻辑：遍历音色 -> 展平该音色的所有标签 -> 计算和用户标签的交集数量
    scores = [
        (name, len(user_tags & {t for v in data.values() for t in v}))
        for name, data in voice_info.items()
    ]

    # 4. 排序(降序) -> 取前3 -> 随机选一个 -> 返回名字
    # if scores 用于防止文件为空报错
    final_voice_name = random.choice(sorted(scores, key=lambda x: x[1], reverse=True)[:2])[0] if scores else None
    original_voice_name, pitch, rate = parse_tts_filename(final_voice_name, all_voice_name_list)
    final_voice_infp = {
        "voice_name": original_voice_name,
        "pitch": pitch,
        "rate": rate
    }
    print(f"ⓘ 根据标签推荐音色: {final_voice_infp}")
    return final_voice_infp

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
        target_loudness: int = -14,
        rate: str = "+10%",
        pitch: str = '+10Hz',
) -> float | None:
    """
    【重构版本】生成、处理并保存高质量音频（带重试机制）。

    流程:
    0. 如果文本为空，则直接生成1秒静音并返回。
    1. 使用 edge-tts 生成原始 MP3。
    2. 使用 librosa 加载并切除首尾静音（如果启用）。
    3. 将处理后的音频保存为临时的 WAV 文件（无损格式，适合处理）。
    4. 使用 ffmpeg 的 loudnorm 对 WAV 文件进行响度归一化。
    5. 返回最终音频的时长。
    *  新增: 如果在步骤1-4中发生任何异常，将重试最多3次，每次重试前等待5秒。
    """
    start_time = time.time()
    # ==================== 处理空文本 (无变化) ====================
    if not text or not text.strip():
        print(f"ⓘ 文本为空，正在为 '{output_filename}' 生成 1 秒静音。")
        try:
            duration_seconds = 1.0
            sample_rate = 24000
            silent_audio = np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)
            output_path = Path(output_filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), silent_audio, sample_rate)
            print(f"✓ 成功生成静音文件: {output_filename}")
            return duration_seconds
        except Exception as e:
            print(f"❌ 在生成静音文件时发生错误: {e}")
            traceback.print_exc()
            return None
    # ===============================================================

    if trim_silence and not LIBROSA_AVAILABLE:
        print("❌ 错误: 请求了静音切除，但 `librosa` 不可用。任务中止。")
        return None

    output_path = Path(output_filename)
    # 使用临时文件来处理中间步骤，避免格式混乱
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        raw_mp3 = temp_dir / "raw.mp3"
        trimmed_wav = temp_dir / "trimmed.wav"

        # ==================== 新增: 重试逻辑 ====================
        max_retries = 3
        retry_delay_seconds = 5

        for attempt in range(max_retries):
            try:
                # 1. 生成原始音频
                async def _generate_task():
                    communicate = edge_tts.Communicate(text, voice_name, volume='+100%', rate=rate, pitch=pitch)
                    await communicate.save(str(raw_mp3))

                # 在新的尝试中，确保异步事件循环是干净的
                # asyncio.run() 每次都会创建并关闭一个新的事件循环，所以这里没问题
                asyncio.run(_generate_task())

                # 2. 加载音频并进行静音切除
                y, sr = librosa.load(str(raw_mp3), sr=None)

                if trim_silence:
                    y_trimmed, index = librosa.effects.trim(y, top_db=25)
                    # 只有在切除的长度有意义时才使用
                    if len(y) - len(y_trimmed) > sr * 0.1:
                        y = y_trimmed
                    else:
                        print("ⓘ 未检测到明显静音，跳过切除。")

                    # 在结尾增加一点静音缓冲，防止声音戛然而止
                    pad_samples = int(sr * 0.2)
                    y = np.concatenate([y, np.zeros(pad_samples)])

                # 3. 保存为临时的 WAV 文件
                sf.write(str(trimmed_wav), y, sr)

                # 4. 进行响度归一化
                output_path.parent.mkdir(parents=True, exist_ok=True)
                success = process_audio_with_loudnorm(str(trimmed_wav), str(output_path), target_loudness)

                if not success:
                    # 如果 loudnorm 明确返回失败，我们可以主动引发异常来触发重试
                    raise RuntimeError("ffmpeg loudnorm 处理失败。")

                print(
                    f'✓ 音频生成完成：{_get_volume_info(output_path)}  "{text}" (耗时 {time.time() - start_time:.2f} 秒)')

                # 5. 成功，返回最终时长
                y_final, sr_final = librosa.load(str(output_path), sr=None)
                return librosa.get_duration(y=y_final, sr=sr_final)

            except Exception as e:
                print(f"❌ 尝试 {attempt + 1}/{max_retries} 失败: {e}")
                if attempt < max_retries - 1:
                    print(f"ⓘ 等待 {retry_delay_seconds} 秒后重试...")
                    time.sleep(retry_delay_seconds)
                else:
                    print(f"❌ 所有 {max_retries} 次尝试均失败。")
                    traceback.print_exc()
                    return None  # 在所有重试失败后返回 None
        # ===============================================================

    # 理论上代码不会执行到这里，因为循环要么成功返回，要么在最后一次失败时返回。
    # 但为了代码健壮性，在函数末尾保留一个返回。
    return None


async def list_chinese_voices():
    # 获取所有语音包
    voices = await edge_tts.list_voices()

    # 筛选包含 "zh-" 的语音 (包括简体、繁体、粤语等)
    chinese_voices = [v for v in voices if "zh-CN" in v["ShortName"]]

    print(f"{'语音代号 (Voice ID)':<35} | {'性别':<5} | {'区域/名称'}")
    print("-" * 80)

    for v in chinese_voices:
        # 提取关键信息
        voice_id = v["ShortName"]
        gender = v["Gender"]
        # FriendlyName 通常包含 "Microsoft Server Speech Text to Speech Voice..." 比较长，这里做简化展示
        friendly_name = v["FriendlyName"].split(" - ")[1] if " - " in v["FriendlyName"] else v["FriendlyName"]

        print(f"{voice_id:<35} | {gender:<5} | {friendly_name}")


# ================================================================
# 演示代码
# ================================================================
if __name__ == "__main__":
    # asyncio.run(list_chinese_voices())


    print("🚀 演示使用专业响度归一化 (`loudnorm`) 生成高质量语音。\n")

    text_list = [
        # "2024年4月14日，上海的天气十分晴朗，许多市民选择在世纪公园散步，享受这难得的春光。",
        # "你答应过我的，为什么现在又变了？……我真不知道该怎么面对这一切。",
        "请别担心，放慢脚步，用心感受每一个瞬间，你会发现，真正的美好其实就在身边。",
    ]
    pitch_list = ['+0Hz', '+10Hz', '+20Hz', '+30Hz', '+40Hz', '+50Hz', '+60Hz', '+70Hz']
    rate_list = ['+0%', '+10%', '+20%', '+30%', '+40%', '+50%', '+60%']
    voice_name_list = [
        "zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural", "zh-CN-YunjianNeural", "zh-CN-YunxiNeural",
        "zh-CN-YunxiaNeural", "zh-CN-YunyangNeural", "zh-CN-liaoning-XiaobeiNeural", "zh-CN-shaanxi-XiaoniNeural"
    ]
    for voice_name in voice_name_list:
        for pitch in pitch_list:
            for rate in rate_list:
                for i, text in enumerate(text_list):
                    output_file = f"tts_output/说话人{voice_name.split('-')[-1].replace('Neural','')}_音调{pitch}_语速{rate}_句子{i + 1}.mp3"
                    print(f"--- 正在生成第 {i + 1}/{len(text_list)} 个文件: {output_file} ---")
                    if os.path.exists(output_file):
                        print(f"⚠️ 文件已存在，跳过: {output_file}\n")
                        continue

                    duration = generate_audio_and_get_duration_sync(
                        text=text,
                        output_filename=output_file,
                        voice_name=voice_name,  # 可以换成你喜欢的语音
                        trim_silence=True,
                        target_loudness=-14,  # 这是关键参数，可以调整，-14更响，-18更轻
                        pitch=pitch,
                        rate=rate,
                    )

            if duration:
                print(f"🎉 文件 '{output_file}' 生成成功，时长: {duration:.2f} 秒。\n")
            else:
                print(f"🔥 文件 '{output_file}' 生成失败。\n")

    print("所有文件生成完毕。请试听 `output_processed_*.mp3` 文件，对比效果。")