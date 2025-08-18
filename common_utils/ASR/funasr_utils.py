import os
from funasr import AutoModel

def run_asr(audio_path):
    """
    使用 FunASR 执行语音识别（中文为主），输出带时间戳的结果
    """
    # 选择推荐的高精度模型
    model = AutoModel(
        # model= "iic/SenseVoiceSmall",   # 中英混合、效果好
        model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",  # 中英混合、效果好
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",  # 语音活动检测
        punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",  # 标点恢复
        spk_model=None,  # 不需要说话人分离
    )

    # 推理
    res = model.generate(
        input=audio_path,
        batch_size_s=300,  # 每次处理 300 秒以内的片段
    )

    # 输出结果
    for i, r in enumerate(res):
        text = r["text"]
        segments = r.get("segments", [])
        print(f"\n===== 第 {i+1} 条结果 =====")
        print(f"识别文本: {text}")
        if segments:
            for seg in segments:
                start, end, seg_text = seg["start"], seg["end"], seg["text"]
                print(f"[{start:.2f}s - {end:.2f}s]: {seg_text}")

if __name__ == "__main__":
    # 这里换成你的音频文件路径（支持 wav/mp3/m4a/flac）
    audio_file = r"test.wav"
    run_asr(audio_file)
