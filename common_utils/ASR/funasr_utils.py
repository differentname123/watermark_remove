import os
from funasr import AutoModel

from common_utils.common_utils import save_json


def run_funasr(audio_path):
    """
    使用 FunASR 执行语音识别（中文为主），输出带时间戳的结果
    """

    base_dir = 'output'
    output_file_name = os.path.splitext(audio_path)[0] + "_asr_funasr.json"
    output_file = os.path.join(base_dir, output_file_name)
    if os.path.exists(output_file):
        return output_file
    # 选择推荐的高精度模型
    model = AutoModel(
        # model= "iic/SenseVoiceSmall",   # 中英混合、效果好
        model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",  # 中英混合、效果好
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",  # 语音活动检测
        # punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",  # 标点恢复
        spk_model=None,  # 不需要说话人分离
    )

    # 推理
    res = model.generate(
        input=audio_path,
        batch_size_s=30,  # 每次处理 300 秒以内的片段
    )

    final_res = []
    # 输出结果
    for i, r in enumerate(res):
        text = r["text"]
        word_list = text.split(' ')
        timestamp_list = r["timestamp"]
        # 判断长度是否一致
        if len(word_list) != len(timestamp_list):
            raise ValueError(f"Word list and timestamp list length mismatch in result {i+1}.")
        # 将结果转换为字典列表
        final_res = [
            {
                "word": word,
                "start": round(ts[0], 3) if ts[0] is not None else None,
                "end": round(ts[1], 3) if ts[1] is not None else None,
                "probability": 1
            }
            for word, ts in zip(word_list, timestamp_list)
        ]

    save_json(output_file, final_res)
    return output_file


if __name__ == "__main__":
    # 这里换成你的音频文件路径（支持 wav/mp3/m4a/flac）
    audio_file = r"test.wav"
    run_funasr(audio_file)
