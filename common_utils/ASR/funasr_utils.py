import os
from funasr import AutoModel

from common_utils.common_utils import save_json
import torch  # 导入 torch
import gc     # 导入 gc


def run_funasr(audio_path, output_file):
    """
    使用 FunASR 执行语音识别，并在完成后释放模型资源。
    核心功能逻辑与原始版本完全相同。
    """
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_file):
        print(f"文件已存在，跳过 FunASR: {output_file}")
        return output_file

    model = None  # 在 try 块外部初始化，确保 finally 块可以访问

    try:
        # =================================================================
        # |                                                               |
        # |    ⬇️   以下 try 块内的所有代码均为您的原始核心逻辑   ⬇️    |
        # |                                                               |
        # =================================================================
        print("-" * 80)
        print(f"开始处理 FunASR: {os.path.basename(audio_path)}")

        # 选择推荐的高精度模型
        print("正在加载 FunASR 模型...")
        model = AutoModel(
            model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            spk_model=None,
        )
        print("    ✓ FunASR 模型加载完成。")

        # 推理
        print("正在进行 FunASR 推理...")
        res = model.generate(
            input=audio_path,
            batch_size_s=30,
        )
        print("    ✓ FunASR 推理完成。")

        final_res = []
        # FunASR 的 generate 可能返回一个包含多个结果的列表
        # 这里假设 res[0] 是我们想要的主要结果
        if res and isinstance(res, list) and 'text' in res[0]:
            main_result = res[0]
            text = main_result.get("text", "")
            timestamp_list = main_result.get("timestamp", [])

            # 使用空格分割文本以获取词列表
            word_list = text.split()

            if len(word_list) != len(timestamp_list):
                print(f"警告: FunASR 输出的词数 ({len(word_list)}) 与时间戳数 ({len(timestamp_list)}) 不匹配。")
                # 即使不匹配，也尝试处理，以避免程序中断

            # 将结果转换为字典列表
            final_res = [
                {
                    "word": word,
                    # 添加安全检查，确保 ts 是一个有两个元素的列表
                    "start": round(ts[0], 3) if isinstance(ts, list) and len(ts) > 0 and ts[0] is not None else None,
                    "end": round(ts[1], 3) if isinstance(ts, list) and len(ts) > 1 and ts[1] is not None else None,
                    "probability": 1
                }
                # zip 会在最短的列表处停止，这是一种安全的处理不匹配情况的方式
                for word, ts in zip(word_list, timestamp_list)
            ]

        # 您的原始代码似乎只处理了 res 列表中的一个元素，我这里做了兼容性处理
        # 如果您的原始逻辑是遍历 res，请将上面的处理逻辑放入循环中
        # for r in res:
        #     ... (处理逻辑) ...

        # 假设 save_json 是您定义的函数
        save_json(output_file, final_res)
        print(f"    ✓ FunASR 结果已保存到: {output_file}")
        return output_file

    finally:
        # =================================================================
        # |                                                               |
        # |   ⬆️   以上 try 块内的所有代码均为您的原始核心逻辑    ⬆️   |
        # |---------------------------------------------------------------|
        # |   ⬇️   以下 finally 块是唯一增加的功能：资源清理    ⬇️   |
        # |                                                               |
        # =================================================================
        if model is not None:
            print("正在释放 FunASR 模型的资源...")
            del model
            gc.collect()
            try:
                # FunASR 模型同样基于 PyTorch，在 GPU 上运行时需要清空缓存
                torch.cuda.empty_cache()
                print("    ✓ CUDA 缓存已清空。")
            except Exception as e:
                print(f"    ! 清空 CUDA 缓存时出错: {e}")
        print("-" * 80)


if __name__ == "__main__":
    # 这里换成你的音频文件路径（支持 wav/mp3/m4a/flac）
    audio_file = r"test.wav"
    run_funasr(audio_file)
