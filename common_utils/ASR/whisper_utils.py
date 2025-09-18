import os
import json
import time
from typing import Tuple, List, Dict
from faster_whisper import WhisperModel
import torch  # 导入 torch
import gc     # 导入 gc


def transcribe_words_to_json(audio_file: str, output_file, MODEL_SIZE="large-v3"):
    """
    只接受 audio_file 参数，使用固定的模型配置执行转录，并在完成后释放模型资源。
    输出: (word_items, json_path)
      - word_items: list of {"start": float|None, "end": float|None, "word": str, "probability": float|None}
      - json_path: 保存的 JSON 文件路径（output/<basename>_words.json）

    会打印过程日志：
      - 检查文件、加载模型、转录开始/结束、每个 segment 的摘要、保存文件、总耗时
    """
    # ---------- 固定配置 ----------
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
    BEAM_SIZE = 5
    json_path = output_file
    output_dir = os.path.dirname(json_path)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(json_path):
        print(f"文件已存在，跳过转录: {json_path}")
        return json_path

    model = None  # 在 try 外部初始化 model 变量
    start_all = time.perf_counter()

    try:
        print("-" * 80)
        print(f"开始处理: {os.path.basename(audio_file)} with model {MODEL_SIZE}")

        print(f"[1/5] 检查音频文件：{audio_file}")
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"音频文件不存在: {audio_file}")
        print("    ✓ 文件存在")

        # 加载模型
        t0 = time.perf_counter()
        print(f"[2/5] 正在加载模型 {MODEL_SIZE}（device={DEVICE}, compute_type={COMPUTE_TYPE}）...")
        # 使用 faster-whisper 的 WhisperModel
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        t1 = time.perf_counter()
        print(f"    ✓ 模型加载完成，耗时: {t1 - t0:.2f}s")

        # 转录
        print(f"[3/5] 开始转录（beam_size={BEAM_SIZE}, word_timestamps=True）...")
        t2 = time.perf_counter()
        segments, info = model.transcribe(audio_file, beam_size=BEAM_SIZE, word_timestamps=True)
        segments = list(segments)
        t3 = time.perf_counter()
        print(f"    ✓ 转录完成，耗时: {t3 - t2:.2f}s")
        print(
            f"    识别语言: {getattr(info, 'language', None)}  语言置信度: {getattr(info, 'language_probability', None)}")
        print(f"    共检测到 {len(segments)} 个 segment")

        # 处理并组装词级结果
        print("[4/5] 处理 segment、提取词级时间戳...")
        t_seg_start = time.perf_counter()
        word_items: List[Dict] = []
        for segment in segments:
            seg_words = getattr(segment, "words", None) or []
            for word in seg_words:
                w_start = getattr(word, "start", None)
                w_end = getattr(word, "end", None)
                w_word = getattr(word, "word", "").strip()
                w_prob = getattr(word, "probability", None)

                word_items.append({
                    "start": round(w_start * 1000) if w_start is not None else None,
                    "end": round(w_end * 1000) if w_end is not None else None,
                    "word": w_word,
                    "probability": round(w_prob, 3) if w_prob is not None else None
                })
        t_seg_end = time.perf_counter()
        print(f"    ✓ 处理完成，提取到 {len(word_items)} 个词条，耗时: {t_seg_end - t_seg_start:.2f}s")

        # 保存为 JSON
        print("[5/5] 正在保存为 JSON 文件...")
        t_save_start = time.perf_counter()
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(word_items, jf, ensure_ascii=False, indent=2)
        t_save_end = time.perf_counter()
        print(f"    ✓ 已保存到: {json_path} （耗时: {t_save_end - t_save_start:.2f}s）")

        total_elapsed = time.perf_counter() - start_all
        print(f"任务完成。总耗时: {total_elapsed:.2f}s。文件: {json_path}")

        return json_path

    finally:
        # ==================== 关键改动在这里 ====================
        # 无论函数是否成功，都尝试释放模型资源
        if model is not None:
            print(f"正在释放模型 {MODEL_SIZE} 的资源...")
            # 对于 faster-whisper，模型对象可能需要特殊的卸载方法，但通常 del 就能工作
            # 如果 faster-whisper 有 model.unload() 之类的方法，应该优先使用
            # 查阅文档后，faster-whisper 没有明确的 unload 方法，所以我们用标准方法
            del model

            # 使用 gc.collect() 来强制 Python 进行垃圾回收
            gc.collect()

            # 如果设备是 cuda，清空 PyTorch 的 CUDA 缓存
            if DEVICE == "cuda":
                try:
                    torch.cuda.empty_cache()
                    print("    ✓ CUDA 缓存已清空。")
                except Exception as e:
                    print(f"    ! 清空 CUDA 缓存时出错: {e}")
        print("-" * 80)
        # =========================================================


# 简单示例（直接运行脚本会执行）
if __name__ == "__main__":
    audio = "test.wav"
    path = transcribe_words_to_json(audio, MODEL_SIZE="large-v2")
