import os
import json
import time
from typing import Tuple, List, Dict
from faster_whisper import WhisperModel

def transcribe_words_to_json(audio_file: str, MODEL_SIZE="large-v3"):
    """
    只接受 audio_file 参数，使用固定的模型配置执行转录。
    输出: (word_items, json_path)
      - word_items: list of {"start": float|None, "end": float|None, "word": str, "probability": float|None}
      - json_path: 保存的 JSON 文件路径（output/<basename>_words.json）

    会打印过程日志：
      - 检查文件、加载模型、转录开始/结束、每个 segment 的摘要、保存文件、总耗时
    """
    # ---------- 固定配置（如需修改，请在此处改） ----------
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
    BEAM_SIZE = 5
    OUTPUT_DIR = "output"
    # -------------------------------------------------------

    start_all = time.perf_counter()

    print(f"[1/5] 检查音频文件：{audio_file}")
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"音频文件不存在: {audio_file}")
    print("    ✓ 文件存在")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型
    t0 = time.perf_counter()
    print(f"[2/5] 正在加载模型 {MODEL_SIZE}（device={DEVICE}, compute_type={COMPUTE_TYPE}）...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    t1 = time.perf_counter()
    print(f"    ✓ 模型加载完成，耗时: {t1 - t0:.2f}s")

    # 转录
    print(f"[3/5] 开始转录（beam_size={BEAM_SIZE}, word_timestamps=True）...")
    t2 = time.perf_counter()
    segments, info = model.transcribe(audio_file, beam_size=BEAM_SIZE, word_timestamps=True)
    segments = list(segments)  # 将生成器转为列表，方便多次遍历
    t3 = time.perf_counter()
    print(f"    ✓ 转录完成（仅模型返回） 耗时: {t3 - t2:.2f}s")
    print(f"    识别语言: {getattr(info, 'language', None)}  语言置信度: {getattr(info,'language_probability', None)}")
    print(f"    共检测到 {len(segments)} 个 segment")

    # 处理并组装词级结果（同时打印每个 segment 的简要日志）
    print("[4/5] 处理 segment、提取词级时间戳...")
    t_seg_start = time.perf_counter()
    word_items: List[Dict] = []
    for idx, segment in enumerate(segments, start=1):
        seg_words = getattr(segment, "words", None) or []
        preview = (segment.text.strip()[:40] + "...") if len(segment.text.strip()) > 40 else segment.text.strip()
        # print(f"    segment {idx}/{len(segments)}: [{segment.start:.2f}s -> {segment.end:.2f}s] \"{preview}\"  words={len(seg_words)}")

        for word in seg_words:
            w_start = None if getattr(word, "start", None) is None else float(word.start)
            w_end = None if getattr(word, "end", None) is None else float(word.end)
            w_word = "" if getattr(word, "word", None) is None else word.word.strip()
            w_prob = None if getattr(word, "probability", None) is None else float(word.probability)

            word_items.append({
                "start": round(w_start, 3) * 1000 if w_start is not None else None,
                "end":   round(w_end, 3) * 1000 if w_end is not None else None,
                "word":  w_word,
                "probability": round(w_prob, 3) if w_prob is not None else None
            })
    t_seg_end = time.perf_counter()
    print(f"    ✓ 处理完成，提取到 {len(word_items)} 个词条，耗时: {t_seg_end - t_seg_start:.2f}s")

    # 保存为 JSON
    print("[5/5] 正在保存为 JSON 文件...")
    t_save_start = time.perf_counter()
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    json_filename = f"{base_name}_asr_whisper_{MODEL_SIZE}.json"
    json_path = os.path.join(OUTPUT_DIR, json_filename)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(word_items, jf, ensure_ascii=False, indent=2)
    t_save_end = time.perf_counter()
    print(f"    ✓ 已保存到: {json_path} （耗时: {t_save_end - t_save_start:.2f}s）")

    total_elapsed = time.perf_counter() - start_all
    print(f"全部完成。总耗时: {total_elapsed:.2f}s （加载模型: {t1 - t0:.2f}s, 转录: {t3 - t2:.2f}s, 处理: {t_seg_end - t_seg_start:.2f}s, 保存: {t_save_end - t_save_start:.2f}s）保存到: {json_path}")

    return json_path


# 简单示例（直接运行脚本会执行）
if __name__ == "__main__":
    audio = "test.wav"
    path = transcribe_words_to_json(audio, MODEL_SIZE="large-v2")
