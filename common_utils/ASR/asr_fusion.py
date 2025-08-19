import collections
import math
import statistics
import unicodedata
from difflib import SequenceMatcher

from common_utils.ASR.funasr_utils import run_funasr
from common_utils.ASR.whisper_utils import transcribe_words_to_json


def read_json(filepath):
    """从文件路径读取JSON数据。"""
    import json
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def save_json(filepath, data):
    """将数据保存为格式化的JSON文件。"""
    import json
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving to {filepath}: {e}")


def fuse_asr_results_final(all_asr_lists):
    """
    文本优先的多路 ASR 融合（ROVER 风格 + 稳健时间聚合）。

    V3 最终版修正:
      1. [V3 核心修正] 在对齐算法中引入时间偏差阈值 (MAX_TIME_SKEW_MS)，
         防止将时间上相距甚远的词错误对齐，从根本上杜绝了“幻影”词的产生。
      2. [V2 修正] 修正了单源幻觉抑制逻辑，通过使用总源数(N)作为覆盖度基准。
      3. [V2 修正] 调整了置信度公式和阈值，以更好地保留高共识但有微小分歧的词。
      4. [V2 新增] 新增了对不自然静音间隙的修复逻辑，使输出更流畅。
    """

    # ----------------------- 可调参数 (V3 最终版) -----------------------
    # 融合门限
    MIN_CAND_PROB = 0.05
    SINGLE_SOURCE_KEEP = 0.65
    FINAL_MIN_CONF = 0.15  # 调整后的丢弃门限
    PUNCT_WEIGHT = 0.65
    VOTE_EXP = 1.2

    # 对齐相关
    LOOKAHEAD = 10
    SIM_EQ = 0.95
    # [V3 核心修正] 只有当两个词的起始时间差在此范围内时，才可能被对齐
    MAX_TIME_SKEW_MS = 1500.0  # 允许的最大时间偏差为 1.5 秒

    # 时间聚合
    MIN_DUR_FRACTION = 0.30
    MERGE_GAP_FRACTION = 0.20
    MAX_GAP_TO_BRIDGE_MS = 250.0  # 修复小于 250ms 的不自然间隙

    # ----------------------- 工具函数 (完整) -----------------------
    def clamp(v, lo=0.0, hi=1.0):
        return lo if v < lo else hi if v > hi else v

    def to_float(x, default=0.0):
        try:
            val = float(x)
            return default if math.isnan(val) else val
        except (ValueError, TypeError):
            return default

    def norm_text(s: str) -> str:
        if s is None: return ""
        s = unicodedata.normalize("NFKC", str(s)).strip()
        return " ".join(s.split())

    def get_prob(item):
        p = item.get("probability", item.get("confidence", 0.5))
        return clamp(to_float(p, 0.5))

    def is_punct_char(ch: str) -> bool:
        if not ch: return False
        return unicodedata.category(ch)[0] in ("P", "S")

    def is_punctuation_token(s: str) -> bool:
        s = (s or "").strip()
        return bool(s) and all(is_punct_char(ch) for ch in s)

    def is_cjk_char(ch: str) -> bool:
        if not ch: return False
        code = ord(ch)
        return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or
                0x20000 <= code <= 0x2A6DF or 0x2A700 <= code <= 0x2B73F or
                0x2B740 <= code <= 0x2B81F or 0x2B820 <= code <= 0x2CEAF or
                0xF900 <= code <= 0xFAFF or 0x2F800 <= code <= 0x2FA1F)

    def contains_cjk(s: str) -> bool:
        return any(is_cjk_char(ch) for ch in s)

    def split_units(word: str):
        w = norm_text(word)
        if not w: return [], []
        units = [ch for ch in w if not ch.isspace()] if contains_cjk(w) else [w]
        flags = [is_punctuation_token(u) for u in units]
        return units, flags

    def text_similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def weight_of(prob: float, is_punct: bool) -> float:
        w = clamp(prob) ** VOTE_EXP
        return w * PUNCT_WEIGHT if is_punct else w

    def median_or_default(xs, default):
        xs = sorted([x for x in xs if x is not None])
        if not xs: return default
        n = len(xs)
        return xs[n // 2] if n % 2 == 1 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

    # ----------------------- 构造各路 token 序列 -----------------------
    all_tokens = []
    all_durations = []
    for si, lst in enumerate(all_asr_lists or []):
        seq = []
        for item in (lst or []):
            start = to_float(item.get("start"))
            end = to_float(item.get("end"), start)
            if end < start: continue
            word = norm_text(item.get("word", ""))
            if not word: continue
            units, flags = split_units(word)
            if not units: continue

            dur = end - start
            prob = get_prob(item)
            per_prob = prob / max(1, len(units))

            for k, u in enumerate(units):
                u_start = start + dur * k / len(units)
                u_end = start + dur * (k + 1) / len(units)
                if u_end > u_start: all_durations.append(u_end - u_start)
                seq.append({
                    "text": u, "start": u_start, "end": u_end, "prob": clamp(per_prob),
                    "src": si, "is_punct": bool(flags[k]),
                })
        all_tokens.append(seq)

    if not any(all_tokens): return []

    typical_dur = statistics.median(all_durations) if all_durations else 200.0
    typical_dur = max(20.0, typical_dur)
    min_seg_dur = max(1e-3, typical_dur * MIN_DUR_FRACTION)
    merge_gap_tol = typical_dur * MERGE_GAP_FRACTION
    N = len(all_tokens)

    # ----------------------- 选择对齐骨架 -----------------------
    def seq_quality(seq):
        if not seq: return -1
        return sum(t["prob"] for t in seq) / len(seq) + 0.03 * len(seq)

    ref_idx = max(range(N), key=lambda i: seq_quality(all_tokens[i]))

    def get_start_time(token_or_none):
        return token_or_none['start'] if token_or_none else float('inf')

    # ----------------------- 带时间约束的对齐合并算法 -----------------------
    def align_and_merge(cols, seq, src_idx):
        if not seq: return cols
        if not cols:
            return [{"rep": t["text"], "cands": [None] * i + [t] + [None] * (N - 1 - i)} for i, t in enumerate(seq)]

        i, j, M, K, out = 0, 0, len(cols), len(seq), []

        while i < M and j < K:
            col_ref_tok = next((c for c in cols[i]["cands"] if c), None)
            col_start = get_start_time(col_ref_tok)
            seq_start = seq[j]['start']

            # [V3 核心修正] 时间偏差过大，视为插入/删除
            if abs(col_start - seq_start) > MAX_TIME_SKEW_MS:
                if col_start < seq_start:
                    out.append(cols[i])
                    i += 1
                else:
                    new_col = {"rep": seq[j]["text"], "cands": [None] * N}
                    new_col["cands"][src_idx] = seq[j]
                    out.append(new_col)
                    j += 1
                continue

            # 时间接近，执行基于文本的对齐
            if text_similarity(cols[i]["rep"], seq[j]["text"]) >= SIM_EQ:
                col = cols[i].copy();
                col["cands"] = col["cands"][:]
                col["cands"][src_idx] = seq[j]
                out.append(col);
                i += 1;
                j += 1
            else:
                # 贪心前看，寻找最佳匹配点
                k_found = next((k for k in range(j + 1, min(K, j + LOOKAHEAD)) if
                                text_similarity(cols[i]["rep"], seq[k]["text"]) >= SIM_EQ and abs(
                                    col_start - seq[k]['start']) <= MAX_TIME_SKEW_MS), None)
                r_found = next((r for r in range(i + 1, min(M, i + LOOKAHEAD)) if
                                text_similarity(cols[r]["rep"], seq[j]["text"]) >= SIM_EQ and abs(get_start_time(
                                    next(c for c in cols[r]["cands"] if c)) - seq_start) <= MAX_TIME_SKEW_MS), None)

                if k_found is not None and (r_found is None or (k_found - j) <= (r_found - i)):
                    for t in seq[j:k_found]:
                        new_col = {"rep": t["text"], "cands": [None] * N};
                        new_col["cands"][src_idx] = t;
                        out.append(new_col)
                    col = cols[i].copy();
                    col["cands"] = col["cands"][:];
                    col["cands"][src_idx] = seq[k_found];
                    out.append(col)
                    i += 1;
                    j = k_found + 1
                elif r_found is not None:
                    out.extend(cols[i:r_found])
                    col = cols[r_found].copy();
                    col["cands"] = col["cands"][:];
                    col["cands"][src_idx] = seq[j];
                    out.append(col)
                    i = r_found + 1;
                    j += 1
                else:  # 视为替换
                    col = cols[i].copy();
                    col["cands"] = col["cands"][:];
                    col["cands"][src_idx] = seq[j];
                    out.append(col)
                    i += 1;
                    j += 1

        out.extend(cols[i:])
        for t in seq[j:]:
            new_col = {"rep": t["text"], "cands": [None] * N};
            new_col["cands"][src_idx] = t;
            out.append(new_col)

        # 更新列代表
        for col in out:
            votes = collections.Counter()
            for cand in col["cands"]:
                if cand: votes[cand["text"]] += weight_of(cand["prob"], cand["is_punct"])
            if votes: col["rep"] = max(votes, key=votes.get)

        return out

    columns = [{"rep": t["text"], "cands": [None] * ref_idx + [t] + [None] * (N - 1 - ref_idx)} for t in
               all_tokens[ref_idx]]
    for si in range(N):
        if si != ref_idx: columns = align_and_merge(columns, all_tokens[si], si)

    if not columns: return []

    # ----------------------- 列内投票 + 稳健时间聚合 -----------------------
    fused = []
    prev_end = 0.0
    for col in columns:
        cand_list = [c for c in col["cands"] if c and (c["prob"] >= MIN_CAND_PROB or c["is_punct"])]
        if not cand_list: continue

        groups = collections.defaultdict(lambda: {"items": [], "w": 0.0, "srcs": set(), "is_punct": False})
        total_w = sum(weight_of(c["prob"], c["is_punct"]) for c in cand_list)
        for c in cand_list:
            grp = groups[c["text"]]
            grp["items"].append(c);
            grp["w"] += weight_of(c["prob"], c["is_punct"])
            grp["srcs"].add(c["src"]);
            grp["is_punct"] = c["is_punct"]

        if not groups: continue

        win_txt, win = max(groups.items(), key=lambda kv: (kv[1]["w"], len(kv[1]["srcs"])))
        win_srcs = win["srcs"]

        support_ratio = len(win_srcs) / N
        avg_p = sum(c["prob"] for c in win["items"]) / len(win["items"])

        if len(win_srcs) == 1 and N >= 2 and not win["is_punct"]: continue
        if win["is_punct"] and support_ratio < 0.5 and avg_p < 0.6: continue

        vote_margin = win["w"] / max(1e-9, total_w)
        conf = 0.30 * avg_p + 0.35 * vote_margin + 0.35 * support_ratio
        if conf < FINAL_MIN_CONF and not win["is_punct"]: continue

        starts = [c["start"] for c in win["items"]];
        ends = [c["end"] for c in win["items"]]
        start_med = median_or_default(starts, prev_end)
        end_med = median_or_default(ends, start_med + typical_dur)

        if start_med < prev_end: start_med = prev_end
        if end_med <= start_med: end_med = start_med + min_seg_dur

        fused.append({"start": start_med, "end": end_med, "word": win_txt, "probability": clamp(conf)})
        prev_end = end_med

    if not fused: return []

    # ----------------------- 后处理：间隙修复与合并 -----------------------
    bridged = []
    if fused:
        bridged.append(fused[0])
        for curr in fused[1:]:
            prev = bridged[-1]
            gap = curr["start"] - prev["end"]
            if 0 < gap < MAX_GAP_TO_BRIDGE_MS:
                half_gap = gap / 2.0
                prev["end"] += half_gap
                curr["start"] -= half_gap
            bridged.append(curr)

    merged = []
    if bridged:
        cur = dict(bridged[0])
        for s in bridged[1:]:
            if s["word"] == cur["word"] and s["start"] <= cur["end"] + merge_gap_tol:
                cur["end"] = max(cur["end"], s["end"])
                cur["probability"] = max(cur["probability"], s["probability"])
            else:
                if (cur["end"] - cur["start"]) >= min_seg_dur or is_punctuation_token(cur["word"]): merged.append(cur)
                cur = dict(s)
        if (cur["end"] - cur["start"]) >= min_seg_dur or is_punctuation_token(cur["word"]): merged.append(cur)

    final_out = []
    if merged:
        final_out.append(merged[0])
        for it in merged[1:]:
            if is_punctuation_token(it["word"]) and is_punctuation_token(final_out[-1]["word"]):
                final_out[-1]["word"] += it["word"]
                final_out[-1]["end"] = max(final_out[-1]["end"], it["end"])
                final_out[-1]["probability"] = max(final_out[-1]["probability"], it["probability"])
            else:
                final_out.append(it)

    return final_out


# --- 示例用法 ---
if __name__ == '__main__':
    audio_file = r"mix.mp3"



    # 假设你的ASR文件都在同一个目录下
    ASR_FILES = [
        run_funasr(audio_file),
        transcribe_words_to_json(audio_file),
        transcribe_words_to_json(audio_file, MODEL_SIZE="large-v3"),
    ]
    OUTPUT_FILE = 'output/fused_transcript_final.json'

    all_asr_lists = [read_json(f) for f in ASR_FILES if read_json(f) is not None]

    if len(all_asr_lists) > 1:
        # 执行最终版融合
        final_result = fuse_asr_results_final(all_asr_lists)

        # 将结果时间戳转换为秒，并格式化
        for item in final_result:
            item['start'] = round(item['start'] / 1000.0, 3)
            item['end'] = round(item['end'] / 1000.0, 3)
            item['probability'] = round(item['probability'], 4)

        save_json(OUTPUT_FILE, final_result)
        print(f"融合成功！最终结果已保存到 {OUTPUT_FILE}")
    else:
        print("未能加载足够的ASR文件进行融合。")