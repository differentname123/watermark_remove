# diarize_with_speechbrain.py
import os
import sys
import math
import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# SpeechBrain 新 API（注意不是 speechbrain.pretrained）
from speechbrain.inference.VAD import VAD
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.processing import diarization as diar

# ========== 配置 ==========
INPUT_MP3 = "mix.mp3"
WAV = "mix.wav"
SAVEDIR = "pretrained_models"   # 下载模型的保存位置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WIN_S = 1.5    # 窗口长度（秒）
HOP_S = 0.75   # 步长（秒）
BATCH_SIZE = 32
MAX_SPEAKERS = 6   # 自动估计时的上限
OUT_RTTM = "mix.rttm"

# ========== 1) mp3 -> wav (16k mono) ==========
if not os.path.exists(WAV):
    print(f"Converting {INPUT_MP3} -> {WAV} (16k mono)...")
    audio = AudioSegment.from_file(INPUT_MP3, format="mp3")
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(WAV, format="wav")

# ========== 2) VAD: get speech boundaries ==========
print("Loading VAD model...")
vad = VAD.from_hparams(
    source="speechbrain/vad-crdnn-libriparty",
    savedir=os.path.join(SAVEDIR, "vad-crdnn-libriparty"),
    run_opts={"device": DEVICE} if DEVICE.startswith("cuda") else None
)
print("Running VAD ...")
boundaries = vad.get_speech_segments(WAV)  # tensor or list of [start,end]
if isinstance(boundaries, torch.Tensor):
    boundaries = boundaries.cpu().numpy().tolist()
if len(boundaries) == 0:
    print("No speech segments found by VAD. Exiting.")
    sys.exit(0)

# ========== 3) load full wave and prepare sliding windows within speech segments ==========
signal, sr = torchaudio.load(WAV)  # signal: [channels, samples]
if signal.size(0) > 1:
    signal = signal.mean(dim=0, keepdim=True)
signal = signal.squeeze(0)  # [samples]
num_samples = signal.shape[0]

win_samples = int(WIN_S * sr)
hop_samples = int(HOP_S * sr)

windows = []        # list of torch tensors (length = win_samples)
win_times = []      # list of (start_time, end_time)
for (s, e) in boundaries:
    s_samp = max(0, int(math.floor(s * sr)))
    e_samp = min(num_samples, int(math.ceil(e * sr)))
    pos = s_samp
    while pos < e_samp:
        end = min(pos + win_samples, e_samp)
        chunk = signal[pos:end]
        # pad to win_samples if last chunk shorter
        if chunk.numel() < win_samples:
            pad = torch.zeros(win_samples - chunk.numel())
            chunk = torch.cat([chunk, pad], dim=0)
        windows.append(chunk)
        win_times.append((pos / sr, end / sr))
        pos += hop_samples

if len(windows) == 0:
    print("No windows created (weird). Exiting.")
    sys.exit(0)

# ========== 4) Load embedding model (ECAPA or xvect) and extract embeddings ==========
print("Loading speaker embedding model (ECAPA)...")
enc = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=os.path.join(SAVEDIR, "spkrec-ecapa-voxceleb"),
    run_opts={"device": DEVICE} if DEVICE.startswith("cuda") else None
)

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
embs = []
enc.to(device)

print(f"Extracting embeddings for {len(windows)} windows (batch_size={BATCH_SIZE}) ...")
for i in range(0, len(windows), BATCH_SIZE):
    batch = torch.stack(windows[i:i+BATCH_SIZE])  # shape (B, samples)
    batch = batch.to(device)
    try:
        # some encoder implementations accept (B, samples)
        out = enc.encode_batch(batch)
    except Exception:
        # try channel dim: (B, 1, samples)
        out = enc.encode_batch(batch.unsqueeze(1))
    # out might be (B, 1, D) or (B, D)
    if out.dim() == 3 and out.shape[1] == 1:
        out = out.squeeze(1)
    out = out.detach().cpu().numpy()
    embs.append(out)
embeddings = np.vstack(embs)   # (n_windows, embed_dim)

# L2 normalize
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-10)

# ========== 5) 估计说话人数（可选） ==========
def estimate_num_speakers(X, max_k=MAX_SPEAKERS):
    # 简单的谱间隙法（heuristic）：计算拉普拉斯特征值的最大间隙
    S = cosine_similarity(X)
    # 保证对称并非负
    S = (S + S.T) / 2.0
    S = np.clip(S, 0, None)
    D = np.diag(S.sum(axis=1))
    L = D - S
    # 计算前几个最小的特征值
    max_check = min(max_k + 1, S.shape[0] - 1)
    if max_check < 2:
        return 1
    vals, _ = np.linalg.eigh(L)
    vals = np.real(vals[:max_check+1])  # take smallest eigenvalues
    gaps = np.diff(vals)
    # 找到最大 gap 的索引 -> clusters = idx+1
    idx = int(np.argmax(gaps)) + 1
    return max(2, idx)

print("Estimating number of speakers (can be overridden)...")
n_speakers = estimate_num_speakers(embeddings, MAX_SPEAKERS)
print(f"Estimated #speakers = {n_speakers}")

# ========== 6) 聚类（谱聚类） ==========
print("Computing affinity matrix and clustering ...")
aff = cosine_similarity(embeddings)
# 归一化到 [0,1]
aff = (aff - aff.min()) / (aff.max() - aff.min() + 1e-9)

clust = SpectralClustering(n_clusters=n_speakers, affinity="precomputed", assign_labels="kmeans")
labels = clust.fit_predict(aff)  # length == n_windows

# ========== 7) 合并相邻相同 label 的窗口，生成 segs 并输出 RTTM ==========
segs = []
cur_label = labels[0]
cur_start = win_times[0][0]
cur_end = win_times[0][1]
for i in range(1, len(labels)):
    l = labels[i]
    s_t, e_t = win_times[i]
    if l == cur_label:
        cur_end = e_t
    else:
        segs.append(["mix", cur_start, cur_end, f"spk{cur_label+1}"])
        cur_label = l
        cur_start = s_t
        cur_end = e_t
segs.append(["mix", cur_start, cur_end, f"spk{cur_label+1}"])

# optional: merge very short segments or distribute overlaps using speechbrain functions
segs = diar.merge_ssegs_same_speaker(segs)  # 会合并相邻同说话人的小段

print("Writing RTTM:", OUT_RTTM)
diar.write_rttm(segs, OUT_RTTM)

print("Diarization finished. RTTM saved to", OUT_RTTM)
print("Segments:")
for r in segs:
    print(r)


from collections import defaultdict

print("Cutting speaker segments into wav files...")

# 读取原始 wav
audio = AudioSegment.from_wav(WAV)

# 保存目录
out_dir = "speaker_segments"
os.makedirs(out_dir, exist_ok=True)

# 每个说话人的片段收集器
spk_segments = defaultdict(list)

for seg in segs:
    _, start, end, spk = seg
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    chunk = audio[start_ms:end_ms]
    # 保存单个片段
    chunk_path = os.path.join(out_dir, f"{spk}_{start_ms}_{end_ms}.wav")
    chunk.export(chunk_path, format="wav")
    spk_segments[spk].append(chunk)

# 额外功能：把每个说话人的所有片段拼接成一个文件
for spk, chunks in spk_segments.items():
    combined = sum(chunks)  # 拼接
    out_path = os.path.join(out_dir, f"{spk}.wav")
    combined.export(out_path, format="wav")
    print(f"已保存 {spk} 的完整语音 -> {out_path}")

print("所有说话人片段已保存到:", out_dir)
