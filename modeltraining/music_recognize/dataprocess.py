import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from gammatone.gtgram import gtgram

# 强制使用 soundfile 后端避免 DLL 加载错误
try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass

# ================= 1. 配置参数 =================
SR = 44100
SEGMENT_LEN = 22050  # 0.5s
START_TIME = 15
NUM_SEG = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WAV_DIR = "data/audio_wav"
DYNAMIC_ANN_DIR = "data/annotations/annotations averaged per song/dynamic (per second annotations)"
SAVE_DIR = "data/processed_data"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# ================= 2. 辅助函数 =================
def pre_emphasis_torch(signal):
    return torch.cat((signal[:, :1], signal[:, 1:] - 0.97 * signal[:, :-1]), dim=1)


def extract_harmony_24(seg_np, sr):
    """提取 24 维 Harmony 特征: Chroma (12) + Delta Chroma (12)"""
    try:
        # A. Chroma STFT (12维)
        chroma = librosa.feature.chroma_stft(
            y=seg_np, sr=sr, n_fft=2048, hop_length=int(sr * 0.01)
        )
        # B. Delta Chroma (12维)
        delta_chroma = librosa.feature.delta(chroma)

        # C. 拼接 (12 + 12 = 24)
        feat = np.concatenate([chroma, delta_chroma], axis=0)

        # D. 插值对齐到时间轴 96 帧 (保持频率轴为 24)
        feat_t = torch.from_numpy(feat).float().unsqueeze(0).unsqueeze(0)
        feat_resized = F.interpolate(
            feat_t, size=(24, 96), mode='bilinear', align_corners=False
        ).squeeze()

        return feat_resized.numpy().T  # 返回形状 (96, 24)
    except Exception as e:
        return np.zeros((96, 24))


# ================= 3. 主处理流程 =================
def process_full_dataset():
    print(f"特征提取开始 设备: {DEVICE}")

    # A. 加载标签
    df_a = pd.read_csv(os.path.join(DYNAMIC_ANN_DIR, "arousal.csv"))
    df_v = pd.read_csv(os.path.join(DYNAMIC_ANN_DIR, "valence.csv"))
    df_a.columns = df_a.columns.str.strip()
    df_v.columns = df_v.columns.str.strip()

    # B. 5:1:2 划分
    all_song_ids = df_a['song_id'].unique()
    train_ids, temp_ids = train_test_split(all_song_ids, train_size=5 / 8, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, train_size=1 / 3, random_state=42)

    # C. MFCC 转换器
    mfcc_transform = T.MFCC(
        sample_rate=SR,
        n_mfcc=44,
        melkwargs={
            "n_fft": 2048,
            "hop_length": int(SR * 0.005),
            "win_length": int(SR * 0.025),
            "window_fn": torch.hamming_window
        }
    ).to(DEVICE)

    sets = {'train': train_ids, 'val': val_ids, 'test': test_ids}

    for set_name, ids in sets.items():
        print(f"\n开始处理 {set_name} 集 (共 {len(ids)} 首)...")
        m_list, g_list, h_list, l_list = [], [], [], []

        for sid in tqdm(ids):
            wav_path = os.path.join(WAV_DIR, f"{sid}.wav")
            if not os.path.exists(wav_path): continue

            try:
                a_vals = df_a[df_a['song_id'] == sid].iloc[0, 1:61].values.astype(float)
                v_vals = df_v[df_v['song_id'] == sid].iloc[0, 1:61].values.astype(float)
            except:
                continue

            waveform, _ = torchaudio.load(wav_path)
            waveform = waveform.to(DEVICE)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            start_sample = START_TIME * SR
            waveform_np = waveform.squeeze().cpu().numpy()

            for i in range(NUM_SEG):
                s = start_sample + i * SEGMENT_LEN
                e = s + SEGMENT_LEN

                # --- 1. MFCC ---
                seg = waveform[:, s:e]
                if seg.shape[1] < SEGMENT_LEN:
                    seg = F.pad(seg, (0, SEGMENT_LEN - seg.shape[1]))

                seg_pre = pre_emphasis_torch(seg)
                mfcc_feat = mfcc_transform(seg_pre).squeeze(0).T
                if mfcc_feat.shape[0] < 96:
                    mfcc_feat = F.pad(mfcc_feat, (0, 0, 0, 96 - mfcc_feat.shape[0]))
                else:
                    mfcc_feat = mfcc_feat[:96, :]
                m_list.append(mfcc_feat.cpu().numpy())

                # --- 2. GTF ---
                seg_np = waveform_np[s:e]
                if len(seg_np) < SEGMENT_LEN:
                    seg_np = np.pad(seg_np, (0, SEGMENT_LEN - len(seg_np)))

                gtf_feat = gtgram(seg_np, SR, 0.025, 0.025, 44, 50).T
                if gtf_feat.shape[0] < 12:
                    gtf_feat = np.pad(gtf_feat, ((0, 12 - gtf_feat.shape[0]), (0, 0)))
                else:
                    gtf_feat = gtf_feat[:12, :]
                g_list.append(gtf_feat)

                # --- 3. Harmony (24维) ---
                h_feat = extract_harmony_24(seg_np, SR)
                h_list.append(h_feat)

                # --- 4. Label ---
                l_list.append([a_vals[i], v_vals[i]])

        # 保存
        np.save(os.path.join(SAVE_DIR, f"{set_name}_mfcc.npy"), np.array(m_list))
        np.save(os.path.join(SAVE_DIR, f"{set_name}_gtf.npy"), np.array(g_list))
        np.save(os.path.join(SAVE_DIR, f"{set_name}_harmony.npy"), np.array(h_list))
        np.save(os.path.join(SAVE_DIR, f"{set_name}_labels.npy"), np.array(l_list))

if __name__ == "__main__":
    process_full_dataset()