import os

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from gammatone.gtgram import gtgram
from pydub import AudioSegment
from tqdm import tqdm

from musicmodel import ArousalModel, ValenceModel


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
MODEL_WEIGHTS_DIR = os.path.join(project_root, "models")

VALENCE_PATH = os.path.join(MODEL_WEIGHTS_DIR, "valence.pth")
AROUSAL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "arousal.pth")

CONFIG = {
    "SR": 44100,
    "SEGMENT_LEN": 22050,
    "START_TIME": 60,
    "NUM_SEG": 90,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "UPLOAD_DIR": os.path.join(current_dir, "upload"),
    "PROCESSED_DIR": os.path.join(current_dir, "processed"),
    "SAVE_PATH": os.path.join(current_dir, "music_emotion_index.csv"),
    "COMPARE_TOLERANCE": 1e-4,
}

for path in (CONFIG["UPLOAD_DIR"], CONFIG["PROCESSED_DIR"]):
    os.makedirs(path, exist_ok=True)


def normalize_index_path(raw_path):
    raw_path = (raw_path or "").strip()
    if not raw_path:
        return ""
    if os.path.isabs(raw_path):
        return os.path.normpath(raw_path)
    return os.path.normpath(os.path.join(current_dir, raw_path))


def to_index_relative_path(abs_path):
    return os.path.relpath(os.path.normpath(abs_path), current_dir)


def load_index_rows():
    csv_path = CONFIG["SAVE_PATH"]
    if not os.path.exists(csv_path):
        return []

    try:
        df_index = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return []

    rows = []
    for row in df_index.to_dict("records"):
        try:
            rows.append(
                {
                    "song_id": str(row.get("song_id", "")).strip(),
                    "valence": float(row.get("valence", 0.0)),
                    "arousal": float(row.get("arousal", 0.0)),
                    "file_path": str(row.get("file_path", "")).strip(),
                }
            )
        except (TypeError, ValueError):
            continue
    return rows


def save_index_rows(rows):
    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                "song_id": str(row["song_id"]).strip(),
                "valence": float(row["valence"]),
                "arousal": float(row["arousal"]),
                "file_path": str(row["file_path"]).strip(),
            }
        )
    df = pd.DataFrame(normalized_rows, columns=["song_id", "valence", "arousal", "file_path"])
    df.to_csv(CONFIG["SAVE_PATH"], index=False, encoding="utf-8-sig")


def list_upload_audio_files():
    files = {}
    for filename in os.listdir(CONFIG["UPLOAD_DIR"]):
        if filename.startswith("."):
            continue
        full_path = os.path.join(CONFIG["UPLOAD_DIR"], filename)
        if not os.path.isfile(full_path):
            continue
        song_id = os.path.splitext(filename)[0]
        files[song_id] = full_path
    return files


class MusicEmotionIndexer:
    def __init__(self):
        self.device = CONFIG["DEVICE"]
        self.model_v = None
        self.model_a = None
        self.mfcc_transform = T.MFCC(
            sample_rate=CONFIG["SR"],
            n_mfcc=44,
            melkwargs={
                "n_fft": 2048,
                "hop_length": int(CONFIG["SR"] * 0.005),
                "win_length": int(CONFIG["SR"] * 0.025),
                "window_fn": torch.hamming_window,
            },
        ).to(self.device)

    @staticmethod
    def pre_emphasis_torch(signal):
        return torch.cat((signal[:, :1], signal[:, 1:] - 0.97 * signal[:, :-1]), dim=1)

    def load_models(self):
        if self.model_v is None:
            self.model_v = ValenceModel().to(self.device)
            self.model_v.load_state_dict(torch.load(VALENCE_PATH, map_location=self.device))
            self.model_v.eval()
        if self.model_a is None:
            self.model_a = ArousalModel().to(self.device)
            self.model_a.load_state_dict(torch.load(AROUSAL_PATH, map_location=self.device))
            self.model_a.eval()

    def convert_to_standard_wav(self, src_path, song_id):
        target_path = os.path.join(CONFIG["PROCESSED_DIR"], f"{song_id}.wav")
        audio = AudioSegment.from_file(src_path)
        audio = audio.set_frame_rate(CONFIG["SR"]).set_channels(1)
        audio.export(target_path, format="wav")
        return target_path

    def _prepare_target_path(self, input_path, song_id):
        input_path = os.path.normpath(input_path)
        if input_path.lower().endswith(".wav") and os.path.abspath(os.path.dirname(input_path)) == os.path.abspath(CONFIG["PROCESSED_DIR"]):
            return input_path
        return self.convert_to_standard_wav(input_path, song_id)

    def predict_file(self, input_path, song_id=None):
        self.load_models()

        if not input_path or not os.path.exists(input_path):
            raise FileNotFoundError(f"音频文件不存在：{input_path}")

        song_id = song_id or os.path.splitext(os.path.basename(input_path))[0]
        target_path = self._prepare_target_path(input_path, song_id)

        y, _ = librosa.load(target_path, sr=CONFIG["SR"])
        waveform_np = y
        waveform_tensor = torch.from_numpy(waveform_np).unsqueeze(0).to(self.device)

        g_list, h_list, m_list = [], [], []
        start_sample = CONFIG["START_TIME"] * CONFIG["SR"]
        required_length = start_sample + CONFIG["NUM_SEG"] * CONFIG["SEGMENT_LEN"]
        if len(waveform_np) < required_length:
            start_sample = 0

        for i in range(CONFIG["NUM_SEG"]):
            start = start_sample + i * CONFIG["SEGMENT_LEN"]
            end = start + CONFIG["SEGMENT_LEN"]

            if start >= len(waveform_np):
                seg_np = np.zeros(CONFIG["SEGMENT_LEN"])
                seg_tensor = torch.zeros((1, CONFIG["SEGMENT_LEN"]), device=self.device)
            elif end > len(waveform_np):
                remain = len(waveform_np) - start
                seg_np = np.pad(waveform_np[start:], (0, CONFIG["SEGMENT_LEN"] - remain))
                seg_tensor = F.pad(waveform_tensor[:, start:], (0, CONFIG["SEGMENT_LEN"] - remain))
            else:
                seg_np = waveform_np[start:end]
                seg_tensor = waveform_tensor[:, start:end]

            seg_pre = self.pre_emphasis_torch(seg_tensor)
            mfcc_feat = self.mfcc_transform(seg_pre).squeeze(0).T
            if mfcc_feat.shape[0] < 96:
                mfcc_feat = F.pad(mfcc_feat, (0, 0, 0, 96 - mfcc_feat.shape[0]))
            else:
                mfcc_feat = mfcc_feat[:96, :]
            m_list.append(mfcc_feat.cpu().numpy())

            gtf_feat = gtgram(seg_np, CONFIG["SR"], 0.025, 0.025, 44, 50).T
            if gtf_feat.shape[0] >= 12:
                g_list.append(gtf_feat[:12, :])
            else:
                g_list.append(np.pad(gtf_feat, ((0, 12 - gtf_feat.shape[0]), (0, 0))))

            chroma = librosa.feature.chroma_stft(
                y=seg_np,
                sr=CONFIG["SR"],
                n_fft=2048,
                hop_length=int(CONFIG["SR"] * 0.01),
            )
            delta = librosa.feature.delta(chroma)
            feat = np.concatenate([chroma, delta], axis=0)
            feat_t = torch.from_numpy(feat).float().unsqueeze(0).unsqueeze(0)
            h_feat = F.interpolate(feat_t, size=(24, 96), mode="bilinear", align_corners=False).squeeze().numpy().T
            h_list.append(h_feat)

        m_batch = torch.FloatTensor(np.array(m_list)).to(self.device).unsqueeze(0)
        g_batch = torch.FloatTensor(np.array(g_list)).to(self.device).unsqueeze(0)
        h_batch = torch.FloatTensor(np.array(h_list)).to(self.device).unsqueeze(0)

        with torch.no_grad():
            pred_v = self.model_v(g_batch, h_batch)
            pred_a = self.model_a(m_batch, g_batch)

        return {
            "song_id": song_id,
            "valence": float(pred_v.mean().item()),
            "arousal": float(pred_a.mean().item()),
            "file_path": to_index_relative_path(target_path),
        }


def values_changed(old_row, new_row, tolerance=None):
    tolerance = CONFIG["COMPARE_TOLERANCE"] if tolerance is None else tolerance
    return (
        abs(float(old_row["valence"]) - float(new_row["valence"])) > tolerance
        or abs(float(old_row["arousal"]) - float(new_row["arousal"])) > tolerance
        or str(old_row["file_path"]).strip() != str(new_row["file_path"]).strip()
    )


def recheck_song_entry(song_id, file_path, indexer=None):
    rows = load_index_rows()
    row_index = next((idx for idx, row in enumerate(rows) if row["song_id"] == song_id), -1)
    if row_index < 0:
        raise ValueError(f"索引中不存在歌曲：{song_id}")

    normalized_path = normalize_index_path(file_path)
    if not os.path.exists(normalized_path):
        raise FileNotFoundError(f"歌曲文件不存在：{normalized_path}")

    indexer = indexer or MusicEmotionIndexer()
    old_row = dict(rows[row_index])
    new_row = indexer.predict_file(normalized_path, song_id=song_id)
    changed = values_changed(old_row, new_row)
    rows[row_index] = new_row
    save_index_rows(rows)
    return {"changed": changed, "old": old_row, "new": new_row}


def refresh_music_index(indexer=None):
    indexer = indexer or MusicEmotionIndexer()
    existing_rows = load_index_rows()
    upload_files = list_upload_audio_files()
    existing_song_ids = {row["song_id"] for row in existing_rows}

    summary = {
        "removed": [],
        "added": [],
        "updated": [],
        "unchanged_count": 0,
    }
    refreshed_rows = []

    for row in existing_rows:
        song_id = row["song_id"]
        normalized_path = normalize_index_path(row["file_path"])
        if not os.path.exists(normalized_path):
            summary["removed"].append(song_id)
            continue

        new_row = indexer.predict_file(normalized_path, song_id=song_id)
        if values_changed(row, new_row):
            summary["updated"].append(
                {
                    "song_id": song_id,
                    "old_valence": float(row["valence"]),
                    "old_arousal": float(row["arousal"]),
                    "new_valence": float(new_row["valence"]),
                    "new_arousal": float(new_row["arousal"]),
                }
            )
        else:
            summary["unchanged_count"] += 1
        refreshed_rows.append(new_row)

    for song_id, src_path in tqdm(sorted(upload_files.items()), desc="refresh_music_index"):
        if song_id in existing_song_ids:
            continue
        new_row = indexer.predict_file(src_path, song_id=song_id)
        refreshed_rows.append(new_row)
        summary["added"].append(song_id)

    refreshed_rows.sort(key=lambda item: item["song_id"].lower())
    save_index_rows(refreshed_rows)
    return summary


def process_and_build_index():
    summary = refresh_music_index()
    print(f"added={len(summary['added'])}")
    print(f"removed={len(summary['removed'])}")
    print(f"updated={len(summary['updated'])}")
    print(f"unchanged={summary['unchanged_count']}")


if __name__ == "__main__":
    process_and_build_index()
