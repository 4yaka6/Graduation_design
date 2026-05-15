import os
import re
import torch
import pickle
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel

# ======================
# 参数
# ======================
IEMOCAP_PATH = ""
SAVE_PATH = "iemocap_text_roberta.pkl"
OUTPUT_DIR = "./"  # 分割后的 pkl 保存目录

TARGET_EMOTIONS = ['ang', 'fru', 'neu', 'hap', 'exc', 'sad']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LEN = 64
USE_CONTEXT = True
CONTEXT_SIZE = 3  # 上下文窗口

# ======================
# RoBERTa
# ======================
tokenizer = RobertaTokenizer.from_pretrained(
    "roberta-base",
    clean_up_tokenization_spaces=True
)
model = RobertaModel.from_pretrained("roberta-base").to(DEVICE)
model.eval()


# ======================
# 解析 Emotion + VA
# ======================
def parse_emoeval(file_path):
    emo_dict = {}
    if not os.path.exists(file_path): return emo_dict

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not (line.startswith('[') and '\t' in line):
                continue

            parts = line.split('\t')
            if len(parts) < 4:
                continue

            utt_id = parts[1]
            emotion = parts[2]

            if emotion not in TARGET_EMOTIONS:
                continue

            matches = re.findall(r'\[(.*?)\]', line)
            valence, arousal = 0.0, 0.0

            if len(matches) >= 2:
                last = matches[-1]
                if ',' in last:
                    try:
                        va = [float(x.strip()) for x in last.split(',')]
                        valence = va[0]
                        arousal = va[1]
                    except:
                        pass

            emo_dict[utt_id] = {
                "emotion": emotion,
                "valence": valence,
                "arousal": arousal
            }
    return emo_dict


# ======================
# 解析文本
# ======================
def parse_transcription(file_path):
    text_dict = {}
    if not os.path.exists(file_path): return text_dict

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' not in line:
                continue
            utt_id = line.split(' ')[0]
            text = line.split(':', 1)[1].strip()
            text_dict[utt_id] = text
    return text_dict


# ======================
# 构造上下文
# ======================
def build_context(utts, idx):
    if not USE_CONTEXT:
        return utts[idx]
    start = max(0, idx - CONTEXT_SIZE + 1)  # 修正范围逻辑，包含当前句在内的前 N 句
    context = utts[start:idx + 1]
    return f" {tokenizer.sep_token} ".join(context)


# ======================
# 分割与保存逻辑
# ======================
def split_and_save(results):
    """
    逻辑：Session 1,2,3 -> Train, Session 4 -> Val, Session 5 -> Test
    """
    train_data = {}
    val_data = {}
    test_data = {}

    for dialog_id, content in results.items():
        sess = content['session']
        if sess in [1, 2, 3]:
            train_data[dialog_id] = content
        elif sess == 4:
            val_data[dialog_id] = content
        elif sess == 5:
            test_data[dialog_id] = content

    # 保存
    save_tasks = [
        ("iemocap_train.pkl", train_data),
        ("iemocap_val.pkl", val_data),
        ("iemocap_test.pkl", test_data)
    ]

    print("\n--- 保存统计 ---")
    for name, data_dict in save_tasks:
        with open(os.path.join(OUTPUT_DIR, name), "wb") as f:
            pickle.dump(data_dict, f)
        print(f"已保存 {name}: {len(data_dict)} 条对话")


# ======================
# 主处理
# ======================
def process_iemocap_text():
    results = {}

    for session in range(1, 6):
        print(f"Processing Session{session}...")
        session_path = os.path.join(IEMOCAP_PATH, f"Session{session}")

        emo_dir = os.path.join(session_path, "dialog", "EmoEvaluation")
        txt_dir = os.path.join(session_path, "dialog", "transcriptions")

        if not os.path.exists(txt_dir): continue

        for file in tqdm(os.listdir(txt_dir)):
            if not file.endswith(".txt") or file.startswith("."):
                continue

            dialog_id = file.replace(".txt", "")
            emo_path = os.path.join(emo_dir, file)
            txt_path = os.path.join(txt_dir, file)

            emo_dict = parse_emoeval(emo_path)
            text_dict = parse_transcription(txt_path)

            utt_ids = sorted(text_dict.keys())
            utterances, emotions, valence, arousal, speakers = [], [], [], [], []

            for utt_id in utt_ids:
                if utt_id not in emo_dict:
                    continue
                utterances.append(text_dict[utt_id])
                emotions.append(emo_dict[utt_id]["emotion"])
                valence.append(emo_dict[utt_id]["valence"])
                arousal.append(emo_dict[utt_id]["arousal"])
                speakers.append(utt_id.split("_")[0][-1])

            if len(utterances) == 0:
                continue

            # 特征提取
            features = []
            for i in range(len(utterances)):
                text = build_context(utterances, i)
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                   padding="max_length", max_length=MAX_LEN).to(DEVICE)
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Mean pooling
                    emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                features.append(emb.cpu())

            results[dialog_id] = {
                "utterances": utterances,
                "features": torch.stack(features),
                "emotions": emotions,
                "valence": valence,
                "arousal": arousal,
                "speakers": speakers,
                "session": session  # <-- 核心：记录 session 用于后续分割
            }

    return results


# ======================
# 运行
# ======================
if __name__ == "__main__":

        all_data = process_iemocap_text()

        # 2. 执行分割并保存三个文件
        split_and_save(all_data)

        # 3. 可选：保存原始全量数据
        with open(SAVE_PATH, "wb") as f:
            pickle.dump(all_data, f)
        print(f"\n全量数据已保存至: {SAVE_PATH}")