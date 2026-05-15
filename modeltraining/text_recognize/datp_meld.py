import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import pickle
import os

# ======================
# 配置参数
# ======================
# 定义输入输出文件对照表
FILES_TO_PROCESS = {
    "train_sent_emo.csv": "meld_train_flattened.pkl",
    "dev_sent_emo.csv": "meld_dev_flattened.pkl",
    "test_sent_emo.csv": "meld_test_flattened.pkl"
}

MODEL_NAME = "roberta-base"
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_CONTEXT = True
CONTEXT_SIZE = 2

# ======================
# 1. 初始化模型
# ======================
print(f"Loading {MODEL_NAME}...")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


# ======================
# 2. 标签映射函数 (保持一致性)
# ======================
def map_meld_emotion(emo):
    mapping = {
        "anger": 0, "disgust": 0, "fear": 5,
        "joy": 3, "neutral": 2, "sadness": 5, "surprise": 4
    }
    return mapping.get(emo.lower(), 2)


# ======================
# 3. 核心处理函数
# ======================
def process_csv(csv_path, save_path):
    if not os.path.exists(csv_path):
        print(f"跳过: 找不到文件 {csv_path}")
        return

    df = pd.read_csv(csv_path)
    # 必须排序以确保上下文逻辑正确
    df = df.sort_values(by=["Dialogue_ID", "Utterance_ID"])
    dialogue_groups = df.groupby("Dialogue_ID")

    flattened_results = []

    print(f"正在处理 {csv_path} -> {save_path}")
    with torch.no_grad():
        for did, group in tqdm(dialogue_groups):
            utterances = group["Utterance"].astype(str).tolist()
            emotions = group["Emotion"].tolist()

            for i in range(len(utterances)):
                # 构造上下文
                if USE_CONTEXT:
                    start = max(0, i - CONTEXT_SIZE + 1)
                    text = " </s> ".join(utterances[start: i + 1])
                else:
                    text = utterances[i]

                # 特征提取
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                   padding="max_length", max_length=MAX_LEN).to(DEVICE)
                outputs = model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()

                flattened_results.append({
                    "feature": cls_emb,
                    "label": map_meld_emotion(emotions[i]),
                    "text": utterances[i],
                    "source": "meld"

                })

    with open(save_path, "wb") as f:
        pickle.dump(flattened_results, f)
    print(f"保存完成，共 {len(flattened_results)} 条数据。")


# ======================
# 4. 执行批量处理
# ======================
if __name__ == "__main__":
    for csv, pkl in FILES_TO_PROCESS.items():
        process_csv(csv, pkl)
    print("\n所有 MELD 文件处理完毕！")