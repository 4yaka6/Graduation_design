import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import RobertaModel,RobertaTokenizer
import seaborn as sns
from torch import amp
from transformers import get_cosine_schedule_with_warmup
import time
import matplotlib.pyplot as plt

class Config:
    # --- 路径 ---
    DATA_DIR = ""
    TRAIN_PATH = os.path.join(DATA_DIR, "iemocap_train.pkl")
    VAL_PATH = os.path.join(DATA_DIR, "iemocap_val.pkl")
    MODEL_SAVE_DIR = "checkpoints"
    BEST_MODEL_NAME = "best_model.pth"

    # --- 模型参数 ---
    INPUT_DIM = 768           # RoBERTa特征维度
    HIDDEN_DIM = 256
    NUM_CLASSES = 6

    # --- 训练参数 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 8
    ACCUMULATION_STEPS =8
    # LR = 5e-5
    EPOCHS = 50
    PATIENCE = 20
    LAMBDA_CLS = 1.0
    LAMBDA_REG = 0.3
    # --- 对话级参数 ---
    MAX_SEQ_LEN = 64
    GRAD_CLIP = 1.0

# 创建保存目录
os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)

# ======================
# 数据集
# ======================
class FineTuneDataset(Dataset):
    def __init__(self, iemocap_pkl=None, meld_pkl=None, tokenizer=None, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        # --- 处理 IEMOCAP (字典嵌套结构) ---
        if iemocap_pkl is not None:
            with open(iemocap_pkl, 'rb') as f:
                iemocap_data = pickle.load(f)

            # 遍历每一个对话 ID (did)
            for did, content in iemocap_data.items():
                utts = content['utterances']
                emos = content['emotions']
                vas = content.get('valence', [3.0] * len(utts))
                ars = content.get('arousal', [3.0] * len(utts))

                for i in range(len(utts)):
                    # 统一映射标签 (IEMOCAP 原始是字符串)
                    label_map = {'ang': 0, 'fru': 1, 'neu': 2, 'hap': 3, 'exc': 4, 'sad': 5}
                    if emos[i] in label_map:
                        self.samples.append({
                            "text": str(utts[i]),
                            "label": label_map[emos[i]],
                            "va": [vas[i], ars[i]]
                        })

        # --- 处理 MELD (假设你重新生成后是列表结构) ---
        if meld_pkl is not None :
            with open(meld_pkl, 'rb') as f:
                meld_data = pickle.load(f)
            # MELD 已经是展平后的 list，直接 extend
            for item in meld_data:
                # 兼容性检查：确保 MELD 里有 'text' 键
                if 'text' in item:
                    self.samples.append({
                        "text": str(item['text']),
                        "label": item['label'],
                        "va": item.get('va', [3.0, 3.0])
                    })

        print(f"数据加载完成！有效单句总数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return (
            encoding['input_ids'].squeeze(0),
            encoding['attention_mask'].squeeze(0),
            torch.tensor(item['label'], dtype=torch.long),
            (torch.tensor(item['va'], dtype=torch.float32) - 1.0) / 4.0
        )

def get_flattened_iemocap(pkl_path):
    """将原本是 Session 结构的 IEMOCAP 彻底展平为单句列表"""
    with open(pkl_path, 'rb') as f:
        full_data = pickle.load(f)

    flattened = []
    # 兼容处理：如果已经是 list 就直接回传，如果是 dict 则展平
    if isinstance(full_data, list): return full_data

    for session_id, d in full_data.items():
        for i in range(len(d["emotions"])):
            if d["emotions"][i] in ['ang', 'fru', 'neu', 'hap', 'exc', 'sad']:
                flattened.append({
                    "feature": d["features"][i],
                    "label": d["emotions"][i],  # 存原始字符，后面在 Dataset 里统一转数字
                    "va": [d["valence"][i], d["arousal"][i]]
                })
    return flattened

# ======================
# 模型
# ======================
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重 [1.2, 1.0, 0.8, 2.0, 1.5, 1.5]
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # 1. 计算标准 Cross Entropy (开启平滑)
        # 注意：PyTorch 的 cross_entropy 自带 label_smoothing 参数 (需 torch >= 1.10)
        log_pt = F.log_softmax(inputs, dim=-1)
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            weight=self.alpha,
            label_smoothing=self.label_smoothing
        )

        # 2. 计算 Focal 项: (1 - pt)^gamma
        # 我们取正确类别的预测概率 pt
        pt = torch.exp(-F.cross_entropy(inputs, targets, reduction='none'))
        focal_term = (1 - pt) ** self.gamma

        # 3. 结合两者
        loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


class RobertaEmotionFineTuner(nn.Module):
    def __init__(self, model_name="roberta-base", hidden_dim=256, num_classes=6):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name, add_pooling_layer=False)
        self.classifier_head = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        self.regressor_head = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # 取最后四层 hidden states
        all_layers = outputs.hidden_states
        last_four_layers = torch.stack(all_layers[-4:])  # [4, Batch, Seq, 768]
        cat_hidden = torch.mean(last_four_layers, dim=0)  # [Batch, Seq, 768]

        # 取 CLS 位置
        cls_output = cat_hidden[:, 0, :]  # [Batch, 768]

        logits = self.classifier_head(cls_output)
        va = self.regressor_head(cls_output)
        return logits, va

def save_confusion_matrix(y_true, y_pred, class_names, epoch):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(Config.MODEL_SAVE_DIR, f'cm_epoch_{epoch}.png'))
    plt.close()

def plot_training_history(train_losses, val_losses, macro_f1s, weighted_f1s):
    epochs = range(1, len(train_losses) + 1)

    # --- 图 1：Loss 变化曲线 ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_history.png', dpi=300)
    plt.close()
    # --- 图 2：F1-Score 变化曲线 ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, macro_f1s, 'g-s', label='Macro F1')
    plt.plot(epochs, weighted_f1s, 'm-s', label='Weighted F1')
    plt.title('F1-Score History on Validation Set')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('f1_history.png', dpi=300)
    plt.close()

def evaluate_finetune(model, loader, device, criterion_cls, criterion_reg, class_names, epoch):
    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0.0

    with torch.no_grad():
        for input_ids, mask, labels, va in loader:
            # 别忘了把 va 也推到 GPU
            input_ids, mask, labels, va = input_ids.to(device), mask.to(device), labels.to(device), va.to(device)

            with torch.amp.autocast('cuda'):  # 更新了较新的 API 写法
                logits, pred_va = model(input_ids, mask)
                loss_cls = criterion_cls(logits, labels)
                loss_reg = criterion_reg(pred_va, va)
                # 严格对齐训练时的 Loss 结构
                loss = Config.LAMBDA_CLS * loss_cls + Config.LAMBDA_REG * loss_reg

            val_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"评估报告 (Epoch {epoch}):")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))

    return weighted_f1, macro_f1, val_loss / len(loader)

def train():
    train_losses = []
    val_losses = []
    macro_f1s = []
    weighted_f1s = []
    # --- 1. 初始化 ---
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", clean_up_tokenization_spaces=True)
    model = RobertaEmotionFineTuner(model_name="roberta-base").to(Config.DEVICE)
    fgm = FGM(model)

    # --- 2. 数据准备 ---
    train_dataset = FineTuneDataset("iemocap_train.pkl", "meld_train_flattened.pkl", tokenizer,
                                    max_len=Config.MAX_SEQ_LEN)
    val_dataset = FineTuneDataset("iemocap_val.pkl", "meld_dev_flattened.pkl", tokenizer, max_len=Config.MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # --- 3. 差异化学习率 & 优化器 ---
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.roberta.named_parameters()], "lr": 2e-6, "weight_decay": 0.05},
        {"params": [p for n, p in model.classifier_head.named_parameters()], "lr": 1e-4, "weight_decay": 0.01},
        {"params": [p for n, p in model.regressor_head.named_parameters()], "lr": 1e-4}
    ]

    # 计算总步数 (考虑梯度累积)
    total_steps = (len(train_loader) // Config.ACCUMULATION_STEPS) * Config.EPOCHS
    warmup_steps = int(total_steps * 0.1)
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # --- 4. 损失函数 (带 Label Smoothing) ---
    weights = torch.tensor([1.19, 2.0, 0.6, 1.11, 1.39, 1.44]).to(Config.DEVICE)
    criterion_cls = FocalLoss(alpha=weights, gamma=2.0, label_smoothing=0.1)
    criterion_reg = nn.MSELoss()

    scaler = torch.amp.GradScaler('cuda')
    class_names = ["Anger", "Frustrated", "Neutral", "Happiness", "Excited", "Sadness"]

    best_macro_f1 = 0
    patience_counter = 0

    print(f"训练开始...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for i, (input_ids, mask, labels, va) in enumerate(train_loader):
            input_ids, mask, labels, va = input_ids.to(Config.DEVICE), mask.to(Config.DEVICE), labels.to(
                Config.DEVICE), va.to(Config.DEVICE)

            # --- 第一步：标准前向 & 反向传播 ---
            with amp.autocast('cuda'):
                logits, pred_va = model(input_ids, mask)
                loss_cls = criterion_cls(logits, labels)
                loss_reg = criterion_reg(pred_va, va)
                loss = (Config.LAMBDA_CLS * loss_cls + Config.LAMBDA_REG * loss_reg) / Config.ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            # --- 第二步：FGM 对抗攻击 ---
            fgm.attack()
            with amp.autocast('cuda'):
                logits_adv, _ = model(input_ids, mask)
                loss_adv = (Config.LAMBDA_CLS * criterion_cls(logits_adv, labels)) / Config.ACCUMULATION_STEPS
            scaler.scale(loss_adv).backward()  # 累加扰动后的梯度
            fgm.restore()

            # --- 第三步：参数更新 ---
            if (i + 1) % Config.ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()

                if scaler.get_scale() >= scale_before:
                    scheduler.step()

                optimizer.zero_grad()

            running_loss += loss.item() * Config.ACCUMULATION_STEPS
        # 计算耗时
        epoch_mins = (time.time() - start_time) / 60
        avg_train_loss = running_loss / len(train_loader)

        # 评估
        val_w_f1, val_m_f1, avg_val_loss = evaluate_finetune(
            model, val_loader, Config.DEVICE, criterion_cls, criterion_reg, class_names, epoch
        )

        # 获取当前底座学习率进行监控
        current_lr = optimizer.param_groups[0]['lr']

        print(f"--- Epoch {epoch:02d} Summary ---")
        print(f"Time: {epoch_mins:.2f} min | LR: {current_lr:.8e}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Macro-F1: {val_m_f1:.4f}")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        macro_f1s.append(val_m_f1)
        weighted_f1s.append(val_w_f1)

        if val_m_f1 > best_macro_f1:
            best_macro_f1 = val_m_f1
            torch.save(model.state_dict(), os.path.join(Config.MODEL_SAVE_DIR, Config.BEST_MODEL_NAME))
            print(f"发现更佳模型，已保存！")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"早停触发，训练结束。")
                break
    plot_training_history(train_losses, val_losses, macro_f1s, weighted_f1s)
if __name__ == "__main__":
    train()