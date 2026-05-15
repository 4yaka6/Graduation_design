import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from sklearn.metrics import r2_score


# ===================== 1. 实验环境固定 =====================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)

CONFIG = {
    "seq_len": 60,
    "batch_size": 32,
    "lr": 3e-5,
    "epochs": 40,
    "delay": 1,
    "weight_decay": 1e-2
}


# ===================== 2. 架构图核心实现 =====================
class ArousalModelPro(nn.Module):
    def __init__(self):
        super(ArousalModelPro, self).__init__()

        # --- MFCC 支路 (图示 x2 模块) ---
        # 输入: (B*T, 1, 96, 44)
        self.mfcc_cnn = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(6), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # -> (6, 48, 22)大致
            nn.Conv2d(6, 6, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(6), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))  # -> (6, 24, 11)大致
        )
        # 展平后进入第一层 LSTM (6通道 * 11宽 = 66)
        self.mfcc_lstm = nn.LSTM(input_size=66, hidden_size=256, batch_first=True)

        # --- GTF 支路 (图示 x1 模块) ---
        # 输入: (B*T, 1, 12, 44)
        self.gtf_cnn = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(6), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))  # -> (6, 6, 22)大致
        )
        # 展平后进入第一层 LSTM (6通道 * 22宽 = 132)
        self.gtf_lstm = nn.LSTM(input_size=132, hidden_size=256, batch_first=True)

        # --- 融合层 (Dense 1x128) ---
        self.fusion_dense = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # --- 全局时序建模 (BiLSTM 128) ---
        self.global_bilstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.out = nn.Linear(128 * 2, 1)

    def forward(self, m, g):
        B, T, _, _ = m.shape

        # MFCC 支路
        m_in = m.view(B * T, 1, 96, 44)
        m_c = self.mfcc_cnn(m_in)  # (B*T, 6, 24, 11)
        m_l_in = m_c.permute(0, 2, 1, 3).flatten(start_dim=2)  # (B*T, 24, 66)
        _, (h_m, _) = self.mfcc_lstm(m_l_in)
        m_feat = h_m[-1]  # 取最后隐藏状态 (B*T, 256)

        # GTF 支路
        g_in = g.view(B * T, 1, 12, 44)
        g_c = self.gtf_cnn(g_in)  # (B*T, 6, 6, 22)
        g_l_in = g_c.permute(0, 2, 1, 3).flatten(start_dim=2)  # (B*T, 6, 132)
        _, (h_g, _) = self.gtf_lstm(g_l_in)
        g_feat = h_g[-1]  # (B*T, 256)

        # 融合
        combined = torch.cat([m_feat, g_feat], dim=-1)  # (B*T, 512)
        dense_out = self.fusion_dense(combined).view(B, T, -1)  # (B, T, 128)

        # 全局时序
        bi_out, _ = self.global_bilstm(dense_out)
        return self.out(bi_out)


# ===================== 3. 数据加载 (双输入版) =====================
class ArousalProDataset(Dataset):
    def __init__(self, set_name="train", data_path="data/processed_data"):
        m = np.load(os.path.join(data_path, f"{set_name}_mfcc.npy")).astype(np.float32)  # (N, 96, 44)
        g = np.load(os.path.join(data_path, f"{set_name}_gtf.npy")).astype(np.float32)  # (N, 12, 44)
        labels = np.load(os.path.join(data_path, f"{set_name}_labels.npy")).astype(np.float32)

        # 延迟补偿
        delay = CONFIG["delay"]
        self.mfcc = m[:-delay] if delay > 0 else m
        self.gtf = g[:-delay] if delay > 0 else g
        target = labels[delay:, 0] if delay > 0 else labels[:, 0]

        # 分别标准化
        self.mfcc = (self.mfcc - self.mfcc.mean()) / (self.mfcc.std() + 1e-8)
        self.gtf = (self.gtf - self.gtf.mean()) / (self.gtf.std() + 1e-8)
        self.target = target

    def __len__(self):
        return len(self.target) // CONFIG["seq_len"]

    def __getitem__(self, idx):
        s, e = idx * CONFIG["seq_len"], (idx + 1) * CONFIG["seq_len"]
        return (torch.FloatTensor(self.mfcc[s:e]),
                torch.FloatTensor(self.gtf[s:e]),
                torch.FloatTensor(self.target[s:e]).unsqueeze(-1))


# ===================== 4. 训练引擎 =====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("models_Arousal", exist_ok=True)

    train_loader = DataLoader(ArousalProDataset("train"), batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(ArousalProDataset("val"), batch_size=CONFIG["batch_size"], shuffle=False)

    model = ArousalModelPro().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    print(f"🚀 启动图示版 Arousal 模型 | 双支路解耦 | 设备: {device}")

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_l = 0.0
        for m_batch, g_batch, y in train_loader:
            m_batch, g_batch, y = m_batch.to(device), g_batch.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(m_batch, g_batch), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪稳住 R2
            optimizer.step()
            train_l += loss.item() * m_batch.size(0)

        model.eval()
        val_l, all_p, all_t = 0.0, [], []
        with torch.no_grad():
            for m_batch, g_batch, y in val_loader:
                m_batch, g_batch, y = m_batch.to(device), g_batch.to(device), y.to(device)
                p = model(m_batch, g_batch)
                val_l += criterion(p, y).item() * m_batch.size(0)
                all_p.append(p.cpu().numpy().flatten())
                all_t.append(y.cpu().numpy().flatten())

        avg_train, avg_val = train_l / len(train_loader.dataset), val_l / len(val_loader.dataset)
        r2 = r2_score(np.concatenate(all_t), np.concatenate(all_p))
        scheduler.step(avg_val)

        print(f"Epoch {epoch:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | R2: {r2:.4f}")
        torch.save(model.state_dict(), f"models_Arousal/ep{epoch:02d}_loss{avg_val:.4f}_r2{r2:.4f}.pth")