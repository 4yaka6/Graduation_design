import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from sklearn.metrics import r2_score


# ===================== 1. 环境固定 =====================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)

CONFIG = {
    "seq_len": 60,
    "batch_size": 16,
    "lr": 4e-4,
    "epochs": 80,
    "delay": 1,
    "weight_decay": 1e-3
}


# ===================== 2. 异构特征拼接架构 =====================
class ValenceModelPro(nn.Module):
    def __init__(self):
        super(ValenceModelPro, self).__init__()

        # --- GTF 支路 (CNN + LSTM) ---
        self.gtf_cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.gtf_lstm = nn.LSTM(176, 128, batch_first=True)

        # --- Harmony 支路 (24维线性对齐) ---
        self.harm_embed = nn.Sequential(
            nn.Linear(24, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # --- 融合层 ---
        self.fusion_dense = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # --- 全局时序建模 ---
        self.global_bilstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.alpha = nn.Parameter(torch.tensor([0.2]))

        # --- 输出头 ---
        self.out_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, g, h):
        B, T, _, _ = g.shape

        # 1. GTF 支路
        g_in = g.view(B * T, 1, 12, 44)
        g_c = self.gtf_cnn(g_in).permute(0, 2, 1, 3).flatten(start_dim=2)
        _, (h_g, _) = self.gtf_lstm(g_c)
        g_feat = h_g[-1]  # (B*T, 128)

        # 2. Harmony 支路 (均值池化)
        h_avg = torch.mean(h, dim=2)  # (B, T, 24)
        h_feat = self.harm_embed(h_avg.view(B * T, 24))  # (B*T, 64)

        # 3. 拼接与融合
        combined = torch.cat([g_feat, h_feat], dim=-1)  # (B*T, 192)
        identity = self.fusion_dense(combined).view(B, T, -1)

        # 4. 全局 BiLSTM
        bi_out, _ = self.global_bilstm(identity)
        fused = identity + torch.tanh(self.alpha) * bi_out

        return self.out_head(fused)


# ===================== 3. 数据加载器 =====================
class ValenceProDataset(Dataset):
    def __init__(self, set_name="train", data_path="data/processed_data"):
        self.set_name = set_name
        g = np.load(os.path.join(data_path, f"{set_name}_gtf.npy")).astype(np.float32)
        h = np.load(os.path.join(data_path, f"{set_name}_harmony.npy")).astype(np.float32)[:, :, :24]
        labels = np.load(os.path.join(data_path, f"{set_name}_labels.npy")).astype(np.float32)

        delay = CONFIG["delay"]
        self.gtf = g[:-delay] if delay > 0 else g
        self.harm = h[:-delay] if delay > 0 else h
        target = labels[delay:, 1] if delay > 0 else labels[:, 1]

        self.gtf = (self.gtf - self.gtf.mean()) / (self.gtf.std() + 1e-8)
        self.harm = (self.harm - self.harm.mean()) / (self.harm.std() + 1e-8)
        self.target = target

    def __len__(self):
        return len(self.target) // CONFIG["seq_len"]

    def __getitem__(self, idx):
        s, e = idx * CONFIG["seq_len"], (idx + 1) * CONFIG["seq_len"]
        return torch.FloatTensor(self.gtf[s:e]), torch.FloatTensor(self.harm[s:e]), torch.FloatTensor(
            self.target[s:e]).unsqueeze(-1)


# ===================== 4. 训练与保存逻辑 =====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "models_Valence_Pro"
    os.makedirs(save_dir, exist_ok=True)

    train_loader = DataLoader(ValenceProDataset("train"), batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(ValenceProDataset("val"), batch_size=CONFIG["batch_size"], shuffle=False)

    model = ValenceModelPro().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    print(f"🚀 启动【最终完善版】训练 | 逐轮保存模式开启")

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for g_batch, h_batch, y in train_loader:
            g_batch, h_batch, y = g_batch.to(device), h_batch.to(device), y.to(device)
            optimizer.zero_grad()
            p = model(g_batch, h_batch)
            loss = criterion(p, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * g_batch.size(0)

        model.eval()
        val_loss, all_p, all_t = 0.0, [], []
        with torch.no_grad():
            for g_batch, h_batch, y in val_loader:
                g_batch, h_batch, y = g_batch.to(device), h_batch.to(device), y.to(device)
                p = model(g_batch, h_batch)
                val_loss += criterion(p, y).item() * g_batch.size(0)
                all_p.append(p.cpu().numpy().flatten())
                all_t.append(y.cpu().numpy().flatten())

        avg_train, avg_val = train_loss / len(train_loader.dataset), val_loss / len(val_loader.dataset)
        r2 = r2_score(np.concatenate(all_t), np.concatenate(all_p))
        scheduler.step(avg_val)

        # 打印状态
        print(f"Epoch {epoch:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | R2: {r2:.4f}")
        # 这里使用 batch 内最后一个 p 和 y 计算均值进行监控
        print(f"Pred Mean: {p.mean().item():.4f}, Target Mean: {y.mean().item():.4f}")

        # 逐轮保存模型
        save_path = os.path.join(save_dir, f"ep{epoch:02d}_loss{avg_val:.4f}_r2{r2:.4f}.pth")
        torch.save(model.state_dict(), save_path)