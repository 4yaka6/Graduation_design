import torch
import torch.nn as nn

class ArousalModel(nn.Module):
    def __init__(self):
        super(ArousalModel, self).__init__()

        # --- MFCC 支路 ---
        self.mfcc_cnn = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(6), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(6, 6, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(6), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.mfcc_lstm = nn.LSTM(input_size=66, hidden_size=256, batch_first=True)

        # --- GTF 支路 ---
        self.gtf_cnn = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(6), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.gtf_lstm = nn.LSTM(input_size=132, hidden_size=256, batch_first=True)

        # --- 融合层 (修改处 1：256 -> 128) ---
        # 这里的输入是 256+256=512，输出必须改为 128 以匹配权重文件 [128, 512]
        self.fusion_dense = nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.global_bilstm = nn.LSTM(
            input_size=128,
            hidden_size=128,  # 这里保持 128 匹配 [512, 128] 的 ih 权重
            batch_first=True,
            bidirectional=True
        )

        # --- 输出层 (修改处 3：128*2 保持不变，因为 BiLSTM 128 对应 256) ---
        self.out = nn.Linear(128 * 2, 1)

    def forward(self, m, g):
        B, T, _, _ = m.shape

        # MFCC 支路推理
        m_in = m.view(B * T, 1, 96, 44)
        m_c = self.mfcc_cnn(m_in)
        m_l_in = m_c.permute(0, 2, 1, 3).flatten(start_dim=2)
        _, (h_m, _) = self.mfcc_lstm(m_l_in)
        m_feat = h_m[-1]

        # GTF 支路推理
        g_in = g.view(B * T, 1, 12, 44)
        g_c = self.gtf_cnn(g_in)
        g_l_in = g_c.permute(0, 2, 1, 3).flatten(start_dim=2)
        _, (h_g, _) = self.gtf_lstm(g_l_in)
        g_feat = h_g[-1]

        # 融合与全局时序
        combined = torch.cat([m_feat, g_feat], dim=-1)
        # dense_out 形状: (B, T, 128)
        dense_out = self.fusion_dense(combined).view(B, T, -1)

        bi_out, _ = self.global_bilstm(dense_out)
        return self.out(bi_out)


class ValenceModel(nn.Module):
    def __init__(self):
        super(ValenceModel, self).__init__()

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


