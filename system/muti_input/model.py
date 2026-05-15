import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import RobertaModel

# 心理学 PAD/VA 映射表
VA_MAP = torch.tensor([
    [-0.51, 0.59],  # Anger
    [-0.64, 0.52],  # Frustrated
    [0.0, 0.0],     # Neutral
    [0.81, 0.51],   # Happiness
    [0.62, 0.82],   # Excited
    [-0.63, -0.27]  # Sadness
])

class ImageModel(nn.Module):
    """简化后的图像模型：输出 6 分类概率并映射为 VA 坐标"""
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.resnet18()
        in_features = self.backbone.fc.in_features
        # 严格保持参数：Dropout(0.5) + Linear
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        logits = self.backbone(x)
        # 计算 6 分类概率 (Softmax)
        probs = F.softmax(logits, dim=1)
        # 加权融合 VA 坐标: (Batch, 6) @ (6, 2) -> (Batch, 2)
        va_coords = torch.mm(probs, VA_MAP.to(logits.device))
        return probs, va_coords

class TextModel(nn.Module):
    """文本模型：保持原参数不变，输出分类 Logits 与回归 VA"""
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
        # 取最后四层平均
        all_layers = outputs.hidden_states
        last_four_layers = torch.stack(all_layers[-4:])
        cat_hidden = torch.mean(last_four_layers, dim=0)
        # 取 CLS 位置
        cls_output = cat_hidden[:, 0, :]

        logits = self.classifier_head(cls_output)
        va = self.regressor_head(cls_output)
        return logits, va