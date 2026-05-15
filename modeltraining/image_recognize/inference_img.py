import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

# ================= 配置与对齐 =================
CLASS_NAMES = ["Anger", "Frustrated", "Neutral", "Happiness", "Excited", "Sadness"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 指向你训练出的最佳模型权重
MODEL_PATH = "checkpoints_image1/best_model_f1_0.6422.pth"
TEST_DIR = "reorganized_dataset/test"
BATCH_SIZE = 32


# ================= 1. 模型结构定义 (必须与训练一致) =================
class FullResNetEmotion(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.resnet18()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ================= 2. 测试集加载器 =================
class FERTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_paths = []
        self.labels = []
        self.transform = transform

        for cls_name in CLASS_NAMES:
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.exists(cls_dir):
                label_idx = CLASS_TO_IDX[cls_name]
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.img_paths.append(os.path.join(cls_dir, img_name))
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


# 推理预处理
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ================= 3. 核心测试逻辑 =================
def run_test():
    print(f"🔍 正在加载测试集: {TEST_DIR}...")
    test_dataset = FERTestDataset(TEST_DIR, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"📦 加载权重: {MODEL_PATH}...")
    model = FullResNetEmotion(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    print("🚀 开始批量推理...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            logits = model(images)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels)

    # --- 指标计算 ---
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', labels=range(len(CLASS_NAMES)), zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', labels=range(len(CLASS_NAMES)), zero_division=0)

    print("\n" + "=" * 50)
    print("📈 图像模态测试集性能报告")
    print("-" * 50)
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Macro F1-Score:   {macro_f1:.4f}")
    print(f"Weighted F1-Score:{weighted_f1:.4f}")
    print("-" * 50)

    # 打印详细分类报告
    report = classification_report(
        all_labels, all_preds,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        zero_division=0
    )
    print(report)

    # --- 绘制混淆矩阵 ---
    plot_cm(all_labels, all_preds)


def plot_cm(y_true, y_pred):
    # 1. 计算基础混淆矩阵（数量）
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))

    # 2. 计算归一化矩阵（百分比）
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    # 处理除以 0 的情况（比如 Frustrated 类样本为 0）
    cm_norm = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(cm.astype('float')), where=cm_sum != 0)

    # 3. 构造标注文字矩阵：格式为 "数量\n(百分比%)"
    # 使用 np.vectorize 或循环遍历来生成字符串数组
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            count = cm[i, j]
            percent = cm_norm[i, j] * 100
            # 如果该行总数为0，则只显示数量
            if cm_sum[i] > 0:
                annot[i, j] = f"{count}\n({percent:.1f}%)"
            else:
                annot[i, j] = f"{count}"

    # 4. 绘图
    plt.figure(figsize=(12, 10), dpi=300)
    sns.heatmap(
        cm_norm,  # 颜色深浅依然由百分比（召回率）决定，这样对比更直观
        annot=annot,  # 使用我们自定义的文字矩阵
        fmt="",  # 因为输入的是字符串，fmt 设为空
        cmap="Blues",  # 修改底色为蓝色
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        annot_kws={"size": 10}  # 调整字体大小以适应两行文字
    )

    plt.title("Confusion Matrix - Image Modality (ResNet18)", fontsize=16, pad=20)
    plt.ylabel("True Emotion", fontsize=12)
    plt.xlabel("Predicted Emotion", fontsize=12)
    plt.tight_layout()

    save_path = "image_confusion_matrix_blue.png"
    plt.savefig(save_path)
    print(f"✅ 混淆矩阵图已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    run_test()