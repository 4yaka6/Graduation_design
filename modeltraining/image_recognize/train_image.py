import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import get_cosine_schedule_with_warmup

torch.backends.cudnn.benchmark = True

# ==========================================
# 1. Config
# ==========================================
CLASS_NAMES = ["Anger", "Frustrated", "Neutral", "Happiness", "Excited", "Sadness"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

DATA_DIR = "reorganized_dataset"
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 5e-5
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "checkpoints_image1"

os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================================
# 2. Dataset & Transforms (保持你的增强逻辑)
# ==========================================
class FERDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_paths = []
        self.labels = []
        self.transform = transform
        if os.path.exists(root_dir):
            for cls_name in os.listdir(root_dir):
                cls_dir = os.path.join(root_dir, cls_name)
                if os.path.isdir(cls_dir) and cls_name in CLASS_TO_IDX:
                    label_idx = CLASS_TO_IDX[cls_name]
                    for img_name in os.listdir(cls_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.img_paths.append(os.path.join(cls_dir, img_name))
                            self.labels.append(label_idx)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, torch.tensor(self.labels[idx], dtype=torch.long)


train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ==========================================
# 3. Model (冻结 + 强分类头)
# ==========================================
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


# ==========================================
# 4. 修复后的 FocalLoss (解决平滑冲突)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # 用不平滑的计算 pt 以保证权重准确
        ce_loss_clean = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss_clean)
        # 用平滑后的计算最终 loss
        ce_loss_smooth = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none',
                                                     label_smoothing=self.label_smoothing)
        return (((1 - pt) ** self.gamma) * ce_loss_smooth).mean()


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE.type):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy());
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, labels=range(len(CLASS_NAMES)), average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, labels=range(len(CLASS_NAMES)), average='weighted', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, labels=range(len(CLASS_NAMES)), target_names=CLASS_NAMES,
                                   zero_division=0)
    return total_loss / len(loader), macro_f1, weighted_f1, acc, report


def plot_metrics(history):
    epochs = range(1, len(history['train_loss']) + 1)

    # 图 1: Loss 变化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 图 2: F1 Scores 变化
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['macro_f1'], 'g-', label='Macro F1')
    plt.plot(epochs, history['weighted_f1'], 'y-', label='Weighted F1')
    plt.title('F1 Scores (Macro vs Weighted)')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_metrics.png'), dpi=300)
    print(f"📊 训练指标图已保存至: {SAVE_DIR}/training_metrics.png")
    plt.close()

# ==========================================
# 5. Main Train (整合早停逻辑)
# ==========================================
def main():
    print(f"🚀 Device: {DEVICE} ")

    history = {
        'train_loss': [], 'val_loss': [],
        'macro_f1': [], 'weighted_f1': []
    }

    train_dataset = FERDataset(os.path.join(DATA_DIR, 'train'), train_transforms)
    val_dataset = FERDataset(os.path.join(DATA_DIR, 'val'), val_transforms)

    class_weights = torch.tensor([0.768, 1.0, 0.944, 0.971, 1.218, 1.268], dtype=torch.float).to(DEVICE)
    criterion = FocalLoss(alpha=class_weights, gamma=1.5, label_smoothing=0.1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
                            persistent_workers=True)

    model = FullResNetEmotion(len(CLASS_NAMES)).to(DEVICE)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)
    scaler = torch.amp.GradScaler(device=DEVICE.type)

    # --- 早停控制变量 ---
    best_f1 = 0
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, start = 0, time.time()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer);
            scaler.update();
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        val_loss, val_f1, w_f1, val_acc, report = evaluate(model, val_loader, criterion)
        print(
            f"\nEpoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Macro F1: {val_f1:.4f}")

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['macro_f1'].append(val_f1)
        history['weighted_f1'].append(w_f1)

        if val_f1 > 0.65 and val_loss < 0.6:
            stable_name = f"stable_ep{epoch + 1}_f1_{val_f1:.4f}_loss_{val_loss:.4f}.pth"
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, stable_name))
            print(f"🛡️ Stable Model Captured: {stable_name}")

        # --- 早停判定逻辑 ---
        if val_f1 > best_f1:
            best_f1 = val_f1
            early_stop_counter = 0  # 重置计数器
            save_name = f"best_model_f1_{val_f1:.4f}.pth"
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, save_name))
            print(f"✅ 模型表现提升，已保存: {save_name}")
        else:
            early_stop_counter += 1
            print(f"⚠️ 表现未提升，早停计数: {early_stop_counter}/{PATIENCE}")

        if early_stop_counter >= PATIENCE:
            print(f"🛑 连续 {PATIENCE} 个 Epoch 未提升，触发早停。")
            break

    plot_metrics(history)
    print(f"🎉 Training Complete. Best Macro F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()