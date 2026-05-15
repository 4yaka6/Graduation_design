import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from traintext import RobertaEmotionFineTuner, FineTuneDataset
import matplotlib.pyplot as plt
import seaborn as sns

def run_detailed_evaluation(model, data_loader, device, class_names, dataset_name):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, mask, labels, _ in data_loader:
            input_ids, mask = input_ids.to(device), mask.to(device)
            # 模型前向传播
            with torch.amp.autocast('cuda'):
                logits, _ = model(input_ids, mask)

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算各项指标
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    # 处理除零风险：如果某类在测试集中不存在，Acc 设为 0
    with np.errstate(divide='ignore', invalid='ignore'):
        cls_acc = cm.diagonal() / cm.sum(axis=1)
        cls_acc = np.nan_to_num(cls_acc)

    report_dict = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # 打印表格
    print("\n" + "=" * 55)
    print(f"  测试报告: {dataset_name}")
    print("=" * 55)
    print(f"{'情感类别':<15} | {'准确率 (Acc)':<12} | {'F1-Score':<12} | {'样本数':<8}")
    print("-" * 55)

    for i, name in enumerate(class_names):
        f1 = report_dict[name]['f1-score']
        acc = cls_acc[i]
        support = report_dict[name]['support']
        print(f"{name:<15} | {acc:<12.4f} | {f1:<12.4f} | {int(support):<8}")

    print("-" * 55)

    # 总体指标
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    total_acc = accuracy_score(all_labels, all_preds)

    print(f"总体准确率 (Overall Acc):  {total_acc:.4f}")
    print(f"加权 F1 (Weighted F1):    {weighted_f1:.4f}")
    print(f"宏平均 F1 (Macro F1):     {macro_f1:.4f}")
    print("=" * 55)

    return all_labels.tolist(), all_preds.tolist()


def plot_final_confusion_matrix(all_true, all_pred, class_names):
    cm = confusion_matrix(all_true, all_pred, labels=range(len(class_names)))

    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_perc = np.nan_to_num(cm_perc)  # 处理除零情况

    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_theme(style="white")

    annot = []
    for i in range(len(class_names)):
        row = []
        for j in range(len(class_names)):
            s = f"{cm[i, j]}\n({cm_perc[i, j]:.1%})"
            row.append(s)
        annot.append(row)

    ax = sns.heatmap(
        cm_perc,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 11, "weight": "bold"},
        cbar_kws={'label': 'Recall Rate'}
    )

    plt.title("Combined Confusion Matrix (IEMOCAP + MELD)", fontsize=15, pad=20)
    plt.ylabel("Actual Emotion", fontsize=12)
    plt.xlabel("Predicted Emotion", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()

    save_path = "final_combined_cm.png"
    plt.savefig(save_path)
    plt.show()
    print(f"混淆矩阵已保存至: {save_path}")

def test():
    # 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Anger", "Frustrated", "Neutral", "Happiness", "Excited", "Sadness"]
    model_path = "checkpoints/loss_0.9681_f1_0.5285.pth"
    iemocap_test_path = "iemocap_test.pkl"
    meld_test_path = "meld_test_flattened.pkl"

    print(f"正在准备测试... 使用设备: {device}")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", clean_up_tokenization_spaces=True)
    model = RobertaEmotionFineTuner(model_name="roberta-base", num_classes=6)

    if not os.path.exists(model_path):
        print(f"找不到权重文件: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("模型权重加载成功。")
    MAX_LEN = 64

    ds_ie = FineTuneDataset(iemocap_pkl=iemocap_test_path, tokenizer=tokenizer, max_len=MAX_LEN)
    ds_me = FineTuneDataset(meld_pkl=meld_test_path, tokenizer=tokenizer, max_len=MAX_LEN)

    loader_ie = DataLoader(ds_ie, batch_size=64, shuffle=False)
    loader_me = DataLoader(ds_me, batch_size=64, shuffle=False)

    labels_ie, preds_ie = run_detailed_evaluation(model, loader_ie, device, class_names, "IEMOCAP Testset")
    labels_me, preds_me = run_detailed_evaluation(model, loader_me, device, class_names, "MELD Testset")

    all_true = labels_ie + labels_me
    all_pred = preds_ie + preds_me

    all_true_np = np.array(all_true)
    all_pred_np = np.array(all_pred)

    cm_total = confusion_matrix(all_true_np, all_pred_np, labels=range(len(class_names)))
    cls_acc_total = cm_total.diagonal() / cm_total.sum(axis=1)
    report_total = classification_report(all_true_np, all_pred_np, target_names=class_names, output_dict=True,
                                         zero_division=0)

    print("  IEMOCAP + MELD 联合测试报告")
    print(f"{'情感类别':<15} | {'准确率 (Acc)':<12} | {'F1-Score':<12} | {'总样本数':<8}")
    print("-" * 65)

    for i, name in enumerate(class_names):
        f1 = report_total[name]['f1-score']
        acc = cls_acc_total[i]
        support = report_total[name]['support']
        print(f"{name:<15} | {acc:<12.4f} | {f1:<12.4f} | {int(support):<8}")

    print("-" * 65)
    print(f"联合总体准确率 (Total Acc): {accuracy_score(all_true_np, all_pred_np):.4f}")
    print(f"联合加权 F1 (Weighted F1): {f1_score(all_true_np, all_pred_np, average='weighted'):.4f}")
    print(f"联合宏平均 F1 (Macro F1):  {f1_score(all_true_np, all_pred_np, average='macro'):.4f}")
    plot_final_confusion_matrix(all_true, all_pred, class_names)

if __name__ == "__main__":
    test()