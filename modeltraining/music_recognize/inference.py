import torch
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score
from CLDNN_BILSTM import CLDNN_BILSTM_ATTENTION  # 确保文件名和类名一致


def evaluate_performance(model_path="best_model.pth", data_path="data/processed_data"):
    # 1. 环境与模型准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLDNN_BILSTM_ATTENTION().to(device)

    if not os.path.exists(model_path):
        print(f"❌ 错误：在 {model_path} 找不到模型文件！")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. 加载验证/测试数据
    try:
        val_mfcc = np.load(os.path.join(data_path, "val_mfcc.npy"), mmap_mode='r')
        val_gtf = np.load(os.path.join(data_path, "val_gtf.npy"), mmap_mode='r')
        val_labels = np.load(os.path.join(data_path, "val_labels.npy"), mmap_mode='r')
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return

    num_songs = len(val_labels) // 60
    all_preds = []
    all_labels = []

    print(f"🚀 正在评估 {num_songs} 首歌曲 (12GB 显存模式)...")

    # 3. 推理循环
    with torch.no_grad():
        # 12G 显存可以支持较大的 Batch，这里设为 16 首歌（即 16*60 个片段）
        batch_size = 16
        for i in range(0, num_songs, batch_size):
            end_idx = min(i + batch_size, num_songs)

            # 提取当前 Batch 的数据
            s_ptr, e_ptr = i * 60, end_idx * 60

            # 转换为 Tensor 并移动到 GPU
            m = torch.from_numpy(np.array(val_mfcc[s_ptr:e_ptr])).float().to(device)
            g = torch.from_numpy(np.array(val_gtf[s_ptr:e_ptr])).float().to(device)
            y = val_labels[s_ptr:e_ptr]  # 真实标签 (Batch*60, 2)

            # 调整形状为 (Batch, 60, H, W) 以匹配模型 forward
            current_batch_size = end_idx - i
            m = m.view(current_batch_size, 60, 96, 44)
            g = g.view(current_batch_size, 60, 12, 44)

            # 推理
            preds = model(m, g)  # 输出 (Batch, 60, 2)

            # 保存结果用于后续计算
            all_preds.append(preds.cpu().numpy().reshape(-1, 2))
            all_labels.append(y)

    # 4. 指标汇总计算
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # Arousal (唤醒度)
    a_rmse = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    a_r2 = r2_score(y_true[:, 0], y_pred[:, 0])

    # Valence (正负效价)
    v_rmse = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    v_r2 = r2_score(y_true[:, 1], y_pred[:, 1])

    # 综合指标
    total_r2 = (a_r2 + v_r2) / 2

    # 5. 格式化输出
    print("\n" + "=" * 40)
    print(f"{'模型评估报告':^34}")
    print("=" * 40)
    print(f"指标项目         | Arousal  | Valence")
    print(f"----------------|----------|----------")
    print(f"RMSE (越小越好)  | {a_rmse:.4f}   | {v_rmse:.4f}")
    print(f"R²   (越大越好)  | {a_r2:.4f}   | {v_r2:.4f}")
    print("-" * 40)
    print(f"✨ 最终总 R² (Avg): {total_r2:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    evaluate_performance()