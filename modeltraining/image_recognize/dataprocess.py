import os
import shutil
import random
from tqdm import tqdm

# 原始数据根目录
raw_data_root = "data"
# 重新划分后的输出目录
output_root = "reorganized_dataset"

# 标签映射表
# 排除 Contempt 和 Disgust
label_mapping = {
    'anger': 'Anger', 'Anger': 'Anger',
    'neutral': 'Neutral', 'Neutral': 'Neutral',
    'sad': 'Sadness', 'Sad': 'Sadness',
    'happy': 'Happiness', 'Happy': 'Happiness',
    'surprise': 'Excited', 'Surprise': 'Excited',
    'fear': 'Anger'
}

# 目标比例 5:2:3
split_ratio = {'train': 0.5, 'val': 0.2, 'test': 0.3}


# ==========================================

def process_and_split():
    # 1. 扫描并汇总所有符合条件的图片路径
    all_category_data = {target: [] for target in set(label_mapping.values())}

    # 遍历原有的 Train 和 Test
    for split in ['Train', 'Test']:
        split_path = os.path.join(raw_data_root, split)
        if not os.path.exists(split_path):
            continue

        for folder_name in os.listdir(split_path):
            if folder_name in label_mapping:
                target_label = label_mapping[folder_name]
                folder_path = os.path.join(split_path, folder_name)

                # 获取该文件夹下所有合法图片
                images = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_category_data[target_label].extend(images)

    # 2. 执行迁移
    print("开始重新规划文件布局...")
    for label, file_list in all_category_data.items():
        random.shuffle(file_list)  # 随机打乱

        total = len(file_list)
        train_idx = int(total * split_ratio['train'])
        val_idx = train_idx + int(total * split_ratio['val'])

        indices = {
            'train': file_list[:train_idx],
            'val': file_list[train_idx:val_idx],
            'test': file_list[val_idx:]
        }

        for stage, paths in indices.items():
            dest_dir = os.path.join(output_root, stage, label)
            os.makedirs(dest_dir, exist_ok=True)

            for p in tqdm(paths, desc=f"{stage} - {label}", leave=False):
                # 为防止同名冲突，给文件名加个前缀（原所属文件夹名）
                prefix = os.path.basename(os.path.dirname(p))
                fname = f"{prefix}_{os.path.basename(p)}"
                shutil.copy(p, os.path.join(dest_dir, fname))

    print(f"\n处理完成！新数据集已保存在: {output_root}")
    # 打印最终统计结果
    for stage in ['train', 'val', 'test']:
        print(f"\n--- {stage.upper()} 统计 ---")
        for label in all_category_data.keys():
            count = len(os.listdir(os.path.join(output_root, stage, label)))
            print(f"{label}: {count}")


if __name__ == "__main__":
    random.seed(42)
    if os.path.exists(output_root):
        print(f"错误: 目标目录 {output_root} 已存在，请重命名或删除后再试。")
    else:
        process_and_split()