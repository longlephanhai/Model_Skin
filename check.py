import os
import pandas as pd 

output_dir = 'dataset_classification'
splits = ['train', 'valid', 'test']

data_report = []

for split in splits:
    split_path = os.path.join(output_dir, split)
    if not os.path.exists(split_path):
        continue
    
    # Lấy danh sách các class (thư mục con)
    classes = os.listdir(split_path)
    for cls in classes:
        cls_path = os.path.join(split_path, cls)
        if os.path.isdir(cls_path):
            # Đếm số lượng file ảnh trong mỗi class
            num_images = len([f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
            data_report.append({
                'Split': split,
                'Class': cls,
                'Count': num_images
            })

# Tạo bảng tổng kết
df = pd.DataFrame(data_report)
pivot_df = df.pivot(index='Class', columns='Split', values='Count').fillna(0).astype(int)

print("\n=== THỐNG KÊ CHI TIẾT DATASET CLASSIFICATION ===")
print(pivot_df)
print("================================================")

# Tính tổng để Long dễ hình dung
print(f"\nTổng cộng ảnh trong Train: {pivot_df['train'].sum() if 'train' in pivot_df else 0}")
print(f"Tổng cộng ảnh trong Valid: {pivot_df['valid'].sum() if 'valid' in pivot_df else 0}")
print(f"Tổng cộng ảnh trong Test: {pivot_df['test'].sum() if 'test' in pivot_df else 0}")