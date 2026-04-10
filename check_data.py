import os
from collections import Counter

# Đường dẫn tới thư mục labels của tập train
label_path = './dataset/train/labels'  # Thay đổi nếu đường dẫn khác
class_names = ['Acne', 'Blackheads', 'Dark-Spots', 'Dry-Skin', 'Englarged-Pores',
               'Eyebags', 'Oily-Skin', 'Skin-Redness', 'Whiteheads', 'Wrinkles']

img_count = Counter()

for label_file in os.listdir(label_path):
    if label_file.endswith('.txt'):
        with open(os.path.join(label_path, label_file), 'r') as f:
            classes_in_file = set([line.split()[0] for line in f.readlines()])
            for cls_id in classes_in_file:
                img_count[int(cls_id)] += 1

print("--- Thống kê số lượng ẢNH chứa mỗi class ---")
for cls_id, count in sorted(img_count.items()):
    print(f"{class_names[cls_id]}: {count} ảnh")
