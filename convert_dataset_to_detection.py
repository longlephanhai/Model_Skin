import os
import shutil

src_root = './data'
dst_root = './dataset_detection'
splits = ['train', 'valid', 'test']

os.makedirs(dst_root, exist_ok=True)

for split in splits:
  print("Processing split:", split)
  src_img_dir = os.path.join(src_root, split, 'images')
  src_label_dir = os.path.join(src_root, split, 'labels')
  
  dst_img_dir = os.path.join(dst_root, split, 'images')
  dst_label_dir = os.path.join(dst_root, split, 'labels')
  
  if not os.path.exists(src_img_dir):
    continue
  
  os.makedirs(dst_img_dir, exist_ok=True)
  os.makedirs(dst_label_dir, exist_ok=True)
  
  for file_name in os.listdir(src_label_dir):
    if not file_name.endswith('.txt'):
      print("Skipping non-txt file:", file_name)
      continue
    
    new_lines = []
    with open(os.path.join(src_label_dir, file_name), 'r') as f:
      for line in f.readlines():
        parts = line.split()
        if len(parts)>=5:
          parts[0] = '0' 
          new_lines.append(' '.join(parts) + '\n')
    
    with open(os.path.join(dst_label_dir, file_name), 'w') as f:
      f.writelines(new_lines)
      
    base_name = file_name.replace('.txt', '')
    
    img_src_path = os.path.join(src_img_dir, base_name + '.jpg')
    if os.path.exists(img_src_path):
      shutil.copy(img_src_path, os.path.join(dst_img_dir, base_name + '.jpg'))
      
print("Dataset conversion completed.")