import cv2
import os 

base_dir = './data'
splits = ['train', 'valid', 'test']
output_dir = 'dataset_classification'
class_names = ['blackheads', 'dark spot', 'nodules', 'papules', 'pustules', 'whiteheads']

for split in splits:
    print("Processing split:", split)
    img_dir = os.path.join(base_dir, split, 'images')
    label_dir = os.path.join(base_dir, split, 'labels')
    
    if not os.path.exists(img_dir) or not os.path.exists(label_dir):
        print(f"Skipping {split} - directory not found.")
        continue
    
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'): continue

        img_name = label_file.replace('.txt', '.jpg')
        img_path = os.path.join(img_dir, img_name)
        
        img = cv2.imread(img_path)
        if img is None: continue
        
        H, W, _ = img.shape
        
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for i, line in enumerate(f.readlines()):
                data = line.split()
                if len(data) < 5: continue
                
                cls, x, y, bw, bh = map(float, data[:5])
                
                x1 = int((x - bw/2) * W)
                y1 = int((y - bh/2) * H)
                x2 = int((x + bw/2) * W)
                y2 = int((y + bh/2) * H)
                

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                
                crop_img = img[y1:y2, x1:x2]

                if crop_img.size == 0: continue
                
                save_dir = os.path.join(output_dir, split, class_names[int(cls)])
                os.makedirs(save_dir, exist_ok=True)
                
                save_path = os.path.join(save_dir, f"{label_file.replace('.txt', '')}_{i}.jpg")
                cv2.imwrite(save_path, crop_img)
        
print("--- Dataset conversion completed! Check folder 'dataset_classification' ---")