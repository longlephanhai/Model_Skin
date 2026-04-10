import os
import torch
import numpy as np
import cv2
from argparse import ArgumentParser
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def get_args():
    parser = ArgumentParser(description="Skin Project Balanced: YOLOv11 + SAHI + RealESRGAN")
    parser.add_argument("--image-path", "-p", type=str, required=True)
    parser.add_argument("--checkpoint", "-c", type=str,
                        default="./runs/detect/SkinProject/Train_v1/weights/best.pt")
    parser.add_argument("--sr-model", type=str,
                        default="./RealESRGAN_x4plus.pth")
    parser.add_argument("--conf", type=float, default=0) 
    return parser.parse_args()

def main():
    args = get_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("--- Đang tải mô hình... ---")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.checkpoint,
        confidence_threshold=args.conf,
        device=device
    )

    rrdb_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path=args.sr_model, model=rrdb_model, tile=400, half=True if "cuda" in device else False)

    # --- 2. CHẠY SAHI VỚI NMS CHẶT CHẼ HƠN ---
    print(f"--- Đang phân tích: {args.image_path} ---")
    result = get_sliced_prediction(
        args.image_path,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.3, # Tăng overlap để gộp box tốt hơn
        overlap_width_ratio=0.3,
        postprocess_type="GREEDYNMM", # Thuật toán gộp box
        postprocess_match_threshold=0.5 # Ngưỡng gộp các box trùng nhau
    )

    img_origin = cv2.imread(args.image_path)
    h_orig, w_orig = img_origin.shape[:2]
    predictions = result.object_prediction_list

    export_dir = "skin_results"
    crops_dir = os.path.join(export_dir, "acne_details_hq")
    os.makedirs(crops_dir, exist_ok=True)

    acne_stats = {}
    valid_predictions = [] # Danh sách chứa các nốt mụn thực sự chất lượng

    # --- 3. LỌC DIỆN TÍCH VÀ XỬ LÝ ---
    for i, pred in enumerate(predictions):
        x1, y1, x2, y2 = map(int, [pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy])
        
        # TỐI ƯU 1: Lọc diện tích (Area Filter)
        # Loại bỏ các box quá bé (ví dụ nhỏ hơn 15x15 pixel)
        width, height = x2 - x1, y2 - y1
        if width < 15 or height < 15: 
            continue

        # TỐI ƯU 2: Lọc theo Confidence Score cực cao cho ảnh chi tiết
        # Chỉ những nốt > 0.5 mới mang đi chạy SR cho bệnh nhân xem
        if pred.score.value < 0.5:
            continue

        valid_predictions.append(pred) # Giữ lại nốt mụn đạt chuẩn
        label = pred.category.name
        acne_stats[label] = acne_stats.get(label, 0) + 1

        # Xử lý Crop và SR như cũ
        pad_x, pad_y = max(width, 70), max(height, 70)
        nx1, ny1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        nx2, ny2 = min(w_orig, x2 + pad_x), min(h_orig, y2 + pad_y)
        crop_bgr = img_origin[ny1:ny2, nx1:nx2]

        try:
            output_sr, _ = upsampler.enhance(crop_bgr, outscale=2)
            cv2.putText(output_sr, f"{label}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imwrite(os.path.join(crops_dir, f"acne_{len(valid_predictions)}_{label}.jpg"), output_sr)
        except:
            continue

    # --- 4. XUẤT ẢNH TỔNG THỂ (Chỉ vẽ các nốt mụn đã lọc) ---
    # Ghi đè lại object_prediction_list của result để export_visuals chỉ vẽ nốt mụn 'xịn'
    result.object_prediction_list = valid_predictions 
    result.export_visuals(export_dir=export_dir, file_name="tong_the_detect_da_loc", hide_conf=True)

    print(f"\n--- ĐÃ LỌC: Còn {len(valid_predictions)} nốt mụn chất lượng ---")
    print(f"Báo cáo chi tiết đã lưu tại {export_dir}")

if __name__ == "__main__":
    main()