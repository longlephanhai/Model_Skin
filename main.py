from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import torch
import cv2
import uuid
import requests
import numpy as np
from torchvision import transforms
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import uvicorn

app = FastAPI()

device     = "cuda:0" if torch.cuda.is_available() else "cpu"
checkpoint = './runs/detect/SkinProject/Train_YOLOv8_Optimized/weights/best.pt'
sr_model   = './RealESRGAN_x4plus.pth'

yolo_model = YOLO(checkpoint)

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=checkpoint,
    confidence_threshold=0.25,
    device=device
)

rrdb_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                     num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4, model_path=sr_model, model=rrdb_model,
    tile=400, half=("cuda" in device)
)

os.makedirs("results", exist_ok=True)


# ── Grad-CAM wrapper ──────────────────────────────────────
class YOLOBackboneNeckWrapper(torch.nn.Module):
    def __init__(self, yolo_inner):
        super().__init__()
        self.layers = yolo_inner.model[:-1]

    def forward(self, x):
        outputs = {}
        y = x
        for i, layer in enumerate(self.layers):
            if layer.f != -1:
                y = outputs[layer.f] if isinstance(layer.f, int) \
                    else [outputs[j] if j != -1 else y for j in layer.f]
            y = layer(y)
            outputs[i] = y
        if isinstance(y, (list, tuple)):
            y = y[0]
        return y.reshape(y.shape[0], -1)


def run_gradcam(img_path: str, export_dir: str, session_id: str, angle: str) -> dict:
    try:
        predict_results = yolo_model.predict(img_path, verbose=False)
        if len(predict_results[0].boxes) == 0:
            return {}

        img_bgr        = cv2.imread(img_path)
        h_orig, w_orig = img_bgr.shape[:2]


        scale = 640 / max(h_orig, w_orig)
        new_h = int(round(h_orig * scale / 32) * 32)
        new_w = int(round(w_orig * scale / 32) * 32)

        img_rgb     = img_bgr[:, :, ::-1].copy()
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        img_float   = np.float32(img_resized) / 255.0

        input_tensor = transforms.ToTensor()(img_resized).unsqueeze(0)
        dev = next(yolo_model.model.parameters()).device
        input_tensor = input_tensor.to(dev)
        input_tensor.requires_grad_(True)

        wrapped = YOLOBackboneNeckWrapper(yolo_model.model)
        wrapped.train()
        for p in wrapped.parameters():
            p.requires_grad_(True)

        target_layer = None
        for layer in wrapped.layers:
            if type(layer).__name__ == 'C2f':
                target_layer = layer

        cam = GradCAM(model=wrapped, target_layers=[target_layer])
        with torch.enable_grad():
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)

        grayscale_cam = cv2.resize(grayscale_cam[0], (new_w, new_h))


        original_path = os.path.join(export_dir, "original.jpg")
        cv2.imwrite(original_path, cv2.resize(img_bgr, (w_orig, h_orig)))


        heatmap_jet  = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heatmap_path = os.path.join(export_dir, "heatmap.jpg")
        cv2.imwrite(heatmap_path, cv2.resize(heatmap_jet, (w_orig, h_orig)))


        overlay = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        scale_x, scale_y = new_w / w_orig, new_h / h_orig

        for box in predict_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1s = int(x1 * scale_x); y1s = int(y1 * scale_y)
            x2s = int(x2 * scale_x); y2s = int(y2 * scale_y)
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            lbl    = f"{yolo_model.names[cls_id]} {conf:.0%}"
            cv2.rectangle(overlay, (x1s, y1s), (x2s, y2s), (255, 255, 255), 2)
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x1s, y1s - th - 6), (x1s + tw + 4, y1s), (255, 255, 255), -1)
            cv2.putText(overlay, lbl, (x1s + 2, y1s - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)

        gradcam_path = os.path.join(export_dir, "gradcam.jpg")
        cv2.imwrite(gradcam_path,
            cv2.resize(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), (w_orig, h_orig)))

        base = f"/results/{session_id}/{angle}"
        return {
            "original_url": f"{base}/original.jpg",
            "heatmap_url":  f"{base}/heatmap.jpg",
            "gradcam_url":  f"{base}/gradcam.jpg",
        }

    except Exception as e:
        print(f"[GradCAM] Loi: {e}")
        return {}

class ImageUrls(BaseModel):
    front: str
    left: str
    right: str


def download_image(url: str, save_path: str):
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            raise HTTPException(status_code=400,
                detail=f"Cannot download image from {url}, status: {r.status_code}")
        with open(save_path, "wb") as f:
            f.write(r.content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400,
            detail=f"Error downloading: {str(e)}")


def process_image(image_url: str, session_id: str, angle: str):
    temp_path = f"temp_{session_id}_{angle}.jpg"
    download_image(image_url, temp_path)

    img_origin = cv2.imread(temp_path)
    if img_origin is None:
        os.remove(temp_path)
        return None

    h_orig, w_orig = img_origin.shape[:2]
    slice_h = min(640, h_orig)
    slice_w = min(640, w_orig)

    if h_orig <= 640 and w_orig <= 640:
        result = get_sliced_prediction(
            temp_path, detection_model,
            slice_height=h_orig, slice_width=w_orig,
            overlap_height_ratio=0.0, overlap_width_ratio=0.0,
            postprocess_type="GREEDYNMM", postprocess_match_threshold=0.5
        )
    else:
        result = get_sliced_prediction(
            temp_path, detection_model,
            slice_height=slice_h, slice_width=slice_w,
            overlap_height_ratio=0.3, overlap_width_ratio=0.3,
            postprocess_type="GREEDYNMM", postprocess_match_threshold=0.5
        )

    predictions = result.object_prediction_list
    export_dir  = f"results/{session_id}/{angle}"
    crops_dir   = os.path.join(export_dir, "acne_crops_sr")
    os.makedirs(crops_dir, exist_ok=True)

    acne_stats = {}
    detections = []

    for i, pred in enumerate(predictions):
        x1, y1, x2, y2 = map(int, [pred.bbox.minx, pred.bbox.miny,
                                     pred.bbox.maxx, pred.bbox.maxy])
        width, height   = x2 - x1, y2 - y1
        conf  = pred.score.value
        label = pred.category.name
        acne_stats[label] = acne_stats.get(label, 0) + 1

        pad_x = max(15, width  // 2)
        pad_y = max(15, height // 2)
        nx1 = max(0, x1 - pad_x);      ny1 = max(0, y1 - pad_y)
        nx2 = min(w_orig, x2 + pad_x); ny2 = min(h_orig, y2 + pad_y)
        crop_bgr = img_origin[ny1:ny2, nx1:nx2]

        try:
            output_sr, _ = upsampler.enhance(crop_bgr, outscale=2)
            sp = os.path.join(crops_dir, f"det_{i+1}_{label}_conf{conf:.2f}.jpg")
            cv2.imwrite(sp, output_sr)
            detections.append({
                "label":      label,
                "confidence": conf,
                "bbox":       [x1, y1, x2, y2],
                "crop_url":   f"/results/{session_id}/{angle}/acne_crops_sr/det_{i+1}_{label}_conf{conf:.2f}.jpg"
            })
        except Exception as e:
            print(f"Loi SR: {e}")

    result.export_visuals(export_dir=export_dir,
        file_name="tong_the_detect", hide_conf=False)

    gradcam_urls = run_gradcam(temp_path, export_dir, session_id, angle)
    os.remove(temp_path)

    return {
        "total":              len(predictions),
        "stats":              acne_stats,
        "detections":         detections,
        "visualization_url":  f"/results/{session_id}/{angle}/tong_the_detect.png",
        "original_url":       f"http://localhost:8000{gradcam_urls.get('original_url')}",
        "heatmap_url":        f"http://localhost:8000{gradcam_urls.get('heatmap_url')}",   
        "gradcam_url":        f"http://localhost:8000{gradcam_urls.get('gradcam_url')}",   
    }

@app.post("/detect")
async def detect(image_urls: ImageUrls):
    session_id   = str(uuid.uuid4())
    front_result = process_image(image_urls.front, session_id, "front")
    left_result  = process_image(image_urls.left,  session_id, "left")
    right_result = process_image(image_urls.right, session_id, "right")

    total_acne = 0; combined_stats = {}
    for r in [front_result, left_result, right_result]:
        if r:
            total_acne += r["total"]
            for lbl, cnt in r["stats"].items():
                combined_stats[lbl] = combined_stats.get(lbl, 0) + cnt

    return {
        "success":    True,
        "session_id": session_id,
        "total_acne": total_acne,
        "stats":      combined_stats,
        "results": {
            "front": front_result,
            "left":  left_result,
            "right": right_result
        }
    }


@app.get("/results/{session_id}/{angle}/{folder}/{filename}")
async def get_result(session_id: str, angle: str, folder: str, filename: str):
    return FileResponse(f"results/{session_id}/{angle}/{folder}/{filename}")

@app.get("/results/{session_id}/{angle}/{filename}")
async def get_visualization(session_id: str, angle: str, filename: str):
    return FileResponse(f"results/{session_id}/{angle}/{filename}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)