from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import torch
import cv2
import uuid
import requests
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import uvicorn

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
checkpoint = './runs/detect/SkinProject/Train_YOLOv8_Optimized/weights/best.pt'
sr_model = './RealESRGAN_x4plus.pth'

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=checkpoint,
    confidence_threshold=0.25,
    device=device
)

rrdb_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                     num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path=sr_model,
    model=rrdb_model,
    tile=400,
    half=(True if "cuda" in device else False)
)

os.makedirs("results", exist_ok=True)




class ImageUrls(BaseModel):
    front: str 
    left: str   
    right: str  


def download_image(url: str, save_path: str):
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400, detail=f"Cannot download image from {url}, status: {response.status_code}")

        with open(save_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error downloading image from {url}: {str(e)}")


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
            temp_path,
            detection_model,
            slice_height=h_orig,
            slice_width=w_orig,
            overlap_height_ratio=0.0,
            overlap_width_ratio=0.0,
            postprocess_type="GREEDYNMM",
            postprocess_match_threshold=0.5
        )
    else:
        result = get_sliced_prediction(
            temp_path,
            detection_model,
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
            postprocess_type="GREEDYNMM",
            postprocess_match_threshold=0.5
        )

    predictions = result.object_prediction_list

    export_dir = f"results/{session_id}/{angle}"
    crops_dir = os.path.join(export_dir, "acne_crops_sr")
    os.makedirs(crops_dir, exist_ok=True)

    acne_stats = {}
    detections = []

    for i, pred in enumerate(predictions):
        x1, y1, x2, y2 = map(int, [
            pred.bbox.minx, pred.bbox.miny,
            pred.bbox.maxx, pred.bbox.maxy
        ])
        width, height = x2 - x1, y2 - y1
        conf = pred.score.value
        label = pred.category.name

        acne_stats[label] = acne_stats.get(label, 0) + 1

        pad_x = max(15, width // 2)
        pad_y = max(15, height // 2)
        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(w_orig, x2 + pad_x)
        ny2 = min(h_orig, y2 + pad_y)

        crop_bgr = img_origin[ny1:ny2, nx1:nx2]

        try:
            output_sr, _ = upsampler.enhance(crop_bgr, outscale=2)
            save_path = os.path.join(
                crops_dir, f"det_{i+1}_{label}_conf{conf:.2f}.jpg")
            cv2.imwrite(save_path, output_sr)

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "crop_url": f"/results/{session_id}/{angle}/acne_crops_sr/det_{i+1}_{label}_conf{conf:.2f}.jpg"
            })
        except Exception as e:
            print(f"Lỗi SR: {e}")

    result.export_visuals(
        export_dir=export_dir,
        file_name="tong_the_detect",
        hide_conf=False
    )

    os.remove(temp_path)

    return {
        "total": len(predictions),
        "stats": acne_stats,
        "detections": detections,
        "visualization_url": f"/results/{session_id}/{angle}/tong_the_detect.png"
    }


@app.post("/detect")
async def detect(image_urls: ImageUrls):
    session_id = str(uuid.uuid4())

    front_result = process_image(image_urls.front, session_id, "front")
    left_result = process_image(image_urls.left, session_id, "left")
    right_result = process_image(image_urls.right, session_id, "right")

    total_acne = 0
    combined_stats = {}

    for result in [front_result, left_result, right_result]:
        if result:
            total_acne += result["total"]
            for label, count in result["stats"].items():
                combined_stats[label] = combined_stats.get(label, 0) + count

    return {
        "success": True,
        "session_id": session_id,
        "total_acne": total_acne,
        "stats": combined_stats,
        "results": {
            "front": front_result,
            "left": left_result,
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