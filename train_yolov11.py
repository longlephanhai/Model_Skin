from ultralytics import YOLO
import torch


def start_training():
    model = YOLO("yolo11m.pt")
    # model = YOLO("./runs/detect/SkinProject/Train_v1/weights/last.pt")

    torch.cuda.empty_cache()

    model.train(
        data="./data.yaml",
        project="SkinProject",
        name="Train_v1",
        exist_ok=True,
        imgsz=640,
        batch=4,
        nbs=32,
        epochs=200,
        device=0,
        workers=4,
        box=7.5,
        cls=5.0,
        dfl=1.5,
        optimizer="AdamW",
        lr0=0.0005,
        cos_lr=True,
        warmup_epochs=5,
        augment=True,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=20.0,
        translate=0.2,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.3,
        copy_paste=0.3,
        close_mosaic=15,
        patience=50,
        save_period=10,
        amp=True,
        verbose=False,
    )


if __name__ == "__main__":
    start_training()
