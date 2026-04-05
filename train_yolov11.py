from ultralytics import YOLO
import torch

def start_training():
    model = YOLO("yolo11s.pt") 
    # model = YOLO("./runs/detect/SkinProject/Train_v1/weights/last.pt")

    model.train(
        # resume=True,
        data="./data.yaml",  
        epochs=100,          
        imgsz=640,           
        batch=16,         
        device=0,            
        workers=4,          
        project="SkinProject",
        name="Train_v1"     
    )

if __name__ == "__main__":
    start_training()