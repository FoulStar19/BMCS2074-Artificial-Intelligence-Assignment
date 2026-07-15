import sys
from ultralytics import YOLO

def train_model():
    model = YOLO(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\models\yolo\runs\train\weights\best.pt")
    
    # Training
    model.train(
        data="C:\\Users\\fouls\\Downloads\\TARUMT\\Y2S1\\AI\\BMCS2074-Artificial-Intelligence-Assignment\\models\\yolo\\dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=11,  
        device=0,  # Use GPU (0 = first GPU)
        workers=4, 
        patience=50,
        project="C:\\Users\\fouls\\Downloads\\TARUMT\\Y2S1\\AI\\BMCS2074-Artificial-Intelligence-Assignment\\models\\yolo\\runs",
        exist_ok=True,
        amp=True,  
        optimizer='auto',
        cos_lr=False,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.3,
        erasing=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    
    print("---------------------------------------------Successfully Training---------------------------------------------")

if __name__ == "__main__":
    # Important for Windows multiprocessing
    from multiprocessing import freeze_support
    freeze_support()  
    train_model()