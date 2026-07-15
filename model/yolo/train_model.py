import sys
from ultralytics import YOLO

def train_model():
    model = YOLO(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\yolo26n.pt")
    
    # Training
    model.train(
        data="C:\\Users\\fouls\\Downloads\\TARUMT\\Y2S1\\AI\\BMCS2074-Artificial-Intelligence-Assignment\\model\\yolo\\dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=11,  
        device=0,  
        workers=4,  
        patience=50,
        project="C:\\Users\\fouls\\Downloads\\TARUMT\\Y2S1\\AI\\BMCS2074-Artificial-Intelligence-Assignment\\model\\yolo\\runs",
        exist_ok=True,
    )
    
    print("---------------------------------------------Successfully Training---------------------------------------------")

if __name__ == "__main__":
    # Important for Windows multiprocessing
    from multiprocessing import freeze_support
    freeze_support()  
    train_model()