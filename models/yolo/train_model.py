import sys
from ultralytics import YOLO

model = YOLO(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\yolov8n.pt")

# Training
model.train(
    data="C:\\Users\\fouls\\Downloads\\TARUMT\\Y2S1\\AI\\BMCS2074-Artificial-Intelligence-Assignment\\models\\yolo\\dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="cpu",
    workers=4,
    patience=100,
    project="C:\\Users\\fouls\\Downloads\\TARUMT\\Y2S1\\AI\\BMCS2074-Artificial-Intelligence-Assignment\\models\\yolo\\runs",
)

print("---------------------------------------------Successfully Training---------------------------------------------")