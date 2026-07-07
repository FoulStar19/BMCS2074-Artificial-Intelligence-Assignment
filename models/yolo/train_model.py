import sys

from ultralytics import YOLO

model = YOLO(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\yolov8n.pt")  # Model initialized without weights

# Training
model.train(
    data="C:\\Users\\fouls\\Downloads\\TARUMT\\Y2S1\\AI\\BMCS2074-Artificial-Intelligence-Assignment\\models\\yolo\\dataset.yaml",      # Path to dataset configuration file
    epochs=500,            # Number of training epochs
    imgsz=640,             # Image size for training
    batch=16,              # Batch size
    device="cpu",              # GPU (0) or CPU ("cpu")
    workers=4,             # Number of data loader workers
    patience=100           # Early stopping patience
)

print("---------------------------------------------Successfully Training---------------------------------------------")