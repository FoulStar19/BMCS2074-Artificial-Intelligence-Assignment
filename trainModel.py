import sys
import os
from ultralytics import YOLO

# Add the backend path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use the dataset.yaml from root folder
dataset_yaml = "dataset.yaml"

# Model path - you can change this to your model
model = YOLO("yolov8n.pt")  # Start with base model

# Training
model.train(
    data=dataset_yaml,          # Path to dataset configuration file
    epochs=500,                 # Number of training epochs
    imgsz=640,                  # Image size for training
    batch=16,                   # Batch size
    device=0,                   # GPU (0) or CPU ("cpu")
    workers=4,                  # Number of data loader workers
    patience=100,               # Early stopping patience
    project='runs/train',       # Project directory
    name='custom_model'         # Name of the experiment
)

# Save the trained model to backend/models
model_path = "backend/models/best_new_global_model.pt"
model.export(format='pt')  # Export in PyTorch format

print("---------------------------------------------Successfully Training---------------------------------------------")