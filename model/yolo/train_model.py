import sys
import os
import json
import glob
from datetime import datetime
from ultralytics import YOLO

class TrainingManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.tracking_file = os.path.join(base_dir, "training_history.json")
        self.load_history()
    
    def load_history(self):
        """Load training history from JSON file"""
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {"versions": [], "last_version": 0}
    
    def save_history(self):
        """Save training history to JSON file"""
        os.makedirs(self.base_dir, exist_ok=True)
        with open(self.tracking_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_next_version(self):
        """Get next version number and increment counter"""
        next_num = self.history["last_version"] + 1
        version_name = f"v{next_num}"
        version_path = os.path.join(self.base_dir, version_name)
        
        # Update history
        self.history["versions"].append({
            "version": version_name,
            "timestamp": datetime.now().isoformat(),
            "path": version_path
        })
        self.history["last_version"] = next_num
        self.save_history()
        
        return version_name, version_path

def train_model():
    base_dir = r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\model\yolo\runs"
    
    # Initialize training manager
    manager = TrainingManager(base_dir)
    version_name, version_path = manager.get_next_version()
    
    # Create version directory
    os.makedirs(version_path, exist_ok=True)
    
    model = YOLO(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\yolo26n.pt")
    
    # Training with dynamic naming
    model.train(
        data="C:\\Users\\fouls\\Downloads\\TARUMT\\Y2S1\\AI\\BMCS2074-Artificial-Intelligence-Assignment\\model\\yolo\\dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=11,  
        device=0,  
        workers=4,  
        patience=50,
        project=version_path,
        exist_ok=True,
    )
    
    print(f"---------------------------------------------Successfully Training - {version_name}---------------------------------------------")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  
    train_model()
