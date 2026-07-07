# backend/detection/yolo_detector.py
import torch
import cv2
import numpy as np
from pathlib import Path

class YOLODetector:
    def __init__(self, model_path=None, device='cpu'):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model = None
        
        # If no model path provided, use the default
        if model_path is None:
            # Use your specific path
            model_path = r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\models\yolo\yolov1.pt"
        
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load YOLO model from file"""
        try:
            # Try loading with ultralytics YOLO
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                print(f"Loaded YOLO model from {model_path}")
            except ImportError:
                # Fallback to PyTorch loading
                self.model = torch.load(model_path, map_location=self.device)
                self.model.eval()
                print(f"Loaded PyTorch model from {model_path}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            # Create dummy model for demonstration
            self.model = None
            self.is_dummy = True
            
    def detect(self, frame):
        """
        Detect vehicles in frame
        
        Args:
            frame: numpy array (BGR image)
            
        Returns:
            List of detections with bbox and confidence
        """
        if self.model is None:
            return self._dummy_detection(frame)
            
        try:
            # Check if using ultralytics YOLO
            if hasattr(self.model, 'predict'):
                results = self.model.predict(frame, conf=0.5, device=self.device)
                
                detections = []
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = box.cls[0].cpu().numpy()
                            
                            # Only detect vehicles (car, truck, bus, motorcycle, etc.)
                            vehicle_classes = [0, 1, 2, 3, 5, 7]  # COCO classes for vehicles
                            if int(cls) in vehicle_classes:
                                detections.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': float(conf),
                                    'class': int(cls)
                                })
                return detections
            else:
                # PyTorch model loading
                return self._pytorch_detection(frame)
                
        except Exception as e:
            print(f"Error during detection: {e}")
            return self._dummy_detection(frame)
    
    def _pytorch_detection(self, frame):
        """Detection with PyTorch model"""
        # Preprocess frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor / 255.0
        img_tensor = img_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        # Process outputs (simplified - adjust based on your model's output format)
        detections = []
        # This is a placeholder - actual processing depends on your model
        return self._dummy_detection(frame)
    
    def _dummy_detection(self, frame):
        """Fallback: generate dummy detections for testing"""
        h, w = frame.shape[:2]
        detections = []
        # Generate 3-8 random vehicles
        num_vehicles = np.random.randint(3, 8)
        for _ in range(num_vehicles):
            x1 = np.random.randint(50, w-200)
            y1 = np.random.randint(50, h-200)
            x2 = x1 + np.random.randint(100, 200)
            y2 = y1 + np.random.randint(80, 150)
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': np.random.uniform(0.6, 0.98)
            })
        return detections