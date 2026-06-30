"""
YOLO-based vehicle detector
"""

import torch
import numpy as np
import cv2
import os
import sys

# Add parent directory to path for model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import ultralytics
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics not installed. Using dummy implementation.")
    ultralytics = None


class YOLODetector:
    """YOLO-based vehicle detector"""
    
    def __init__(self, model_path=None, device='cpu', confidence_threshold=0.5):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
            device: 'cpu' or 'cuda'
            confidence_threshold: Minimum confidence for detections
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.confidence_threshold = confidence_threshold
        
        # Vehicle classes (COCO dataset)
        self.vehicle_classes = [
            'car', 'truck', 'bus', 'motorcycle', 'bicycle',
            'train', 'boat', 'airplane'
        ]
        
        # Initialize YOLO model
        if ultralytics is not None:
            try:
                if model_path and os.path.exists(model_path):
                    self.model = YOLO(model_path)
                else:
                    # Use YOLOv8n pretrained model
                    self.model = YOLO('yolov8n.pt')
                
                # Move to device
                self.model.to(self.device)
                print(f"YOLO model loaded on {self.device}")
                
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
                self.model = None
        else:
            self.model = None
            print("YOLO model not available. Using dummy implementation.")
    
    def detect(self, frame):
        """
        Detect vehicles in a frame using YOLO
        
        Args:
            frame: Input image (numpy array)
            
        Returns:
            List of detections with bbox and confidence
        """
        if self.model is None:
            return self._dummy_detection(frame)
        
        if frame is None:
            return []
        
        # Run inference
        try:
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class id and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Get class name
                        class_name = self.model.names[class_id]
                        
                        # Filter for vehicle classes
                        if class_name.lower() in self.vehicle_classes:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class': class_name,
                                'class_id': class_id
                            })
            
            return detections
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return self._dummy_detection(frame)
    
    def _dummy_detection(self, frame):
        """Dummy detection for testing when YOLO is not available"""
        h, w = frame.shape[:2]
        
        # Create some random detections for demonstration
        num_detections = np.random.randint(2, 8)
        detections = []
        
        for _ in range(num_detections):
            x = np.random.randint(0, w - 100)
            y = np.random.randint(0, h - 100)
            w_box = np.random.randint(80, 200)
            h_box = np.random.randint(100, 250)
            
            # Ensure box stays within frame
            if x + w_box > w:
                w_box = w - x
            if y + h_box > h:
                h_box = h - y
            
            detections.append({
                'bbox': [x, y, x + w_box, y + h_box],
                'confidence': np.random.uniform(0.5, 0.95),
                'class': np.random.choice(['car', 'truck', 'bus']),
                'class_id': 0
            })
        
        return detections
    
    def detect_batch(self, frames):
        """
        Detect vehicles in multiple frames
        
        Args:
            frames: List of input images
            
        Returns:
            List of detections for each frame
        """
        if self.model is None:
            return [self._dummy_detection(frame) for frame in frames]
        
        detections_list = []
        for frame in frames:
            detections_list.append(self.detect(frame))
        
        return detections_list
    
    def save_model(self, path):
        """Save model weights"""
        if self.model is not None:
            self.model.export(path)
            print(f"Model saved to {path}")
        else:
            print("No model to save")
    
    def load_model(self, path):
        """Load model weights"""
        if ultralytics is not None and os.path.exists(path):
            self.model = YOLO(path)
            self.model.to(self.device)
            print(f"Model loaded from {path}")
        else:
            print(f"Model file not found: {path}")