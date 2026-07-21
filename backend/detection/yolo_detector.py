"""
YOLO-based vehicle detector with tracking integration
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import os
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.tracking.tracker import VehicleTracker


class YOLODetector:
    """
    YOLO-based vehicle detection using Ultralytics YOLO with tracking
    """
    
    def __init__(self, model_path=None, device='cpu', conf_threshold=0.25, 
                 enable_tracking=True, max_lost_frames=15):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights (.pt file)
            device: 'cpu' or 'cuda'
            conf_threshold: Confidence threshold for detections
            enable_tracking: Enable vehicle tracking
            max_lost_frames: Maximum frames before track is lost
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = None
        self.is_dummy = False
        self.enable_tracking = enable_tracking
        
        # Initialize tracker
        self.tracker = VehicleTracker(max_lost_frames=max_lost_frames)
        
        # Load dataset configuration
        self.dataset_config = self._load_dataset_config()
        
        # Vehicle classes from dataset config
        self.vehicle_classes = self.dataset_config.get('names', {
            0: 'car',
            1: 'truck',
            2: 'bus',
            3: 'motorcycle',
            4: 'bicycle'
        })
        
        # Class colors from dataset config
        self.class_colors = self.dataset_config.get('colors', {
            0: [0, 0, 255],    # Red (BGR)
            1: [0, 255, 0],    # Green (BGR)
            2: [255, 0, 0],    # Blue (BGR)
            3: [0, 255, 255],  # Yellow (BGR)
            4: [255, 0, 255]   # Magenta (BGR)
        })
        
        # Only these classes are considered vehicles
        self.vehicle_class_ids = list(self.vehicle_classes.keys())
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No model found. Using dummy detector.")
            self.is_dummy = True
    
    def _load_dataset_config(self):
        """Load dataset configuration from dataset.yaml"""
        # Check multiple possible locations
        yaml_paths = [
            "dataset.yaml",
            "model/yolo/dataset.yaml",
            "config/dataset.yaml",
            Path(__file__).parent.parent / "dataset.yaml",
            Path(__file__).parent.parent / "model" / "yolo" / "dataset.yaml",
            Path(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\dataset.yaml"),
        ]
        
        for yaml_path in yaml_paths:
            if os.path.exists(yaml_path):
                try:
                    with open(yaml_path, 'r') as f:
                        config = yaml.safe_load(f)
                    print(f"✅ Loaded dataset config from {yaml_path}")
                    return config
                except Exception as e:
                    print(f"Error loading {yaml_path}: {e}")
        
        # Return default configuration
        print("Using default dataset configuration")
        return {
            'nc': 5,
            'names': {
                0: 'car',
                1: 'truck',
                2: 'bus',
                3: 'motorcycle',
                4: 'bicycle'
            },
            'colors': {
                0: [0, 0, 255],    # Red (BGR)
                1: [0, 255, 0],    # Green (BGR)
                2: [255, 0, 0],    # Blue (BGR)
                3: [0, 255, 255],  # Yellow (BGR)
                4: [255, 0, 255]   # Magenta (BGR)
            }
        }
    
    def get_color_for_class(self, class_id):
        """Get color for a specific class from dataset config"""
        color = self.class_colors.get(class_id, [0, 255, 0])
        if isinstance(color, list):
            return tuple(color)
        return (0, 255, 0)
    
    def get_class_name(self, class_id):
        """Get class name from dataset config"""
        return self.vehicle_classes.get(class_id, f'class_{class_id}')
    
    def load_model(self, model_path):
        """
        Load YOLO model from weights file
        
        Args:
            model_path: Path to model weights
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            # Move to device if CUDA is available
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')
            print(f"✅ YOLO model loaded from {model_path} on {self.device}")
            self.is_dummy = False
        except ImportError:
            print("⚠️ Ultralytics not installed. Using dummy mode.")
            self.is_dummy = True
        except Exception as e:
            print(f"⚠️ Error loading model: {e}. Using dummy mode.")
            self.is_dummy = True
    
    def detect(self, frame, enable_tracking=None):
        """
        Detect vehicles in a frame with optional tracking
        
        Args:
            frame: Input image (numpy array)
            enable_tracking: Override tracking setting
            
        Returns:
            List of detection dictionaries with track_ids and speeds
        """
        # If dummy mode, return simulated detections
        if self.is_dummy or self.model is None:
            return self._dummy_detect(frame)
        
        use_tracking = enable_tracking if enable_tracking is not None else self.enable_tracking
        
        try:
            # Run YOLO inference
            results = self.model(
                frame, 
                conf=self.conf_threshold,
                verbose=False,
                stream=False
            )
            
            detections = []
            
            # Process results
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Only include vehicle classes
                        if class_id in self.vehicle_class_ids:
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class': class_id,
                                'class_name': self.get_class_name(class_id)
                            })
            
            # Apply tracking if enabled
            if use_tracking and detections:
                detections = self.tracker.update(frame, detections)
            
            return detections
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return self._dummy_detect(frame)
    
    def detect_frame(self, frame, enable_tracking=None):
        """Alias for detect method"""
        return self.detect(frame, enable_tracking)
    
    def reset_tracker(self):
        """Reset the vehicle tracker"""
        self.tracker.reset()
    
    def draw_trails(self, frame, trail_length=20):
        """Draw tracking trails on frame"""
        return self.tracker.draw_trails(frame, trail_length)
    
    def _dummy_detect(self, frame):
        """
        Generate dummy detections for testing
        
        Args:
            frame: Input frame
            
        Returns:
            List of dummy detections
        """
        h, w = frame.shape[:2]
        
        # Generate 2-5 random detections
        import random
        num_detections = random.randint(2, 5)
        detections = []
        
        for i in range(num_detections):
            # Random bounding box
            x = random.randint(50, w - 150)
            y = random.randint(50, h - 150)
            width = random.randint(60, 150)
            height = random.randint(60, 150)
            
            # Random class ID (0-4)
            class_id = random.randint(0, 4)
            
            detections.append({
                'bbox': [x, y, x + width, y + height],
                'confidence': random.uniform(0.5, 0.95),
                'class': class_id,
                'class_name': self.get_class_name(class_id),
                'track_id': i,
                'speed': random.uniform(20, 80)
            })
        
        # Apply tracking if enabled
        if self.enable_tracking:
            detections = self.tracker.update(frame, detections)
        
        return detections