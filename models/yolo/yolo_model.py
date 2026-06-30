"""
YOLO model wrapper for vehicle detection
"""

import torch
import cv2
import numpy as np
import os

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Ultralytics not installed. Install with: pip install ultralytics")


class YOLOVehicleModel:
    """YOLO model wrapper for vehicle detection"""
    
    def __init__(self, model_name='yolov8n.pt', device='cuda', conf_threshold=0.5, iou_threshold=0.45):
        """
        Initialize YOLO model
        
        Args:
            model_name: Name or path of YOLO model
            device: 'cuda' or 'cpu'
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics is required for YOLO model")
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        # Vehicle classes
        self.vehicle_classes = [
            'car', 'truck', 'bus', 'motorcycle', 'bicycle',
            'train', 'boat', 'airplane'
        ]
    
    def detect(self, image):
        """
        Detect vehicles in image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            List of detections
        """
        if image is None:
            return []
        
        # Run inference
        results = self.model(image, 
                           conf=self.conf_threshold, 
                           iou=self.iou_threshold,
                           verbose=False)
        
        detections = []
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.model.names[class_id]
                    
                    # Filter for vehicle classes
                    if class_name.lower() in self.vehicle_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class': class_name,
                            'class_id': class_id
                        })
        
        return detections
    
    def detect_batch(self, images):
        """
        Detect vehicles in batch of images
        
        Args:
            images: List of input images
            
        Returns:
            List of detections for each image
        """
        if not images:
            return []
        
        results = self.model(images, 
                           conf=self.conf_threshold,
                           iou=self.iou_threshold,
                           verbose=False)
        
        all_detections = []
        
        for i, result in enumerate(results):
            detections = []
            if result and result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.model.names[class_id]
                    
                    if class_name.lower() in self.vehicle_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class': class_name,
                            'class_id': class_id
                        })
            
            all_detections.append(detections)
        
        return all_detections
    
    def track_video(self, video_source, output_path=None):
        """
        Track vehicles in video
        
        Args:
            video_source: Video file path or webcam index
            output_path: Output video path (optional)
            
        Returns:
            Generator yielding processed frames
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        # Video writer
        writer = None
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run tracking
                results = self.model.track(frame, 
                                         persist=True,
                                         conf=self.conf_threshold,
                                         iou=self.iou_threshold,
                                         verbose=False)
                
                # Annotate frame
                if results and len(results) > 0:
                    annotated_frame = results[0].plot()
                else:
                    annotated_frame = frame
                
                # Write to output
                if writer:
                    writer.write(annotated_frame)
                
                yield annotated_frame, results
        
        finally:
            cap.release()
            if writer:
                writer.release()
    
    def train(self, data_yaml, epochs=100, batch_size=16, imgsz=640):
        """
        Train YOLO model
        
        Args:
            data_yaml: Path to data configuration file
            epochs: Number of training epochs
            batch_size: Batch size
            imgsz: Image size
        """
        self.model.train(data=data_yaml, epochs=epochs, batch=batch_size, imgsz=imgsz)
    
    def save_model(self, path):
        """Save model"""
        self.model.export(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model"""
        self.model = YOLO(path)
        self.model.to(self.device)
        print(f"Model loaded from {path}")