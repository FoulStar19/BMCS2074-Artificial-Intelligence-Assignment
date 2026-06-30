import cv2
import numpy as np
import torch
from PIL import Image

class ImageProcessor:
    @staticmethod
    def process_image(image_array, model, model_type="YOLO", confidence_threshold=0.5):
        """Process image with the specified model type"""
        
        if model_type == "CNN":
            return ImageProcessor._process_cnn(image_array, model, confidence_threshold)
        elif model_type == "ML":
            return ImageProcessor._process_ml(image_array, model, confidence_threshold)
        elif model_type == "YOLO":
            return ImageProcessor._process_yolo(image_array, model, confidence_threshold)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def _process_cnn(image_array, model, confidence_threshold):
        """Process image with CNN model"""
        # Preprocess for CNN
        img = cv2.resize(image_array, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            output = model(img)
            # For demo, simulate detections
            detections = []
            result_img = image_array.copy()
            
            # Simulate bounding boxes
            boxes = [
                [50, 50, 200, 350],
                [300, 100, 450, 400],
                [100, 200, 300, 500]
            ]
            
            for i, box in enumerate(boxes[:2]):  # Show only some
                x1, y1, x2, y2 = box
                conf = 0.85 + np.random.rand() * 0.1
                if conf >= confidence_threshold:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class': 0
                    })
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(result_img, f'Person {conf:.2f}%', 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 255, 0), 2)
        
        return result_img, detections
    
    @staticmethod
    def _process_ml(image_array, model, confidence_threshold):
        """Process image with ML model"""
        # Extract HOG features
        from skimage.feature import hog
        from skimage import exposure
        
        # Resize and convert to grayscale
        img = cv2.resize(image_array, (64, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Extract HOG features
        hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=False)
        
        # For demo, simulate detections
        detections = []
        result_img = image_array.copy()
        
        # Simulate bounding boxes (sliding window approach)
        boxes = [
            [30, 30, 180, 330],
            [280, 80, 430, 380],
            [80, 180, 280, 480]
        ]
        
        for i, box in enumerate(boxes[:2]):
            x1, y1, x2, y2 = box
            conf = 0.80 + np.random.rand() * 0.1
            if conf >= confidence_threshold:
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class': 0
                })
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(result_img, f'Person {conf:.2f}%', 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 0, 255), 2)
        
        return result_img, detections
    
    @staticmethod
    def _process_yolo(image_array, model, confidence_threshold):
        """Process image with YOLO model"""
        # Run inference
        results = model(image_array)
        
        detections = []
        result_img = image_array.copy()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if conf >= confidence_threshold:
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': cls
                        })
                        
                        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(result_img, f'Person {conf:.2f}%', 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (0, 255, 0), 2)
        
        return result_img, detections