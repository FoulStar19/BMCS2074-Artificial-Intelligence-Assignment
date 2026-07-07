"""
CNN-based vehicle detector
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import sys
import os

# Add parent directory to path for model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class VehicleCNN(nn.Module):
    """Simple CNN for vehicle detection classification"""
    
    def __init__(self, num_classes=2):
        super(VehicleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.sigmoid(x)


class CNNDetector:
    """CNN-based vehicle detector with sliding window approach"""
    
    def __init__(self, model_path=None, device='cpu', confidence_threshold=0.5):
        """
        Initialize CNN detector
        
        Args:
            model_path: Path to trained model weights
            device: 'cpu' or 'cuda'
            confidence_threshold: Minimum confidence for detections
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Initialize model
        self.model = VehicleCNN(num_classes=2).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            # Use dummy weights for demonstration
            print("No model weights found. Using random initialization.")
        
        self.model.eval()
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Sliding window parameters
        self.window_sizes = [(64, 64), (96, 96), (128, 128)]
        self.stride = 16
        
    def detect(self, frame):
        """
        Detect vehicles in a frame using sliding window approach
        
        Args:
            frame: Input image (numpy array)
            
        Returns:
            List of detections with bbox and confidence
        """
        if frame is None:
            return []
        
        detections = []
        h, w = frame.shape[:2]
        
        # Convert to PIL Image for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Sliding window detection
        for win_w, win_h in self.window_sizes:
            for y in range(0, h - win_h, self.stride):
                for x in range(0, w - win_w, self.stride):
                    # Extract window
                    window = frame_rgb[y:y+win_h, x:x+win_w]
                    if window.size == 0:
                        continue
                    
                    # Preprocess
                    pil_window = Image.fromarray(window)
                    input_tensor = self.transform(pil_window).unsqueeze(0).to(self.device)
                    
                    # Inference
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        confidence = output.squeeze().cpu().numpy()
                    
                    # Check if vehicle detected (class 1) with confidence > threshold
                    if confidence[1] > self.confidence_threshold:
                        detections.append({
                            'bbox': [x, y, x + win_w, y + win_h],
                            'confidence': float(confidence[1]),
                            'class': 'vehicle'
                        })
        
        # Non-Maximum Suppression (NMS) to remove overlapping detections
        detections = self._non_max_suppression(detections, iou_threshold=0.4)
        
        return detections
    
    def _non_max_suppression(self, detections, iou_threshold=0.4):
        """
        Apply Non-Maximum Suppression to remove overlapping boxes
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Sort by confidence descending
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for det in detections:
            bbox = det['bbox']
            overlap = False
            
            for kept in filtered:
                bbox2 = kept['bbox']
                if self._calculate_iou(bbox, bbox2) > iou_threshold:
                    overlap = True
                    break
            
            if not overlap:
                filtered.append(det)
        
        return filtered
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union of two bounding boxes
        
        Args:
            bbox1, bbox2: [x1, y1, x2, y2]
            
        Returns:
            IoU score
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def train(self, train_loader, val_loader=None, epochs=10, lr=0.001):
        """
        Train the CNN model
        
        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            epochs: Number of training epochs
            lr: Learning rate
        """
        import torch.optim as optim
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
            
            if val_loader:
                val_loss = self._validate(val_loader, criterion)
                print(f"Validation Loss: {val_loss:.4f}")
        
        self.model.eval()
    
    def _validate(self, val_loader, criterion):
        """Validation step"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_model(self, path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {path}")