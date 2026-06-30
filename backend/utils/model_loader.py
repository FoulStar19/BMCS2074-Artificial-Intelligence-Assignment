import os
import pickle
import torch
import numpy as np
from ultralytics import YOLO
import cv2
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelLoader:
    def __init__(self, models_dir=None):
        if models_dir is None:
            self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        else:
            self.models_dir = models_dir
        
        self.loaded_models = {}
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def load_model(self, model_type):
        """Load model based on type: CNN, ML, or YOLO"""
        if model_type in self.loaded_models:
            return self.loaded_models[model_type]
        
        if model_type == "CNN":
            model = self.load_cnn_model()
        elif model_type == "ML":
            model = self.load_ml_model()
        elif model_type == "YOLO":
            model = self.load_yolo_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.loaded_models[model_type] = model
        return model
    
    def load_cnn_model(self):
        """Load custom CNN model"""
        try:
            from backend.models.cnn.cnn_model import HumanDetectionCNN
            
            model_path = os.path.join(self.models_dir, "cnn_model.pth")
            
            if os.path.exists(model_path):
                model = HumanDetectionCNN(num_classes=2)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                print("CNN model loaded successfully!")
                return model
            else:
                print("CNN model not found. Using dummy model.")
                return DummyCNNModel()
        except ImportError as e:
            print(f"CNN model module not found: {e}")
            return DummyCNNModel()
    
    def load_ml_model(self):
        """Load traditional ML model (SVM/Random Forest)"""
        try:
            model_path = os.path.join(self.models_dir, "ml_model.pkl")
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Load the model wrapper
                from backend.models.ml.ml_model import HumanDetectionML
                
                # Create wrapper and load model
                if 'model_type' in model_data:
                    ml_model = HumanDetectionML(model_type=model_data['model_type'])
                else:
                    ml_model = HumanDetectionML()
                
                ml_model.model = model_data['model']
                ml_model.scaler = model_data['scaler']
                ml_model.is_trained = True
                
                print("ML model loaded successfully!")
                return ml_model
            else:
                print("ML model not found. Using dummy model.")
                return DummyMLModel()
        except Exception as e:
            print(f"Error loading ML model: {e}")
            return DummyMLModel()
    
    def load_yolo_model(self):
        """Load YOLO model"""
        try:
            # Try custom trained model first
            custom_model_path = os.path.join(self.models_dir, "yolov8_custom.pt")
            
            if os.path.exists(custom_model_path):
                print("Loading custom YOLO model...")
                return YOLO(custom_model_path)
            
            # Try standard model
            standard_model_path = os.path.join(self.models_dir, "yolov8n.pt")
            
            if os.path.exists(standard_model_path):
                print("Loading standard YOLO model...")
                return YOLO(standard_model_path)
            
            # Download model
            print("Downloading YOLO model...")
            return YOLO('yolov8n.pt')
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return DummyYOLOModel()

# ==========================
# Dummy Models for Demo
# ==========================

class DummyCNNModel:
    """Dummy CNN model for demonstration"""
    def __call__(self, image):
        return self.predict(image)
    
    def predict(self, image):
        return [{'bbox': [50, 50, 200, 350], 'confidence': 0.92, 'class': 0}]
    
    def eval(self):
        return self

class DummyMLModel:
    """Dummy ML model for demonstration"""
    def __call__(self, image):
        return self.predict(image)
    
    def predict(self, image):
        return [{'bbox': [50, 50, 200, 350], 'confidence': 0.85, 'class': 0}]

class DummyYOLOModel:
    """Dummy YOLO model for demonstration"""
    def __call__(self, image):
        return self.predict(image)
    
    def predict(self, image):
        return [{'bbox': [50, 50, 200, 350], 'confidence': 0.95, 'class': 0}]