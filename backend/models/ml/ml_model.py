import numpy as np
import cv2
from skimage.feature import hog
from skimage import exposure
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

class HumanDetectionML:
    """Machine Learning model for human detection using HOG features"""
    
    def __init__(self, model_type='svm'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        if model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.is_trained = False
    
    def extract_hog_features(self, image):
        """Extract HOG features from image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to standard size
        gray = cv2.resize(gray, (64, 128))
        
        # Extract HOG features
        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            block_norm='L2-Hys'
        )
        
        return features
    
    def extract_batch_features(self, images):
        """Extract HOG features from batch of images"""
        features_list = []
        
        for image in images:
            features = self.extract_hog_features(image)
            features_list.append(features)
        
        return np.array(features_list)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        # Extract features
        X_train_features = self.extract_batch_features(X_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Validate if validation data is provided
        if X_val is not None and y_val is not None:
            X_val_features = self.extract_batch_features(X_val)
            X_val_scaled = self.scaler.transform(X_val_features)
            val_accuracy = self.model.score(X_val_scaled, y_val)
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            return val_accuracy
        
        return None
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Extract features
        X_features = self.extract_batch_features(X)
        X_scaled = self.scaler.transform(X_features)
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_features = self.extract_batch_features(X)
        X_scaled = self.scaler.transform(X_features)
        
        return self.model.predict_proba(X_scaled)
    
    def save_model(self, model_path):
        """Save model to file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load model from file"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {model_path}")