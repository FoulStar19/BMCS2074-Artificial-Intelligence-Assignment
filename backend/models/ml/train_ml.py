import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.models.ml.ml_model import HumanDetectionML

def load_images_from_directory(directory, max_images=None):
    """Load images from directory"""
    images = []
    labels = []
    
    # Walk through directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    
                    # Label based on filename or directory
                    if 'person' in file.lower() or 'human' in file.lower():
                        labels.append(1)
                    else:
                        labels.append(0)
    
    # Limit dataset if specified
    if max_images and len(images) > max_images:
        indices = np.random.choice(len(images), max_images, replace=False)
        images = [images[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    print(f"Loaded {len(images)} images from {directory}")
    return images, np.array(labels)

def train_ml():
    """Main training function for ML model"""
    
    # Configuration
    config = {
        'model_type': 'svm',  # 'svm' or 'random_forest'
        'test_size': 0.2,
        'max_images': 1000,  # Limit for faster training
        'model_save_path': 'models/ml_model.pkl'
    }
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    print("Loading training data...")
    
    # Load training images
    train_images, train_labels = load_images_from_directory(
        'dataset/Training/images',
        max_images=config['max_images']
    )
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_images, train_labels,
        test_size=config['test_size'],
        random_state=42,
        stratify=train_labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Positive samples (train): {np.sum(y_train)}")
    print(f"Negative samples (train): {len(y_train) - np.sum(y_train)}")
    
    # Initialize and train model
    print(f"\nTraining {config['model_type']} model...")
    model = HumanDetectionML(model_type=config['model_type'])
    
    # Train model with validation
    val_accuracy = model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['No Person', 'Person']))
    
    # Create confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {config["model_type"].upper()}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('models/ml_confusion_matrix.png')
    plt.show()
    
    # Save model
    model.save_model(config['model_save_path'])
    
    # Test on test set if available
    if os.path.exists('dataset/Testing/images'):
        print("\nTesting on test set...")
        test_images, test_labels = load_images_from_directory(
            'dataset/Testing/images',
            max_images=config['max_images']
        )
        
        if len(test_images) > 0:
            y_test_pred = model.predict(test_images)
            test_accuracy = accuracy_score(test_labels, y_test_pred)
            print(f"Test Accuracy: {test_accuracy:.4f}")
            
            print("\nTest Classification Report:")
            print(classification_report(test_labels, y_test_pred, target_names=['No Person', 'Person']))
    
    print(f"\nTraining completed! Model saved to {config['model_save_path']}")
    
    return model

if __name__ == "__main__":
    train_ml()