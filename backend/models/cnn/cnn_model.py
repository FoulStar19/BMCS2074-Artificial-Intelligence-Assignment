import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.models.cnn.cnn_model import HumanDetectionCNN

class HumanDataset(Dataset):
    """Custom dataset for human detection"""
    def __init__(self, image_dir, transform=None, is_training=True):
        self.image_dir = image_dir
        self.transform = transform
        self.is_training = is_training
        
        # Load images
        self.images = []
        self.labels = []
        
        # Walk through directory
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    self.images.append(img_path)
                    
                    # Label: 1 for person, 0 for no person
                    # This assumes your dataset has subfolders or naming convention
                    # You can modify this based on your dataset structure
                    if 'person' in file.lower() or 'human' in file.lower():
                        self.labels.append(1)
                    else:
                        self.labels.append(0)
        
        print(f"Loaded {len(self.images)} images for {'training' if is_training else 'validation'}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_cnn():
    """Main training function for CNN"""
    
    # Configuration
    config = {
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'image_size': 224,
        'num_classes': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_save_path': 'models/cnn_model.pth',
        'checkpoint_dir': 'models/checkpoints',
        'early_stopping_patience': 10
    }
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    print(f"Using device: {config['device']}")
    
    # Data augmentation and transforms
    train_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = HumanDataset(
        'dataset/Training/images',
        transform=train_transform,
        is_training=True
    )
    
    val_dataset = HumanDataset(
        'dataset/Validation/images',
        transform=val_transform,
        is_training=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = HumanDetectionCNN(num_classes=config['num_classes']).to(config['device'])
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    patience_counter = 0
    
    # Training loop
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 50)
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for images, labels in train_pbar:
            images, labels = images.to(config['device']), labels.to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.4f}'
            })
        
        train_acc = train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for images, labels in val_pbar:
                images, labels = images.to(config['device']), labels.to(config['device'])
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"✓ Best model saved with val_acc: {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint
        checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, checkpoint_path)
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
    
    # Save training history
    with open('models/cnn_training_history.json', 'w') as f:
        json.dump(history, f)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('models/cnn_training_curves.png')
    plt.show()
    
    return model, history

if __name__ == "__main__":
    train_cnn()