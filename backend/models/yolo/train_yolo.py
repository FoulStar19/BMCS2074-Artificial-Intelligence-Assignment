import os
import sys
import yaml
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def create_dataset_yaml():
    """Create dataset.yaml file for YOLO training"""
    dataset_config = {
        'path': os.path.abspath('dataset'),
        'train': 'Training/images',
        'val': 'Validation/images',
        'test': 'Testing/images',
        'nc': 1,  # Number of classes
        'names': ['Person']
    }
    
    # Save YAML file
    with open('dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print("dataset.yaml created successfully!")
    return 'dataset.yaml'

def train_yolo():
    """Main training function for YOLO"""
    
    # Configuration
    config = {
        'model_name': 'yolov8n.pt',  # Base model: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
        'epochs': 500,
        'imgsz': 640,
        'batch': 16,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 4,
        'patience': 100,  # Early stopping patience
        'project': 'runs/train',
        'name': 'custom_yolo_model',
        'save_period': 10,  # Save checkpoint every 10 epochs
        'model_save_path': 'models/yolov8_custom.pt'
    }
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs(config['project'], exist_ok=True)
    
    print(f"Using device: {config['device']}")
    
    # Create dataset.yaml if not exists
    if not os.path.exists('dataset.yaml'):
        dataset_yaml = create_dataset_yaml()
    else:
        dataset_yaml = 'dataset.yaml'
        print("dataset.yaml already exists")
    
    # Verify dataset
    print("\nVerifying dataset structure...")
    required_dirs = [
        'dataset/Training/images',
        'dataset/Validation/images',
        'dataset/Testing/images'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        else:
            # Count images
            img_files = list(Path(dir_path).glob('*'))
            img_files = [f for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            print(f"  {dir_path}: {len(img_files)} images")
    
    # Initialize model
    print(f"\nLoading base model: {config['model_name']}")
    if os.path.exists(config['model_name']):
        model = YOLO(config['model_name'])
    else:
        # Download model if not exists
        print("Downloading model...")
        model = YOLO(config['model_name'])
    
    # Model info
    print(f"\nModel info:")
    print(f"  Type: {config['model_name']}")
    print(f"  Input size: {config['imgsz']}x{config['imgsz']}")
    print(f"  Classes: 1 (Person)")
    
    # Start training
    print("\n" + "="*60)
    print("Starting YOLO Training...")
    print("="*60)
    start_time = time.time()
    
    try:
        results = model.train(
            data=dataset_yaml,
            epochs=config['epochs'],
            imgsz=config['imgsz'],
            batch=config['batch'],
            device=config['device'],
            workers=config['workers'],
            patience=config['patience'],
            project=config['project'],
            name=config['name'],
            save_period=config['save_period'],
            pretrained=True,
            optimizer='auto',
            lr0=0.01,  # Initial learning rate
            lrf=0.01,  # Final learning rate factor
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            seed=42
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes!")
        
        # Display results
        print("\n" + "="*60)
        print("Training Results Summary")
        print("="*60)
        
        # Best model path
        best_model_path = f"{config['project']}/{config['name']}/weights/best.pt"
        if os.path.exists(best_model_path):
            print(f"Best model saved to: {best_model_path}")
            print("Copying best model to models directory...")
            
            # Copy best model to models directory
            import shutil
            shutil.copy2(best_model_path, config['model_save_path'])
            print(f"Model saved to: {config['model_save_path']}")
            
            # Also save last model
            last_model_path = f"{config['project']}/{config['name']}/weights/last.pt"
            if os.path.exists(last_model_path):
                shutil.copy2(last_model_path, 'models/yolov8_custom_last.pt')
                print("Last checkpoint saved to: models/yolov8_custom_last.pt")
            
            # Validate the model
            print("\nValidating best model...")
            model = YOLO(config['model_save_path'])
            val_results = model.val(data=dataset_yaml)
            
            print("\nValidation Results:")
            print(f"  mAP50-95: {val_results.box.map:.4f}")
            print(f"  mAP50: {val_results.box.map50:.4f}")
            print(f"  mAP75: {val_results.box.map75:.4f}")
        
        # Plot training metrics
        try:
            # Load results from training
            results_path = f"{config['project']}/{config['name']}/results.csv"
            if os.path.exists(results_path):
                import pandas as pd
                df = pd.read_csv(results_path)
                
                # Plot metrics
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Loss plots
                axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
                axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss')
                axes[0, 0].plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].set_title('Training Losses')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
                
                # Validation loss
                axes[0, 1].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
                axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss')
                axes[0, 1].plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].set_title('Validation Losses')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
                
                # Metrics
                axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
                axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].set_title('Precision and Recall')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
                
                # mAP
                axes[1, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
                axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('mAP')
                axes[1, 1].set_title('mAP Metrics')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
                
                plt.tight_layout()
                plt.savefig('models/yolo_training_curves.png')
                print("Training curves saved to: models/yolo_training_curves.png")
                plt.show()
                
        except Exception as e:
            print(f"Could not plot training curves: {e}")
        
        print("\n" + "="*60)
        print("YOLO Training Completed Successfully!")
        print("="*60)
        
        return results
        
    except Exception as e:
        print(f"\nError during training: {e}")
        return None

def test_yolo_model():
    """Test the trained YOLO model on test set"""
    model_path = 'models/yolov8_custom.pt'
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    print(f"\nTesting model: {model_path}")
    model = YOLO(model_path)
    
    # Run inference on test set
    results = model.val(data='dataset.yaml')
    
    print("\nTest Results:")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP75: {results.box.map75:.4f}")
    
    return results

if __name__ == "__main__":
    # Train model
    train_yolo()
    
    # Test model
    test_yolo_model()