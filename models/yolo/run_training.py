"""
Complete pipeline runner for YOLO training
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed"""
    try:
        import ultralytics
        import torch
        import cv2
        import yaml
        print("✅ All dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nPlease install required packages:")
        print("pip install ultralytics torch opencv-python pyyaml scikit-learn matplotlib pandas plotly streamlit")
        return False


def run_prepare_data():
    """Run data preparation"""
    print("\n" + "="*60)
    print("STEP 1: Preparing Dataset")
    print("="*60)
    
    try:
        # Import and run prepare_data
        from prepare_data import prepare_vehicle_dataset
        prepare_vehicle_dataset()
        return True
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return False


def run_training():
    """Run training with versioning"""
    print("\n" + "="*60)
    print("STEP 2: Training Model")
    print("="*60)
    
    try:
        from train_with_versioning import VersionedModelTrainer
        
        trainer = VersionedModelTrainer()
        
        # Check if dataset.yaml exists
        if not os.path.exists('dataset.yaml'):
            print("❌ dataset.yaml not found. Please run prepare_data first.")
            return False
        
        # Train
        version, results = trainer.train(
            data_yaml='dataset.yaml',
            model_name='yolov8n.pt',
            epochs=50,  # Start with 50 epochs for quick testing
            batch_size=16,
            imgsz=640
        )
        
        print(f"\n✅ Training completed! Version: v{version}")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


def run_retraining():
    """Run retraining"""
    print("\n" + "="*60)
    print("STEP 3: Retraining Model")
    print("="*60)
    
    try:
        from train_with_versioning import VersionedModelTrainer
        
        trainer = VersionedModelTrainer()
        
        # List existing versions
        trainer.list_versions()
        
        # Get version to retrain
        if trainer.metadata['latest_version'] is None:
            print("No models to retrain. Train first.")
            return False
        
        version_to_retrain = trainer.metadata['latest_version']
        print(f"\nRetraining version {version_to_retrain} with additional epochs...")
        
        version, results = trainer.retrain(
            version_to_retrain=version_to_retrain,
            data_yaml='dataset.yaml',
            additional_epochs=30
        )
        
        print(f"\n✅ Retraining completed! New version: v{version}")
        return True
        
    except Exception as e:
        print(f"❌ Retraining failed: {e}")
        return False


def run_demo():
    """Run a demo using the latest model"""
    print("\n" + "="*60)
    print("STEP 4: Running Demo")
    print("="*60)
    
    try:
        from train_with_versioning import VersionedModelTrainer
        import cv2
        
        trainer = VersionedModelTrainer()
        
        # Get latest model
        weights_path = trainer.get_latest_model()
        if weights_path is None:
            print("No model found. Train first.")
            return False
        
        # Load model
        model = trainer.load_model()
        
        # Test on a sample image
        test_image = 'dataset/val/images/vehicle_0000.jpg'
        if os.path.exists(test_image):
            results = model(test_image)
            results[0].show()
            print(f"✅ Demo completed!")
        else:
            print("No test image found. Model loaded successfully.")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


def run_streamlit_app():
    """Launch the Streamlit app"""
    print("\n" + "="*60)
    print("STEP 5: Launching Streamlit App")
    print("="*60)
    
    app_path = 'app.py'
    if os.path.exists(app_path):
        print("\nStarting Streamlit app...")
        print("Press Ctrl+C to stop the app")
        time.sleep(2)
        subprocess.run(['streamlit', 'run', app_path])
    else:
        print(f"❌ app.py not found at {app_path}")


def main():
    """Main runner"""
    print("\n" + "="*60)
    print("🚗 YOLO Vehicle Detection Training Pipeline")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Menu
    while True:
        print("\n" + "="*60)
        print("Select an option:")
        print("1. Full Pipeline (Prepare Data + Train + Retrain + Demo)")
        print("2. Prepare Data Only")
        print("3. Train Model (First time)")
        print("4. Retrain Model (Add more epochs)")
        print("5. List All Versions")
        print("6. Run Demo")
        print("7. Launch Streamlit App")
        print("8. Exit")
        print("="*60)
        
        choice = input("Enter your choice (1-8): ").strip()
        
        if choice == '1':
            # Full pipeline
            if run_prepare_data():
                if run_training():
                    run_retraining()
                    run_demo()
                    if input("\nLaunch Streamlit app? (y/n): ").lower() == 'y':
                        run_streamlit_app()
        
        elif choice == '2':
            run_prepare_data()
        
        elif choice == '3':
            if not os.path.exists('dataset.yaml'):
                print("❌ dataset.yaml not found. Running prepare_data first...")
                run_prepare_data()
            run_training()
        
        elif choice == '4':
            run_retraining()
        
        elif choice == '5':
            try:
                from train_with_versioning import VersionedModelTrainer
                trainer = VersionedModelTrainer()
                trainer.list_versions()
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '6':
            run_demo()
        
        elif choice == '7':
            run_streamlit_app()
        
        elif choice == '8':
            print("👋 Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()