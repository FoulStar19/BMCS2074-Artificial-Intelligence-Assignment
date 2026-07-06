"""
YOLO model training with versioning
"""

import os
import torch
import yaml
import json
import shutil
from ultralytics import YOLO
from datetime import datetime
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class VersionedModelTrainer:
    """Train YOLO model with version control"""
    
    def __init__(self, base_dir='models/yolo'):
        """
        Initialize trainer
        
        Args:
            base_dir: Base directory for storing models
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create versions directory
        self.versions_dir = self.base_dir / 'versions'
        self.versions_dir.mkdir(exist_ok=True)
        
        # Create metadata file
        self.metadata_file = self.base_dir / 'metadata.json'
        self.metadata = self._load_metadata()
        
        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
    
    def _load_metadata(self):
        """Load or create metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'versions': [],
            'latest_version': None,
            'total_models': 0
        }
    
    def _save_metadata(self):
        """Save metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_next_version(self):
        """Get next version number"""
        existing_versions = [v['version'] for v in self.metadata['versions']]
        if not existing_versions:
            return 1
        return max(existing_versions) + 1
    
    def train(self, data_yaml, model_name='yolov8n.pt', epochs=100, 
              batch_size=16, imgsz=640, project='vehicle_detection'):
        """
        Train with versioning
        
        Args:
            data_yaml: Path to dataset YAML
            model_name: Base model name
            epochs: Number of epochs
            batch_size: Batch size
            imgsz: Image size
            project: Project name
            
        Returns:
            Version number and training results
        """
        # Get next version
        version = self.get_next_version()
        version_name = f"v{version}"
        
        # Create version directory
        version_dir = self.versions_dir / version_name
        version_dir.mkdir(exist_ok=True)
        
        # Copy dataset YAML to version directory
        if os.path.exists(data_yaml):
            shutil.copy2(data_yaml, version_dir / 'dataset.yaml')
        
        # Prepare training args
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'device': self.device,
            'project': str(version_dir / project),
            'name': version_name,
            'exist_ok': True,
            'verbose': True,
            'patience': 50,
            'save': True,
            'save_period': 10,
            'seed': 42
        }
        
        # Save training config
        with open(version_dir / 'config.json', 'w') as f:
            json.dump(train_args, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"🚀 STARTING TRAINING - Version {version}")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Image Size: {imgsz}")
        print(f"Device: {self.device}")
        print(f"Output: {version_dir}")
        print(f"{'='*60}\n")
        
        # Initialize model
        model = YOLO(model_name)
        
        # Train
        try:
            results = model.train(**train_args)
            
            # Find best weights
            weights_dir = version_dir / project / version_name / 'weights'
            best_weights = weights_dir / 'best.pt'
            
            # Try alternative path if not found
            if not best_weights.exists():
                alt_weights = version_dir / project / 'weights' / 'best.pt'
                if alt_weights.exists():
                    best_weights = alt_weights
            
            if not best_weights.exists():
                # Try to find any weights file
                weight_files = list(version_dir.rglob('best.pt'))
                if weight_files:
                    best_weights = weight_files[0]
                else:
                    raise FileNotFoundError(f"Best weights not found")
            
            # Copy best weights to version root
            shutil.copy2(str(best_weights), str(version_dir / 'best.pt'))
            
            # Save results summary
            results_dict = {
                'version': version,
                'name': version_name,
                'model': model_name,
                'epochs': epochs,
                'batch_size': batch_size,
                'imgsz': imgsz,
                'device': self.device,
                'train_date': datetime.now().isoformat(),
                'best_weights': str(best_weights),
                'metrics': self._extract_metrics(results)
            }
            
            # Save results
            with open(version_dir / 'results.json', 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            # Update metadata
            version_info = {
                'version': version,
                'name': version_name,
                'path': str(version_dir),
                'created': datetime.now().isoformat(),
                'model_type': model_name,
                'epochs': epochs,
                'metrics': results_dict['metrics']
            }
            
            self.metadata['versions'].append(version_info)
            self.metadata['latest_version'] = version
            self.metadata['total_models'] += 1
            self._save_metadata()
            
            # Generate training report
            self._generate_report(version_dir, results)
            
            print(f"\n✅ TRAINING COMPLETED! Version {version}")
            print(f"📁 Location: {version_dir}")
            print(f"🏆 Best weights: {best_weights}")
            
            return version, results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
    
    def _extract_metrics(self, results):
        """Extract metrics from training results"""
        metrics = {}
        
        try:
            if hasattr(results, 'metrics'):
                metrics_dict = results.metrics
                if isinstance(metrics_dict, dict):
                    metrics = metrics_dict
        except:
            pass
        
        # Add placeholder metrics if not available
        if not metrics:
            metrics = {
                'mAP50': 0.0,
                'mAP50-95': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        
        return metrics
    
    def _generate_report(self, version_dir, results):
        """Generate training report with plots"""
        report_dir = version_dir / 'report'
        report_dir.mkdir(exist_ok=True)
        
        # Try to generate plots if results contain metrics history
        try:
            # Look for results.csv in various locations
            results_csv = None
            possible_paths = [
                version_dir / 'vehicle_detection' / f'{version_dir.name}' / 'results.csv',
                version_dir / 'vehicle_detection' / 'results.csv',
                version_dir / 'results.csv'
            ]
            
            for path in possible_paths:
                if path.exists():
                    results_csv = path
                    break
            
            if results_csv:
                df = pd.read_csv(results_csv)
                
                # Plot metrics
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Find available columns
                loss_cols = [c for c in df.columns if 'loss' in c.lower()]
                metric_cols = [c for c in df.columns if 'mAP' in c or 'precision' in c or 'recall' in c]
                
                if len(loss_cols) > 0:
                    axes[0,0].plot(df[loss_cols[0]], label=loss_cols[0])
                    axes[0,0].set_title('Loss')
                    axes[0,0].legend()
                
                if len(loss_cols) > 1:
                    axes[0,1].plot(df[loss_cols[1]], label=loss_cols[1])
                    axes[0,1].set_title('Validation Loss')
                    axes[0,1].legend()
                
                if len(metric_cols) > 0:
                    axes[1,0].plot(df[metric_cols[0]], label=metric_cols[0])
                    axes[1,0].set_title(metric_cols[0])
                    axes[1,0].legend()
                
                if len(metric_cols) > 1:
                    axes[1,1].plot(df[metric_cols[1]], label=metric_cols[1])
                    axes[1,1].set_title(metric_cols[1])
                    axes[1,1].legend()
                
                plt.tight_layout()
                plt.savefig(report_dir / 'training_metrics.png', dpi=150)
                plt.close()
                
                print(f"📊 Training plots saved to {report_dir}")
        
        except Exception as e:
            print(f"Note: Could not generate plots: {e}")
    
    def retrain(self, version_to_retrain, data_yaml, additional_epochs=50, 
                model_name=None):
        """
        Retrain an existing model with additional epochs
        
        Args:
            version_to_retrain: Version number to retrain
            data_yaml: Path to dataset YAML
            additional_epochs: Additional epochs to train
            model_name: New base model (optional)
            
        Returns:
            New version number
        """
        # Find existing version
        version_info = None
        for v in self.metadata['versions']:
            if v['version'] == version_to_retrain:
                version_info = v
                break
        
        if not version_info:
            raise ValueError(f"Version {version_to_retrain} not found")
        
        # Use existing weights as starting point
        weights_path = Path(version_info['path']) / 'best.pt'
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        
        if model_name is None:
            model_name = str(weights_path)
        
        print(f"\n🔄 RETRAINING VERSION {version_to_retrain}")
        print(f"📁 Using weights: {weights_path}")
        print(f"➕ Additional epochs: {additional_epochs}")
        
        # Train with additional epochs
        return self.train(
            data_yaml=data_yaml,
            model_name=model_name,
            epochs=additional_epochs,
            batch_size=16,
            imgsz=640
        )
    
    def list_versions(self):
        """List all trained versions"""
        print(f"\n{'='*60}")
        print("📊 TRAINED MODELS")
        print(f"{'='*60}")
        
        if not self.metadata['versions']:
            print("No models trained yet.")
            return
        
        # Create DataFrame
        df_data = []
        for v in self.metadata['versions']:
            row = {
                'Version': f"v{v['version']}",
                'Created': v['created'][:19].replace('T', ' '),
                'Epochs': v['epochs'],
                'Model': v['model_type'],
                'mAP50': v.get('metrics', {}).get('mAP50', 'N/A'),
                'mAP50-95': v.get('metrics', {}).get('mAP50-95', 'N/A')
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        print(f"\n📁 Base directory: {self.base_dir}")
        print(f"📊 Total models: {len(self.metadata['versions'])}")
        print(f"⭐ Latest version: v{self.metadata['latest_version']}")
        print(f"{'='*60}")
    
    def get_latest_model(self):
        """Get the latest trained model path"""
        if self.metadata['latest_version'] is None:
            return None
        
        version_info = None
        for v in self.metadata['versions']:
            if v['version'] == self.metadata['latest_version']:
                version_info = v
                break
        
        if version_info:
            weights_path = Path(version_info['path']) / 'best.pt'
            if weights_path.exists():
                return str(weights_path)
        
        return None
    
    def load_model(self, version=None):
        """
        Load a specific version or latest model
        
        Args:
            version: Version number (None for latest)
            
        Returns:
            YOLO model instance
        """
        if version is None:
            weights_path = self.get_latest_model()
            if weights_path is None:
                raise ValueError("No model found")
        else:
            # Find version
            version_info = None
            for v in self.metadata['versions']:
                if v['version'] == version:
                    version_info = v
                    break
            
            if not version_info:
                raise ValueError(f"Version {version} not found")
            
            weights_path = Path(version_info['path']) / 'best.pt'
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights not found: {weights_path}")
        
        print(f"Loading model from: {weights_path}")
        model = YOLO(str(weights_path))
        return model


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train YOLO vehicle detection model with versioning')
    parser.add_argument('--data', type=str, default='dataset.yaml',
                       help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Base model name or path')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--retrain', type=int, default=None,
                       help='Version to retrain (adds more epochs)')
    parser.add_argument('--list', action='store_true',
                       help='List all trained versions')
    parser.add_argument('--test', type=str, default=None,
                       help='Test image path for loaded model')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = VersionedModelTrainer()
    
    # List versions
    if args.list:
        trainer.list_versions()
        return
    
    # Check data YAML
    if not os.path.exists(args.data):
        print(f"Error: Data YAML file '{args.data}' not found!")
        print("Please run prepare_data.py first to create dataset.yaml")
        return
    
    # Train or retrain
    try:
        if args.retrain:
            print(f"🔄 Retraining version {args.retrain}...")
            version, results = trainer.retrain(
                version_to_retrain=args.retrain,
                data_yaml=args.data,
                additional_epochs=args.epochs,
                model_name=args.model
            )
            print(f"✅ Retraining completed! New version: v{version}")
        else:
            version, results = trainer.train(
                data_yaml=args.data,
                model_name=args.model,
                epochs=args.epochs,
                batch_size=args.batch,
                imgsz=args.imgsz
            )
        
        # Show summary
        trainer.list_versions()
        
        # Test if image provided
        if args.test and os.path.exists(args.test):
            model = trainer.load_model(version)
            results = model(args.test)
            results[0].show()
            print(f"✅ Model tested on {args.test}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()