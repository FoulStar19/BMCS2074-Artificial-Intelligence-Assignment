"""
Model Manager - Centralized model discovery, loading, and management
"""

import os
from pathlib import Path
import yaml
import torch
import sys

# Add parent directory to path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)


class ModelManager:
    """
    Manages model discovery, loading, and lifecycle
    """
    
    def __init__(self, base_dir=None):
        """
        Initialize ModelManager
        
        Args:
            base_dir: Base directory for model discovery
        """
        self.base_dir = Path(base_dir) if base_dir else Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.models = {}
        self.loaded_model = None
        self.model_type = None
        self.model_path = None
        self.device = 'cpu'
        
    def discover_models(self):
        """
        Discover available models from runs directories
        
        Returns:
            Dictionary of model versions and paths
        """
        models = {}
        
        # Get the absolute path to the project root
        # Navigate from backend/core to project root
        project_root = self.base_dir.parent.parent if self.base_dir.name == 'core' else self.base_dir
        
        # Define search paths - use project_root as base
        search_paths = [
            project_root / "model" / "yolo" / "runs",
            project_root / "model" / "runs",
            project_root / "yolo" / "runs",
            project_root / "runs",
        ]
        
        # Also search from current directory
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        search_paths.extend([
            current_dir.parent.parent / "model" / "yolo" / "runs",
            current_dir.parent.parent / "model" / "runs",
            current_dir.parent.parent / "yolo" / "runs",
            current_dir.parent.parent / "runs",
        ])
        
        # Add explicit path from train_model.py
        explicit_path = Path(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\model\yolo\runs")
        search_paths.append(explicit_path)
        
        print(f"🔍 Searching for models in {len(search_paths)} locations...")
        print(f"📁 Project root: {project_root}")
        
        for runs_path in search_paths:
            if runs_path.exists():
                print(f"✅ Checking: {runs_path}")
                
                # Look for version directories
                for item in runs_path.iterdir():
                    if item.is_dir():
                        # Check for version directories (v1, v2, etc.)
                        if item.name.startswith('v') and item.name[1:].isdigit():
                            print(f"  📁 Found version: {item.name}")
                            
                            # Check for weights directory directly in version folder
                            weights_dir = item / "weights"
                            if weights_dir.exists():
                                for pt_file in weights_dir.glob("*.pt"):
                                    key = f"{item.name}/{pt_file.stem}"
                                    models[key] = str(pt_file)
                                    print(f"    ✅ Found: {key}")
                            
                            # Look for train subdirectories in version folder
                            for sub_dir in item.iterdir():
                                if sub_dir.is_dir() and (sub_dir.name.startswith('train') or sub_dir.name == 'train'):
                                    weights_dir = sub_dir / "weights"
                                    if weights_dir.exists():
                                        for pt_file in weights_dir.glob("*.pt"):
                                            key = f"{item.name}/{sub_dir.name}/{pt_file.stem}"
                                            models[key] = str(pt_file)
                                            print(f"    ✅ Found: {key}")
                            
                            # Look for other subdirectories that might contain weights
                            for sub_dir in item.iterdir():
                                if sub_dir.is_dir() and sub_dir.name not in ['train']:
                                    weights_dir = sub_dir / "weights"
                                    if weights_dir.exists():
                                        for pt_file in weights_dir.glob("*.pt"):
                                            key = f"{item.name}/{sub_dir.name}/{pt_file.stem}"
                                            models[key] = str(pt_file)
                                            print(f"    ✅ Found: {key}")
                        
                        # Look for train directories directly
                        elif item.name.startswith('train') or item.name == 'train':
                            weights_dir = item / "weights"
                            if weights_dir.exists():
                                for pt_file in weights_dir.glob("*.pt"):
                                    key = f"{item.name}/{pt_file.stem}"
                                    models[key] = str(pt_file)
                                    print(f"  ✅ Found: {key}")
        
        # Specific check for the known model path
        known_model_path = Path(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\model\yolo\runs\v1\train\weights\best.pt")
        if known_model_path.exists():
            key = "v1/train/best"
            if key not in models:
                models[key] = str(known_model_path)
                print(f"✅ Found specific model: {key}")
        else:
            print(f"⚠️ Known model path not found: {known_model_path}")
        
        # Check if the v1 directory exists but doesn't have the train subdirectory
        v1_path = Path(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\model\yolo\runs\v1")
        if v1_path.exists():
            print(f"📁 Checking v1 directory: {v1_path}")
            # Look recursively for any .pt files in v1
            for pt_file in v1_path.glob("**/*.pt"):
                # Create a relative key from v1
                relative_path = pt_file.relative_to(v1_path)
                key = f"v1/{relative_path.parent.name}/{relative_path.stem}"
                if key not in models:
                    models[key] = str(pt_file)
                    print(f"  ✅ Found: {key}")
        
        # Fallback: look for any .pt files in the project
        if not models:
            print("🔍 No models found in runs, searching for .pt files...")
            fallback_paths = [
                project_root,
                project_root / "model",
                project_root / "model" / "yolo",
                project_root / "yolo",
            ]
            
            for path in fallback_paths:
                if path.exists():
                    for pt_file in path.glob("**/*.pt"):
                        if "venv" not in str(pt_file) and "site-packages" not in str(pt_file):
                            # Try to create a meaningful key
                            if "runs" in str(pt_file):
                                parts = str(pt_file).split("runs")[-1].strip("\\/").split("\\")
                                if len(parts) >= 2:
                                    key = f"{parts[0]}/{parts[1]}"
                                    if key not in models:
                                        models[key] = str(pt_file)
                                        print(f"  ✅ Found: {key}")
                            else:
                                key = f"{pt_file.parent.name}/{pt_file.stem}"
                                if key not in models:
                                    models[key] = str(pt_file)
                                    print(f"  ✅ Found: {key}")
        
        # Sort models for consistent display
        self.models = dict(sorted(models.items()))
        print(f"📊 Discovered {len(self.models)} model(s)")
        
        if self.models:
            print("📋 Available models:")
            for key, path in self.models.items():
                print(f"  - {key}: {path}")
        
        return self.models
    
    def load_model(self, model_type, model_path, device='cpu', conf_threshold=0.25):
        """
        Load a model based on type and path
        
        Args:
            model_type: 'YOLO' or 'CNN'
            model_path: Path to model weights
            device: 'cpu' or 'cuda'
            conf_threshold: Confidence threshold
            
        Returns:
            Detector instance or None if failed
        """
        self.model_type = model_type
        self.model_path = model_path
        self.device = device
        
        try:
            if model_type == 'YOLO':
                from backend.detection.yolo_detector import YOLODetector
                detector = YOLODetector(
                    model_path=model_path,
                    device=device,
                    conf_threshold=conf_threshold
                )
            else:
                from backend.detection.cnn_detector import CNNDetector
                detector = CNNDetector(
                    model_path=model_path,
                    device=device,
                    confidence_threshold=conf_threshold
                )
            
            self.loaded_model = detector
            return detector
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.loaded_model = None
            return None
    
    def get_model_info(self):
        """
        Get information about the currently loaded model
        
        Returns:
            Dictionary with model info
        """
        if self.loaded_model is None:
            return {
                'loaded': False,
                'type': None,
                'path': None,
                'device': None
            }
        
        return {
            'loaded': True,
            'type': self.model_type,
            'path': self.model_path,
            'device': self.device,
            'is_dummy': getattr(self.loaded_model, 'is_dummy', False)
        }
    
    def unload_model(self):
        """Unload the currently loaded model to free memory"""
        if self.loaded_model is not None:
            self.loaded_model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def load_dataset_config():
        """
        Load dataset configuration from YAML file
        
        Returns:
            Configuration dictionary
        """
        yaml_paths = [
            Path("dataset.yaml"),
            Path("model/yolo/dataset.yaml"),
            Path("config/dataset.yaml"),
            Path("dataset/dataset.yaml"),
        ]
        
        current_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        project_root = current_dir.parent.parent if current_dir.name == 'core' else current_dir.parent
        
        yaml_paths.extend([
            project_root / "dataset.yaml",
            project_root / "model" / "yolo" / "dataset.yaml",
            project_root / "config" / "dataset.yaml",
            current_dir / "dataset.yaml",
            current_dir / "model" / "yolo" / "dataset.yaml",
        ])
        
        explicit_path = Path(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\model\yolo\dataset.yaml")
        yaml_paths.append(explicit_path)
        
        for yaml_path in yaml_paths:
            if yaml_path.exists():
                try:
                    with open(yaml_path, 'r') as f:
                        config = yaml.safe_load(f)
                    print(f"✅ Loaded dataset config from: {yaml_path}")
                    return config
                except Exception as e:
                    print(f"⚠️ Error loading {yaml_path}: {e}")
        
        print("ℹ️ Using default dataset configuration")
        return {
            'nc': 5,
            'names': {
                0: 'car',
                1: 'truck',
                2: 'bus',
                3: 'motorcycle',
                4: 'bicycle'
            },
            'colors': {
                0: [0, 0, 255],
                1: [0, 255, 0],
                2: [255, 0, 0],
                3: [0, 255, 255],
                4: [255, 0, 255]
            }
        }