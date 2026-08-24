"""
Model Manager - handles model loading, discovery, and lifecycle management.
Optimized with lazy loading and proper resource cleanup.
"""

import os
import yaml
import gc
from pathlib import Path
from typing import Dict, Optional, Any

import torch

from backend.detection.yolo_detector import YOLODetector


class ModelManager:
    """Manages model lifecycle, discovery, and loading."""

    def __init__(self):
        self.loaded_model = None
        self.model_type = None
        self.model_path = None
        self.device = "cpu"
        self.conf_threshold = 0.25

    @staticmethod
    def load_dataset_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load dataset configuration from YAML."""
        if config_path is None:
            possible_paths = [
                Path("config/dataset.yaml"),
                Path("model/yolo/dataset.yaml"),
                Path("dataset.yaml"),
                Path(__file__).parent.parent.parent / "config" / "dataset.yaml",
                Path(__file__).parent.parent.parent / "model" / "yolo" / "dataset.yaml",
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading config: {e}")

        # Default configuration
        return {
            "nc": 5,
            "names": {0: "car", 1: "truck", 2: "bus", 3: "motorcycle", 4: "bicycle"},
            "colors": {
                0: [0, 0, 255],
                1: [0, 255, 0],
                2: [255, 0, 0],
                3: [0, 255, 255],
                4: [255, 0, 255],
            },
        }

    def discover_models(self, runs_dir: str = "model/yolo/runs") -> Dict[str, str]:
        """Discover available trained models."""
        models = {}
        runs_path = Path(runs_dir)

        if not runs_path.exists():
            # Try alternative paths
            alt_paths = [
                Path("runs"),
                Path("model/runs"),
                Path("yolo/runs"),
                Path(__file__).parent.parent.parent / "model" / "yolo" / "runs",
            ]
            for path in alt_paths:
                if path.exists():
                    runs_path = path
                    break

        if runs_path.exists():
            for run_dir in runs_path.iterdir():
                if run_dir.is_dir():
                    weights_dir = run_dir / "weights"
                    if weights_dir.exists():
                        weight_files = list(weights_dir.glob("best.pt")) + list(weights_dir.glob("*.pt"))
                        for weight_file in weight_files:
                            # If we have both best.pt and other .pt, prefer best.pt
                            if weight_file.name == "best.pt":
                                version_name = f"{run_dir.name}/best"
                            else:
                                version_name = f"{run_dir.name}/{weight_file.stem}"
                            models[version_name] = str(weight_file)

        # If no models found, check specific path from user
        specific_path = Path("model/yolo/runs/v1/train/weights/best.pt")
        if specific_path.exists() and "v1/best" not in models:
            models["v1/best"] = str(specific_path)

        return models

    def load_model(self, model_type: str = "YOLO", model_path: str = None,
                   device: str = "cpu", conf_threshold: float = 0.25):
        """Load a model with the specified parameters."""
        # Unload any existing model first
        self.unload_model()

        self.model_type = model_type
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold

        if model_type.upper() == "YOLO":
            detector = YOLODetector(
                model_path=model_path,
                device=device,
                conf_threshold=conf_threshold,
                enable_tracking=True,
                max_lost_frames=15,
            )
            self.loaded_model = detector
            return detector
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def unload_model(self):
        """Unload the currently loaded model and free resources."""
        if self.loaded_model is not None:
            try:
                if hasattr(self.loaded_model, "unload"):
                    self.loaded_model.unload()
                elif hasattr(self.loaded_model, "model"):
                    # Try to free model resources
                    model = getattr(self.loaded_model, "model", None)
                    if model is not None and hasattr(model, "to"):
                        model.to("cpu")
                        del model
            except Exception as e:
                print(f"Error unloading model: {e}")

            self.loaded_model = None
            self.model_type = None
            self.model_path = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.loaded_model is not None

    def get_model(self):
        """Get the currently loaded model."""
        return self.loaded_model