"""
Model Manager - handles model loading, discovery, and lifecycle management.

Primary detectors (benchmarked head-to-head): YOLO and Faster R-CNN.
The CNN is no longer a primary detector -- it's loaded separately via
load_classifier() as an optional verification pass on top of whichever
primary detector is active (see CNNDetector.classify_crop).
"""

import os
import yaml
import gc
from pathlib import Path
from typing import Dict, Optional, Any

import torch

from backend.detection.yolo_detector import YOLODetector
from backend.detection.faster_rcnn_detector import FasterRCNNDetector
# cnn_backend.py's class is named CNNVehicleClassifier; alias it to
# CNNDetector, the name the rest of this file (and its docstrings)
# already use for the CNN verification/legacy-primary path.
from backend.detection.cnn_backend import CNNVehicleClassifier as CNNDetector


class ModelManager:
    """Manages model lifecycle, discovery, and loading."""

    def __init__(self):
        self.loaded_model = None
        self.loaded_classifier = None
        self.model_type = None
        self.model_path = None
        self.device = "cpu"
        self.conf_threshold = 0.25
        self.base_dir = Path(__file__).resolve().parent
        self.models = {}

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
        """Discover available trained models: YOLO .pt weights, Faster
        R-CNN .pth checkpoints, and the CNN classifier checkpoint."""
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
                                for pt_file in weights_dir.glob("best.pt"):
                                    key = f"YOLO/{item.name}/{pt_file.stem}"
                                    models[key] = str(pt_file)
                                    print(f"    ✅ Found: {key}")

                            # Look for train subdirectories in version folder
                            for sub_dir in item.iterdir():
                                if sub_dir.is_dir() and (sub_dir.name.startswith('train') or sub_dir.name == 'train'):
                                    weights_dir = sub_dir / "weights"
                                    if weights_dir.exists():
                                        for pt_file in weights_dir.glob("best.pt"):
                                            key = f"YOLO/{item.name}/{sub_dir.name}/{pt_file.stem}"
                                            models[key] = str(pt_file)
                                            print(f"    ✅ Found: {key}")

                            # Look for other subdirectories that might contain weights
                            for sub_dir in item.iterdir():
                                if sub_dir.is_dir() and sub_dir.name not in ['train']:
                                    weights_dir = sub_dir / "weights"
                                    if weights_dir.exists():
                                        for pt_file in weights_dir.glob("best.pt"):
                                            key = f"YOLO/{item.name}/{sub_dir.name}/{pt_file.stem}"
                                            models[key] = str(pt_file)
                                            print(f"    ✅ Found: {key}")

                        # Look for train directories directly
                        elif item.name.startswith('train') or item.name == 'train':
                            weights_dir = item / "weights"
                            if weights_dir.exists():
                                for pt_file in weights_dir.glob("best.pt"):
                                    key = f"YOLO/{item.name}/{pt_file.stem}"
                                    models[key] = str(pt_file)
                                    print(f"  ✅ Found: {key}")

        # YOLO selection is intentionally restricted to best.pt.
        # Use project-relative paths; never depend on another user's PC path.
        yolo_best_candidates = [
            project_root / "model" / "yolo" / "runs",
            project_root / "model" / "runs",
            project_root / "yolo" / "runs",
            project_root / "runs",
        ]

        for root in yolo_best_candidates:
            if root.exists():
                for pt_file in root.glob("**/best.pt"):
                    key = f"YOLO/{pt_file.parent.parent.name}/{pt_file.parent.name}/best"
                    if key not in models:
                        models[key] = str(pt_file)
                        print(f"  ✅ Found best.pt: {key}")

        # Faster R-CNN checkpoint(s) - the second primary detector,
        # trained via train_faster_rcnn.py on the same split as YOLO.
        # Keys are prefixed "FasterRCNN/" so the sidebar can filter by
        # backend, mirroring the "YOLO/" prefix above.
        frcnn_search_paths = [
            project_root / "model" / "faster_rcnn" / "saved_model",
            current_dir.parent.parent / "model" / "faster_rcnn" / "saved_model",
        ]
        for frcnn_dir in frcnn_search_paths:
            if frcnn_dir.exists():
                for pth_file in frcnn_dir.glob("*.pth"):
                    key = f"FasterRCNN/{pth_file.stem}"
                    if key not in models:
                        models[key] = str(pth_file)
                        print(f"✅ Found: {key}")
                break

        # CNN checkpoint(s) - kept discoverable, but no longer selectable
        # as a primary model_type in the sidebar. It's surfaced separately
        # as an optional verification classifier (see load_classifier()).
        # Keys are prefixed "CNN/".
        cnn_search_paths = [
            project_root / "model" / "cnn" / "saved_model",
            current_dir.parent.parent / "model" / "cnn" / "saved_model",
        ]
        for cnn_dir in cnn_search_paths:
            if cnn_dir.exists():
                for pth_file in cnn_dir.glob("*.pth"):
                    key = f"CNN/{pth_file.stem}"
                    if key not in models:
                        models[key] = str(pth_file)
                        print(f"✅ Found: {key}")
                break

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
        Load a PRIMARY detector based on type and path.

        Args:
            model_type: 'YOLO' or 'Faster R-CNN' (also accepts 'FasterRCNN'/'FRCNN')
            model_path: Path to model weights
            device: 'cpu' or 'cuda'
            conf_threshold: Confidence threshold

        Returns:
            Detector instance or None if failed
        """
        self.model_type = model_type
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold

        normalized_type = (model_type or "").upper().replace(" ", "").replace("-", "").replace("_", "")

        if normalized_type == "YOLO":
            detector = YOLODetector(
                model_path=model_path,
                device=device,
                conf_threshold=conf_threshold,
                enable_tracking=True,
                max_lost_frames=15,
            )
        elif normalized_type in ("FASTERRCNN", "FRCNN"):
            detector = FasterRCNNDetector(
                model_path=model_path,
                device=device,
                conf_threshold=conf_threshold,
                enable_tracking=True,
                max_lost_frames=15,
            )
        elif normalized_type == "CNN":
            # Kept for backward compatibility only; the UI no longer
            # offers CNN as a primary model_type (use load_classifier()).
            detector = CNNDetector(
                model_path=model_path,
                device=device,
                conf_threshold=conf_threshold,
                enable_tracking=True,
                max_lost_frames=15,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.loaded_model = detector
        return detector

    def load_classifier(self, model_path: Optional[str], device: str = "cpu",
                         conf_threshold: float = 0.70) -> Optional[CNNDetector]:
        """Load the CNN as an auxiliary crop-classifier used to verify
        boxes found by the primary detector (see CNNDetector.classify_crop).
        This is independent of load_model()/self.loaded_model."""
        if not model_path or not os.path.exists(model_path):
            print(f"⚠️ CNN classifier checkpoint not found: {model_path}")
            return None

        try:
            classifier = CNNDetector(
                model_path=model_path,
                device=device,
                conf_threshold=conf_threshold,
                enable_tracking=False,
            )
            self.loaded_classifier = classifier
            return classifier
        except Exception as e:
            print(f"Error loading CNN classifier: {e}")
            self.loaded_classifier = None
            return None

    def unload_classifier(self):
        """Unload the CNN verification classifier, if one is loaded."""
        if self.loaded_classifier is not None:
            try:
                if hasattr(self.loaded_classifier, "unload"):
                    self.loaded_classifier.unload()
            except Exception as e:
                print(f"Error unloading classifier: {e}")
            self.loaded_classifier = None

    def unload_model(self):
        """Unload the currently loaded primary detector AND any loaded
        verification classifier, freeing all GPU memory in one call."""
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

        self.unload_classifier()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()