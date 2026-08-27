"""
YOLO-based vehicle detector with tracking integration.
Optimized for performance with proper GPU memory management.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any

import torch
import yaml
import numpy as np

from backend.tracking.tracker import VehicleTracker


class YOLODetector:
    """YOLO-based vehicle detection using Ultralytics YOLO."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        enable_tracking: bool = True,
        max_lost_frames: int = 15
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = None
        self.is_dummy = False
        self.enable_tracking = enable_tracking

        # Initialize tracker
        self.tracker = VehicleTracker(max_lost_frames=max_lost_frames)

        # Load configuration
        self.dataset_config = self._load_dataset_config()

        # Class mappings
        self.vehicle_classes = self.dataset_config.get(
            "names",
            {0: "car", 1: "truck", 2: "bus", 3: "motorcycle"}
        )

        # dataset.yaml stores colors as [R, G, B]. Convert once here to
        # (B, G, R) tuples so every consumer (cv2 drawing code) gets
        # ready-to-use colors, regardless of whether the config loaded
        # from disk or fell back to the defaults below.
        raw_colors = self.dataset_config.get(
            "colors",
            {0: [255, 0, 0], 1: [0, 255, 0], 2: [0, 0, 255], 3: [255, 255, 0]}
        )
        self.class_colors = {
            int(class_id): self._rgb_to_bgr(rgb)
            for class_id, rgb in raw_colors.items()
        }

        self.vehicle_class_ids = list(self.vehicle_classes.keys())

        # Load model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"⚠️ Model not found: {model_path}. Using dummy detector.")
            self.is_dummy = True

    @staticmethod
    def _rgb_to_bgr(rgb) -> tuple:
        """Convert an [R, G, B] list/tuple (as stored in dataset.yaml) to a
        (B, G, R) tuple for use with cv2 drawing functions."""
        r, g, b = rgb
        return (b, g, r)

    def _load_dataset_config(self) -> Dict[str, Any]:
        """Load dataset configuration."""
        possible_paths = [
            "config/dataset.yaml",
            "model/yolo/dataset.yaml",
            "dataset.yaml",
            Path(__file__).parent.parent.parent / "config" / "dataset.yaml",
            Path(__file__).parent.parent.parent / "model" / "yolo" / "dataset.yaml",
        ]

        for yaml_path in possible_paths:
            if os.path.exists(yaml_path):
                try:
                    with open(yaml_path, "r") as f:
                        return yaml.safe_load(f)
                except Exception as e:
                    print(f"Error loading {yaml_path}: {e}")

        # Default configuration (kept in sync with dataset.yaml: 4 classes,
        # no bicycle, colors stored as [R, G, B])
        return {
            "nc": 4,
            "names": {0: "car", 1: "truck", 2: "bus", 3: "motorcycle"},
            "colors": {
                0: [255, 0, 0],
                1: [0, 255, 0],
                2: [0, 0, 255],
                3: [255, 255, 0],
            },
        }

    def load_model(self, model_path: str):
        """Load YOLO model from path."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)

            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model.to("cuda")

            print(f"✅ YOLO model loaded from {model_path} on {self.device}")
            self.is_dummy = False

        except ImportError:
            print("⚠️ Ultralytics not installed. Using dummy mode.")
            self.is_dummy = True
        except Exception as e:
            print(f"⚠️ Error loading model: {e}. Using dummy mode.")
            self.is_dummy = True

    def unload(self):
        """Release model resources."""
        if self.model is not None:
            try:
                # Move to CPU before deletion
                if hasattr(self.model, "model"):
                    underlying = self.model.model
                    if underlying is not None and hasattr(underlying, "to"):
                        underlying.to("cpu")
            except Exception as e:
                print(f"Error moving model to CPU: {e}")

            self.model = None

        self.is_dummy = True

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def detect(self, frame: np.ndarray, enable_tracking: Optional[bool] = None) -> List[Dict]:
        """Detect vehicles in frame."""
        if self.is_dummy or self.model is None:
            return self._dummy_detect(frame)

        use_tracking = enable_tracking if enable_tracking is not None else self.enable_tracking

        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.conf_threshold,
                verbose=False,
                stream=False
            )

            detections = []

            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())

                        if class_id in self.vehicle_class_ids:
                            detections.append({
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "confidence": confidence,
                                "class": class_id,
                                "class_name": self.get_class_name(class_id),
                            })

            # Apply tracking
            if use_tracking and detections:
                detections = self.tracker.update(frame, detections)

            return detections

        except Exception as e:
            print(f"Error during detection: {e}")
            return self._dummy_detect(frame)

    def detect_frame(self, frame: np.ndarray, enable_tracking: Optional[bool] = None) -> List[Dict]:
        """Alias for detect method."""
        return self.detect(frame, enable_tracking)

    def get_class_name(self, class_id: int) -> str:
        """Get class name by ID."""
        return self.vehicle_classes.get(class_id, f"class_{class_id}")

    def get_color_for_class(self, class_id: int) -> tuple:
        """Get (B, G, R) color for class ID, ready for cv2 drawing."""
        return self.class_colors.get(class_id, (0, 255, 0))

    def reset_tracker(self):
        """Reset the tracker."""
        self.tracker.reset()

    def draw_trails(self, frame: np.ndarray, trail_length: int = 20) -> np.ndarray:
        """Draw tracking trails on frame."""
        return self.tracker.draw_trails(frame, trail_length)

    def _dummy_detect(self, frame: np.ndarray) -> List[Dict]:
        """Generate dummy detections for testing."""
        import random

        h, w = frame.shape[:2]
        num_detections = random.randint(2, 5)
        detections = []

        for i in range(num_detections):
            x = random.randint(50, max(51, w - 150))
            y = random.randint(50, max(51, h - 150))
            width = random.randint(60, 150)
            height = random.randint(60, 150)
            class_id = random.randint(0, len(self.vehicle_classes) - 1)

            detections.append({
                "bbox": [x, y, x + width, y + height],
                "confidence": random.uniform(0.5, 0.95),
                "class": class_id,
                "class_name": self.get_class_name(class_id),
                "track_id": i,
                "speed": random.uniform(20, 80),
            })

        if self.enable_tracking:
            detections = self.tracker.update(frame, detections)

        return detections