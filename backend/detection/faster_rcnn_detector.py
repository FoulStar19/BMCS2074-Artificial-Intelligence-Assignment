"""
Faster R-CNN based vehicle detector with tracking integration.

This is now the second primary detection backend alongside YOLODetector
(replacing CNNDetector in that role), trained on the exact same
train/validation split via train_faster_rcnn.py, so the two backends can
be benchmarked head-to-head. It exposes the same interface YOLODetector
does so ModelManager / VideoProcessingService can use either
interchangeably.

Target location: backend/detection/faster_rcnn_detector.py
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
import torch
import yaml

from backend.tracking.tracker import VehicleTracker


class FasterRCNNDetector:
    """Torchvision Faster R-CNN (ResNet-50 FPN) vehicle detector."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        conf_threshold: float = 0.5,
        enable_tracking: bool = True,
        max_lost_frames: int = 15,
    ):
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.conf_threshold = conf_threshold
        self.model = None
        self.is_dummy = False
        self.enable_tracking = enable_tracking

        # Same tracker YOLODetector/CNNDetector use, so all backends
        # produce detections with persistent track_id values.
        self.tracker = VehicleTracker(max_lost_frames=max_lost_frames)

        self.dataset_config = self._load_dataset_config()
        self.vehicle_classes = self.dataset_config.get(
            "names", {0: "car", 1: "truck", 2: "bus", 3: "motorcycle"}
        )

        raw_colors = self.dataset_config.get(
            "colors", {0: [255, 0, 0], 1: [0, 255, 0], 2: [0, 0, 255], 3: [255, 255, 0]}
        )
        self.class_colors = {
            int(class_id): self._rgb_to_bgr(rgb) for class_id, rgb in raw_colors.items()
        }
        self.vehicle_class_ids = list(self.vehicle_classes.keys())

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"⚠️ Faster R-CNN checkpoint not found: {model_path}. Using dummy detector.")
            self.is_dummy = True

    @staticmethod
    def _rgb_to_bgr(rgb) -> tuple:
        r, g, b = rgb
        return (b, g, r)

    def _load_dataset_config(self) -> Dict[str, Any]:
        """Load dataset configuration (same search order as YOLODetector,
        so both backends see identical class ids/names/colors)."""
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

        return {
            "nc": 4,
            "names": {0: "car", 1: "truck", 2: "bus", 3: "motorcycle"},
            "colors": {0: [255, 0, 0], 1: [0, 255, 0], 2: [0, 0, 255], 3: [255, 255, 0]},
        }

    def load_model(self, model_path: str):
        """Load a checkpoint produced by train_faster_rcnn.py."""
        try:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            class_names = checkpoint.get("class_names", list(self.vehicle_classes.values()))
            num_classes = checkpoint.get("num_classes", len(class_names) + 1)

            model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()

            self.model = model
            self.checkpoint_class_names = class_names
            self.is_dummy = False

            print(f"✅ Faster R-CNN model loaded from {model_path} on {self.device}")

        except Exception as e:
            print(f"⚠️ Error loading Faster R-CNN model: {e}. Using dummy mode.")
            self.is_dummy = True

    def unload(self):
        """Release model resources."""
        if self.model is not None:
            try:
                self.model.to("cpu")
            except Exception as e:
                print(f"Error moving model to CPU: {e}")
            self.model = None

        self.is_dummy = True

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def detect(self, frame: np.ndarray, enable_tracking: Optional[bool] = None) -> List[Dict]:
        """Detect vehicles in a BGR (OpenCV-style) frame."""
        if self.is_dummy or self.model is None:
            return self._dummy_detect(frame)

        use_tracking = enable_tracking if enable_tracking is not None else self.enable_tracking

        try:
            from torchvision.transforms import functional as TF

            # BGR (cv2) -> RGB, HWC uint8 -> CHW float tensor in [0, 1]
            rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
            tensor = TF.to_tensor(rgb_frame).to(self.device)

            with torch.no_grad():
                output = self.model([tensor])[0]

            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()

            detections = []
            for box, label, score in zip(boxes, labels, scores):
                if score < self.conf_threshold:
                    continue

                # Faster R-CNN labels are 1-indexed (0 = background);
                # shift back to the 0-indexed dataset.yaml class ids.
                class_id = int(label) - 1
                if class_id not in self.vehicle_class_ids:
                    continue

                x1, y1, x2, y2 = box.astype(int)
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(score),
                    "class": class_id,
                    "class_name": self.get_class_name(class_id),
                })

            if use_tracking and detections:
                detections = self.tracker.update(frame, detections)

            return detections

        except Exception as e:
            print(f"Error during Faster R-CNN detection: {e}")
            return self._dummy_detect(frame)

    def detect_frame(self, frame: np.ndarray, enable_tracking: Optional[bool] = None) -> List[Dict]:
        """Alias for detect method (matches YOLODetector's interface)."""
        return self.detect(frame, enable_tracking)

    def get_class_name(self, class_id: int) -> str:
        return self.vehicle_classes.get(class_id, f"class_{class_id}")

    def get_color_for_class(self, class_id: int) -> tuple:
        return self.class_colors.get(class_id, (0, 255, 0))

    def reset_tracker(self):
        self.tracker.reset()

    def draw_trails(self, frame: np.ndarray, trail_length: int = 20) -> np.ndarray:
        return self.tracker.draw_trails(frame, trail_length)

    def _dummy_detect(self, frame: np.ndarray) -> List[Dict]:
        """Generate dummy detections for testing (mirrors YOLODetector)."""
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