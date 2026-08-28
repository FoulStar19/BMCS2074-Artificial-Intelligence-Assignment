"""
CNN vehicle-classification backend.

Consolidates what used to be three separate runtime modules:
  - cnn_model.py                (TrafficCNN architecture)
  - cnn_classifier.py           (crop verification classifier)
  - sliding_window_detector.py  (full-frame car detector)

All three wrapped the same MobileNetV2 checkpoint, the same
Resize/ToTensor/Normalize preprocessing, and the same checkpoint-loading
code, so keeping them in separate files just duplicated that logic
three times. This module keeps one copy of the shared pieces and
exposes two runtime classes:

  CNNVehicleClassifier    - classifies an already-cropped vehicle image.
                            Used as the optional verification pass on
                            top of YOLO / Faster R-CNN
                            (see ModelManager.load_classifier).

  SlidingWindowCarDetector - runs the same CNN as a full-frame car
                            detector via an image pyramid + sliding
                            window + NMS. Used by the "CNN Scanner" tab
                            for single images -- NOT wired into video
                            processing (see the class docstring below
                            for why).

Target location: backend/detection/cnn_backend.py
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class TrafficCNN(nn.Module):
    """MobileNetV2 transfer-learning classifier for car/truck/bus/motorcycle."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.model = models.mobilenet_v2(weights=weights)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# SHARED CHECKPOINT / PREPROCESSING HELPERS
# ============================================================

_DEFAULT_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_STD = [0.229, 0.224, 0.225]

# Same search order ModelManager.discover_models() uses for "CNN/" keys,
# so a checkpoint found by the sidebar resolves the same way here.
_CNN_CHECKPOINT_CANDIDATES = [
    Path("model/cnn/saved_model/best_traffic_cnn.pth"),
    Path("config/cnn/best_traffic_cnn.pth"),
    Path(__file__).resolve().parent.parent.parent / "model" / "cnn" / "saved_model" / "best_traffic_cnn.pth",
]


def _resolve_checkpoint_path(checkpoint_path: Optional[str]) -> Path:
    if checkpoint_path:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"CNN checkpoint not found:\n{path}")
        return path

    for candidate in _CNN_CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "CNN checkpoint not found. Pass checkpoint_path explicitly, or "
        "train one first with `python cnn_pipeline.py train`."
    )


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint.setdefault("mean", _DEFAULT_MEAN)
    checkpoint.setdefault("std", _DEFAULT_STD)
    checkpoint.setdefault("image_size", 224)
    return checkpoint


def _build_eval_transform(image_size: int, mean, std) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _build_model_from_checkpoint(checkpoint: Dict[str, Any], device: torch.device) -> TrafficCNN:
    class_names = checkpoint["class_names"]
    model = TrafficCNN(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def box_iou(box_a, box_b) -> float:
    """Plain-Python IoU for two [x1, y1, x2, y2] boxes. Used by this
    module's NMS and by cnn_pipeline.py's evaluate-detector command."""
    left, top = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    right, bottom = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


# ============================================================
# CROP CLASSIFIER (verification pass over YOLO / Faster R-CNN boxes)
# ============================================================

class CNNVehicleClassifier:
    """Classifies an already-cropped vehicle image into
    car / truck / bus / motorcycle.

    Accepts both `checkpoint_path` and the `model_path` alias -- the
    name model_manager.py's load_classifier() actually calls with --
    plus a `conf_threshold` alias, so it's a drop-in for the old
    `CNNDetector` name that model_manager.py referenced but never
    imported.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.0,
        conf_threshold: Optional[float] = None,
        **_ignored: Any,
    ):
        self.checkpoint_path = _resolve_checkpoint_path(checkpoint_path or model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.confidence_threshold = conf_threshold if conf_threshold is not None else confidence_threshold

        checkpoint = _load_checkpoint(self.checkpoint_path, self.device)
        self.class_names = checkpoint["class_names"]
        self.image_size = checkpoint["image_size"]
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]

        self.model = _build_model_from_checkpoint(checkpoint, self.device)
        self.transform = _build_eval_transform(self.image_size, self.mean, self.std)

    def classify_pil(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        if image is None:
            return None

        input_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probabilities = torch.softmax(self.model(input_tensor), dim=1)[0]

        confidence, index = probabilities.max(dim=0)
        confidence_value = float(confidence.item())
        class_name = self.class_names[index.item()]

        if confidence_value < self.confidence_threshold:
            return {"class": "unknown", "confidence": confidence_value}
        return {"class": class_name, "confidence": confidence_value}

    def classify_crop(self, crop: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Classify a BGR (OpenCV-style) crop."""
        if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
            return None

        if crop.ndim == 3:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

        return self.classify_pil(Image.fromarray(crop_rgb))

    def classify_bbox(self, frame: np.ndarray, bbox: List[int]) -> Optional[Dict[str, Any]]:
        """Classify the region of `frame` inside `bbox` = [x1, y1, x2, y2]."""
        if frame is None or len(bbox) != 4:
            return None

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None

        return self.classify_crop(frame[y1:y2, x1:x2])

    def unload(self):
        """Matches the unload() interface YOLODetector / FasterRCNNDetector expose."""
        if self.model is not None:
            self.model.to("cpu")
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# FULL-FRAME CAR DETECTOR (image pyramid + sliding window + NMS)
# ============================================================

class SlidingWindowCarDetector:
    """Detects cars in a full frame by classifying a pyramid of sliding
    windows with the same CNN and running NMS on the "car" class score.

    This is intentionally NOT wired into the video pipeline: at six
    window sizes with 1/3-size strides, a single 640px-wide frame can
    mean hundreds of forward passes. That's much too slow for
    real-time video -- that's what YOLO / Faster R-CNN are for. It's
    exposed as a single-image tool (see the "CNN Scanner" tab in the
    app) for quick spot-checks when you don't have a bounding-box
    detector handy.
    """

    WINDOW_SIZES = (48, 64, 96, 128, 160, 192)

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.70,
        nms_iou_threshold: float = 0.40,
        max_frame_width: int = 640,
        batch_size: int = 64,
    ):
        self.checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_frame_width = max_frame_width
        self.batch_size = batch_size

        checkpoint = _load_checkpoint(self.checkpoint_path, self.device)
        self.class_names = checkpoint["class_names"]
        self.car_index = self.class_names.index("car")

        self.model = _build_model_from_checkpoint(checkpoint, self.device)
        self.transform = _build_eval_transform(checkpoint["image_size"], checkpoint["mean"], checkpoint["std"])

    def _windows(self, image: np.ndarray):
        height, width = image.shape[:2]
        for size in self.WINDOW_SIZES:
            if size > width or size > height:
                continue
            stride = max(16, size // 3)
            for y in range(0, height - size + 1, stride):
                for x in range(0, width - size + 1, stride):
                    yield x, y, x + size, y + size

    def _nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ordered = sorted(detections, key=lambda item: item["confidence"], reverse=True)
        kept: List[Dict[str, Any]] = []
        for detection in ordered:
            if all(box_iou(detection["bbox"], chosen["bbox"]) <= self.nms_iou_threshold for chosen in kept):
                kept.append(detection)
        return kept

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Return car detections as {"bbox", "confidence", "class"} dicts."""
        if frame is None or frame.size == 0:
            return []

        original_height, original_width = frame.shape[:2]
        scale = min(1.0, self.max_frame_width / original_width)
        image = (
            cv2.resize(frame, (round(original_width * scale), round(original_height * scale)))
            if scale < 1.0 else frame
        )
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = list(self._windows(image_rgb))
        detections: List[Dict[str, Any]] = []

        for start in range(0, len(boxes), self.batch_size):
            batch_boxes = boxes[start:start + self.batch_size]
            batch = torch.stack([
                self.transform(Image.fromarray(image_rgb[y1:y2, x1:x2]))
                for x1, y1, x2, y2 in batch_boxes
            ]).to(self.device)

            with torch.no_grad():
                scores = torch.softmax(self.model(batch), dim=1)[:, self.car_index].cpu().numpy()

            for box, score in zip(batch_boxes, scores):
                if score >= self.confidence_threshold:
                    x1, y1, x2, y2 = box
                    detections.append({
                        "bbox": [round(x1 / scale), round(y1 / scale), round(x2 / scale), round(y2 / scale)],
                        "confidence": float(score),
                        "class": "car",
                    })

        return self._nms(detections)

    def unload(self):
        if self.model is not None:
            self.model.to("cpu")
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()