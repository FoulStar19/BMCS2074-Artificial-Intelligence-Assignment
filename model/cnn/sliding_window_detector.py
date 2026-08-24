"""Sliding-window inference for the binary car/background CNN."""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    from .cnn_model import TrafficCNN
except ImportError:  # Allows both ``python file.py`` and package imports.
    from cnn_model import TrafficCNN


def box_iou(box_a, box_b):
    left, top = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    right, bottom = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = ((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]) +
             (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) - intersection)
    return intersection / union if union else 0.0


class SlidingWindowCarDetector:
    """Detect cars with a binary classifier, image pyramid, and NMS."""

    def __init__(self, checkpoint_path=None, device=None, confidence_threshold=0.70,
                 nms_iou_threshold=0.40, max_frame_width=640, batch_size=64):
        cnn_dir = Path(__file__).resolve().parent
        checkpoint_path = Path(checkpoint_path or cnn_dir / "saved_model" / "best_car_detector.pth")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_frame_width = max_frame_width
        self.batch_size = batch_size
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.class_names = checkpoint["class_names"]
        self.car_index = self.class_names.index("car")
        self.model = TrafficCNN(len(self.class_names), pretrained=False).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        image_size = checkpoint.get("image_size", 224)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), transforms.ToTensor(),
            transforms.Normalize(checkpoint["mean"], checkpoint["std"]),
        ])

    def _windows(self, image):
        height, width = image.shape[:2]
        for size in (48, 64, 96, 128, 160, 192):
            if size > width or size > height:
                continue
            stride = max(16, size // 3)
            for y in range(0, height - size + 1, stride):
                for x in range(0, width - size + 1, stride):
                    yield x, y, x + size, y + size

    def _nms(self, detections):
        ordered = sorted(detections, key=lambda item: item["confidence"], reverse=True)
        kept = []
        for detection in ordered:
            if all(box_iou(detection["bbox"], chosen["bbox"]) <= self.nms_iou_threshold for chosen in kept):
                kept.append(detection)
        return kept

    def detect(self, frame):
        """Return car detections as ``bbox``, ``confidence``, and ``class`` dictionaries."""
        if frame is None or frame.size == 0:
            return []
        original_height, original_width = frame.shape[:2]
        scale = min(1.0, self.max_frame_width / original_width)
        if scale < 1.0:
            image = cv2.resize(frame, (round(original_width * scale), round(original_height * scale)))
        else:
            image = frame
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = list(self._windows(image_rgb))
        detections = []
        for start in range(0, len(boxes), self.batch_size):
            batch_boxes = boxes[start:start + self.batch_size]
            batch = torch.stack([self.transform(Image.fromarray(image_rgb[y1:y2, x1:x2]))
                                 for x1, y1, x2, y2 in batch_boxes]).to(self.device)
            with torch.no_grad():
                scores = torch.softmax(self.model(batch), dim=1)[:, self.car_index].cpu().numpy()
            for box, score in zip(batch_boxes, scores):
                if score >= self.confidence_threshold:
                    x1, y1, x2, y2 = box
                    detections.append({"bbox": [round(x1 / scale), round(y1 / scale), round(x2 / scale), round(y2 / scale)],
                                       "confidence": float(score), "class": "car"})
        return self._nms(detections)
