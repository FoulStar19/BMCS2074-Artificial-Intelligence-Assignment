"""Application adapter for the trained CNN sliding-window car detector.

Wraps SlidingWindowCarDetector so it matches YOLODetector's interface
closely enough for ModelManager / VideoProcessingService to use either
detector interchangeably.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from backend.tracking.tracker import VehicleTracker
from model.cnn.sliding_window_detector import SlidingWindowCarDetector


class CNNDetector:
    """Expose the CNN sliding-window detector through the YOLODetector API."""

    # The CNN is trained as a binary car/background classifier. Its single
    # "car" output is mapped onto class id 0 so it lines up with
    # dataset.yaml (0: car) and with the integer class ids YOLODetector
    # produces -- everything downstream (class_colors, class_names,
    # class_id filtering) keys off integers, not the string "car".
    CAR_CLASS_ID = 0

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        conf_threshold: float = 0.70,
        enable_tracking: bool = True,
        max_lost_frames: int = 15,
    ):
        project_root = Path(__file__).resolve().parents[2]
        checkpoint = Path(model_path) if model_path else (
            project_root / "model" / "cnn" / "saved_model" / "best_car_detector.pth"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"CNN checkpoint not found: {checkpoint}. Run model/cnn/train_cnn.py first."
            )

        selected_device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.device = selected_device
        self.is_dummy = False
        self.enable_tracking = enable_tracking

        # Same tracker YOLODetector uses, so both backends produce
        # detections with persistent track_id values that
        # VideoProcessingService can interpolate and estimate speed from.
        self.tracker = VehicleTracker(max_lost_frames=max_lost_frames)

        self.detector = SlidingWindowCarDetector(
            checkpoint_path=checkpoint,
            device=selected_device,
            confidence_threshold=conf_threshold,
        )

    @property
    def conf_threshold(self) -> float:
        """Mirror the sliding-window detector's threshold under the name
        VideoProcessingService looks for (``hasattr(detector, 'conf_threshold')``).
        Without this, the confidence slider in the UI silently had no
        effect on the CNN backend."""
        return self.detector.confidence_threshold

    @conf_threshold.setter
    def conf_threshold(self, value: float):
        self.detector.confidence_threshold = value

    def detect(self, frame: np.ndarray, enable_tracking: Optional[bool] = None) -> List[Dict]:
        """Detect cars in frame, with class ids and tracking matching YOLODetector."""
        raw_detections = self.detector.detect(frame)

        detections = [
            {
                "bbox": item["bbox"],
                "confidence": item["confidence"],
                "class": self.CAR_CLASS_ID,
                "class_name": "car",
            }
            for item in raw_detections
        ]

        use_tracking = enable_tracking if enable_tracking is not None else self.enable_tracking
        if use_tracking and detections:
            detections = self.tracker.update(frame, detections)

        return detections

    def detect_frame(self, frame: np.ndarray, enable_tracking: Optional[bool] = None) -> List[Dict]:
        """Alias for detect, matching YOLODetector's interface."""
        return self.detect(frame, enable_tracking)

    def get_class_name(self, class_id: int) -> str:
        return "car" if class_id == self.CAR_CLASS_ID else f"class_{class_id}"

    def reset_tracker(self):
        self.tracker.reset()

    def draw_trails(self, frame: np.ndarray, trail_length: int = 20) -> np.ndarray:
        return self.tracker.draw_trails(frame, trail_length)

    def unload(self):
        """Release model resources. Reaches the real nn.Module nested at
        self.detector.model -- ModelManager.unload_model() otherwise has
        no way to find it and would leak GPU memory on model switches."""
        if getattr(self, "detector", None) is not None:
            model = getattr(self.detector, "model", None)
            if model is not None and hasattr(model, "to"):
                model.to("cpu")
            self.detector = None

        self.is_dummy = True

        if torch.cuda.is_available():
            torch.cuda.empty_cache()