"""Application adapter for the trained CNN sliding-window car detector."""

from pathlib import Path

import torch

from model.cnn.sliding_window_detector import SlidingWindowCarDetector


class CNNDetector:
    """Expose the CNN detector through the same ``detect`` API as YOLO."""

    def __init__(self, model_path=None, device="cpu", confidence_threshold=0.70):
        project_root = Path(__file__).resolve().parents[2]
        checkpoint = Path(model_path) if model_path else (
            project_root / "model" / "cnn" / "saved_model" / "best_car_detector.pth"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"CNN checkpoint not found: {checkpoint}. Run model/cnn/train_cnn.py first."
            )
        selected_device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.detector = SlidingWindowCarDetector(
            checkpoint_path=checkpoint,
            device=selected_device,
            confidence_threshold=confidence_threshold,
        )

    def detect(self, frame):
        return self.detector.detect(frame)
