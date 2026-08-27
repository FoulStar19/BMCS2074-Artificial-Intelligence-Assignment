"""
CNN Vehicle Classifier

Loads the trained MobileNetV2 CNN and classifies
a cropped vehicle image.

Classes:
    car
    truck
    bus
    motorcycle
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from cnn_model import TrafficCNN


class CNNVehicleClassifier:

    def __init__(
        self,
        checkpoint_path=None,
        device=None,
        confidence_threshold=0.0,
    ):

        cnn_dir = (
            Path(__file__)
            .resolve()
            .parent
        )

        # ----------------------------------------------------
        # Model checkpoint
        # ----------------------------------------------------

        self.checkpoint_path = Path(
            checkpoint_path
            or (
                cnn_dir
                / "saved_model"
                / "best_traffic_cnn.pth"
            )
        )

        if not self.checkpoint_path.exists():

            raise FileNotFoundError(
                f"CNN model not found:\n"
                f"{self.checkpoint_path}\n\n"
                "Train the CNN first."
            )

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.confidence_threshold = (
            confidence_threshold
        )

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device
        )

        self.class_names = checkpoint[
            "class_names"
        ]

        self.image_size = checkpoint.get(
            "image_size",
            224
        )

        self.mean = checkpoint.get(
            "mean",
            [0.485, 0.456, 0.406]
        )

        self.std = checkpoint.get(
            "std",
            [0.229, 0.224, 0.225]
        )

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        self.model = TrafficCNN(
            num_classes=len(
                self.class_names
            ),
            pretrained=False
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.eval()

        # ----------------------------------------------------
        # Image preprocessing
        # ----------------------------------------------------

        self.transform = transforms.Compose([

            transforms.Resize(
                (
                    self.image_size,
                    self.image_size
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=self.mean,
                std=self.std
            ),
        ])

    # ========================================================
    # CLASSIFY PIL IMAGE
    # ========================================================

    def classify_pil(self, image):

        if image is None:

            return None

        image = image.convert(
            "RGB"
        )

        input_tensor = (
            self.transform(image)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():

            outputs = self.model(
                input_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[0]

        confidence, index = (
            probabilities.max(
                dim=0
            )
        )

        confidence_value = (
            float(
                confidence.item()
            )
        )

        class_name = (
            self.class_names[
                index.item()
            ]
        )

        if (
            confidence_value
            < self.confidence_threshold
        ):

            return {
                "class": "unknown",
                "confidence":
                    confidence_value,
            }

        return {
            "class": class_name,
            "confidence":
                confidence_value,
        }

    # ========================================================
    # CLASSIFY OPENCV CROP
    # ========================================================

    def classify_crop(self, crop):

        if crop is None:

            return None

        if not isinstance(
            crop,
            np.ndarray
        ):

            raise TypeError(
                "crop must be a "
                "NumPy array."
            )

        if crop.size == 0:

            return None

        # OpenCV uses BGR.
        # PIL expects RGB.

        if len(crop.shape) == 3:

            crop_rgb = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            )

        else:

            crop_rgb = cv2.cvtColor(
                crop,
                cv2.COLOR_GRAY2RGB
            )

        image = Image.fromarray(
            crop_rgb
        )

        return self.classify_pil(
            image
        )

    # ========================================================
    # CLASSIFY YOLO BOUNDING BOX
    # ========================================================

    def classify_bbox(
        self,
        frame,
        bbox
    ):

        if frame is None:

            return None

        if len(bbox) != 4:

            raise ValueError(
                "Bounding box must contain "
                "[x1, y1, x2, y2]."
            )

        height, width = (
            frame.shape[:2]
        )

        x1, y1, x2, y2 = map(
            int,
            bbox
        )

        # Keep coordinates inside frame.

        x1 = max(
            0,
            min(x1, width - 1)
        )

        y1 = max(
            0,
            min(y1, height - 1)
        )

        x2 = max(
            0,
            min(x2, width)
        )

        y2 = max(
            0,
            min(y2, height)
        )

        if x2 <= x1 or y2 <= y1:

            return None

        crop = frame[
            y1:y2,
            x1:x2
        ]

        return self.classify_crop(
            crop
        )