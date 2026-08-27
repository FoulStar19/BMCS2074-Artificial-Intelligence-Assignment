"""Predict a vehicle class for one cropped vehicle image.

Classes:
    car
    truck
    bus
    motorcycle
"""

import argparse
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

from cnn_model import TrafficCNN


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Predict the vehicle class "
            "for one cropped image."
        )
    )

    parser.add_argument(
        "image",
        help="Path to one cropped vehicle image"
    )

    args = parser.parse_args()

    # ========================================================
    # PATHS
    # ========================================================

    cnn_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    checkpoint_path = (
        cnn_dir
        / "saved_model"
        / "best_traffic_cnn.pth"
    )

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"\nCNN model not found:\n"
            f"{checkpoint_path}\n\n"
            "Train the CNN first using "
            "train_cnn.py."
        )

    # ========================================================
    # CHECK INPUT IMAGE
    # ========================================================

    image_path = Path(
        args.image
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"\nInput image not found:\n"
            f"{image_path}"
        )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    class_names = checkpoint[
        "class_names"
    ]

    image_size = checkpoint.get(
        "image_size",
        224
    )

    mean = checkpoint.get(
        "mean",
        [0.485, 0.456, 0.406]
    )

    std = checkpoint.get(
        "std",
        [0.229, 0.224, 0.225]
    )

    num_classes = checkpoint.get(
        "num_classes",
        len(class_names)
    )

    print(
        f"Classes: {class_names}"
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = TrafficCNN(
        num_classes=num_classes,
        pretrained=False
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # ========================================================
    # IMAGE TRANSFORMATION
    # ========================================================

    transform = transforms.Compose([

        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=mean,
            std=std
        ),
    ])

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    # ========================================================
    # PREDICTION
    # ========================================================

    with torch.no_grad():

        outputs = model(
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

    predicted_class = (
        class_names[
            index.item()
        ]
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print("=" * 60)
    print("CNN PREDICTION")
    print("=" * 60)

    print(
        f"Image: {image_path.name}"
    )

    print(
        f"Prediction: {predicted_class}"
    )

    print(
        f"Confidence: "
        f"{confidence.item() * 100:.2f}%"
    )

    print()
    print("Class probabilities:")

    for class_index, class_name in enumerate(
        class_names
    ):

        probability = (
            probabilities[
                class_index
            ].item()
            * 100
        )

        print(
            f"  {class_name:<12}: "
            f"{probability:.2f}%"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()