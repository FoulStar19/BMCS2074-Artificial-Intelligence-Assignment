"""Evaluate the trained 4-class CNN on the validation crop dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn_model import TrafficCNN


# ============================================================
# MAIN
# ============================================================

def main():

    cnn_dir = Path(
        __file__
    ).resolve().parent

    # --------------------------------------------------------
    # Model checkpoint
    # --------------------------------------------------------

    checkpoint_path = (
        cnn_dir
        / "saved_model"
        / "best_traffic_cnn.pth"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"\nTrained CNN model not found:\n"
            f"{checkpoint_path}\n\n"
            "Train the CNN first using train_cnn.py."
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

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

    print()
    print(
        f"Classes: {class_names}"
    )

    print(
        f"Number of classes: {num_classes}"
    )

    # --------------------------------------------------------
    # Validation transform
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    dataset_dir = (
        cnn_dir.parents[1]
        / "dataset"
        / "cnn_4class"
        / "val"
    )

    if not dataset_dir.exists():

        raise FileNotFoundError(
            f"\nValidation dataset not found:\n"
            f"{dataset_dir}\n\n"
            "Run prepare_cnn_dataset.py first."
        )

    dataset = datasets.ImageFolder(
        dataset_dir,
        transform=transform
    )

    # --------------------------------------------------------
    # Check class mapping
    # --------------------------------------------------------

    print()
    print(
        f"Dataset classes: "
        f"{dataset.classes}"
    )

    print(
        f"Checkpoint classes: "
        f"{class_names}"
    )

    if dataset.classes != class_names:

        raise ValueError(
            "\nValidation dataset classes do not "
            "match checkpoint classes.\n"
            f"Dataset: {dataset.classes}\n"
            f"Checkpoint: {class_names}"
        )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    labels = []
    predictions = []

    with torch.no_grad():

        for images, targets in loader:

            images = images.to(device)

            outputs = model(
                images
            )

            predicted = outputs.argmax(
                dim=1
            )

            labels.extend(
                targets.tolist()
            )

            predictions.extend(
                predicted.cpu().tolist()
            )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    if len(labels) == 0:

        raise ValueError(
            "Validation dataset contains no samples."
        )

    correct = sum(
        actual == predicted
        for actual, predicted
        in zip(
            labels,
            predictions
        )
    )

    accuracy = (
        correct
        / len(labels)
    )

    print()
    print("=" * 70)
    print("CNN VALIDATION RESULTS")
    print("=" * 70)

    print()
    print(
        f"Validation samples: "
        f"{len(labels)}"
    )

    print(
        f"Correct predictions: "
        f"{correct}"
    )

    print(
        f"Validation accuracy: "
        f"{accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    report = classification_report(
        labels,
        predictions,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0
    )

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=range(len(class_names))
    )

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(matrix)

    # --------------------------------------------------------
    # Display confusion matrix
    # --------------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names
    )

    display.plot(
        xticks_rotation=45
    )

    plt.title(
        "CNN Vehicle Classification Confusion Matrix"
    )

    plt.tight_layout()

    output_path = (
        cnn_dir
        / "saved_model"
        / "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=160
    )

    plt.close()

    print()
    print(
        f"Saved confusion matrix:\n"
        f"{output_path}"
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()