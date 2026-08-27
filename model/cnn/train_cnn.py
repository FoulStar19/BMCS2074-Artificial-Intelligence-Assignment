"""
Train the 4-class CNN used for vehicle classification.

Classes:
    0 - car
    1 - truck
    2 - bus
    3 - motorcycle

Algorithm:
    MobileNetV2 Transfer Learning

Dataset:
    dataset/cnn_4class/
        train/
            car/
            truck/
            bus/
            motorcycle/
        val/
            car/
            truck/
            bus/
            motorcycle/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn_model import TrafficCNN


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

MEAN = [
    0.485,
    0.456,
    0.406
]

STD = [
    0.229,
    0.224,
    0.225
]

REQUIRED_CLASSES = [
    "bus",
    "car",
    "motorcycle",
    "truck"
]

NUM_CLASSES = 4


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device
):
    """
    Evaluate the CNN on the validation dataset.
    """

    model.eval()

    loss_sum = 0.0
    correct = 0
    samples = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss_sum += (
                loss.item()
                * labels.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            samples += labels.size(0)

    if samples == 0:
        return 0.0, 0.0

    average_loss = loss_sum / samples
    accuracy = correct / samples

    return average_loss, accuracy


# ============================================================
# SAVE TRAINING CURVES
# ============================================================

def save_curves(
    history,
    output_path
):
    """
    Save training and validation
    loss/accuracy curves.
    """

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    axes[0].plot(
        epochs,
        history["train_loss"],
        label="Train"
    )

    axes[0].plot(
        epochs,
        history["val_loss"],
        label="Validation"
    )

    axes[0].set_title(
        "Training and Validation Loss"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].legend()

    axes[0].grid(
        alpha=0.3
    )

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    axes[1].plot(
        epochs,
        history["train_accuracy"],
        label="Train"
    )

    axes[1].plot(
        epochs,
        history["val_accuracy"],
        label="Validation"
    )

    axes[1].set_title(
        "Training and Validation Accuracy"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Accuracy"
    )

    axes[1].set_ylim(
        0,
        1
    )

    axes[1].legend()

    axes[1].grid(
        alpha=0.3
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160
    )

    plt.close(
        figure
    )


# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

def calculate_class_weights(dataset):
    """
    Calculate inverse-frequency class weights.

    This helps reduce the effect of class imbalance,
    especially for the minority bus class.
    """

    targets = torch.tensor(
        dataset.targets,
        dtype=torch.long
    )

    class_counts = torch.bincount(
        targets,
        minlength=NUM_CLASSES
    ).float()

    if torch.any(class_counts == 0):

        raise ValueError(
            "At least one class has zero "
            "training samples."
        )

    total_samples = class_counts.sum()

    class_weights = (
        total_samples
        / (
            NUM_CLASSES
            * class_counts
        )
    )

    return class_counts, class_weights


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train 4-class vehicle "
            "classification CNN."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true"
    )

    args = parser.parse_args()

    # ========================================================
    # PATHS
    # ========================================================

    cnn_dir = Path(
        __file__
    ).resolve().parent

    dataset_dir = (
        cnn_dir.parents[1]
        / "dataset"
        / "cnn_4class"
    )

    output_dir = (
        cnn_dir
        / "saved_model"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_dir = (
        dataset_dir
        / "train"
    )

    val_dir = (
        dataset_dir
        / "val"
    )

    # ========================================================
    # CHECK DATASET
    # ========================================================

    if not train_dir.exists():

        raise FileNotFoundError(
            f"Training dataset not found:\n"
            f"{train_dir}\n\n"
            "Run prepare_cnn_dataset.py first."
        )

    if not val_dir.exists():

        raise FileNotFoundError(
            f"Validation dataset not found:\n"
            f"{val_dir}\n\n"
            "Run prepare_cnn_dataset.py first."
        )

    # ========================================================
    # TRANSFORMS
    # ========================================================

    train_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            10
        ),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            MEAN,
            STD
        ),
    ])

    eval_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            MEAN,
            STD
        ),
    ])

    # ========================================================
    # LOAD DATASET
    # ========================================================

    train_data = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    val_data = datasets.ImageFolder(
        val_dir,
        transform=eval_transform
    )

    # ========================================================
    # CHECK CLASS NAMES
    # ========================================================

    expected_classes = sorted(
        REQUIRED_CLASSES
    )

    actual_train_classes = (
        train_data.classes
    )

    actual_val_classes = (
        val_data.classes
    )

    print()
    print(
        f"Expected classes: "
        f"{expected_classes}"
    )

    print(
        f"Training classes: "
        f"{actual_train_classes}"
    )

    print(
        f"Validation classes: "
        f"{actual_val_classes}"
    )

    if actual_train_classes != expected_classes:

        raise ValueError(
            "\nTraining dataset classes "
            "do not match expected classes.\n"
            f"Expected: {expected_classes}\n"
            f"Found: {actual_train_classes}\n"
            "\nPlease check the CNN dataset "
            "folder structure."
        )

    if actual_val_classes != expected_classes:

        raise ValueError(
            "\nValidation dataset classes "
            "do not match expected classes.\n"
            f"Expected: {expected_classes}\n"
            f"Found: {actual_val_classes}\n"
            "\nPlease check the CNN dataset "
            "folder structure."
        )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"Device: {device}"
    )

    print(
        f"Number of classes: "
        f"{NUM_CLASSES}"
    )

    print(
        f"Training images: "
        f"{len(train_data)}"
    )

    print(
        f"Validation images: "
        f"{len(val_data)}"
    )

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    class_counts, class_weights = (
        calculate_class_weights(
            train_data
        )
    )

    print()
    print(
        "Training class distribution:"
    )

    for class_index, class_name in enumerate(
        train_data.classes
    ):

        print(
            f"  {class_name:<12}: "
            f"{int(class_counts[class_index])}"
        )

    print()
    print(
        "Class weights:"
    )

    for class_index, class_name in enumerate(
        train_data.classes
    ):

        print(
            f"  {class_name:<12}: "
            f"{class_weights[class_index]:.4f}"
        )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = TrafficCNN(
        num_classes=NUM_CLASSES,
        pretrained=(
            not args.no_pretrained
        )
    ).to(device)

    # ========================================================
    # LOSS WITH CLASS WEIGHTS
    # ========================================================

    class_weights = class_weights.to(
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    # ========================================================
    # TRAINING HISTORY
    # ========================================================

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    best_accuracy = -1.0

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("=" * 70)
    print(
        "Starting 4-class CNN training"
    )
    print("=" * 70)

    for epoch in range(
        1,
        args.epochs + 1
    ):

        model.train()

        loss_sum = 0.0
        correct = 0
        samples = 0

        # ----------------------------------------------------
        # TRAINING LOOP
        # ----------------------------------------------------

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            loss_sum += (
                loss.item()
                * labels.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            correct += (
                predictions == labels
            ).sum().item()

            samples += (
                labels.size(0)
            )

        train_loss = (
            loss_sum / samples
        )

        train_accuracy = (
            correct / samples
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "val_loss"
        ].append(
            val_loss
        )

        history[
            "train_accuracy"
        ].append(
            train_accuracy
        )

        history[
            "val_accuracy"
        ].append(
            val_accuracy
        )

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print(
            f"Epoch "
            f"{epoch:02d}/{args.epochs}: "
            f"train loss={train_loss:.4f}, "
            f"train accuracy={train_accuracy:.4f}, "
            f"val loss={val_loss:.4f}, "
            f"val accuracy={val_accuracy:.4f}"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_accuracy > best_accuracy:

            best_accuracy = (
                val_accuracy
            )

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    train_data.classes,

                "image_size":
                    IMAGE_SIZE,

                "mean":
                    MEAN,

                "std":
                    STD,

                "num_classes":
                    NUM_CLASSES,

                "best_val_accuracy":
                    best_accuracy
            }

            model_path = (
                output_dir
                / "best_traffic_cnn.pth"
            )

            torch.save(
                checkpoint,
                model_path
            )

            print(
                f"  ✓ Best model saved: "
                f"{model_path}"
            )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_path = (
        output_dir
        / "history.json"
    )

    history_path.write_text(
        json.dumps(
            history,
            indent=2
        ),
        encoding="utf-8"
    )

    # ========================================================
    # SAVE CURVES
    # ========================================================

    curves_path = (
        output_dir
        / "training_curves.png"
    )

    save_curves(
        history,
        curves_path
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "CNN TRAINING COMPLETED"
    )
    print("=" * 70)

    print(
        f"Best validation accuracy: "
        f"{best_accuracy:.4f}"
    )

    print(
        f"Model saved to:"
        f"\n{output_dir / 'best_traffic_cnn.pth'}"
    )

    print(
        f"\nHistory saved to:"
        f"\n{history_path}"
    )

    print(
        f"\nTraining curves saved to:"
        f"\n{curves_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()