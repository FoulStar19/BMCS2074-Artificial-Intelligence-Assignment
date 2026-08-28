"""
CNN pipeline command-line tool.

Consolidates five previously-separate scripts that all worked on the
same MobileNetV2 vehicle-classifier checkpoint:

    prepare_cnn_dataset.py -> `prepare-dataset`
    train_cnn.py           -> `train`
    evaluate_cnn.py        -> `evaluate`
    predict_cnn.py         -> `predict`
    evaluate_detector.py   -> `evaluate-detector`
    saved_model.py         -> (folded into each subcommand's own
                               output_dir.mkdir(..., exist_ok=True))

Usage:
    python cnn_pipeline.py prepare-dataset
    python cnn_pipeline.py train --epochs 20
    python cnn_pipeline.py evaluate
    python cnn_pipeline.py predict path/to/crop.jpg
    python cnn_pipeline.py evaluate-detector --confidence 0.7

Target location: model/cnn/cnn_pipeline.py
(same directory the original five scripts lived in)
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# The runtime model/classifier code now lives alongside the other
# detector backends so the app can import it directly; reach across
# for it here too, rather than keeping a second copy.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.detection.cnn_backend import TrafficCNN, box_iou, SlidingWindowCarDetector  # noqa: E402


CNN_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = CNN_DIR / "saved_model"

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASS_NAMES = {0: "car", 1: "truck", 2: "bus", 3: "motorcycle"}
REQUIRED_CLASSES = sorted(CLASS_NAMES.values())
NUM_CLASSES = len(CLASS_NAMES)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# prepare-dataset
# ============================================================

CROP_MARGIN = 0.10
MIN_CROP_WIDTH = 20
MIN_CROP_HEIGHT = 20
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _read_yolo_labels(label_path: Path, image_width: int, image_height: int):
    """Read YOLO-format labels -> [(class_id, x1, y1, x2, y2), ...] in
    pixel coordinates, with a small margin, clipped to the image, and
    dropped below a minimum crop size."""
    annotations = []
    if not label_path.exists():
        return annotations

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            cx, cy, box_width, box_height = (float(p) for p in parts[1:])
        except ValueError:
            continue
        if class_id not in CLASS_NAMES:
            continue

        x1 = int((cx - box_width / 2) * image_width)
        y1 = int((cy - box_height / 2) * image_height)
        x2 = int((cx + box_width / 2) * image_width)
        y2 = int((cy + box_height / 2) * image_height)

        margin_x = int((x2 - x1) * CROP_MARGIN)
        margin_y = int((y2 - y1) * CROP_MARGIN)
        x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
        x2, y2 = min(image_width, x2 + margin_x), min(image_height, y2 + margin_y)

        if (x2 - x1) < MIN_CROP_WIDTH or (y2 - y1) < MIN_CROP_HEIGHT:
            continue

        annotations.append((class_id, x1, y1, x2, y2))

    return annotations


def _process_split(image_dir: Path, label_dir: Path, output_split_dir: Path, split_label: str):
    print(f"\n{'=' * 70}\nPROCESSING {split_label.upper()} DATASET\n{'=' * 70}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found:\n{image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found:\n{label_dir}")

    counters = {class_id: 0 for class_id in CLASS_NAMES}
    images_processed = images_skipped = total_crops = 0
    image_files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    print(f"Images found: {len(image_files)}")

    for image_path in image_files:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: could not read image: {image_path.name}")
            images_skipped += 1
            continue

        height, width = image.shape[:2]
        annotations = _read_yolo_labels(label_dir / f"{image_path.stem}.txt", width, height)
        if not annotations:
            images_skipped += 1
            continue

        for object_index, (class_id, x1, y1, x2, y2) in enumerate(annotations):
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            class_name = CLASS_NAMES[class_id]
            output_path = output_split_dir / class_name / f"{image_path.stem}_object_{object_index}_{class_name}.jpg"
            if cv2.imwrite(str(output_path), crop):
                counters[class_id] += 1
                total_crops += 1

        images_processed += 1

    print("\nVehicle crop distribution:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_id}: {class_name:<12} {counters[class_id]} crops")
    print(f"\nImages processed: {images_processed}")
    print(f"Images skipped:   {images_skipped}")
    print(f"Total vehicle crops: {total_crops}")
    return counters


def cmd_prepare_dataset(args):
    train_images, train_labels = DATASET_DIR / "Training" / "images", DATASET_DIR / "Training" / "labels"
    val_images, val_labels = DATASET_DIR / "Validation" / "images", DATASET_DIR / "Validation" / "labels"
    output_dir = DATASET_DIR / "cnn_4class"

    for path in (train_images, train_labels, val_images, val_labels):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found:\n{path}")

    if output_dir.exists():
        print("Removing previous CNN 4-class dataset...")
        shutil.rmtree(output_dir)
    for split in ("train", "val"):
        for class_name in CLASS_NAMES.values():
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    train_counts = _process_split(train_images, train_labels, output_dir / "train", "training")
    val_counts = _process_split(val_images, val_labels, output_dir / "val", "validation")

    print(f"\n{'=' * 70}\nFINAL CNN DATASET CHECK\n{'=' * 70}")
    empty_classes = [
        name for cid, name in CLASS_NAMES.items()
        if train_counts[cid] == 0 or val_counts[cid] == 0
    ]
    if empty_classes:
        print(f"WARNING: classes with zero samples in train or val: {empty_classes}")
    else:
        print("All four classes have training and validation samples.")
    print(f"\nCNN dataset written to:\n{output_dir}")


# ============================================================
# train
# ============================================================

def _calculate_class_weights(dataset):
    targets = torch.tensor(dataset.targets, dtype=torch.long)
    class_counts = torch.bincount(targets, minlength=NUM_CLASSES).float()
    if torch.any(class_counts == 0):
        raise ValueError("At least one class has zero training samples.")
    class_weights = class_counts.sum() / (NUM_CLASSES * class_counts)
    return class_counts, class_weights


def _run_classification_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    loss_sum = correct = samples = 0
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if is_training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            loss_sum += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            samples += labels.size(0)

    if samples == 0:
        return 0.0, 0.0
    return loss_sum / samples, correct / samples


def _save_curves(history, output_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def cmd_train(args):
    dataset_dir = DATASET_DIR / "cnn_4class"
    train_dir, val_dir = dataset_dir / "train", dataset_dir / "val"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_dir in (train_dir, val_dir):
        if not split_dir.exists():
            raise FileNotFoundError(f"{split_dir} not found. Run `prepare-dataset` first.")

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=eval_transform)

    for label, classes in (("Training", train_data.classes), ("Validation", val_data.classes)):
        if classes != REQUIRED_CLASSES:
            raise ValueError(f"{label} dataset classes {classes} != expected {REQUIRED_CLASSES}")

    device = _device()
    print(f"Device: {device}")
    print(f"Training images: {len(train_data)} | Validation images: {len(val_data)}")

    class_counts, class_weights = _calculate_class_weights(train_data)
    print("\nClass distribution / weights:")
    for i, name in enumerate(train_data.classes):
        print(f"  {name:<12}: {int(class_counts[i])} samples, weight {class_weights[i]:.4f}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = TrafficCNN(num_classes=NUM_CLASSES, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_accuracy = -1.0

    print(f"\n{'=' * 70}\nStarting 4-class CNN training\n{'=' * 70}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = _run_classification_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = _run_classification_epoch(model, val_loader, criterion, device)

        for key, value in (("train_loss", train_loss), ("val_loss", val_loss),
                           ("train_accuracy", train_accuracy), ("val_accuracy", val_accuracy)):
            history[key].append(value)

        print(f"Epoch {epoch:02d}/{args.epochs}: train loss={train_loss:.4f} acc={train_accuracy:.4f} | "
              f"val loss={val_loss:.4f} acc={val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model_path = OUTPUT_DIR / "best_traffic_cnn.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": train_data.classes,
                "image_size": IMAGE_SIZE,
                "mean": MEAN,
                "std": STD,
                "num_classes": NUM_CLASSES,
                "best_val_accuracy": best_accuracy,
            }, model_path)
            print(f"  \u2713 Best model saved: {model_path}")

    (OUTPUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    _save_curves(history, OUTPUT_DIR / "training_curves.png")

    print(f"\n{'=' * 70}\nCNN TRAINING COMPLETED\nBest validation accuracy: {best_accuracy:.4f}\n{'=' * 70}")


# ============================================================
# evaluate  (classifier accuracy on held-out crops)
# ============================================================

def cmd_evaluate(args):
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

    checkpoint_path = OUTPUT_DIR / "best_traffic_cnn.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Trained CNN not found:\n{checkpoint_path}\nRun `train` first.")

    device = _device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", IMAGE_SIZE)
    mean, std = checkpoint.get("mean", MEAN), checkpoint.get("std", STD)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_dir = DATASET_DIR / "cnn_4class" / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation dataset not found:\n{val_dir}\nRun `prepare-dataset` first.")

    dataset = datasets.ImageFolder(val_dir, transform=transform)
    if dataset.classes != class_names:
        raise ValueError(f"Dataset classes {dataset.classes} != checkpoint classes {class_names}")

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model = TrafficCNN(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels, predictions = [], []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(device))
            labels.extend(targets.tolist())
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())

    if not labels:
        raise ValueError("Validation dataset contains no samples.")

    accuracy = sum(a == p for a, p in zip(labels, predictions)) / len(labels)
    print(f"\n{'=' * 70}\nCNN VALIDATION RESULTS\n{'=' * 70}")
    print(f"Validation samples: {len(labels)} | Accuracy: {accuracy:.4f}\n")
    print(classification_report(labels, predictions, labels=range(len(class_names)),
                                 target_names=class_names, zero_division=0))

    matrix = confusion_matrix(labels, predictions, labels=range(len(class_names)))
    print(matrix)

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(xticks_rotation=45)
    plt.title("CNN Vehicle Classification Confusion Matrix")
    plt.tight_layout()
    output_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"\nSaved confusion matrix: {output_path}")


# ============================================================
# predict  (single cropped image -> class)
# ============================================================

def cmd_predict(args):
    checkpoint_path = OUTPUT_DIR / "best_traffic_cnn.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CNN model not found:\n{checkpoint_path}\nRun `train` first.")

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found:\n{image_path}")

    device = _device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", IMAGE_SIZE)
    mean, std = checkpoint.get("mean", MEAN), checkpoint.get("std", STD)

    model = TrafficCNN(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensor), dim=1)[0]

    confidence, index = probabilities.max(dim=0)
    print(f"\n{'=' * 60}\nCNN PREDICTION\n{'=' * 60}")
    print(f"Image: {image_path.name}")
    print(f"Prediction: {class_names[index.item()]}")
    print(f"Confidence: {confidence.item() * 100:.2f}%\n")
    print("Class probabilities:")
    for i, name in enumerate(class_names):
        print(f"  {name:<12}: {probabilities[i].item() * 100:.2f}%")


# ============================================================
# evaluate-detector  (SlidingWindowCarDetector vs YOLO-format labels)
# ============================================================

def _read_ground_truth(label_path: Path, width: int, height: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5 or values[0] != "0":
            continue
        cx, cy, box_width, box_height = map(float, values[1:])
        boxes.append([
            int((cx - box_width / 2) * width), int((cy - box_height / 2) * height),
            int((cx + box_width / 2) * width), int((cy + box_height / 2) * height),
        ])
    return boxes


def _average_precision(predictions, ground_truth, iou_threshold=0.5):
    matched = {image_id: set() for image_id in ground_truth}
    tp_flags, fp_flags = [], []
    for item in sorted(predictions, key=lambda v: v["confidence"], reverse=True):
        candidates = ground_truth[item["image_id"]]
        best_iou, best_index = 0.0, None
        for index, box in enumerate(candidates):
            if index not in matched[item["image_id"]]:
                iou = box_iou(item["bbox"], box)
                if iou > best_iou:
                    best_iou, best_index = iou, index
        if best_iou >= iou_threshold:
            matched[item["image_id"]].add(best_index)
            tp_flags.append(1); fp_flags.append(0)
        else:
            tp_flags.append(0); fp_flags.append(1)

    total_gt = sum(len(boxes) for boxes in ground_truth.values())
    if not total_gt:
        return 0.0

    precision, recall, tp, fp = [], [], 0, 0
    for t, f in zip(tp_flags, fp_flags):
        tp += t; fp += f
        precision.append(tp / (tp + fp))
        recall.append(tp / total_gt)

    return sum(
        max((p for p, r in zip(precision, recall) if r >= level), default=0.0)
        for level in (i / 100 for i in range(101))
    ) / 101


def cmd_evaluate_detector(args):
    validation_dir = DATASET_DIR / "Validation"
    image_paths = sorted(
        p for p in (validation_dir / "images").glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.max_images:
        image_paths = image_paths[:args.max_images]

    detector = SlidingWindowCarDetector(confidence_threshold=args.confidence)
    ground_truth, predictions, matched_ious = {}, [], []

    started = time.perf_counter()
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        truth = _read_ground_truth(validation_dir / "labels" / f"{image_path.stem}.txt", width, height)
        ground_truth[image_path.stem] = truth

        used = set()
        for detection in detector.detect(image):
            predictions.append({"image_id": image_path.stem, **detection})
            best_iou, best_index = 0.0, None
            for index, box in enumerate(truth):
                if index not in used:
                    iou = box_iou(detection["bbox"], box)
                    if iou > best_iou:
                        best_iou, best_index = iou, index
            if best_iou >= 0.5:
                used.add(best_index)
                matched_ious.append(best_iou)
    elapsed = time.perf_counter() - started

    ap50 = _average_precision(predictions, ground_truth)
    total_truth = sum(len(boxes) for boxes in ground_truth.values())

    matched = {image_id: set() for image_id in ground_truth}
    tp = 0
    for item in sorted(predictions, key=lambda v: v["confidence"], reverse=True):
        best_iou, best_index = 0.0, None
        for index, box in enumerate(ground_truth[item["image_id"]]):
            if index not in matched[item["image_id"]]:
                iou = box_iou(item["bbox"], box)
                if iou > best_iou:
                    best_iou, best_index = iou, index
        if best_iou >= 0.5:
            matched[item["image_id"]].add(best_index)
            tp += 1

    fp, fn = len(predictions) - tp, total_truth - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    results = {
        "images": len(image_paths),
        "ground_truth_cars": total_truth,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_iou": sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
        "mAP50": ap50,
        "fps": len(image_paths) / elapsed if elapsed else 0.0,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "detection_metrics.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved: {output_path}")


# ============================================================
# CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(description="CNN vehicle-classifier pipeline (prepare/train/evaluate/predict)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare-dataset", help="Crop vehicles out of the YOLO dataset into class folders")

    train_parser = subparsers.add_parser("train", help="Train the 4-class CNN")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--workers", type=int, default=0)
    train_parser.add_argument("--no-pretrained", action="store_true")

    subparsers.add_parser("evaluate", help="Evaluate classifier accuracy on the validation crops")

    predict_parser = subparsers.add_parser("predict", help="Classify a single cropped vehicle image")
    predict_parser.add_argument("image", help="Path to a cropped vehicle image")

    detector_parser = subparsers.add_parser(
        "evaluate-detector", help="Evaluate the sliding-window full-frame car detector"
    )
    detector_parser.add_argument("--confidence", type=float, default=0.70)
    detector_parser.add_argument("--max-images", type=int, default=0, help="0 evaluates every validation image")

    return parser


def main():
    args = build_parser().parse_args()
    {
        "prepare-dataset": cmd_prepare_dataset,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
        "evaluate-detector": cmd_evaluate_detector,
    }[args.command](args)


if __name__ == "__main__":
    main()