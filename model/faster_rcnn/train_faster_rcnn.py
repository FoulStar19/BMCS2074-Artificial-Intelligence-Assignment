from pathlib import Path
import json
import time

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = Path("dataset/Training_new")
VAL_DIR = Path("dataset/Validation_new")

OUTPUT_DIR = Path("model/faster_rcnn/saved_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["car", "truck", "bus", "motorcycle"]

# Faster R-CNN class IDs:
# 0 = background
# 1 = car
# 2 = truck
# 3 = bus
# 4 = motorcycle

NUM_CLASSES = len(CLASS_NAMES) + 1

BATCH_SIZE = 2
NUM_EPOCHS = 5
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATASET
# ============================================================

class YOLODetectionDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.label_dir = self.root_dir / "labels"

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        self.images = sorted(
            [
                p for p in self.image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in valid_extensions
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes = []
        labels = []

        if label_path.exists():

            with open(label_path, "r", encoding="utf-8") as f:

                for line_number, line in enumerate(f, start=1):

                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split()

                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])

                    if class_id < 0 or class_id >= len(CLASS_NAMES):
                        continue

                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    box_width = float(parts[3]) * width
                    box_height = float(parts[4]) * height

                    xmin = x_center - box_width / 2
                    ymin = y_center - box_height / 2
                    xmax = x_center + box_width / 2
                    ymax = y_center + box_height / 2

                    xmin = max(0, min(xmin, width))
                    ymin = max(0, min(ymin, height))
                    xmax = max(0, min(xmax, width))
                    ymax = max(0, min(ymax, height))

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    boxes.append([xmin, ymin, xmax, ymax])

                    # +1 because 0 is background
                    labels.append(class_id + 1)

        boxes = torch.tensor(
            boxes,
            dtype=torch.float32
        )

        labels = torch.tensor(
            labels,
            dtype=torch.int64
        )

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        area = (
            (boxes[:, 2] - boxes[:, 0])
            * (boxes[:, 3] - boxes[:, 1])
            if len(boxes) > 0
            else torch.zeros((0,), dtype=torch.float32)
        )

        iscrowd = torch.zeros(
            (len(boxes),),
            dtype=torch.int64
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
            "area": area,
            "iscrowd": iscrowd,
        }

        image = TF.to_tensor(image)

        return image, target


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================
# MODEL
# ============================================================

def create_model():

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    model = fasterrcnn_resnet50_fpn(
        weights=weights
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FASTER R-CNN TRAINING")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Number of classes including background: {NUM_CLASSES}")

    train_dataset = YOLODetectionDataset(TRAIN_DIR)
    val_dataset = YOLODetectionDataset(VAL_DIR)

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = create_model()
    model.to(DEVICE)

    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    print()
    print("=" * 70)
    print("Starting training")
    print("=" * 70)

    history = {
        "train_loss": []
    }

    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):

        model.train()

        epoch_loss = 0.0
        start_time = time.time()

        for batch_index, (images, targets) in enumerate(train_loader):

            images = [
                image.to(DEVICE)
                for image in images
            ]

            targets = [
                {
                    key: value.to(DEVICE)
                    for key, value in target.items()
                }
                for target in targets
            ]

            optimizer.zero_grad()

            loss_dict = model(
                images,
                targets
            )

            losses = sum(
                loss for loss in loss_dict.values()
            )

            losses.backward()

            optimizer.step()

            epoch_loss += losses.item()

            if (batch_index + 1) % 50 == 0:

                print(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS} "
                    f"Batch {batch_index + 1}/{len(train_loader)} "
                    f"Loss: {losses.item():.4f}"
                )

        average_loss = epoch_loss / len(train_loader)

        elapsed = time.time() - start_time

        history["train_loss"].append(average_loss)

        print()
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} "
            f"Train Loss: {average_loss:.4f} "
            f"Time: {elapsed / 60:.2f} min"
        )

        if average_loss < best_loss:

            best_loss = average_loss

            checkpoint_path = (
                OUTPUT_DIR / "best_faster_rcnn.pth"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "num_classes": NUM_CLASSES,
                    "epoch": epoch + 1,
                    "train_loss": average_loss,
                },
                checkpoint_path
            )

            print(
                f"  ✓ Best model saved: {checkpoint_path}"
            )

    with open(
        OUTPUT_DIR / "training_history.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("FASTER R-CNN TRAINING COMPLETED")
    print("=" * 70)

    print(f"Best training loss: {best_loss:.4f}")
    print(
        f"Model saved to: "
        f"{OUTPUT_DIR / 'best_faster_rcnn.pth'}"
    )


if __name__ == "__main__":
    main()