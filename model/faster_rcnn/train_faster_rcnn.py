"""
Faster R-CNN training script.

Trains on the EXACT same train/validation split as the YOLO model by
reading the same dataset.yaml (the one in model/yolo/dataset.yaml or
config/dataset.yaml). This means the two detectors are trained and
evaluated on identical images, so metrics between them (YOLO vs
Faster R-CNN) are directly comparable -- no more "Training_new" vs
"Training" mismatch.

Optimizations over the original script:
  - Reads dataset.yaml instead of hardcoded folders -> same split as YOLO
  - Mixed precision (AMP) training
  - Train/val loss tracked every epoch (best checkpoint picked on val loss,
    not train loss, to avoid saving an overfit model)
  - LR scheduling (StepLR) + gradient clipping
  - Early stopping via --patience
  - pin_memory / configurable num_workers for faster data loading
  - Reproducible run (fixed seed)
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ============================================================
# DATASET CONFIG (shared with YOLO)
# ============================================================

DEFAULT_DATASET_YAML_CANDIDATES = [
    Path("config/dataset.yaml"),
    Path("model/yolo/dataset.yaml"),
    Path("dataset.yaml"),
]


def find_dataset_yaml(explicit_path=None) -> Path:
    """Locate the same dataset.yaml that train_model.py (YOLO) uses."""
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"dataset.yaml not found at: {path}")
        return path

    for candidate in DEFAULT_DATASET_YAML_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find dataset.yaml. Pass --dataset-yaml explicitly, "
        "or place it at one of: "
        + ", ".join(str(p) for p in DEFAULT_DATASET_YAML_CANDIDATES)
    )


def load_dataset_yaml(dataset_yaml_path: Path):
    """Load dataset.yaml and derive train/val image+label directories.

    dataset.yaml stores `train`/`val` as paths to the *images* folder
    (same convention Ultralytics YOLO uses), e.g.:
        train: .../dataset/Training/images
        val:   .../dataset/Validation/images

    The matching labels folder is assumed to be a sibling `labels/`
    directory, again matching the YOLO layout.
    """
    with open(dataset_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    train_images = Path(cfg["train"])
    val_images = Path(cfg["val"])

    # If the absolute path baked into dataset.yaml doesn't exist on this
    # machine (e.g. it was written on someone else's Windows path), fall
    # back to interpreting it relative to the dataset.yaml location.
    if not train_images.exists():
        train_images = dataset_yaml_path.parent / Path(cfg["train"]).name
    if not val_images.exists():
        val_images = dataset_yaml_path.parent / Path(cfg["val"]).name

    train_dir = train_images.parent  # .../Training  (parent of images/)
    val_dir = val_images.parent      # .../Validation

    names = cfg.get("names", {})
    # names may be a dict {0: "car", ...} or a list ["car", ...]
    if isinstance(names, dict):
        class_names = [names[k] for k in sorted(names.keys())]
    else:
        class_names = list(names)

    return train_dir, val_dir, class_names, cfg


# ============================================================
# DATASET
# ============================================================

class YOLODetectionDataset(Dataset):
    """Reads YOLO-format label .txt files but returns Faster R-CNN style
    targets (absolute pixel xyxy boxes, 1-indexed labels)."""

    def __init__(self, root_dir, class_names, augment=False):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.label_dir = self.root_dir / "labels"
        self.class_names = class_names
        self.augment = augment

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.images = sorted(
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_extensions
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes, labels = [], []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= len(self.class_names):
                        continue

                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    box_width = float(parts[3]) * width
                    box_height = float(parts[4]) * height

                    xmin = max(0, min(x_center - box_width / 2, width))
                    ymin = max(0, min(y_center - box_height / 2, height))
                    xmax = max(0, min(x_center + box_width / 2, width))
                    ymax = max(0, min(y_center + box_height / 2, height))

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    boxes.append([xmin, ymin, xmax, ymax])
                    # +1 because 0 is background for Faster R-CNN
                    labels.append(class_id + 1)

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        if len(boxes_t) > 0:
            area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        else:
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([index]),
            "area": area,
            "iscrowd": torch.zeros((len(boxes_t),), dtype=torch.int64),
        }

        image_t = TF.to_tensor(image)

        if self.augment and random.random() < 0.5 and len(boxes_t) > 0:
            image_t = TF.hflip(image_t)
            flipped = target["boxes"].clone()
            flipped[:, 0] = width - target["boxes"][:, 2]
            flipped[:, 2] = width - target["boxes"][:, 0]
            target["boxes"] = flipped

        return image_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================
# MODEL
# ============================================================

def create_model(num_classes: int):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ============================================================
# TRAIN / VAL EPOCH
# ============================================================

def run_train_epoch(model, loader, optimizer, device, scaler, grad_clip):
    model.train()
    total_loss = 0.0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += losses.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def run_val_epoch(model, loader, device):
    """Compute validation LOSS (not mAP) for early-stopping / model
    selection. torchvision detection models only return the loss dict
    when in train() mode with targets supplied, so we force train()
    but stay inside no_grad -- the backbone uses frozen batchnorm so
    this doesn't corrupt running stats."""
    model.train()
    total_loss = 0.0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

    return total_loss / max(1, len(loader))


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on the YOLO dataset split")
    parser.add_argument("--dataset-yaml", type=str, default=None,
                         help="Path to dataset.yaml (defaults to the same file YOLO uses)")
    parser.add_argument("--output-dir", type=str, default="model/faster_rcnn/saved_model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=7,
                         help="Stop if val loss doesn't improve for this many epochs")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--amp", action="store_true", default=True,
                         help="Use mixed precision (default on, CUDA only)")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--augment", action="store_true", default=True,
                         help="Random horizontal flip augmentation for training")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    torch.backends.cudnn.benchmark = True

    dataset_yaml_path = find_dataset_yaml(args.dataset_yaml)
    train_dir, val_dir, class_names, _cfg = load_dataset_yaml(dataset_yaml_path)
    num_classes = len(class_names) + 1  # +1 for background

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FASTER R-CNN TRAINING")
    print("=" * 70)
    print(f"Dataset config : {dataset_yaml_path}")
    print(f"Train dir      : {train_dir}")
    print(f"Val dir        : {val_dir}")
    print(f"Device         : {device}")
    print(f"Classes        : {class_names}")
    print(f"Mixed precision: {use_amp}")

    train_dataset = YOLODetectionDataset(train_dir, class_names, augment=args.augment)
    val_dataset = YOLODetectionDataset(val_dir, class_names, augment=False)

    print(f"Training images  : {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=pin_memory,
    )

    model = create_model(num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, args.epochs // 3), gamma=0.1
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {"train_loss": [], "val_loss": [], "lr": [], "epoch_time_sec": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print()
    print("=" * 70)
    print("Starting training")
    print("=" * 70)

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = run_train_epoch(model, train_loader, optimizer, device, scaler if use_amp else None, args.grad_clip)
        val_loss = run_val_epoch(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        history["epoch_time_sec"].append(elapsed)

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"LR: {current_lr:.6f} "
            f"Time: {elapsed / 60:.2f} min"
        )

        # Save best checkpoint by VAL loss (better generalization signal than train loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            checkpoint_path = output_dir / "best_faster_rcnn.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "num_classes": num_classes,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "dataset_yaml": str(dataset_yaml_path),
                },
                checkpoint_path,
            )
            print(f"  \u2713 Best model saved (val_loss={val_loss:.4f}): {checkpoint_path}")
        else:
            epochs_without_improvement += 1

        # Always keep a "last" checkpoint too, useful for resuming
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "num_classes": num_classes,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "dataset_yaml": str(dataset_yaml_path),
            },
            output_dir / "last_faster_rcnn.pth",
        )

        with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if epochs_without_improvement >= args.patience:
            print()
            print(f"Early stopping: no val loss improvement for {args.patience} epochs.")
            break

    print()
    print("=" * 70)
    print("FASTER R-CNN TRAINING COMPLETED")
    print("=" * 70)
    print(f"Best val loss  : {best_val_loss:.4f}")
    print(f"Model saved to : {output_dir / 'best_faster_rcnn.pth'}")


if __name__ == "__main__":
    main()