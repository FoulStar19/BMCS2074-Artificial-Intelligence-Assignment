"""
Faster R-CNN validation evaluation.

Uses the SAME dataset.yaml (and therefore the same validation images) as
YOLO, and reports precision / recall / F1 / mean IoU AND per-class AP@0.5
+ mAP@0.5 -- the same headline metric Ultralytics prints for YOLO -- so
the two backends can be compared on equal footing.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

from train_faster_rcnn import find_dataset_yaml, load_dataset_yaml  # reuse the exact same loader


CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5


# ============================================================
# DATASET
# ============================================================

class YOLODetectionDataset(Dataset):
    def __init__(self, root_dir, class_names):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.label_dir = self.root_dir / "labels"
        self.class_names = class_names

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

                    x_center, y_center, box_w, box_h = (float(p) for p in parts[1:])
                    x_center *= width
                    y_center *= height
                    box_w *= width
                    box_h *= height

                    xmin = max(0, min(x_center - box_w / 2, width))
                    ymin = max(0, min(y_center - box_h / 2, height))
                    xmax = max(0, min(x_center + box_w / 2, width))
                    ymax = max(0, min(y_center + box_h / 2, height))

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id + 1)  # 0 = background

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return TF.to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ============================================================
# PER-IMAGE COUNTING METRICS (precision / recall / F1 / IoU)
# ============================================================

def calculate_image_metrics(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
    keep = pred_scores >= CONFIDENCE_THRESHOLD
    pred_boxes, pred_labels, pred_scores = pred_boxes[keep], pred_labels[keep], pred_scores[keep]

    if len(pred_scores) > 0:
        order = torch.argsort(pred_scores, descending=True)
        pred_boxes, pred_labels = pred_boxes[order], pred_labels[order]

    matched_gt = set()
    tp = fp = 0
    matched_ious = []

    for i in range(len(pred_boxes)):
        pred_box = pred_boxes[i]
        pred_label = pred_labels[i].item()
        best_iou, best_gt_index = 0.0, None

        for j in range(len(gt_boxes)):
            if j in matched_gt or gt_labels[j].item() != pred_label:
                continue
            iou = box_iou(pred_box.unsqueeze(0), gt_boxes[j].unsqueeze(0))[0, 0].item()
            if iou > best_iou:
                best_iou, best_gt_index = iou, j

        if best_gt_index is not None and best_iou >= IOU_THRESHOLD:
            tp += 1
            matched_gt.add(best_gt_index)
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, matched_ious


# ============================================================
# AP@0.5 / mAP@0.5 (VOC-style, all-point interpolation)
# ============================================================

def compute_ap(recalls, precisions):
    """All-point interpolated average precision (same convention as
    Pascal VOC 2010+ / roughly what YOLO's mAP uses)."""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1]))


def compute_map(all_predictions, all_ground_truths, num_fg_classes):
    """
    all_predictions: dict[class_id] -> list of (image_id, score, box)
    all_ground_truths: dict[class_id] -> dict[image_id] -> list of boxes
    """
    aps = {}

    for class_id in range(1, num_fg_classes + 1):
        preds = sorted(all_predictions.get(class_id, []), key=lambda x: -x[1])
        gts = all_ground_truths.get(class_id, {})
        num_gt = sum(len(v) for v in gts.values())

        if num_gt == 0:
            aps[class_id] = 0.0
            continue

        matched = {img_id: np.zeros(len(boxes), dtype=bool) for img_id, boxes in gts.items()}
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        for i, (image_id, score, box) in enumerate(preds):
            gt_boxes = gts.get(image_id, [])
            if not gt_boxes:
                fp[i] = 1
                continue

            gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
            ious = box_iou(torch.tensor(box, dtype=torch.float32).unsqueeze(0), gt_tensor)[0]
            best_iou, best_j = ious.max(0)
            best_iou, best_j = best_iou.item(), best_j.item()

            if best_iou >= IOU_THRESHOLD and not matched[image_id][best_j]:
                tp[i] = 1
                matched[image_id][best_j] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / num_gt
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        aps[class_id] = compute_ap(recalls, precisions)

    mean_ap = float(np.mean(list(aps.values()))) if aps else 0.0
    return aps, mean_ap


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN on the YOLO validation split")
    parser.add_argument("--dataset-yaml", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="model/faster_rcnn/saved_model/best_faster_rcnn.pth")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N images (debugging)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_yaml_path = find_dataset_yaml(args.dataset_yaml)
    _train_dir, val_dir, class_names, _cfg = load_dataset_yaml(dataset_yaml_path)
    num_classes = len(class_names) + 1

    checkpoint_path = Path(args.checkpoint)

    print("=" * 70)
    print("FASTER R-CNN VALIDATION EVALUATION")
    print("=" * 70)
    print(f"Dataset config      : {dataset_yaml_path}")
    print(f"Validation dir      : {val_dir}")
    print(f"Device              : {device}")
    print(f"Checkpoint          : {checkpoint_path}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"IoU threshold       : {IOU_THRESHOLD}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n{checkpoint_path}")

    dataset = YOLODetectionDataset(val_dir, class_names)
    print(f"Validation images available: {len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError("No validation images found.")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    print()
    print("Loading Faster R-CNN checkpoint...")
    model = create_model(num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("Checkpoint loaded successfully.")
    print(f"Checkpoint epoch    : {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint val_loss : {checkpoint.get('val_loss', checkpoint.get('train_loss', 'unknown'))}")

    print()
    print("=" * 70)
    print("STARTING EVALUATION")
    print("=" * 70)

    total_tp = total_fp = total_fn = 0
    all_ious = []
    class_stats = {c: {"tp": 0, "fp": 0, "fn": 0, "ious": []} for c in range(1, num_classes)}

    # For AP/mAP
    all_predictions = defaultdict(list)          # class_id -> [(image_id, score, box)]
    all_ground_truths = defaultdict(lambda: defaultdict(list))  # class_id -> {image_id: [boxes]}

    processed = 0

    with torch.no_grad():
        for images, targets in loader:
            image = images[0].to(device)
            target = targets[0]
            image_id = int(target["image_id"].item())

            output = model([image])[0]
            pred_boxes = output["boxes"].cpu()
            pred_labels = output["labels"].cpu()
            pred_scores = output["scores"].cpu()

            gt_boxes = target["boxes"]
            gt_labels = target["labels"]

            for cls in range(1, num_classes):
                cls_gt_boxes = gt_boxes[gt_labels == cls].tolist()
                if cls_gt_boxes:
                    all_ground_truths[cls][image_id] = cls_gt_boxes

            for box, label, score in zip(pred_boxes.tolist(), pred_labels.tolist(), pred_scores.tolist()):
                all_predictions[label].append((image_id, score, box))

            tp, fp, fn, image_ious = calculate_image_metrics(
                pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn
            all_ious.extend(image_ious)

            for class_id in range(1, num_classes):
                mask_p = pred_labels == class_id
                mask_g = gt_labels == class_id
                c_tp, c_fp, c_fn, c_ious = calculate_image_metrics(
                    pred_boxes[mask_p], pred_labels[mask_p], pred_scores[mask_p],
                    gt_boxes[mask_g], gt_labels[mask_g],
                )
                class_stats[class_id]["tp"] += c_tp
                class_stats[class_id]["fp"] += c_fp
                class_stats[class_id]["fn"] += c_fn
                class_stats[class_id]["ious"].extend(c_ious)

            processed += 1
            pred_count = (pred_scores >= CONFIDENCE_THRESHOLD).sum().item()
            print(f"Image {processed}: GT={len(gt_boxes)}, Pred={pred_count}, TP={tp}, FP={fp}, FN={fn}")

            if args.limit is not None and processed >= args.limit:
                break

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0

    aps, mean_ap = compute_map(all_predictions, all_ground_truths, num_classes - 1)

    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Images evaluated : {processed}")
    print(f"True Positives   : {total_tp}")
    print(f"False Positives  : {total_fp}")
    print(f"False Negatives  : {total_fn}")
    print()
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"F1-score         : {f1:.4f}")
    print(f"Mean IoU         : {mean_iou:.4f}")
    print(f"mAP@0.5          : {mean_ap:.4f}   <- directly comparable to YOLO's mAP50")

    print()
    print("=" * 70)
    print("PER-CLASS RESULTS")
    print("=" * 70)
    for class_id, name in enumerate(class_names, start=1):
        stats = class_stats[class_id]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        c_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        c_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        c_f1 = 2 * c_precision * c_recall / (c_precision + c_recall) if (c_precision + c_recall) > 0 else 0.0
        c_iou = sum(stats["ious"]) / len(stats["ious"]) if stats["ious"] else 0.0
        c_ap = aps.get(class_id, 0.0)

        print()
        print(name)
        print(f"  TP        : {tp}")
        print(f"  FP        : {fp}")
        print(f"  FN        : {fn}")
        print(f"  Precision : {c_precision:.4f}")
        print(f"  Recall    : {c_recall:.4f}")
        print(f"  F1-score  : {c_f1:.4f}")
        print(f"  Mean IoU  : {c_iou:.4f}")
        print(f"  AP@0.5    : {c_ap:.4f}")

    print()
    print("=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()