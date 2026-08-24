"""Evaluate the CNN as a full-frame car detector against YOLO-format labels."""

import argparse
import json
import time
from pathlib import Path

import cv2

from sliding_window_detector import SlidingWindowCarDetector, box_iou


def read_ground_truth(label_path, width, height):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5 or values[0] != "0":
            continue
        cx, cy, box_width, box_height = map(float, values[1:])
        boxes.append([int((cx - box_width / 2) * width), int((cy - box_height / 2) * height),
                      int((cx + box_width / 2) * width), int((cy + box_height / 2) * height)])
    return boxes


def average_precision(predictions, ground_truth, iou_threshold=0.5):
    """Compute AP@IoU from all scored detections using one-to-one matching."""
    matched = {image_id: set() for image_id in ground_truth}
    true_positive, false_positive = [], []
    for item in sorted(predictions, key=lambda value: value["confidence"], reverse=True):
        image_id = item["image_id"]
        candidates = ground_truth[image_id]
        best_iou, best_index = 0.0, None
        for index, box in enumerate(candidates):
            if index not in matched[image_id] and box_iou(item["bbox"], box) > best_iou:
                best_iou, best_index = box_iou(item["bbox"], box), index
        if best_iou >= iou_threshold:
            matched[image_id].add(best_index)
            true_positive.append(1); false_positive.append(0)
        else:
            true_positive.append(0); false_positive.append(1)
    total_ground_truth = sum(len(boxes) for boxes in ground_truth.values())
    if not total_ground_truth:
        return 0.0
    precision, recall, tp, fp = [], [], 0, 0
    for current_tp, current_fp in zip(true_positive, false_positive):
        tp += current_tp; fp += current_fp
        precision.append(tp / (tp + fp)); recall.append(tp / total_ground_truth)
    return sum(max((value for p, value in zip(precision, recall) if p >= level), default=0.0)
               for level in (index / 100 for index in range(101))) / 101


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--max-images", type=int, default=0, help="0 evaluates every validation image.")
    args = parser.parse_args()
    cnn_dir = Path(__file__).resolve().parent
    validation_dir = cnn_dir.parents[1] / "dataset" / "Validation"
    image_paths = sorted(path for path in (validation_dir / "images").glob("*")
                         if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    detector = SlidingWindowCarDetector(confidence_threshold=args.confidence)
    ground_truth, predictions, matched_ious = {}, [], []
    started = time.perf_counter()
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        truth = read_ground_truth(validation_dir / "labels" / f"{image_path.stem}.txt", width, height)
        ground_truth[image_path.stem] = truth
        detections = detector.detect(image)
        used = set()
        for detection in detections:
            predictions.append({"image_id": image_path.stem, **detection})
            best_iou, best_index = 0.0, None
            for index, box in enumerate(truth):
                score = box_iou(detection["bbox"], box)
                if index not in used and score > best_iou:
                    best_iou, best_index = score, index
            if best_iou >= 0.5:
                used.add(best_index); matched_ious.append(best_iou)
    elapsed = time.perf_counter() - started
    ap50 = average_precision(predictions, ground_truth)
    total_truth = sum(len(boxes) for boxes in ground_truth.values())
    ordered = sorted(predictions, key=lambda value: value["confidence"], reverse=True)
    matched = {image_id: set() for image_id in ground_truth}
    tp = 0
    for item in ordered:
        best_iou, best_index = 0.0, None
        for index, box in enumerate(ground_truth[item["image_id"]]):
            score = box_iou(item["bbox"], box)
            if index not in matched[item["image_id"]] and score > best_iou:
                best_iou, best_index = score, index
        if best_iou >= 0.5:
            matched[item["image_id"]].add(best_index); tp += 1
    fp, fn = len(ordered) - tp, total_truth - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    results = {"images": len(image_paths), "ground_truth_cars": total_truth, "precision": precision,
               "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
               "mean_iou": sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
               "mAP50": ap50, "fps": len(image_paths) / elapsed if elapsed else 0.0}
    output = cnn_dir / "saved_model" / "detection_metrics.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
