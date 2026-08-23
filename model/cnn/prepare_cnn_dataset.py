"""Create a derived CNN dataset without changing the original YOLO dataset.

The original dataset and its five class definitions are preserved. This script
uses only class 0 (car) labels to create a binary, derived dataset: car crops
and non-overlapping background crops.
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATASET = PROJECT_ROOT / "dataset"
OUTPUT_DIR = SOURCE_DATASET / "cnn_car_background"
SPLITS = {"train": "Training", "val": "Validation"}
CAR_CLASS_ID = 0


def read_car_boxes(label_path, width, height):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5 or int(parts[0]) != CAR_CLASS_ID:
            continue
        cx, cy, box_width, box_height = map(float, parts[1:])
        x1 = max(0, int((cx - box_width / 2) * width))
        y1 = max(0, int((cy - box_height / 2) * height))
        x2 = min(width, int((cx + box_width / 2) * width))
        y2 = min(height, int((cy + box_height / 2) * height))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def iou(box_a, box_b):
    left, top = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    right, bottom = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = ((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]) +
             (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) - intersection)
    return intersection / union if union else 0.0


def background_box(width, height, car_boxes, reference_box, attempts=100):
    """Find a background crop similar in size to a car crop, with zero overlap."""
    crop_width = min(width, max(32, reference_box[2] - reference_box[0]))
    crop_height = min(height, max(32, reference_box[3] - reference_box[1]))
    if crop_width >= width or crop_height >= height:
        return None
    for _ in range(attempts):
        x1 = random.randint(0, width - crop_width)
        y1 = random.randint(0, height - crop_height)
        candidate = (x1, y1, x1 + crop_width, y1 + crop_height)
        if all(iou(candidate, car_box) == 0 for car_box in car_boxes):
            return candidate
    return None


def create_split(split, source_name):
    image_dir = SOURCE_DATASET / source_name / "images"
    label_dir = SOURCE_DATASET / source_name / "labels"
    car_dir = OUTPUT_DIR / split / "car"
    background_dir = OUTPUT_DIR / split / "background"
    cars = backgrounds = 0
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        boxes = read_car_boxes(label_dir / f"{image_path.stem}.txt", width, height)
        for number, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cv2.imwrite(str(car_dir / f"{image_path.stem}_car_{number}.jpg"), image[y1:y2, x1:x2])
            cars += 1
            negative_box = background_box(width, height, boxes, box)
            if negative_box:
                bx1, by1, bx2, by2 = negative_box
                cv2.imwrite(str(background_dir / f"{image_path.stem}_background_{number}.jpg"),
                            image[by1:by2, bx1:bx2])
                backgrounds += 1
    print(f"{split}: {cars} car crops, {backgrounds} background crops")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-output", action="store_true",
                        help="Delete only the derived cnn_car_background folder first.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    if args.clean_output and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for split in SPLITS:
        for class_name in ("car", "background"):
            (OUTPUT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)
    for split, source_name in SPLITS.items():
        create_split(split, source_name)
    print(f"Derived CNN dataset created at: {OUTPUT_DIR}")
    print("The original YOLO images, labels, and five class definitions were not changed.")


if __name__ == "__main__":
    main()
