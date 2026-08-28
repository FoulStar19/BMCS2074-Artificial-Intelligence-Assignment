"""
Quick sanity check that the Faster R-CNN dataset loader reads the SAME
train split YOLO uses (via dataset.yaml) and produces sane boxes/labels.
"""

from pathlib import Path
from PIL import Image
import torch

from train_faster_rcnn import find_dataset_yaml, load_dataset_yaml


def load_yolo_target(label_path, image_width, image_height, class_names):
    boxes = []
    labels = []

    if not label_path.exists():
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

    with open(label_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid label at {label_path}, line {line_number}: {line}")

            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])

            if class_id < 0 or class_id >= len(class_names):
                raise ValueError(f"Invalid class ID {class_id} in {label_path}")

            x_center *= image_width
            y_center *= image_height
            width *= image_width
            height *= image_height

            xmin = max(0, min(x_center - width / 2, image_width))
            ymin = max(0, min(y_center - height / 2, image_height))
            xmax = max(0, min(x_center + width / 2, image_width))
            ymax = max(0, min(y_center + height / 2, image_height))

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_id + 1)  # Faster R-CNN: 0 = background

    if boxes:
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
    return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)


def main():
    dataset_yaml_path = find_dataset_yaml()
    train_dir, _val_dir, class_names, _cfg = load_dataset_yaml(dataset_yaml_path)

    image_dir = train_dir / "images"
    label_dir = train_dir / "labels"
    images = sorted(image_dir.glob("*"))

    print("=" * 70)
    print("FASTER R-CNN DATASET TEST (same split as YOLO)")
    print("=" * 70)
    print(f"Dataset config    : {dataset_yaml_path}")
    print(f"Training images   : {len(images)}")

    if not images:
        raise RuntimeError("No training images found.")

    total_boxes = 0

    for image_path in images[:5]:
        label_path = label_dir / f"{image_path.stem}.txt"

        with Image.open(image_path) as image:
            width, height = image.size

        boxes, labels = load_yolo_target(label_path, width, height, class_names)

        print()
        print(f"Image : {image_path.name}")
        print(f"Size  : {width} x {height}")
        print(f"Label : {label_path.name}")
        print(f"Boxes : {len(boxes)}")
        print(f"Labels: {labels.tolist()}")
        if len(boxes) > 0:
            print("First box:", boxes[0].tolist())

        total_boxes += len(boxes)

    print()
    print("=" * 70)
    print("TEST PASSED")
    print("=" * 70)
    print(f"Images tested : {min(5, len(images))}")
    print(f"Total boxes   : {total_boxes}")
    print()
    print("Class mapping:")
    for i, name in enumerate(class_names, start=1):
        print(f"  Faster R-CNN label {i}: {name}")


if __name__ == "__main__":
    main()