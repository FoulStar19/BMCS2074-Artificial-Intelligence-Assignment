from pathlib import Path
from PIL import Image
import torch

DATASET_ROOT = Path("dataset")
TRAIN_DIR = DATASET_ROOT / "Training_new"

CLASS_NAMES = ["car", "truck", "bus", "motorcycle"]


def load_yolo_target(label_path, image_width, image_height):
    boxes = []
    labels = []

    if not label_path.exists():
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(
            (0,), dtype=torch.int64
        )

    with open(label_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid label at {label_path}, line {line_number}: {line}"
                )

            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])

            if class_id < 0 or class_id >= len(CLASS_NAMES):
                raise ValueError(
                    f"Invalid class ID {class_id} in {label_path}"
                )

            # YOLO normalized coordinates -> pixel coordinates
            x_center *= image_width
            y_center *= image_height
            width *= image_width
            height *= image_height

            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2

            # Clip boxes to image boundaries
            xmin = max(0, min(xmin, image_width))
            ymin = max(0, min(ymin, image_height))
            xmax = max(0, min(xmax, image_width))
            ymax = max(0, min(ymax, image_height))

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])

            # Faster R-CNN labels are normally 1-based because 0 is background
            labels.append(class_id + 1)

    if boxes:
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)

    return boxes, labels


def main():
    image_dir = TRAIN_DIR / "images"
    label_dir = TRAIN_DIR / "labels"

    images = sorted(image_dir.glob("*"))

    print("=" * 70)
    print("FASTER R-CNN DATASET TEST")
    print("=" * 70)

    print(f"Training images found: {len(images)}")

    if not images:
        raise RuntimeError("No training images found.")

    total_boxes = 0

    for image_path in images[:5]:
        label_path = label_dir / f"{image_path.stem}.txt"

        with Image.open(image_path) as image:
            width, height = image.size

        boxes, labels = load_yolo_target(
            label_path,
            width,
            height
        )

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
    for i, name in enumerate(CLASS_NAMES, start=1):
        print(f"  Faster R-CNN label {i}: {name}")


if __name__ == "__main__":
    main()