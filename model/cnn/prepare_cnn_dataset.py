"""
Prepare a 4-class CNN classification dataset from the YOLO-format dataset.

Classes:
0 - car
1 - truck
2 - bus
3 - motorcycle

The CNN dataset is created by cropping each annotated vehicle
from the YOLO bounding boxes.

Input:
    dataset/Training_new/images
    dataset/Training_new/labels
    dataset/Validation_new/images
    dataset/Validation_new/labels

Output:
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

The original YOLO dataset is NOT modified.
"""

from pathlib import Path
import shutil

import cv2


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset"

TRAIN_IMAGES = DATASET_DIR / "Training_new" / "images"
TRAIN_LABELS = DATASET_DIR / "Training_new" / "labels"

VAL_IMAGES = DATASET_DIR / "Validation_new" / "images"
VAL_LABELS = DATASET_DIR / "Validation_new" / "labels"

# Final CNN dataset
OUTPUT_DIR = DATASET_DIR / "cnn_4class"


# ============================================================
# CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "motorcycle",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


# ============================================================
# CROP SETTINGS
# ============================================================

# Add a small amount of context around each vehicle.
# 0.10 = 10% extra width/height around bounding box.
CROP_MARGIN = 0.10

# Ignore extremely small crops.
MIN_CROP_WIDTH = 20
MIN_CROP_HEIGHT = 20


# ============================================================
# READ YOLO LABELS
# ============================================================

def read_labels(label_path, image_width, image_height):
    """
    Read YOLO-format labels and convert them to pixel bounding boxes.

    YOLO format:
        class_id center_x center_y width height

    All coordinates are normalized between 0 and 1.

    Returns:
        List of tuples:
        (class_id, x1, y1, x2, y2)
    """

    annotations = []

    if not label_path.exists():
        return annotations

    try:

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        for line in lines:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            try:

                class_id = int(parts[0])

                cx = float(parts[1])
                cy = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])

            except ValueError:
                continue

            # Only use the four final CNN classes
            if class_id not in CLASS_NAMES:
                continue

            # Convert normalized YOLO coordinates
            # into pixel coordinates.

            x1 = int(
                (cx - box_width / 2)
                * image_width
            )

            y1 = int(
                (cy - box_height / 2)
                * image_height
            )

            x2 = int(
                (cx + box_width / 2)
                * image_width
            )

            y2 = int(
                (cy + box_height / 2)
                * image_height
            )

            # ------------------------------------------------
            # Add small margin around vehicle
            # ------------------------------------------------

            width = x2 - x1
            height = y2 - y1

            margin_x = int(
                width * CROP_MARGIN
            )

            margin_y = int(
                height * CROP_MARGIN
            )

            x1 -= margin_x
            y1 -= margin_y
            x2 += margin_x
            y2 += margin_y

            # ------------------------------------------------
            # Keep coordinates inside image
            # ------------------------------------------------

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(
                image_width,
                x2
            )

            y2 = min(
                image_height,
                y2
            )

            # ------------------------------------------------
            # Check crop size
            # ------------------------------------------------

            crop_width = x2 - x1
            crop_height = y2 - y1

            if (
                crop_width < MIN_CROP_WIDTH
                or crop_height < MIN_CROP_HEIGHT
            ):
                continue

            annotations.append(
                (
                    class_id,
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

    except Exception as error:

        print(
            f"Warning: Could not read "
            f"{label_path}: {error}"
        )

    return annotations


# ============================================================
# PREPARE OUTPUT DIRECTORIES
# ============================================================

def create_output_directories():

    if OUTPUT_DIR.exists():

        print(
            "Removing previous CNN 4-class dataset..."
        )

        try:

            shutil.rmtree(
                OUTPUT_DIR
            )

        except PermissionError:

            print()
            print(
                "ERROR: Cannot remove previous CNN dataset."
            )

            print(
                "Please close File Explorer, "
                "OneDrive or any program using:"
            )

            print(
                OUTPUT_DIR
            )

            raise

    for split in ["train", "val"]:

        for class_name in CLASS_NAMES.values():

            directory = (
                OUTPUT_DIR
                / split
                / class_name
            )

            directory.mkdir(
                parents=True,
                exist_ok=True
            )


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    split_name,
    image_dir,
    label_dir,
    output_split
):

    print()
    print("=" * 70)
    print(
        f"PROCESSING {split_name.upper()} DATASET"
    )
    print("=" * 70)

    if not image_dir.exists():

        raise FileNotFoundError(
            f"Image directory not found:\n"
            f"{image_dir}"
        )

    if not label_dir.exists():

        raise FileNotFoundError(
            f"Label directory not found:\n"
            f"{label_dir}"
        )

    # Count vehicle crops per class
    counters = {
        class_id: 0
        for class_id in CLASS_NAMES
    }

    images_processed = 0
    images_skipped = 0
    total_crops = 0

    image_files = sorted(
        [
            file
            for file in image_dir.iterdir()
            if (
                file.is_file()
                and file.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )

    print(
        f"Images found: {len(image_files)}"
    )

    # ========================================================
    # PROCESS EACH IMAGE
    # ========================================================

    for image_path in image_files:

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"Warning: Could not read image: "
                f"{image_path.name}"
            )

            images_skipped += 1
            continue

        image_height, image_width = (
            image.shape[:2]
        )

        # ----------------------------------------------------
        # Read bounding boxes
        # ----------------------------------------------------

        annotations = read_labels(
            label_path,
            image_width,
            image_height
        )

        if not annotations:

            images_skipped += 1
            continue

        # ----------------------------------------------------
        # Create one CNN sample for each vehicle
        # ----------------------------------------------------

        object_index = 0

        for (
            class_id,
            x1,
            y1,
            x2,
            y2,
        ) in annotations:

            class_name = CLASS_NAMES[
                class_id
            ]

            # Crop vehicle
            crop = image[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            # ------------------------------------------------
            # Unique filename
            # ------------------------------------------------

            output_filename = (
                f"{image_path.stem}"
                f"_object_{object_index}"
                f"_{class_name}.jpg"
            )

            output_path = (
                OUTPUT_DIR
                / output_split
                / class_name
                / output_filename
            )

            success = cv2.imwrite(
                str(output_path),
                crop
            )

            if success:

                counters[class_id] += 1
                total_crops += 1

            object_index += 1

        images_processed += 1

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("Vehicle crop distribution:")

    for class_id, class_name in CLASS_NAMES.items():

        print(
            f"  {class_id}: "
            f"{class_name:<12} "
            f"{counters[class_id]} crops"
        )

    print()
    print(
        f"Images processed: {images_processed}"
    )

    print(
        f"Images skipped:   {images_skipped}"
    )

    print(
        f"Total vehicle crops: {total_crops}"
    )

    return counters


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("4-CLASS CNN DATASET PREPARATION")
    print("=" * 70)

    print()
    print("Final CNN classes:")

    for class_id, class_name in CLASS_NAMES.items():

        print(
            f"  {class_id}: {class_name}"
        )

    print()
    print("Input dataset:")

    print(
        f"  Training images : {TRAIN_IMAGES}"
    )

    print(
        f"  Training labels : {TRAIN_LABELS}"
    )

    print(
        f"  Validation images : {VAL_IMAGES}"
    )

    print(
        f"  Validation labels : {VAL_LABELS}"
    )

    print()
    print("Output:")

    print(
        f"  {OUTPUT_DIR}"
    )

    # ========================================================
    # CHECK INPUT DIRECTORIES
    # ========================================================

    required_paths = [
        TRAIN_IMAGES,
        TRAIN_LABELS,
        VAL_IMAGES,
        VAL_LABELS,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired path not found:\n"
                f"{path}"
            )

    # ========================================================
    # CREATE OUTPUT FOLDERS
    # ========================================================

    create_output_directories()

    # ========================================================
    # PROCESS TRAINING
    # ========================================================

    train_counts = process_split(
        "training",
        TRAIN_IMAGES,
        TRAIN_LABELS,
        "train"
    )

    # ========================================================
    # PROCESS VALIDATION
    # ========================================================

    val_counts = process_split(
        "validation",
        VAL_IMAGES,
        VAL_LABELS,
        "val"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL CNN DATASET CHECK")
    print("=" * 70)

    total_train = 0
    total_val = 0

    print()
    print("Training:")

    for class_id, class_name in CLASS_NAMES.items():

        count = train_counts[class_id]

        total_train += count

        print(
            f"  {class_name:<12}: {count}"
        )

    print()
    print("Validation:")

    for class_id, class_name in CLASS_NAMES.items():

        count = val_counts[class_id]

        total_val += count

        print(
            f"  {class_name:<12}: {count}"
        )

    print()
    print(
        f"Total training crops:   {total_train}"
    )

    print(
        f"Total validation crops: {total_val}"
    )

    print(
        f"Total CNN samples:      "
        f"{total_train + total_val}"
    )

    # ========================================================
    # CHECK FOR EMPTY CLASSES
    # ========================================================

    empty_classes = []

    for class_id, class_name in CLASS_NAMES.items():

        if (
            train_counts[class_id] == 0
            or val_counts[class_id] == 0
        ):

            empty_classes.append(
                class_name
            )

    print()

    if empty_classes:

        print(
            "WARNING: The following classes "
            "have zero samples in training "
            "or validation:"
        )

        for class_name in empty_classes:

            print(
                f"  - {class_name}"
            )

    else:

        print(
            "All four classes have training "
            "and validation samples."
        )

    print()
    print(
        "CNN DATASET CREATED SUCCESSFULLY"
    )

    print("=" * 70)

    print()
    print(
        "Original YOLO dataset remains unchanged."
    )

    print()
    print("CNN dataset location:")

    print(
        OUTPUT_DIR
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()