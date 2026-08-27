from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou


# ============================================================
# CONFIGURATION
# ============================================================

VAL_DIR = Path("dataset/Validation_new")

CHECKPOINT = Path(
    "model/faster_rcnn/saved_model/best_faster_rcnn.pth"
)

CLASS_NAMES = [
    "car",
    "truck",
    "bus",
    "motorcycle"
]

# Faster R-CNN:
# 0 = background
# 1 = car
# 2 = truck
# 3 = bus
# 4 = motorcycle

NUM_CLASSES = len(CLASS_NAMES) + 1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------------------
# IMPORTANT
# First run: only evaluate 5 validation images.
#
# After the test works correctly, change:
#
# TEST_LIMIT = None
#
# to evaluate all 154 validation images.
# ------------------------------------------------------------

TEST_LIMIT = None

CONFIDENCE_THRESHOLD = 0.5

IOU_THRESHOLD = 0.5


# ============================================================
# DATASET
# ============================================================

class YOLODetectionDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = Path(root_dir)

        self.image_dir = self.root_dir / "images"
        self.label_dir = self.root_dir / "labels"

        valid_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        }

        self.images = sorted(
            [
                p
                for p in self.image_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in valid_extensions
            ]
        )

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        label_path = (
            self.label_dir
            / f"{image_path.stem}.txt"
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        width, height = image.size

        boxes = []
        labels = []

        # ----------------------------------------------------
        # Load YOLO labels
        # ----------------------------------------------------

        if label_path.exists():

            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as f:

                for line_number, line in enumerate(
                    f,
                    start=1
                ):

                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split()

                    if len(parts) != 5:

                        print(
                            f"WARNING: Invalid label "
                            f"in {label_path}, "
                            f"line {line_number}"
                        )

                        continue

                    try:

                        class_id = int(parts[0])

                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        box_width = float(parts[3])
                        box_height = float(parts[4])

                    except ValueError:

                        print(
                            f"WARNING: Invalid values "
                            f"in {label_path}, "
                            f"line {line_number}"
                        )

                        continue

                    # ------------------------------------------------
                    # Validate class
                    # ------------------------------------------------

                    if (
                        class_id < 0
                        or class_id >= len(CLASS_NAMES)
                    ):

                        print(
                            f"WARNING: Invalid class ID "
                            f"{class_id} in {label_path}"
                        )

                        continue

                    # ------------------------------------------------
                    # YOLO normalized coordinates
                    # ->
                    # Pixel coordinates
                    # ------------------------------------------------

                    x_center *= width
                    y_center *= height

                    box_width *= width
                    box_height *= height

                    xmin = (
                        x_center
                        - box_width / 2
                    )

                    ymin = (
                        y_center
                        - box_height / 2
                    )

                    xmax = (
                        x_center
                        + box_width / 2
                    )

                    ymax = (
                        y_center
                        + box_height / 2
                    )

                    # ------------------------------------------------
                    # Clip boxes to image boundaries
                    # ------------------------------------------------

                    xmin = max(
                        0,
                        min(xmin, width)
                    )

                    ymin = max(
                        0,
                        min(ymin, height)
                    )

                    xmax = max(
                        0,
                        min(xmax, width)
                    )

                    ymax = max(
                        0,
                        min(ymax, height)
                    )

                    # ------------------------------------------------
                    # Ignore invalid boxes
                    # ------------------------------------------------

                    if (
                        xmax <= xmin
                        or ymax <= ymin
                    ):
                        continue

                    boxes.append(
                        [
                            xmin,
                            ymin,
                            xmax,
                            ymax
                        ]
                    )

                    # ------------------------------------------------
                    # Faster R-CNN uses:
                    #
                    # 0 = background
                    #
                    # Therefore:
                    #
                    # YOLO 0 -> Faster R-CNN 1
                    # YOLO 1 -> Faster R-CNN 2
                    # YOLO 2 -> Faster R-CNN 3
                    # YOLO 3 -> Faster R-CNN 4
                    # ------------------------------------------------

                    labels.append(
                        class_id + 1
                    )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        if boxes:

            boxes = torch.tensor(
                boxes,
                dtype=torch.float32
            )

            labels = torch.tensor(
                labels,
                dtype=torch.int64
            )

        else:

            boxes = torch.zeros(
                (0, 4),
                dtype=torch.float32
            )

            labels = torch.zeros(
                (0,),
                dtype=torch.int64
            )

        # ----------------------------------------------------
        # Area
        # ----------------------------------------------------

        if len(boxes) > 0:

            area = (
                (boxes[:, 2] - boxes[:, 0])
                *
                (boxes[:, 3] - boxes[:, 1])
            )

        else:

            area = torch.zeros(
                (0,),
                dtype=torch.float32
            )

        iscrowd = torch.zeros(
            (len(boxes),),
            dtype=torch.int64
        )

        target = {

            "boxes": boxes,

            "labels": labels,

            "image_id": torch.tensor(
                [index],
                dtype=torch.int64
            ),

            "area": area,

            "iscrowd": iscrowd
        }

        # ----------------------------------------------------
        # Convert image to tensor
        # ----------------------------------------------------

        image = TF.to_tensor(image)

        return image, target


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    return tuple(zip(*batch))


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )

    in_features = (
        model
        .roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features,
            NUM_CLASSES
        )
    )

    return model


# ============================================================
# MATCH PREDICTIONS WITH GROUND TRUTH
# ============================================================

def calculate_image_metrics(
    pred_boxes,
    pred_labels,
    pred_scores,
    gt_boxes,
    gt_labels
):

    # --------------------------------------------------------
    # Confidence filtering
    # --------------------------------------------------------

    keep = (
        pred_scores
        >= CONFIDENCE_THRESHOLD
    )

    pred_boxes = pred_boxes[keep]

    pred_labels = pred_labels[keep]

    pred_scores = pred_scores[keep]

    # --------------------------------------------------------
    # Sort predictions by confidence
    # --------------------------------------------------------

    if len(pred_scores) > 0:

        order = torch.argsort(
            pred_scores,
            descending=True
        )

        pred_boxes = pred_boxes[order]

        pred_labels = pred_labels[order]

    matched_gt = set()

    tp = 0
    fp = 0

    matched_ious = []

    # --------------------------------------------------------
    # Match each prediction to the best ground-truth box
    # of the SAME class.
    # --------------------------------------------------------

    for i in range(
        len(pred_boxes)
    ):

        pred_box = pred_boxes[i]

        pred_label = (
            pred_labels[i].item()
        )

        best_iou = 0.0

        best_gt_index = None

        for j in range(
            len(gt_boxes)
        ):

            if j in matched_gt:
                continue

            gt_label = (
                gt_labels[j].item()
            )

            # Classes must match
            if pred_label != gt_label:
                continue

            iou = box_iou(
                pred_box.unsqueeze(0),
                gt_boxes[j].unsqueeze(0)
            )[0, 0].item()

            if iou > best_iou:

                best_iou = iou

                best_gt_index = j

        # ----------------------------------------------------
        # True Positive
        # ----------------------------------------------------

        if (
            best_gt_index is not None
            and best_iou >= IOU_THRESHOLD
        ):

            tp += 1

            matched_gt.add(
                best_gt_index
            )

            matched_ious.append(
                best_iou
            )

        # ----------------------------------------------------
        # False Positive
        # ----------------------------------------------------

        else:

            fp += 1

    # --------------------------------------------------------
    # False negatives = GT boxes not matched
    # --------------------------------------------------------

    fn = (
        len(gt_boxes)
        - len(matched_gt)
    )

    return (
        tp,
        fp,
        fn,
        matched_ious
    )

   
# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "FASTER R-CNN VALIDATION EVALUATION"
    )

    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD}"
    )

    print(
        f"IoU threshold: "
        f"{IOU_THRESHOLD}"
    )

    print(
        f"Test limit: {TEST_LIMIT}"
    )

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = YOLODetectionDataset(
        VAL_DIR
    )

    print(
        f"Validation images available: "
        f"{len(dataset)}"
    )

    if len(dataset) == 0:

        raise RuntimeError(
            "No validation images found."
        )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()

    print(
        "Loading Faster R-CNN checkpoint..."
    )

    model = create_model()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(DEVICE)

    model.eval()

    print(
        "Checkpoint loaded successfully."
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Checkpoint train loss: "
        f"{checkpoint.get('train_loss', 'unknown')}"
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "STARTING EVALUATION"
    )

    print("=" * 70)

    total_tp = 0

    total_fp = 0

    total_fn = 0

    all_ious = []

    # --------------------------------------------------------
    # Per-class statistics
    # --------------------------------------------------------

    class_stats = {

        class_id: {

            "tp": 0,

            "fp": 0,

            "fn": 0,

            "ious": []

        }

        for class_id in range(
            1,
            NUM_CLASSES
        )
    }

    processed = 0

    # --------------------------------------------------------
    # No gradients needed
    # --------------------------------------------------------

    with torch.no_grad():

        for images, targets in loader:

            image = images[0].to(
                DEVICE
            )

            target = targets[0]

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            output = model(
                [image]
            )[0]

            pred_boxes = (
                output["boxes"]
                .cpu()
            )

            pred_labels = (
                output["labels"]
                .cpu()
            )

            pred_scores = (
                output["scores"]
                .cpu()
            )

            gt_boxes = (
                target["boxes"]
            )

            gt_labels = (
                target["labels"]
            )

            # ------------------------------------------------
            # Calculate image metrics
            # ------------------------------------------------

            (
                tp,
                fp,
                fn,
                image_ious
            ) = calculate_image_metrics(
                pred_boxes,
                pred_labels,
                pred_scores,
                gt_boxes,
                gt_labels
            )

            total_tp += tp

            total_fp += fp

            total_fn += fn

            all_ious.extend(
                image_ious
            )

            # ------------------------------------------------
            # Per-class metrics
            # ------------------------------------------------

            for class_id in range(
                1,
                NUM_CLASSES
            ):

                class_pred_mask = (
                    pred_labels
                    == class_id
                )

                class_gt_mask = (
                    gt_labels
                    == class_id
                )

                (
                    c_tp,
                    c_fp,
                    c_fn,
                    c_ious
                ) = calculate_image_metrics(
                    pred_boxes[
                        class_pred_mask
                    ],
                    pred_labels[
                        class_pred_mask
                    ],
                    pred_scores[
                        class_pred_mask
                    ],
                    gt_boxes[
                        class_gt_mask
                    ],
                    gt_labels[
                        class_gt_mask
                    ]
                )

                class_stats[
                    class_id
                ]["tp"] += c_tp

                class_stats[
                    class_id
                ]["fp"] += c_fp

                class_stats[
                    class_id
                ]["fn"] += c_fn

                class_stats[
                    class_id
                ]["ious"].extend(
                    c_ious
                )

            processed += 1

            pred_count = (
                pred_scores
                >= CONFIDENCE_THRESHOLD
            ).sum().item()

            print(
                f"Image {processed}: "
                f"GT={len(gt_boxes)}, "
                f"Pred={pred_count}, "
                f"TP={tp}, "
                f"FP={fp}, "
                f"FN={fn}"
            )

            # ------------------------------------------------
            # Stop after TEST_LIMIT
            # ------------------------------------------------

            if (
                TEST_LIMIT is not None
                and processed >= TEST_LIMIT
            ):

                break

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    precision = (

        total_tp
        /
        (total_tp + total_fp)

        if (
            total_tp + total_fp
        ) > 0

        else 0.0
    )

    recall = (

        total_tp
        /
        (total_tp + total_fn)

        if (
            total_tp + total_fn
        ) > 0

        else 0.0
    )

    f1 = (

        2
        * precision
        * recall
        /
        (precision + recall)

        if (
            precision + recall
        ) > 0

        else 0.0
    )

    mean_iou = (

        sum(all_ious)
        /
        len(all_ious)

        if all_ious

        else 0.0
    )

    # ========================================================
    # PRINT OVERALL RESULTS
    # ========================================================

    print()

    print("=" * 70)

    print(
        "OVERALL RESULTS"
    )

    print("=" * 70)

    print(
        f"Images evaluated : "
        f"{processed}"
    )

    print(
        f"True Positives    : "
        f"{total_tp}"
    )

    print(
        f"False Positives   : "
        f"{total_fp}"
    )

    print(
        f"False Negatives   : "
        f"{total_fn}"
    )

    print()

    print(
        f"Precision         : "
        f"{precision:.4f}"
    )

    print(
        f"Recall            : "
        f"{recall:.4f}"
    )

    print(
        f"F1-score          : "
        f"{f1:.4f}"
    )

    print(
        f"Mean IoU          : "
        f"{mean_iou:.4f}"
    )

    # ========================================================
    # PER-CLASS RESULTS
    # ========================================================

    print()

    print("=" * 70)

    print(
        "PER-CLASS RESULTS"
    )

    print("=" * 70)

    for class_id, name in enumerate(
        CLASS_NAMES,
        start=1
    ):

        stats = (
            class_stats[class_id]
        )

        tp = stats["tp"]

        fp = stats["fp"]

        fn = stats["fn"]

        class_precision = (

            tp
            /
            (tp + fp)

            if (
                tp + fp
            ) > 0

            else 0.0
        )

        class_recall = (

            tp
            /
            (tp + fn)

            if (
                tp + fn
            ) > 0

            else 0.0
        )

        class_f1 = (

            2
            * class_precision
            * class_recall
            /
            (
                class_precision
                + class_recall
            )

            if (
                class_precision
                + class_recall
            ) > 0

            else 0.0
        )

        class_iou = (

            sum(stats["ious"])
            /
            len(stats["ious"])

            if stats["ious"]

            else 0.0
        )

        print()

        print(name)

        print(
            f"  TP        : {tp}"
        )

        print(
            f"  FP        : {fp}"
        )

        print(
            f"  FN        : {fn}"
        )

        print(
            f"  Precision : "
            f"{class_precision:.4f}"
        )

        print(
            f"  Recall    : "
            f"{class_recall:.4f}"
        )

        print(
            f"  F1-score  : "
            f"{class_f1:.4f}"
        )

        print(
            f"  Mean IoU  : "
            f"{class_iou:.4f}"
        )

    # ========================================================
    # FINISHED
    # ========================================================

    print()

    print("=" * 70)

    print(
        "EVALUATION TEST COMPLETED"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()