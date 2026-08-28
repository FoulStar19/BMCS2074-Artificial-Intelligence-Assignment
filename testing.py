"""
Check for images with no matching label file in your training/validation sets.
Run from anywhere with Python (no special packages needed):
    python check_label_mismatch.py
"""

import os

base_path = r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\dataset"

for split in ["Training", "Validation"]:
    img_dir = os.path.join(base_path, split, "images")
    lbl_dir = os.path.join(base_path, split, "labels")

    if not os.path.exists(img_dir):
        print(f"{split}: images dir not found at {img_dir}")
        continue

    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    missing_labels = []
    empty_labels = []

    for img in images:
        base, _ = os.path.splitext(img)
        lbl_path = os.path.join(lbl_dir, base + ".txt")
        if not os.path.exists(lbl_path):
            missing_labels.append(img)
        elif os.path.getsize(lbl_path) == 0:
            empty_labels.append(img)

    print(f"\n=== {split} ===")
    print(f"Total images: {len(images)}")
    print(f"Images with NO label file (treated as background/no-object): {len(missing_labels)}")
    print(f"Images with EMPTY label file (also background/no-object): {len(empty_labels)}")
    pct_background = (len(missing_labels) + len(empty_labels)) / len(images) * 100 if images else 0
    print(f"=> {pct_background:.1f}% of {split} images have zero labeled objects")

    if missing_labels[:5]:
        print("Example missing-label images:", missing_labels[:5])