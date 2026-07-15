import os
import shutil
import random

# Base path 
base_path = r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\dataset"

# Input folders
images_dir = os.path.join(base_path, "images")
labels_dir = os.path.join(base_path, "labels")

# Output folders
train_img_dir = os.path.join(base_path, "Training", "images")
train_lbl_dir = os.path.join(base_path, "Training", "labels")
val_img_dir = os.path.join(base_path, "Validation", "images")
val_lbl_dir = os.path.join(base_path, "Validation", "labels")

# Check if directories exist, otherwise raise an error
if not os.path.exists(images_dir):
    print(f"Error: The images directory does not exist: {images_dir}")
    exit(1)  # Exit the script if the directory doesn't exist
if not os.path.exists(labels_dir):
    print(f"Error: The labels directory does not exist: {labels_dir}")
    exit(1)

# Create output folders if they don't exist
for folder in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    os.makedirs(folder, exist_ok=True)

# Get list of image files
try:
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
except FileNotFoundError:
    print(f"Error: Could not find the images folder at: {images_dir}")
    exit(1)

# Shuffle image files for random split
random.shuffle(image_files)

# Split into 70% train and 15% val and 15% test
split_index = int(0.85 * len(image_files))
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# Function to copy images and labels
def copy_files(file_list, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    for img_file in file_list:
        base, _ = os.path.splitext(img_file)
        label_file = base + ".txt"

        # Full paths
        src_img = os.path.join(src_img_dir, img_file)
        dst_img = os.path.join(dst_img_dir, img_file)

        src_lbl = os.path.join(src_lbl_dir, label_file)
        dst_lbl = os.path.join(dst_lbl_dir, label_file)

        # Check if the image exists before copying
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        else:
            print(f"Warning: Image file not found: {src_img}")
        
        # Check if the label exists before copying
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            print(f"Warning: Label file not found: {src_lbl}")

# Do the copying
copy_files(train_files, images_dir, labels_dir, train_img_dir, train_lbl_dir)
copy_files(val_files, images_dir, labels_dir, val_img_dir, val_lbl_dir)

print("Dataset split complete!")
print(f"Training set: {len(train_files)} images")
print(f"Validation set: {len(val_files)} images")
