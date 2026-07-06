"""
Data preparation script for YOLO vehicle detection
"""

import os
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path
import random
import json


class VehicleDataPreparer:
    """Prepare dataset for YOLO training"""
    
    def __init__(self, data_dir='dataset', output_dir='dataset'):
        """
        Initialize data preparer
        
        Args:
            data_dir: Source directory with raw data (default: 'dataset')
            output_dir: Output directory for prepared dataset (default: 'dataset')
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'val'
        self.images_dir = 'images'
        self.labels_dir = 'labels'
        
    def create_directory_structure(self):
        """Create directory structure for YOLO dataset"""
        # Train directories
        (self.train_dir / self.images_dir).mkdir(parents=True, exist_ok=True)
        (self.train_dir / self.labels_dir).mkdir(parents=True, exist_ok=True)
        
        # Validation directories
        (self.val_dir / self.images_dir).mkdir(parents=True, exist_ok=True)
        (self.val_dir / self.labels_dir).mkdir(parents=True, exist_ok=True)
        
        print("✅ Directory structure created successfully")
        return self
    
    def check_existing_dataset(self):
        """Check if dataset already exists and has images"""
        train_images = list((self.train_dir / self.images_dir).glob('*.*'))
        val_images = list((self.val_dir / self.images_dir).glob('*.*'))
        
        if train_images or val_images:
            print(f"📊 Existing dataset found:")
            print(f"  - Training images: {len(train_images)}")
            print(f"  - Validation images: {len(val_images)}")
            return True
        return False
    
    def convert_bbox_to_yolo(self, bbox, img_width, img_height):
        """
        Convert bounding box to YOLO format
        
        Args:
            bbox: [x1, y1, x2, y2] in pixel coordinates
            img_width: Image width
            img_height: Image height
            
        Returns:
            [center_x, center_y, width, height] normalized
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are in correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Clamp to image boundaries
        x1 = max(0, min(x1, img_width))
        x2 = max(0, min(x2, img_width))
        y1 = max(0, min(y1, img_height))
        y2 = max(0, min(y2, img_height))
        
        # Calculate center and dimensions
        center_x = (x1 + x2) / 2.0 / img_width
        center_y = (y1 + y2) / 2.0 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        # Clamp to valid range
        center_x = max(0, min(center_x, 1.0))
        center_y = max(0, min(center_y, 1.0))
        width = max(0, min(width, 1.0))
        height = max(0, min(height, 1.0))
        
        return [center_x, center_y, width, height]
    
    def load_annotations(self, annotation_file):
        """
        Load annotations from file
        
        Args:
            annotation_file: Path to annotation file
            
        Returns:
            List of annotations
        """
        annotations = []
        
        if not os.path.exists(annotation_file):
            return annotations
            
        if annotation_file.endswith('.txt'):
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x1, y1, x2, y2 = map(float, parts[1:5])
                        annotations.append({
                            'class_id': class_id,
                            'bbox': [x1, y1, x2, y2]
                        })
        elif annotation_file.endswith('.json'):
            import json
            with open(annotation_file, 'r') as f:
                data = json.load(f)
                if 'annotations' in data:
                    for ann in data['annotations']:
                        bbox = ann.get('bbox', [])
                        if len(bbox) >= 4:
                            annotations.append({
                                'class_id': ann.get('category_id', 0),
                                'bbox': bbox[:4]
                            })
                
        return annotations
    
    def scan_for_data(self):
        """Scan for existing images and annotations"""
        # Look in multiple possible locations
        possible_dirs = [
            self.data_dir,
            self.data_dir / 'images',
            self.data_dir / 'labels',
            Path('./raw_data'),
            Path('./raw_data/images'),
            Path('./raw_data/labels'),
            Path('../raw_data'),
        ]
        
        image_files = []
        
        # Image extensions
        img_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        for dir_path in possible_dirs:
            if not dir_path.exists():
                continue
                
            # Find images
            for ext in img_extensions:
                image_files.extend(list(dir_path.glob(f'*{ext}')))
                # Also search subdirectories
                image_files.extend(list(dir_path.glob(f'**/*{ext}')))
        
        # Remove duplicates
        image_files = list(set(image_files))
        
        # Match images with annotations
        matched = []
        for img_path in image_files:
            # Look for corresponding annotation
            stem = img_path.stem
            ann_paths = [
                img_path.parent / f'{stem}.txt',
                img_path.parent / f'{stem}.json',
                self.data_dir / 'labels' / f'{stem}.txt',
                self.data_dir / 'labels' / f'{stem}.json',
                Path('./raw_data') / f'{stem}.txt',
                Path('./raw_data') / f'{stem}.json',
            ]
            
            found_ann = None
            for ann_path in ann_paths:
                if ann_path.exists():
                    found_ann = ann_path
                    break
            
            if found_ann:
                matched.append((img_path, found_ann))
            else:
                # If no annotation found, create dummy annotation
                print(f"⚠️ No annotation found for {img_path.name}, creating dummy...")
                dummy_ann = self.create_dummy_annotation(img_path)
                if dummy_ann:
                    matched.append((img_path, dummy_ann))
        
        return matched
    
    def create_dummy_annotation(self, img_path):
        """Create dummy annotation for an image"""
        # Read image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            return None
            
        h, w = img.shape[:2]
        
        # Create dummy annotation file
        ann_path = img_path.parent / f'{img_path.stem}.txt'
        
        # Create 1-3 dummy vehicle annotations
        num_vehicles = random.randint(1, 4)
        with open(ann_path, 'w') as f:
            for _ in range(num_vehicles):
                # Random vehicle position
                x1 = random.randint(50, w - 150)
                y1 = random.randint(50, h - 150)
                x2 = x1 + random.randint(80, 200)
                y2 = y1 + random.randint(60, 150)
                
                # Random class (0-4 for car, truck, bus, motorcycle, bicycle)
                class_id = random.randint(0, 4)
                
                # Convert to YOLO format
                yolo_bbox = self.convert_bbox_to_yolo([x1, y1, x2, y2], w, h)
                f.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
        
        return ann_path
    
    def prepare_dataset(self, class_names=None, test_size=0.2):
        """
        Prepare dataset for YOLO training
        
        Args:
            class_names: List of class names
            test_size: Validation split ratio
        """
        if class_names is None:
            class_names = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
        
        # Create directories
        self.create_directory_structure()
        
        # Check if dataset already exists
        if self.check_existing_dataset():
            response = input("Dataset already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Keeping existing dataset.")
                return self
        
        # Scan for data
        print("🔍 Scanning for data...")
        matched_data = self.scan_for_data()
        
        if not matched_data:
            print("⚠️ No data found! Creating dummy dataset...")
            self.create_dummy_dataset()
            matched_data = self.scan_for_data()
        
        if not matched_data:
            print("❌ Failed to create dataset. Please check your data directory.")
            return None
        
        print(f"📊 Found {len(matched_data)} image-annotation pairs")
        
        # Split data
        random.shuffle(matched_data)
        split_idx = int(len(matched_data) * (1 - test_size))
        train_data = matched_data[:split_idx]
        val_data = matched_data[split_idx:]
        
        print(f"  - Training: {len(train_data)}")
        print(f"  - Validation: {len(val_data)}")
        
        # Process training data
        self._process_data(train_data, self.train_dir)
        
        # Process validation data
        self._process_data(val_data, self.val_dir)
        
        print(f"\n✅ Dataset preparation completed!")
        print(f"📁 Dataset location: {self.output_dir}")
        print(f"📄 Your dataset.yaml file remains unchanged")
        
        return self
    
    def create_dummy_dataset(self, num_images=50):
        """Create dummy dataset for testing"""
        print(f"Creating {num_images} dummy images...")
        
        # Create raw_data directory if it doesn't exist
        raw_dir = Path('./raw_data')
        raw_dir.mkdir(exist_ok=True)
        
        for i in range(num_images):
            # Create a background - FIXED: values now within 0-255
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Add sky gradient - FIXED: values within 0-255
            for y in range(720):
                r = min(255, 135 + (y // 20) * 2)
                g = min(255, 200 + (y // 20) * 1)
                b = min(255, 235 + (y // 20) * 1)
                img[y, :] = (b, g, r)  # OpenCV uses BGR
            
            # Add road
            road_y = 450
            img[road_y:, :] = (80, 80, 80)
            img[road_y:road_y+5, :] = (255, 255, 255)
            
            # Add vehicles
            num_vehicles = random.randint(2, 6)
            
            for j in range(num_vehicles):
                x = random.randint(50, 1100)
                y = random.randint(400, 680)
                w = random.randint(60, 150)
                h = random.randint(40, 100)
                
                color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
                
                # Draw vehicle
                cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
                
                # Add windows
                cv2.rectangle(img, (x + 10, y + 8), (x + w//2 - 5, y + h//2 - 5), 
                            (200, 200, 200), -1)
                cv2.rectangle(img, (x + w//2 + 5, y + 8), (x + w - 10, y + h//2 - 5), 
                            (200, 200, 200), -1)
                
                # Add wheels
                cv2.circle(img, (x + 15, y + h - 5), 8, (30, 30, 30), -1)
                cv2.circle(img, (x + w - 15, y + h - 5), 8, (30, 30, 30), -1)
                
                class_id = random.randint(0, 4)
                
                # Save annotation
                ann_path = raw_dir / f'vehicle_{i:04d}.txt'
                with open(ann_path, 'a') as f:
                    cx = (x + w/2) / 1280
                    cy = (y + h/2) / 720
                    wn = w / 1280
                    hn = h / 720
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n")
            
            # Save image
            img_path = raw_dir / f'vehicle_{i:04d}.jpg'
            cv2.imwrite(str(img_path), img)
        
        print(f"✅ Created {num_images} dummy images in {raw_dir}")
    
    def _process_data(self, data_pairs, output_dir):
        """
        Process images and annotations for YOLO format
        
        Args:
            data_pairs: List of (image_path, annotation_path) tuples
            output_dir: Output directory
        """
        images_dir = output_dir / self.images_dir
        labels_dir = output_dir / self.labels_dir
        
        for img_path, ann_path in data_pairs:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"⚠️ Could not read image {img_path}")
                continue
                
            height, width = image.shape[:2]
            
            # Copy image to output directory
            img_name = img_path.name
            dst_img_path = images_dir / img_name
            shutil.copy2(str(img_path), str(dst_img_path))
            
            # Read annotations and convert to YOLO format
            annotations = self.load_annotations(str(ann_path))
            
            # Create YOLO label file
            label_name = img_path.stem + '.txt'
            label_path = labels_dir / label_name
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    class_id = ann['class_id']
                    bbox = ann['bbox']
                    
                    # Convert to YOLO format
                    yolo_bbox = self.convert_bbox_to_yolo(bbox, width, height)
                    
                    # Skip invalid bounding boxes
                    if yolo_bbox[2] <= 0 or yolo_bbox[3] <= 0:
                        continue
                    
                    # Write to file
                    f.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
            
            # Progress indicator
            print(f"  Processed: {img_name}")


def prepare_vehicle_dataset(data_dir='dataset', output_dir='dataset'):
    """
    Convenience function to prepare the vehicle dataset
    
    Args:
        data_dir: Source directory with raw data
        output_dir: Output directory for prepared dataset
    """
    preparer = VehicleDataPreparer(data_dir=data_dir, output_dir=output_dir)
    preparer.prepare_dataset()
    return preparer


def main():
    """Main function to prepare dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset for YOLO training')
    parser.add_argument('--data_dir', type=str, default='dataset',
                       help='Source directory with data')
    parser.add_argument('--output_dir', type=str, default='dataset',
                       help='Output directory for prepared dataset')
    parser.add_argument('--classes', type=str, default='car,truck,bus,motorcycle,bicycle',
                       help='Comma-separated class names')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--dummy', action='store_true',
                       help='Force creation of dummy dataset')
    
    args = parser.parse_args()
    
    # Parse class names
    class_names = [c.strip() for c in args.classes.split(',')]
    
    # Initialize preparer
    preparer = VehicleDataPreparer(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # If dummy flag is set, create dummy dataset
    if args.dummy:
        preparer.create_dummy_dataset()
    
    # Prepare dataset
    preparer.prepare_dataset(
        class_names=class_names,
        test_size=args.test_size
    )


if __name__ == "__main__":
    main()