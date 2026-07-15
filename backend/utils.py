"""
Utility functions for traffic analysis
"""

import numpy as np
import cv2
import yaml
import os
from pathlib import Path


def load_dataset_config(config_path=None):
    """
    Load dataset configuration from YAML file
    
    Args:
        config_path: Path to dataset.yaml file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Check common locations
        possible_paths = [
            Path("models/yolo/dataset.yaml"),
            Path("dataset.yaml"),
            Path("config/dataset.yaml"),
            Path("runs/train/dataset.yaml")
        ]
        
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
    
    # Return default configuration
    return {
        'nc': 5,
        'names': {
            0: 'car',
            1: 'truck',
            2: 'bus',
            3: 'motorcycle',
            4: 'bicycle'
        },
        'colors': {
            0: [255, 0, 0],
            1: [0, 255, 0],
            2: [0, 0, 255],
            3: [255, 255, 0],
            4: [255, 0, 255]
        }
    }


def get_model_versions(runs_dir="runs"):
    """
    Get all available model versions from runs directory
    
    Args:
        runs_dir: Path to runs directory
        
    Returns:
        Dictionary of version names and paths
    """
    models = {}
    runs_path = Path(runs_dir)
    
    if runs_path.exists():
        for run_dir in runs_path.iterdir():
            if run_dir.is_dir():
                weights_dir = run_dir / "weights"
                if weights_dir.exists():
                    for weight_file in weights_dir.glob("*.pt"):
                        version_name = f"{run_dir.name}/{weight_file.stem}"
                        models[version_name] = str(weight_file)
    
    return models


def get_class_colors(config):
    """
    Get class colors from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping class IDs to colors
    """
    colors = config.get('colors', {})
    
    # Convert to tuple format if needed
    for class_id, color in colors.items():
        if isinstance(color, list):
            colors[class_id] = tuple(color)
    
    return colors


def calculate_traffic_density(detections, frame_shape):
    """
    Calculate traffic density based on detections
    
    Args:
        detections: List of detection dictionaries
        frame_shape: Shape of the frame (height, width)
        
    Returns:
        Traffic density as percentage
    """
    if not detections:
        return 0.0
    
    # Calculate total area occupied by vehicles
    vehicle_area = 0
    for det in detections:
        bbox = det['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        vehicle_area += width * height
    
    total_area = frame_shape[0] * frame_shape[1]
    density = (vehicle_area / total_area) * 100
    
    return min(density, 100.0)


def calculate_speed_statistics(detections):
    """
    Calculate speed statistics from detections
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Dictionary with speed statistics
    """
    speeds = [det.get('speed', 0) for det in detections if det.get('speed', 0) > 0]
    
    if not speeds:
        return {
            'avg': 0,
            'max': 0,
            'min': 0,
            'std': 0,
            'count': 0
        }
    
    return {
        'avg': np.mean(speeds),
        'max': np.max(speeds),
        'min': np.min(speeds),
        'std': np.std(speeds),
        'count': len(speeds)
    }


def draw_vehicle_trails(frame, tracks, trail_length=20):
    """
    Draw vehicle tracking trails on frame
    
    Args:
        frame: Input frame
        tracks: Dictionary of track histories
        trail_length: Number of points to draw
        
    Returns:
        Annotated frame
    """
    annotated_frame = frame.copy()
    
    for track_id, history in tracks.items():
        if len(history) < 2:
            continue
        
        # Get recent points
        points = list(history)[-trail_length:]
        
        # Draw trail
        for i in range(1, len(points)):
            pt1 = (int(points[i-1][0]), int(points[i-1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            alpha = i / len(points)
            
            # Fade color based on position in trail
            color = (int(0 * alpha), int(255 * alpha), int(0 * alpha))
            cv2.line(annotated_frame, pt1, pt2, color, 2)
        
        # Draw current position
        current_pos = points[-1]
        cv2.circle(annotated_frame, 
                  (int(current_pos[0]), int(current_pos[1])), 
                  5, (0, 255, 0), -1)
    
    return annotated_frame