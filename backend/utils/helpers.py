"""
Utility functions for traffic analysis.
Optimized with efficient computations and error handling.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import yaml


def load_dataset_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load dataset configuration from YAML file."""
    if config_path is None:
        possible_paths = [
            Path("config/dataset.yaml"),
            Path("model/yolo/dataset.yaml"),
            Path("dataset.yaml"),
            Path("runs/train/dataset.yaml"),
            Path(__file__).parent.parent.parent / "config" / "dataset.yaml",
            Path(__file__).parent.parent.parent / "model" / "yolo" / "dataset.yaml",
        ]

        for path in possible_paths:
            if path.exists():
                config_path = path
                break

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")

    # Default configuration
    return {
        'nc': 5,
        'names': {0: 'car', 1: 'truck', 2: 'bus', 3: 'motorcycle', 4: 'bicycle'},
        'colors': {
            0: [255, 0, 0],
            1: [0, 255, 0],
            2: [0, 0, 255],
            3: [255, 255, 0],
            4: [255, 0, 255]
        }
    }


def get_model_versions(runs_dir: str = "runs") -> Dict[str, str]:
    """Get all available model versions from runs directory."""
    models = {}
    runs_path = Path(runs_dir)

    if not runs_path.exists():
        alt_paths = [
            Path("model/yolo/runs"),
            Path("model/runs"),
            Path("yolo/runs"),
            Path(__file__).parent.parent.parent / "model" / "yolo" / "runs",
        ]
        for path in alt_paths:
            if path.exists():
                runs_path = path
                break

    if runs_path.exists():
        for run_dir in runs_path.iterdir():
            if run_dir.is_dir():
                weights_dir = run_dir / "weights"
                if weights_dir.exists():
                    for weight_file in weights_dir.glob("*.pt"):
                        version_name = f"{run_dir.name}/{weight_file.stem}"
                        models[version_name] = str(weight_file)

    return models


def get_class_colors(config: Dict[str, Any]) -> Dict[int, Tuple[int, int, int]]:
    """Get class colors from configuration."""
    colors = config.get('colors', {})

    for class_id, color in colors.items():
        if isinstance(color, list):
            colors[class_id] = tuple(color)

    default_colors = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 255, 0),
        4: (255, 0, 255)
    }

    for class_id in range(5):
        if class_id not in colors:
            colors[class_id] = default_colors.get(class_id, (0, 255, 0))

    return colors


def calculate_traffic_density(detections: List[Dict], frame_shape: Tuple[int, int]) -> float:
    """Calculate traffic density based on detections."""
    if not detections:
        return 0.0

    vehicle_area = 0
    for det in detections:
        bbox = det.get('bbox', [0, 0, 100, 100])
        width = max(0, bbox[2] - bbox[0])
        height = max(0, bbox[3] - bbox[1])
        vehicle_area += width * height

    total_area = frame_shape[0] * frame_shape[1]
    density = (vehicle_area / total_area) * 100

    return min(density, 100.0)


def calculate_speed_statistics(detections: List[Dict]) -> Dict[str, Any]:
    """Calculate speed statistics from detections."""
    speeds = [det.get('speed', 0) for det in detections if det.get('speed', 0) > 0]

    if not speeds:
        return {
            'avg': 0.0,
            'max': 0.0,
            'min': 0.0,
            'std': 0.0,
            'count': 0,
            'median': 0.0
        }

    return {
        'avg': float(np.mean(speeds)),
        'max': float(np.max(speeds)),
        'min': float(np.min(speeds)),
        'std': float(np.std(speeds)),
        'count': len(speeds),
        'median': float(np.median(speeds))
    }


def draw_vehicle_trails(frame: np.ndarray, tracks: Dict[int, List[Tuple]],
                        trail_length: int = 20) -> np.ndarray:
    """Draw vehicle tracking trails on frame."""
    annotated_frame = frame.copy()

    for track_id, history in tracks.items():
        if len(history) < 2:
            continue

        points = list(history)[-trail_length:]

        for i in range(1, len(points)):
            pt1 = (int(points[i-1][0]), int(points[i-1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            alpha = i / len(points)
            color = (int(0 * alpha), int(255 * alpha), int(0 * alpha))
            cv2.line(annotated_frame, pt1, pt2, color, 2)

        current_pos = points[-1]
        cv2.circle(annotated_frame, (int(current_pos[0]), int(current_pos[1])), 5, (0, 255, 0), -1)

        cv2.putText(annotated_frame, f"ID:{track_id}",
                   (int(current_pos[0]) + 10, int(current_pos[1])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return annotated_frame


def normalize_box(box: List[int], frame_shape: Tuple[int, int]) -> List[float]:
    """Convert (x1, y1, x2, y2) to normalized (xc, yc, w, h)."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return [xc, yc, bw, bh]


def denormalize_box(xc: float, yc: float, bw: float, bh: float,
                    frame_shape: Tuple[int, int]) -> List[int]:
    """Convert normalized (xc, yc, w, h) to (x1, y1, x2, y2)."""
    h, w = frame_shape[:2]
    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)
    return [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """Calculate Intersection over Union of two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def get_video_properties(video_path: str) -> Dict[str, Any]:
    """Get video properties."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'error': 'Could not open video'}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    properties = {
        'fps': fps,
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': total_frames,
        'duration': int(total_frames / fps) if fps > 0 else 0
    }

    cap.release()
    return properties