"""
Backend module for Traffic AI Detection System
"""

from .detection.yolo_detector import YOLODetector
from .detection.cnn_detector import CNNDetector
from .speed_estimator import SpeedEstimator
from .video_processor import VideoProcessor
from .utils import (
    load_dataset_config,
    get_model_versions,
    get_class_colors,
    calculate_traffic_density,
    calculate_speed_statistics,
    draw_vehicle_trails
)

__all__ = [
    'YOLODetector',
    'CNNDetector',
    'SpeedEstimator',
    'VideoProcessor',
    'load_dataset_config',
    'get_model_versions',
    'get_class_colors',
    'calculate_traffic_density',
    'calculate_speed_statistics',
    'draw_vehicle_trails'
]