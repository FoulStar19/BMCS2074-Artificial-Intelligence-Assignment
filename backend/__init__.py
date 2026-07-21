"""
Backend module for Traffic AI Detection System
"""

from .detection.yolo_detector import YOLODetector
from .detection.cnn_detector import CNNDetector
from .tracking.speed_estimator import SpeedEstimator
from .tracking.tracker import VehicleTracker
from .core.model_manager import ModelManager
from .core.video_processor_service import VideoProcessingService
from .core.session_manager import SessionManager
from .analytics.report_generator import ReportGenerator
from .utils.helpers import (
    load_dataset_config,
    get_model_versions,
    get_class_colors,
    calculate_traffic_density,
    calculate_speed_statistics,
    draw_vehicle_trails,
    normalize_box,
    denormalize_box,
    calculate_iou,
    get_video_properties
)
from .ui import components as ui_components

__all__ = [
    'YOLODetector',
    'CNNDetector',
    'SpeedEstimator',
    'VehicleTracker',
    'ModelManager',
    'VideoProcessingService',
    'SessionManager',
    'ReportGenerator',
    'ui_components',
    'load_dataset_config',
    'get_model_versions',
    'get_class_colors',
    'calculate_traffic_density',
    'calculate_speed_statistics',
    'draw_vehicle_trails',
    'normalize_box',
    'denormalize_box',
    'calculate_iou',
    'get_video_properties'
]