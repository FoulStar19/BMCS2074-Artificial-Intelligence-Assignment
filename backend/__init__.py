"""
Backend module for Traffic AI Detection System.
"""

from backend.core.model_manager import ModelManager
from backend.core.session_manager import SessionManager
from backend.core.video_processor_service import VideoProcessingService
from backend.detection.yolo_detector import YOLODetector
from backend.tracking.tracker import VehicleTracker
from backend.tracking.speed_estimator import SpeedEstimator
from backend.analytics.report_generator import ReportGenerator, display_metrics

__all__ = [
    'ModelManager',
    'SessionManager',
    'VideoProcessingService',
    'YOLODetector',
    'VehicleTracker',
    'SpeedEstimator',
    'ReportGenerator',
    'display_metrics',
]