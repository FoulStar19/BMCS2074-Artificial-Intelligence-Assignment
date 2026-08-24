# backend/detection/__init__.py
"""
Detection module for vehicle detection.
"""

from backend.detection.yolo_detector import YOLODetector
from backend.detection.cnn_detector import CNNDetector

__all__ = [
    'YOLODetector',
    'CNNDetector'
]