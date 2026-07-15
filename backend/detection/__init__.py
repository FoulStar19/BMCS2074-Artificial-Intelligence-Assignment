"""
Detection module for vehicle detection models
"""

from .yolo_detector import YOLODetector
from .cnn_detector import CNNDetector

__all__ = ['YOLODetector', 'CNNDetector']