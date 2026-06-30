"""
Detection module for vehicle detection models
"""

from .cnn_detector import CNNDetector
from .yolo_detector import YOLODetector

__all__ = ['CNNDetector', 'YOLODetector']