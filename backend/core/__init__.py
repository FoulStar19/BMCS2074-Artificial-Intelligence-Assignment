# backend/core/__init__.py
"""
Core modules for the backend
"""

from .model_manager import ModelManager
from .video_processor_service import VideoProcessingService
from .session_manager import SessionManager

__all__ = [
    'ModelManager',
    'VideoProcessingService',
    'SessionManager'
]