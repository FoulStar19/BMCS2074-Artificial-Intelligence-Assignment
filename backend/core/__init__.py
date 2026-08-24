"""
Core backend modules.
"""

from backend.core.model_manager import ModelManager
from backend.core.session_manager import SessionManager
from backend.core.video_processor_service import VideoProcessingService

__all__ = [
    'ModelManager',
    'SessionManager',
    'VideoProcessingService',
]