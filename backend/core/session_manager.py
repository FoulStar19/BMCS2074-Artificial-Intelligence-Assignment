"""
Session State Manager - centralized session state management for Streamlit.
Optimized with type hints and efficient state management.
"""

import gc
from typing import Any, Optional

import streamlit as st


class SessionManager:
    """Manages Streamlit session state with typed getters/setters."""

    # Session keys
    PROCESSING = "processing"
    DETECTIONS_HISTORY = "detections_history"
    SPEED_HISTORY = "speed_history"
    DENSITY_HISTORY = "density_history"
    PROCESSED_VIDEO_PATH = "processed_video_path"
    DETECTOR = "detector"
    CURRENT_RESULTS = "current_results"
    IS_PROCESSING = "is_processing"
    VIDEO_PROCESSED = "video_processed"
    VIDEO_PATH = "video_path"
    CUDA_AVAILABLE = "cuda_available"
    MODEL_MANAGER = "model_manager"
    VIDEO_SERVICE = "video_service"
    VIDEO_BYTES_CACHE = "video_bytes_cache"  # (path, bytes)

    _defaults = {
        PROCESSING: False,
        DETECTIONS_HISTORY: [],
        SPEED_HISTORY: [],
        DENSITY_HISTORY: [],
        PROCESSED_VIDEO_PATH: None,
        DETECTOR: None,
        CURRENT_RESULTS: None,
        IS_PROCESSING: False,
        VIDEO_PROCESSED: False,
        VIDEO_PATH: None,
        CUDA_AVAILABLE: False,
        MODEL_MANAGER: None,
        VIDEO_SERVICE: None,
        VIDEO_BYTES_CACHE: None,
    }

    @classmethod
    def initialize(cls):
        """Initialize all session state keys with defaults."""
        for key, default_value in cls._defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a value from session state."""
        return st.session_state.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any):
        """Set a value in session state."""
        st.session_state[key] = value

    @classmethod
    def update(cls, **kwargs):
        """Update multiple session state values."""
        for key, value in kwargs.items():
            if key in st.session_state:
                st.session_state[key] = value

    @classmethod
    def release_heavy_runtime_objects(cls):
        """
        Release heavy runtime objects (detector, model, video service)
        to free GPU memory and prevent memory leaks.
        """
        # Release video service
        video_service = cls.get(cls.VIDEO_SERVICE)
        if video_service is not None:
            if hasattr(video_service, "detector"):
                video_service.detector = None
            if hasattr(video_service, "track_dict"):
                video_service.track_dict.clear()
            if hasattr(video_service, "prev_positions"):
                video_service.prev_positions.clear()
            if hasattr(video_service, "speed_history"):
                video_service.speed_history.clear()

        # Unload model
        model_manager = cls.get(cls.MODEL_MANAGER)
        if model_manager is not None:
            model_manager.unload_model()

        # Clear references
        cls.set(cls.VIDEO_SERVICE, None)
        cls.set(cls.DETECTOR, None)

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @classmethod
    def reset_all(cls):
        """Reset all session state to defaults."""
        for key, default_value in cls._defaults.items():
            st.session_state[key] = default_value
        gc.collect()