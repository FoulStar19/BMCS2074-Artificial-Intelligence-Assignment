"""
Session State Manager - Centralized session state management for Streamlit
"""

import streamlit as st


class SessionManager:
    """
    Manages Streamlit session state with typed getters/setters
    """
    
    # Session state keys
    PROCESSING = 'processing'
    DETECTIONS_HISTORY = 'detections_history'
    SPEED_HISTORY = 'speed_history'
    DENSITY_HISTORY = 'density_history'
    PROCESSED_VIDEO_PATH = 'processed_video_path'
    DETECTOR = 'detector'
    CURRENT_RESULTS = 'current_results'
    IS_PROCESSING = 'is_processing'
    VIDEO_PROCESSED = 'video_processed'
    VIDEO_PATH = 'video_path'
    CUDA_AVAILABLE = 'cuda_available'
    MODEL_MANAGER = 'model_manager'
    VIDEO_SERVICE = 'video_service'
    
    @classmethod
    def initialize(cls):
        """Initialize all session state variables"""
        defaults = {
            cls.PROCESSING: False,
            cls.DETECTIONS_HISTORY: [],
            cls.SPEED_HISTORY: [],
            cls.DENSITY_HISTORY: [],
            cls.PROCESSED_VIDEO_PATH: None,
            cls.DETECTOR: None,
            cls.CURRENT_RESULTS: None,
            cls.IS_PROCESSING: False,
            cls.VIDEO_PROCESSED: False,
            cls.VIDEO_PATH: None,
            cls.CUDA_AVAILABLE: False,
            cls.MODEL_MANAGER: None,
            cls.VIDEO_SERVICE: None,
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @classmethod
    def get(cls, key, default=None):
        """Get a value from session state"""
        return st.session_state.get(key, default)
    
    @classmethod
    def set(cls, key, value):
        """Set a value in session state"""
        st.session_state[key] = value
    
    @classmethod
    def update(cls, **kwargs):
        """Update multiple values in session state"""
        for key, value in kwargs.items():
            if key in st.session_state:
                st.session_state[key] = value
    
    # Convenience properties
    
    @property
    def processing(self):
        return self.get(self.PROCESSING, False)
    
    @processing.setter
    def processing(self, value):
        self.set(self.PROCESSING, value)
    
    @property
    def current_results(self):
        return self.get(self.CURRENT_RESULTS)
    
    @current_results.setter
    def current_results(self, value):
        self.set(self.CURRENT_RESULTS, value)
    
    @property
    def processed_video_path(self):
        return self.get(self.PROCESSED_VIDEO_PATH)
    
    @processed_video_path.setter
    def processed_video_path(self, value):
        self.set(self.PROCESSED_VIDEO_PATH, value)
    
    @property
    def video_processed(self):
        return self.get(self.VIDEO_PROCESSED, False)
    
    @video_processed.setter
    def video_processed(self, value):
        self.set(self.VIDEO_PROCESSED, value)
    
    @property
    def is_processing(self):
        return self.get(self.IS_PROCESSING, False)
    
    @is_processing.setter
    def is_processing(self, value):
        self.set(self.IS_PROCESSING, value)
    
    @property
    def model_manager(self):
        return self.get(self.MODEL_MANAGER)
    
    @model_manager.setter
    def model_manager(self, value):
        self.set(self.MODEL_MANAGER, value)
    
    @property
    def video_service(self):
        return self.get(self.VIDEO_SERVICE)
    
    @video_service.setter
    def video_service(self, value):
        self.set(self.VIDEO_SERVICE, value)