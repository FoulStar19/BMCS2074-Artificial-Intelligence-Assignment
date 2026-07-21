"""
Speed estimation module for vehicles using tracking and optical flow
"""

import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import time


@dataclass
class SpeedRecord:
    """Data class for speed records"""
    track_id: int
    speed: float
    timestamp: float
    position: Tuple[int, int]


class SpeedEstimator:
    """
    Estimate vehicle speed using tracking history and optical flow
    """
    
    def __init__(self, 
                 fps: float = 30.0,
                 calibration_factor: float = 0.05,
                 smoothing_window: int = 5,
                 min_samples: int = 3):
        """
        Initialize speed estimator
        
        Args:
            fps: Frames per second of video
            calibration_factor: Meters per pixel
            smoothing_window: Number of frames for smoothing
            min_samples: Minimum samples for reliable speed
        """
        self.fps = fps
        self.calibration_factor = calibration_factor
        self.smoothing_window = smoothing_window
        self.min_samples = min_samples
        
        # Speed history per track
        self.speed_history: Dict[int, deque] = {}
        self.track_history: Dict[int, deque] = {}
        
        # Optical flow
        self.prev_gray = None
        self.flow = None
        
        # Statistics
        self.all_speeds: List[float] = []
        self.speed_records: List[SpeedRecord] = []
    
    def estimate(self, 
                detections: List[dict], 
                frame: np.ndarray,
                use_optical_flow: bool = True) -> List[dict]:
        """
        Estimate speed for detected vehicles
        
        Args:
            detections: List of detection dictionaries with track_id
            frame: Current frame
            use_optical_flow: Enable optical flow for motion estimation
            
        Returns:
            List of detections with speed information
        """
        if not detections:
            return detections
        
        # Compute optical flow if enabled
        if use_optical_flow:
            self._compute_optical_flow(frame)
        
        # Estimate speed for each detection
        for det in detections:
            track_id = det.get('track_id')
            bbox = det.get('bbox', [0, 0, 100, 100])
            
            if track_id is None:
                continue
            
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            current_pos = (center_x, center_y)
            
            # Initialize history for new tracks
            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=self.smoothing_window + 5)
                self.speed_history[track_id] = deque(maxlen=self.smoothing_window)
            
            # Update track history
            self.track_history[track_id].append((current_pos, time.time()))
            
            # Calculate speed
            speed = self._estimate_speed_for_track(
                track_id, 
                current_pos, 
                frame.shape
            )
            
            # Store speed
            if speed > 0:
                self.speed_history[track_id].append(speed)
                self.all_speeds.append(speed)
                self.speed_records.append(
                    SpeedRecord(track_id, speed, time.time(), current_pos)
                )
            
            # Add speed to detection
            det['speed'] = speed
            det['speed_unit'] = 'km/h'
        
        return detections
    
    def _compute_optical_flow(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Compute dense optical flow"""
        if frame is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        
        # Use Farneback for dense optical flow
        self.flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        self.prev_gray = gray
        return self.flow
    
    def _estimate_speed_for_track(self, 
                                  track_id: int, 
                                  current_pos: Tuple[int, int],
                                  frame_shape: Tuple[int, int]) -> float:
        """
        Estimate speed for a single track
        
        Returns:
            Speed in km/h
        """
        history = self.track_history.get(track_id, deque())
        
        if len(history) < self.min_samples:
            return 0.0
        
        # Method 1: Position-based tracking
        positions = list(history)
        pos1 = positions[-2]  # Second last position
        pos2 = positions[-1]  # Last position
        
        dx = pos2[0][0] - pos1[0][0]
        dy = pos2[0][1] - pos1[0][1]
        displacement_pixels = np.sqrt(dx**2 + dy**2)
        
        dt = pos2[1] - pos1[1]
        if dt <= 0:
            return 0.0
        
        # Convert to real-world speed
        displacement_meters = displacement_pixels * self.calibration_factor
        speed_mps = displacement_meters / dt
        speed_kmh = speed_mps * 3.6
        
        # Method 2: Optical flow (for small displacements)
        flow_speed = self._get_optical_flow_speed(current_pos)
        
        # Combine methods
        if flow_speed > 0 and displacement_pixels < 10:
            # Weighted average for smooth transitions
            weight = min(displacement_pixels / 15, 0.7)
            speed_kmh = weight * speed_kmh + (1 - weight) * flow_speed
        
        # Apply temporal smoothing
        if track_id in self.speed_history and len(self.speed_history[track_id]) > 0:
            history_avg = np.mean(self.speed_history[track_id])
            alpha = 0.6
            speed_kmh = alpha * speed_kmh + (1 - alpha) * history_avg
        
        return round(max(0, speed_kmh), 2)
    
    def _get_optical_flow_speed(self, position: Tuple[int, int]) -> float:
        """Get speed from optical flow at a position"""
        if self.flow is None:
            return 0.0
        
        x, y = position
        h, w = self.flow.shape[:2]
        
        if not (0 <= x < w and 0 <= y < h):
            return 0.0
        
        flow_vector = self.flow[y, x]
        flow_magnitude = np.sqrt(flow_vector[0]**2 + flow_vector[1]**2)
        
        # Convert to km/h
        flow_speed = flow_magnitude * self.calibration_factor * 3.6
        
        return flow_speed
    
    def get_statistics(self) -> dict:
        """Get speed statistics"""
        if not self.all_speeds:
            return {
                'avg_speed': 0.0,
                'max_speed': 0.0,
                'min_speed': 0.0,
                'std_speed': 0.0,
                'total_samples': 0
            }
        
        return {
            'avg_speed': np.mean(self.all_speeds),
            'max_speed': np.max(self.all_speeds),
            'min_speed': np.min(self.all_speeds),
            'std_speed': np.std(self.all_speeds),
            'total_samples': len(self.all_speeds)
        }
    
    def get_speed_distribution(self, bins: int = 10) -> dict:
        """Get speed distribution histogram"""
        if not self.all_speeds:
            return {'bins': [], 'counts': []}
        
        hist, bin_edges = np.histogram(self.all_speeds, bins=bins)
        return {
            'bins': bin_edges.tolist(),
            'counts': hist.tolist()
        }
    
    def reset(self):
        """Reset estimator state"""
        self.speed_history.clear()
        self.track_history.clear()
        self.all_speeds.clear()
        self.speed_records.clear()
        self.prev_gray = None
        self.flow = None
    
    def set_calibration(self, calibration_factor: float):
        """Set calibration factor"""
        self.calibration_factor = calibration_factor