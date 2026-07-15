"""
Speed estimation module for vehicles using tracking and optical flow
"""

import numpy as np
import cv2
from collections import deque
import time


class SpeedEstimator:
    """
    Estimate vehicle speed using optical flow and object tracking
    """
    
    def __init__(self, fps=30, calibration_factor=0.05, smoothing_window=5):
        """
        Initialize speed estimator
        
        Args:
            fps: Frames per second of video
            calibration_factor: Conversion factor from pixels to real-world distance (meters per pixel)
            smoothing_window: Number of frames for speed smoothing
        """
        self.fps = fps
        self.calibration_factor = calibration_factor
        self.smoothing_window = smoothing_window
        
        # Store previous positions for tracking
        self.track_history = {}
        self.speed_history = {}
        self.next_track_id = 0
        
        # For optical flow
        self.prev_gray = None
        self.prev_detections = None
    
    def estimate_speed(self, detections, frame, use_optical_flow=True):
        """
        Estimate speed for detected vehicles
        
        Args:
            detections: List of detection dictionaries
            frame: Current frame
            use_optical_flow: Whether to use optical flow for motion estimation
            
        Returns:
            List of detections with speed information
        """
        if not detections:
            return detections
        
        # Convert frame to grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow if previous frame exists
        flow = None
        if use_optical_flow and self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
        
        # Process each detection
        for det in detections:
            bbox = det['bbox']
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            # Get or create track ID
            track_id = self._get_track_id(center_x, center_y)
            
            # Estimate speed using tracking and/or optical flow
            speed = self._estimate_speed_single(
                track_id, center_x, center_y, flow, gray
            )
            
            # Add speed to detection
            det['track_id'] = track_id
            det['speed'] = speed
            det['speed_unit'] = 'km/h'
        
        # Update previous frame
        self.prev_gray = gray
        self.prev_detections = detections
        
        return detections
    
    def _get_track_id(self, x, y):
        """
        Get or create track ID based on position
        """
        # Simple assignment based on closest tracked object
        min_dist = 150
        best_id = None
        
        for track_id, history in self.track_history.items():
            if history:
                last_pos = history[-1]
                dist = np.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_id = track_id
        
        if best_id is None:
            best_id = self.next_track_id
            self.next_track_id += 1
            self.track_history[best_id] = deque(maxlen=self.smoothing_window * 3)
            self.speed_history[best_id] = deque(maxlen=self.smoothing_window)
        
        # Update history
        self.track_history[best_id].append((x, y, time.time()))
        
        return best_id
    
    def _estimate_speed_single(self, track_id, x, y, flow, gray):
        """
        Estimate speed for a single tracked object
        """
        history = self.track_history.get(track_id, deque())
        
        if len(history) < 2:
            return 0.0
        
        # Method 1: Position-based tracking
        pos1 = history[-2]
        pos2 = history[-1]
        
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        displacement_pixels = np.sqrt(dx**2 + dy**2)
        
        # Calculate time difference
        dt = pos2[2] - pos1[2]
        if dt <= 0:
            return 0.0
        
        # Convert to real-world distance and speed
        displacement_meters = displacement_pixels * self.calibration_factor
        speed_mps = displacement_meters / dt
        speed_kmh = speed_mps * 3.6
        
        # Method 2: Optical flow (if available)
        flow_speed = 0
        if flow is not None:
            if 0 <= y < flow.shape[0] and 0 <= x < flow.shape[1]:
                flow_vector = flow[y, x]
                flow_magnitude = np.sqrt(flow_vector[0]**2 + flow_vector[1]**2)
                flow_speed = flow_magnitude * self.calibration_factor * 3.6
        
        # Combine both methods (weighted average)
        if flow_speed > 0:
            # Use flow speed if tracking is unreliable (small displacement)
            if displacement_pixels < 5:
                speed_kmh = flow_speed
            else:
                # Weighted average
                weight = min(displacement_pixels / 20, 0.7)
                speed_kmh = weight * speed_kmh + (1 - weight) * flow_speed
        
        # Apply smoothing
        self.speed_history[track_id].append(speed_kmh)
        
        if len(self.speed_history[track_id]) > 0:
            smoothed_speed = np.mean(self.speed_history[track_id])
            return round(max(0, smoothed_speed), 2)
        
        return round(max(0, speed_kmh), 2)
    
    def reset(self):
        """Reset tracking history"""
        self.track_history.clear()
        self.speed_history.clear()
        self.next_track_id = 0
        self.prev_gray = None
        self.prev_detections = None
    
    def set_calibration_factor(self, factor):
        """Set calibration factor"""
        self.calibration_factor = factor