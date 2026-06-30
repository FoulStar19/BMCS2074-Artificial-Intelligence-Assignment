"""
Speed estimation module for vehicles
"""

import numpy as np
import cv2
from collections import deque
import time


class SpeedEstimator:
    """Estimate vehicle speed using optical flow and object tracking"""
    
    def __init__(self, fps=30, calibration_factor=0.05, smoothing_window=5):
        """
        Initialize speed estimator
        
        Args:
            fps: Frames per second of video
            calibration_factor: Conversion factor from pixels to real-world distance
            smoothing_window: Number of frames for speed smoothing
        """
        self.fps = fps
        self.calibration_factor = calibration_factor  # meters per pixel
        self.smoothing_window = smoothing_window
        
        # Store previous positions for tracking
        self.track_history = {}
        self.speed_history = {}
        self.next_track_id = 0
    
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
            return []
        
        # Process each detection
        for det in detections:
            bbox = det['bbox']
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            # Get or create track ID
            track_id = self._get_track_id(center_x, center_y)
            
            # Estimate speed
            speed = self._estimate_speed_single(track_id, center_x, center_y, frame)
            
            # Add speed to detection
            det['track_id'] = track_id
            det['speed'] = speed
            det['speed_unit'] = 'km/h'
        
        return detections
    
    def _get_track_id(self, x, y):
        """
        Get or create track ID based on position
        
        Args:
            x, y: Center position of detection
            
        Returns:
            Track ID
        """
        # Simple assignment based on closest tracked object
        min_dist = 100  # Maximum distance for same object
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
            self.track_history[best_id] = deque(maxlen=self.smoothing_window * 2)
            self.speed_history[best_id] = deque(maxlen=self.smoothing_window)
        
        # Update history
        self.track_history[best_id].append((x, y, time.time()))
        
        return best_id
    
    def _estimate_speed_single(self, track_id, x, y, frame):
        """
        Estimate speed for a single tracked object
        
        Args:
            track_id: Track ID
            x, y: Current position
            frame: Current frame
            
        Returns:
            Speed in km/h
        """
        history = self.track_history.get(track_id, deque())
        
        if len(history) < 2:
            return 0.0
        
        # Calculate displacement
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
        speed_kmh = speed_mps * 3.6  # Convert m/s to km/h
        
        # Apply smoothing
        self.speed_history[track_id].append(speed_kmh)
        
        if len(self.speed_history[track_id]) > 0:
            smoothed_speed = np.mean(self.speed_history[track_id])
            return round(smoothed_speed, 2)
        
        return round(speed_kmh, 2)
    
    def estimate_speed_optical_flow(self, frame, prev_frame, detections):
        """
        Estimate speed using optical flow
        
        Args:
            frame: Current frame
            prev_frame: Previous frame
            detections: List of detections
            
        Returns:
            List of detections with speed information
        """
        if prev_frame is None:
            return detections
        
        # Convert to grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        for det in detections:
            bbox = det['bbox']
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            # Get optical flow at the center point
            if 0 <= center_y < flow.shape[0] and 0 <= center_x < flow.shape[1]:
                flow_vector = flow[center_y, center_x]
                speed_pixels = np.sqrt(flow_vector[0]**2 + flow_vector[1]**2)
                
                # Convert to km/h
                speed_kmh = speed_pixels * self.calibration_factor * 3.6
                det['speed'] = round(speed_kmh, 2)
                det['speed_unit'] = 'km/h'
        
        return detections
    
    def estimate_speed_tracking(self, detections_prev, detections_curr):
        """
        Estimate speed by tracking objects between frames
        
        Args:
            detections_prev: Detections from previous frame
            detections_curr: Detections from current frame
            
        Returns:
            List of detections with speed information
        """
        # Simple tracking using IoU matching
        for det_curr in detections_curr:
            bbox_curr = det_curr['bbox']
            best_iou = 0
            best_det = None
            
            for det_prev in detections_prev:
                bbox_prev = det_prev['bbox']
                iou = self._calculate_iou(bbox_curr, bbox_prev)
                
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_det = det_prev
            
            if best_det:
                # Calculate displacement
                cx_curr = (bbox_curr[0] + bbox_curr[2]) // 2
                cy_curr = (bbox_curr[1] + bbox_curr[3]) // 2
                cx_prev = (best_det['bbox'][0] + best_det['bbox'][2]) // 2
                cy_prev = (best_det['bbox'][1] + best_det['bbox'][3]) // 2
                
                displacement = np.sqrt((cx_curr - cx_prev)**2 + (cy_curr - cy_prev)**2)
                speed_kmh = displacement * self.calibration_factor * 3.6
                
                det_curr['speed'] = round(speed_kmh, 2)
                det_curr['speed_unit'] = 'km/h'
                det_curr['matched_id'] = id(best_det)
        
        return detections_curr
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU of two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def reset(self):
        """Reset tracking history"""
        self.track_history.clear()
        self.speed_history.clear()
        self.next_track_id = 0