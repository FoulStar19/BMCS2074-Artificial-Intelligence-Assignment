"""
Vehicle tracking module using optical flow and Kalman filtering
"""

import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import time


@dataclass
class TrackedVehicle:
    """Data class for a tracked vehicle"""
    track_id: int
    bbox: List[int]  # [x1, y1, x2, y2]
    center: Tuple[int, int]
    class_id: int
    confidence: float
    speed: float = 0.0
    history: deque = None
    last_seen: float = 0.0
    active: bool = True
    
    def __post_init__(self):
        if self.history is None:
            self.history = deque(maxlen=30)
        self.last_seen = time.time()


class VehicleTracker:
    """
    Advanced vehicle tracker using optical flow and Kalman filtering
    """
    
    def __init__(self, 
                 max_lost_frames: int = 15,
                 min_distance: float = 50.0,
                 kalman: bool = True,
                 optical_flow: bool = True):
        """
        Initialize vehicle tracker
        
        Args:
            max_lost_frames: Maximum frames before track is lost
            min_distance: Minimum distance for matching tracks
            kalman: Enable Kalman filtering
            optical_flow: Enable optical flow for motion estimation
        """
        self.max_lost_frames = max_lost_frames
        self.min_distance = min_distance
        self.kalman_enabled = kalman
        self.optical_flow_enabled = optical_flow
        
        self.next_track_id = 0
        self.tracks: Dict[int, TrackedVehicle] = {}
        self.prev_gray = None
        self.flow = None
        
        # Kalman filter for each track
        self.kalman_filters: Dict[int, cv2.KalmanFilter] = {}
        
    def _init_kalman_filter(self, center_x: float, center_y: float) -> cv2.KalmanFilter:
        """Initialize Kalman filter for a track"""
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        kf.statePre = np.array([[center_x], [center_y], [0], [0]], np.float32)
        kf.statePost = np.array([[center_x], [center_y], [0], [0]], np.float32)
        return kf
    
    def _compute_optical_flow(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Compute optical flow for the current frame"""
        if frame is None:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        
        # Compute dense optical flow
        self.flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        self.prev_gray = gray
        return self.flow
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _match_detections(self, 
                         detections: List[dict], 
                         current_centers: List[Tuple[int, int]]) -> Dict[int, int]:
        """
        Match detections to existing tracks using Hungarian algorithm
        Returns mapping of track_id -> detection_index
        """
        if not self.tracks or not detections:
            return {}
        
        # Build cost matrix
        track_centers = [(t.center[0], t.center[1]) for t in self.tracks.values() if t.active]
        track_ids = [tid for tid, t in self.tracks.items() if t.active]
        
        if not track_centers:
            return {}
        
        cost_matrix = np.zeros((len(track_centers), len(current_centers)))
        
        for i, (tx, ty) in enumerate(track_centers):
            for j, (dx, dy) in enumerate(current_centers):
                dist = np.sqrt((tx - dx)**2 + (ty - dy)**2)
                cost_matrix[i, j] = dist
        
        # Hungarian algorithm (simple greedy for efficiency)
        matches = {}
        used_detections = set()
        
        # Sort by minimum cost
        for i in range(len(track_centers)):
            min_cost = float('inf')
            best_j = -1
            
            for j in range(len(current_centers)):
                if j in used_detections:
                    continue
                if cost_matrix[i, j] < min_cost:
                    min_cost = cost_matrix[i, j]
                    best_j = j
            
            if best_j >= 0 and min_cost < self.min_distance:
                matches[track_ids[i]] = best_j
                used_detections.add(best_j)
        
        return matches
    
    def update(self, 
               frame: np.ndarray, 
               detections: List[dict]) -> List[dict]:
        """
        Update tracker with new detections
        
        Args:
            frame: Current frame
            detections: List of detection dictionaries
            
        Returns:
            List of detections with tracking IDs and speeds
        """
        if frame is None:
            return detections
        
        # Compute optical flow
        if self.optical_flow_enabled:
            self._compute_optical_flow(frame)
        
        # Prepare detection data
        detection_centers = []
        for det in detections:
            bbox = det.get('bbox', [0, 0, 100, 100])
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            detection_centers.append((center_x, center_y))
        
        # Match detections to existing tracks
        matches = self._match_detections(detections, detection_centers)
        
        # Update matched tracks
        for track_id, det_idx in matches.items():
            det = detections[det_idx]
            bbox = det['bbox']
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            track = self.tracks[track_id]
            track.bbox = bbox
            track.center = (center_x, center_y)
            track.class_id = det.get('class', 0)
            track.confidence = det.get('confidence', 0.8)
            track.last_seen = time.time()
            track.history.append((center_x, center_y, time.time()))
            
            # Update Kalman filter
            if self.kalman_enabled and track_id in self.kalman_filters:
                kf = self.kalman_filters[track_id]
                measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
                kf.correct(measurement)
                prediction = kf.predict()
                # FIX: Extract scalar values from prediction array
                pred_x = float(prediction[0, 0]) if prediction.ndim > 1 else float(prediction[0])
                pred_y = float(prediction[1, 0]) if prediction.ndim > 1 else float(prediction[1])
                # Smooth position with Kalman
                track.center = (int(pred_x), int(pred_y))
            
            # Calculate speed
            speed = self._calculate_speed(track)
            det['track_id'] = track_id
            det['speed'] = speed
            det['class_name'] = det.get('class_name', f'Class {track.class_id}')
            
            # Mark detection as used
            detections[det_idx] = det
        
        # Create new tracks for unmatched detections
        used_indices = set(matches.values())
        for i, det in enumerate(detections):
            if i in used_indices:
                continue
            
            bbox = det['bbox']
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            track_id = self.next_track_id
            self.next_track_id += 1
            
            track = TrackedVehicle(
                track_id=track_id,
                bbox=bbox,
                center=(center_x, center_y),
                class_id=det.get('class', 0),
                confidence=det.get('confidence', 0.8)
            )
            track.history.append((center_x, center_y, time.time()))
            
            self.tracks[track_id] = track
            
            if self.kalman_enabled:
                self.kalman_filters[track_id] = self._init_kalman_filter(center_x, center_y)
            
            det['track_id'] = track_id
            det['speed'] = 0.0
            det['class_name'] = det.get('class_name', f'Class {track.class_id}')
            
            detections[i] = det
        
        # Remove old tracks
        current_time = time.time()
        inactive_tracks = []
        for track_id, track in self.tracks.items():
            if current_time - track.last_seen > self.max_lost_frames * 0.1:  # ~100ms per frame
                track.active = False
                inactive_tracks.append(track_id)
        
        # Remove inactive tracks after some time
        for track_id in inactive_tracks[:len(inactive_tracks)//2]:
            del self.tracks[track_id]
            if track_id in self.kalman_filters:
                del self.kalman_filters[track_id]
        
        return detections
    
    def _calculate_speed(self, track: TrackedVehicle) -> float:
        """Calculate speed from track history"""
        if len(track.history) < 2:
            return 0.0
        
        # Use last two positions
        pos1 = track.history[-2]
        pos2 = track.history[-1]
        
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        displacement = np.sqrt(dx**2 + dy**2)
        dt = pos2[2] - pos1[2]
        
        if dt <= 0:
            return 0.0
        
        # Convert to km/h (1 pixel ~ 0.05 meters)
        speed_mps = (displacement * 0.05) / dt
        speed_kmh = speed_mps * 3.6
        
        # Use optical flow for small displacements
        if self.optical_flow_enabled and self.flow is not None and displacement < 5:
            x, y = track.center
            if 0 <= y < self.flow.shape[0] and 0 <= x < self.flow.shape[1]:
                flow_vector = self.flow[y, x]
                flow_magnitude = np.sqrt(flow_vector[0]**2 + flow_vector[1]**2)
                flow_speed = flow_magnitude * 0.05 * 3.6
                if flow_speed > 0:
                    speed_kmh = flow_speed
        
        # Smooth speed with exponential moving average
        if hasattr(track, 'smoothed_speed'):
            alpha = 0.7
            speed_kmh = alpha * speed_kmh + (1 - alpha) * track.smoothed_speed
        
        track.smoothed_speed = speed_kmh
        
        return round(max(0, speed_kmh), 2)
    
    def reset(self):
        """Reset tracker state"""
        self.tracks.clear()
        self.kalman_filters.clear()
        self.next_track_id = 0
        self.prev_gray = None
        self.flow = None
    
    def get_active_tracks(self) -> List[TrackedVehicle]:
        """Get all active tracks"""
        return [t for t in self.tracks.values() if t.active]
    
    def draw_trails(self, frame: np.ndarray, trail_length: int = 20) -> np.ndarray:
        """Draw tracking trails on frame"""
        annotated_frame = frame.copy()
        
        for track in self.tracks.values():
            if not track.active or len(track.history) < 2:
                continue
            
            # Get recent points
            points = list(track.history)[-trail_length:]
            
            # Draw trail
            for i in range(1, len(points)):
                pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                alpha = i / len(points)
                color = (int(0 * alpha), int(255 * alpha), int(0 * alpha))
                cv2.line(annotated_frame, pt1, pt2, color, 2)
            
            # Draw current position
            if points:
                current_pos = points[-1]
                cv2.circle(annotated_frame, 
                          (int(current_pos[0]), int(current_pos[1])), 
                          5, (0, 255, 0), -1)
        
        return annotated_frame