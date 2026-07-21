"""
Video processing module for handling video streams
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path


class VideoProcessor:
    """
    Process video streams with detection and tracking
    """
    
    def __init__(self, detector, speed_estimator=None):  # REMOVED frame_skip parameter
        """
        Initialize video processor
        
        Args:
            detector: Detection model instance
            speed_estimator: Speed estimator instance
        """
        self.detector = detector
        self.speed_estimator = speed_estimator
        # REMOVED: self.frame_skip = frame_skip
        
        # Statistics
        self.total_frames = 0
        self.processed_frames = 0
        self.total_detections = 0
        self.start_time = 0
    
    def process_frame_sync(self, frame):
        """
        Process a single frame synchronously
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame and detections
        """
        if frame is None:
            return None, []
        
        # Make a copy for annotation
        annotated_frame = frame.copy()
        
        # Detect vehicles
        detections = self.detector.detect(frame)
        
        # Estimate speeds if available
        if self.speed_estimator and detections:
            detections = self.speed_estimator.estimate_speed(detections, frame)
        
        # Annotate frame with detections
        if detections:
            for det in detections:
                bbox = det['bbox']
                confidence = det.get('confidence', 0)
                speed = det.get('speed', 0)
                class_name = det.get('class_name', 'vehicle')
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, 
                            (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]), 
                            (0, 255, 0), 2)
                
                # Draw label with class name, confidence, and speed
                label = f"{class_name} {confidence:.2f}"
                if speed > 0:
                    label += f" {speed:.1f}km/h"
                
                cv2.putText(annotated_frame, label,
                          (bbox[0], bbox[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (0, 255, 0), 2)
        
        # Add FPS and detection count
        h, w = annotated_frame.shape[:2]
        cv2.putText(annotated_frame, f"Detections: {len(detections)}",
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                  0.7, (0, 255, 255), 2)
        
        self.processed_frames += 1
        self.total_detections += len(detections)
        
        return annotated_frame, detections
    
    def process_video(self, video_path, output_path=None, progress_callback=None):
        """
        Process entire video
        
        Args:
            video_path: Path to video file
            output_path: Path to save processed video (optional)
            progress_callback: Callback for progress updates
            
        Returns:
            List of results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if output path provided
        out = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_count = 0
        self.start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                self.total_frames += 1
                
                # Process ALL frames - REMOVED frame skip condition
                processed_frame, detections = self.process_frame_sync(frame)
                
                # Store results
                results.append({
                    'frame': frame_count,
                    'detections': detections,
                    'count': len(detections)
                })
                
                # Write output
                if out:
                    out.write(processed_frame)
                
                # Update progress
                if progress_callback and total_frames > 0:
                    progress = frame_count / total_frames
                    progress_callback(progress, frame_count, total_frames)
        
        finally:
            cap.release()
            if out:
                out.release()
        
        return results
    
    def get_statistics(self):
        """Get processing statistics"""
        elapsed = time.time() - self.start_time
        fps = self.processed_frames / elapsed if elapsed > 0 else 0
        
        return {
            'total_frames': self.total_frames,
            'processed_frames': self.processed_frames,
            'total_detections': self.total_detections,
            'average_detections': self.total_detections / self.processed_frames if self.processed_frames > 0 else 0,
            'fps': fps,
            'elapsed_time': elapsed
        }
    
    def reset(self):
        """Reset processor state"""
        self.total_frames = 0
        self.processed_frames = 0
        self.total_detections = 0
        self.start_time = 0
        
        if self.speed_estimator:
            self.speed_estimator.reset()