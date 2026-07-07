"""
Video processing module
"""

import cv2
import numpy as np
import time
from queue import Queue
from threading import Thread
import os


class VideoProcessor:
    """Process video streams with detection and tracking"""
    
    def __init__(self, detector, speed_estimator=None, frame_skip=2, max_queue_size=32):
        """
        Initialize video processor
        
        Args:
            detector: Detection model instance
            speed_estimator: Speed estimator instance
            frame_skip: Process every Nth frame
            max_queue_size: Maximum size for frame queue
        """
        self.detector = detector
        self.speed_estimator = speed_estimator
        self.frame_skip = frame_skip
        self.max_queue_size = max_queue_size
        
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.result_queue = Queue(maxsize=max_queue_size)
        
        self.processing = False
        self.thread = None
        
        # Statistics
        self.total_frames = 0
        self.processed_frames = 0
        self.total_detections = 0
        self.start_time = 0
        
        # Buffer for previous frames (for optical flow)
        self.prev_frame = None
        self.prev_detections = None
    
    def process_video(self, video_source, callback=None, stop_event=None):
        """
        Process video from source
        
        Args:
            video_source: Video file path or webcam index
            callback: Function to call with processed frames
            stop_event: Event to signal stopping
            
        Yields:
            Processed frames with detections
        """
        # Open video capture
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        self.processing = True
        self.start_time = time.time()
        frame_count = 0
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.prev_frame = None
        self.prev_detections = None
        
        try:
            while self.processing:
                # Check stop event
                if stop_event and stop_event.is_set():
                    break
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                self.total_frames += 1
                
                # Skip frames for performance
                if frame_count % self.frame_skip != 0:
                    if callback:
                        callback(frame, None, frame_count)
                    continue
                
                # Process frame
                try:
                    processed_frame, detections = self._process_frame(frame)
                    self.processed_frames += 1
                    self.total_detections += len(detections)
                    
                    # Call callback with results
                    if callback:
                        callback(processed_frame, detections, frame_count)
                    
                    # Update previous frame for optical flow
                    self.prev_frame = frame
                    self.prev_detections = detections
                    
                    # Update stats in metadata
                    elapsed = time.time() - self.start_time
                    metadata = {
                        'frame_count': frame_count,
                        'processed_frames': self.processed_frames,
                        'total_detections': self.total_detections,
                        'fps': self.processed_frames / elapsed if elapsed > 0 else 0,
                        'elapsed_time': elapsed
                    }
                    
                    yield processed_frame, detections, metadata
                    
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    if callback:
                        callback(frame, None, frame_count)
                    yield frame, None, None
        
        finally:
            cap.release()
            self.processing = False
        
        print(f"Video processing completed: {self.processed_frames} frames processed")
    
    def _process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with detections drawn
        """
        # Make a copy for annotation
        annotated_frame = frame.copy()
        
        # Detect vehicles
        detections = self.detector.detect(frame)
        
        # Estimate speeds if available
        if self.speed_estimator:
            detections = self.speed_estimator.estimate_speed(detections, frame)
        
        # Annotate frame
        if detections:
            for det in detections:
                bbox = det['bbox']
                confidence = det.get('confidence', 0)
                speed = det.get('speed', 0)
                class_name = det.get('class', 'vehicle')
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, 
                            (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]), 
                            (0, 255, 0), 2)
                
                # Draw confidence and speed
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
        
        return annotated_frame, detections
    
    def process_frame_sync(self, frame):
        """
        Process a single frame synchronously
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame and detections
        """
        return self._process_frame(frame)
    
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
    
    def save_video(self, video_source, output_path, progress_callback=None):
        """
        Process and save video with detections
        
        Args:
            video_source: Input video source
            output_path: Output video path
            progress_callback: Callback for progress updates
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.start_time = time.time()
        frame_count = 0
        processed_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                self.total_frames += 1
                
                # Process every Nth frame
                if frame_count % self.frame_skip == 0:
                    processed_frame, _ = self._process_frame(frame)
                    processed_count += 1
                else:
                    processed_frame = frame
                
                # Write frame
                out.write(processed_frame)
                
                # Progress callback
                if progress_callback and total_frames > 0:
                    progress = frame_count / total_frames
                    progress_callback(progress, frame_count, total_frames)
        
        finally:
            cap.release()
            out.release()
        
        print(f"Video saved to {output_path}")
    
    def stop(self):
        """Stop processing"""
        self.processing = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
    
    def reset(self):
        """Reset processor state"""
        self.total_frames = 0
        self.processed_frames = 0
        self.total_detections = 0
        self.start_time = 0
        self.prev_frame = None
        self.prev_detections = None
        
        if self.speed_estimator:
            self.speed_estimator.reset()