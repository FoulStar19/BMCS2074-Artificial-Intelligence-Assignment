"""
Video processing module for handling video streams with tracking and interpolation
Supports multiple vehicle classes: car, truck, bus, motorcycle, bicycle
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
from collections import defaultdict
import sys


class VideoProcessor:
    """
    Process video streams with detection, tracking, and interpolation
    Supports multiple vehicle classes with color-coded bounding boxes
    """
    
    def __init__(self, detector, speed_estimator=None, frame_skip=1, 
                 conf_threshold=0.25, min_width=0, class_ids=None, device="cpu"):
        """
        Initialize video processor
        
        Args:
            detector: Detection model instance (YOLODetector or similar)
            speed_estimator: Speed estimator instance (optional)
            frame_skip: Process every Nth frame
            conf_threshold: Confidence threshold for detections
            min_width: Minimum bounding box width filter
            class_ids: List of class IDs to track (None for all classes)
            device: Device to use ('cpu' or 'cuda')
        """
        self.detector = detector
        self.speed_estimator = speed_estimator
        self.frame_skip = frame_skip
        self.conf_threshold = conf_threshold
        self.min_width = min_width
        self.class_ids = class_ids
        self.device = device
        
        # Class definitions from dataset.yaml
        self.class_names = {
            0: 'car',
            1: 'truck', 
            2: 'bus',
            3: 'motorcycle',
            4: 'bicycle'
        }
        
        # Class colors for visualization (BGR format)
        self.class_colors = {
            0: [0, 0, 255],      # Red for car
            1: [0, 255, 0],      # Green for truck
            2: [255, 0, 0],      # Blue for bus
            3: [255, 255, 0],    # Cyan for motorcycle
            4: [255, 0, 255]     # Magenta for bicycle
        }
        
        # Override colors from detector if available
        if hasattr(detector, 'class_colors'):
            self.class_colors = detector.class_colors
        if hasattr(detector, 'vehicle_classes'):
            self.class_names = detector.vehicle_classes
        
        # Tracking and interpolation data
        self.track_dict = defaultdict(list)
        self.interp_results = None
        
        # Statistics
        self.total_frames = 0
        self.processed_frames = 0
        self.total_detections = 0
        self.class_counts = defaultdict(int)
        self.start_time = 0
        self.results = []
        self.speeds_list = []
        self.density_list = []
        
        print(f"VideoProcessor initialized with device: {device}")
    
    def get_class_color(self, class_id):
        """Get color for a specific class"""
        return self.class_colors.get(class_id, [0, 255, 0])
    
    def get_class_name(self, class_id):
        """Get name for a specific class"""
        return self.class_names.get(class_id, f'class_{class_id}')
    
    def draw_boxes(self, frame, detections):
        """
        Draw color-coded boxes with class labels and track IDs
        """
        for det in detections:
            bbox = det.get('bbox', [0, 0, 100, 100])
            track_id = det.get('track_id', 0)
            class_id = det.get('class', 0)
            confidence = det.get('confidence', 0)
            speed = det.get('speed', 0)
            class_name = det.get('class_name', self.get_class_name(class_id))
            
            color = self.get_class_color(class_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Prepare tag text
            tag_text = class_name
            if track_id > 0:
                tag_text += f" #{int(track_id)}"
            if confidence > 0:
                tag_text += f" {confidence:.2f}"
            if speed > 0:
                tag_text += f" {speed:.1f}km/h"
            
            # Text params
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            pad = 3
            
            text_size, _ = cv2.getTextSize(tag_text, font, font_scale, thickness)
            tx = bbox[0] + pad
            ty = bbox[1] + text_size[1] + pad
            
            if ty + pad > bbox[3]:
                ty = bbox[3] - pad
            
            # Draw text background
            cv2.rectangle(
                frame,
                (tx - pad, ty - text_size[1] - pad),
                (tx + text_size[0] + pad, ty + pad),
                color,
                -1
            )
            cv2.putText(
                frame,
                tag_text,
                (tx, ty),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )
        
        return frame
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            Detections list
        """
        if frame is None:
            return []
        
        # Detect vehicles
        detections = self.detector.detect(frame)
        
        # Add speed estimation if available
        if self.speed_estimator and detections:
            if hasattr(self.speed_estimator, 'estimate_speed'):
                detections = self.speed_estimator.estimate_speed(detections, frame)
            elif hasattr(self.speed_estimator, 'estimate_speeds'):
                detections = self.speed_estimator.estimate_speeds(detections, frame)
        
        # Update statistics
        self.processed_frames += 1
        self.total_detections += len(detections)
        
        # Collect speeds
        for det in detections:
            speed = det.get('speed', 0)
            if speed > 0:
                self.speeds_list.append(speed)
        
        # Update class counts
        for det in detections:
            class_id = det.get('class', 0)
            self.class_counts[class_id] += 1
        
        return detections
    
    def process_video(self, video_path, output_path=None, 
                      save_labels=False, interpolate=False, 
                      progress_callback=None):
        """
        Process entire video
        
        Args:
            video_path: Path to video file
            output_path: Path to save processed video (optional)
            save_labels: Whether to save YOLO label files
            interpolate: Whether to perform track interpolation
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
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Reset state
        self.track_dict.clear()
        self.interp_results = None
        self.class_counts.clear()
        self.results = []
        self.speeds_list = []
        self.density_list = []
        
        frame_count = 0
        self.start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                self.total_frames += 1
                
                # Process frame (skip frames for performance)
                if frame_count % self.frame_skip == 0:
                    detections = self.process_frame(frame)
                    
                    # Annotate frame with detections
                    annotated_frame = frame.copy()
                    if detections:
                        self.draw_boxes(annotated_frame, detections)
                    
                    # Calculate density for this frame
                    if detections:
                        density = self.calculate_density(detections, frame.shape)
                        self.density_list.append(density)
                    else:
                        self.density_list.append(0)
                else:
                    annotated_frame = frame
                    detections = []
                    self.density_list.append(0)
                
                # Store results
                self.results.append({
                    'frame': frame_count,
                    'detections': detections,
                    'count': len(detections)
                })
                
                # Write output
                if out:
                    out.write(annotated_frame)
                
                # Update progress
                if progress_callback and total_frames > 0:
                    progress = frame_count / total_frames
                    progress_callback(progress, frame_count, total_frames)
        
        finally:
            cap.release()
            if out:
                out.release()
        
        return self.results
    
    def calculate_density(self, detections, frame_shape):
        """
        Calculate traffic density based on detections
        
        Args:
            detections: List of detection dictionaries
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            Density percentage
        """
        if not detections:
            return 0.0
        
        vehicle_area = 0
        for det in detections:
            bbox = det.get('bbox', [0, 0, 100, 100])
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            vehicle_area += width * height
        
        total_area = frame_shape[0] * frame_shape[1]
        density = (vehicle_area / total_area) * 100
        
        return min(density, 100.0)
    
    def get_statistics(self):
        """Get processing statistics"""
        elapsed = time.time() - self.start_time
        fps = self.processed_frames / elapsed if elapsed > 0 else 0
        
        # Calculate speed statistics
        speeds = [s for s in self.speeds_list if s > 0]
        avg_speed = np.mean(speeds) if speeds else 0
        max_speed = np.max(speeds) if speeds else 0
        min_speed = np.min(speeds) if speeds else 0
        
        # Calculate density statistics
        avg_density = np.mean(self.density_list) if self.density_list else 0
        max_density = np.max(self.density_list) if self.density_list else 0
        
        return {
            'total_frames': self.total_frames,
            'processed_frames': self.processed_frames,
            'total_detections': self.total_detections,
            'average_detections': self.total_detections / self.processed_frames if self.processed_frames > 0 else 0,
            'fps': fps,
            'elapsed_time': elapsed,
            'tracked_objects': len(self.track_dict) if self.track_dict else 0,
            'class_counts': dict(self.class_counts),
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'min_speed': min_speed,
            'avg_density': avg_density,
            'max_density': max_density
        }
    
    def reset(self):
        """Reset processor state"""
        self.total_frames = 0
        self.processed_frames = 0
        self.total_detections = 0
        self.start_time = 0
        self.track_dict.clear()
        self.interp_results = None
        self.class_counts.clear()
        self.results = []
        self.speeds_list = []
        self.density_list = []
        
        if self.speed_estimator and hasattr(self.speed_estimator, 'reset'):
            self.speed_estimator.reset()


# ==================
#  WRAPPER FOR APP.PY
# ==================

class VideoProcessorWrapper:
    """
    Wrapper class for video processing to work with app.py
    """
    def __init__(self, detector, speed_estimator=None, frame_skip=2):
        self.detector = detector
        self.speed_estimator = speed_estimator
        self.frame_skip = frame_skip
        self.processor = VideoProcessor(
            detector=detector,
            speed_estimator=speed_estimator,
            frame_skip=frame_skip
        )
    
    def process_frame_sync(self, frame):
        """Process a single frame synchronously"""
        return self.processor.process_frame(frame)
    
    def process_video(self, video_path, output_path=None, progress_callback=None):
        """Process entire video"""
        results = self.processor.process_video(
            video_path=video_path,
            output_path=output_path,
            progress_callback=progress_callback
        )
        
        # Convert to format expected by app.py
        stats = self.processor.get_statistics()
        
        # Extract data for app.py format
        frames = []
        detections = []
        speeds = []
        density = []
        
        for result in results:
            frames.append(result['frame'])
            detections.append(result['count'])
            
            # Calculate average speed for this frame
            frame_speeds = [d.get('speed', 0) for d in result.get('detections', []) if d.get('speed', 0) > 0]
            avg_speed = np.mean(frame_speeds) if frame_speeds else 0
            speeds.append(avg_speed)
            
            # Use density from processor
            if hasattr(self.processor, 'density_list') and len(self.processor.density_list) > len(density):
                density.append(self.processor.density_list[len(density)])
            else:
                density.append(result['count'] * 3)
        
        return {
            'frames': frames,
            'detections': detections,
            'speeds': speeds,
            'density': density,
            'total_vehicles': stats['total_detections'],
            'avg_speed': stats['avg_speed'],
            'max_speed': stats['max_speed'],
            'min_speed': stats['min_speed'],
            'processing_time': stats['elapsed_time'],
            'frames_processed': stats['processed_frames']
        }


# ==================
#  PROCESS VIDEO WITH MODELS (for app.py)
# ==================

def process_video_with_models(
    video_path,
    detector,
    dataset_config,
    frame_skip=2,
    progress_callback=None,
    target_fps=60
):
    """
    Process video with detection and speed estimation (compatible with app.py)
    
    Args:
        video_path: Path to video file
        detector: Detection model instance
        dataset_config: Dataset configuration dictionary
        frame_skip: Process every Nth frame
        progress_callback: Callback for progress updates
        target_fps: Target FPS for output video
    
    Returns:
        Tuple of (output_video_path, results_dict)
    """
    # Create output directory
    output_dir = Path("outputs/processed_videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"processed_{timestamp}.mp4"
    
    # Create speed estimator if needed
    speed_estimator = None
    try:
        from .speed_estimator import SpeedEstimator
        speed_estimator = SpeedEstimator(fps=target_fps, calibration_factor=0.05)
    except:
        pass
    
    # Create wrapper
    wrapper = VideoProcessorWrapper(
        detector=detector,
        speed_estimator=speed_estimator,
        frame_skip=frame_skip
    )
    
    # Process video
    results = wrapper.process_video(
        video_path=video_path,
        output_path=str(output_path),
        progress_callback=progress_callback
    )
    
    return str(output_path), results