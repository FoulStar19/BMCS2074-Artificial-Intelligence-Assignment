import cv2
import numpy as np
from .image_processor import ImageProcessor

class VideoProcessor:
    def __init__(self, model):
        self.model = model
        self.image_processor = ImageProcessor()
    
    def process_video(self, video_path, confidence_threshold=0.5, frame_skip=2):
        """Process video and yield frames with detections"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for performance
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Process frame
            result_frame, detections = self.image_processor.process_image(
                frame, 
                self.model,
                confidence_threshold
            )
            
            yield result_frame, detections
            frame_count += 1
        
        cap.release()
    
    def process_video_to_file(self, video_path, output_path, confidence_threshold=0.5, frame_skip=2):
        """Process video and save to file"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                # Process frame
                result_frame, _ = self.image_processor.process_image(
                    frame, 
                    self.model,
                    confidence_threshold
                )
                out.write(result_frame)
                processed_frames += 1
            
            frame_count += 1
        
        cap.release()
        out.release()
        return processed_frames