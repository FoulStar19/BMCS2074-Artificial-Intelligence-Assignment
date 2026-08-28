"""
Video Processing Service - Handles video I/O, processing, and result management
"""

import gc
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# Add parent directory to path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)


class VideoProcessingService:
    """
    Service for processing videos with detection and tracking
    """
    
    def __init__(self, detector, dataset_config, classifier=None):
        """
        Initialize the video processing service

        Args:
            detector: Primary detection model instance (YOLODetector or
                FasterRCNNDetector)
            dataset_config: Dataset configuration
            classifier: Optional CNNDetector used as a verification pass --
                each box `detector` finds is cropped and re-checked with
                classifier.classify_crop(); boxes it disagrees with are
                dropped. Pass None to skip verification (default).
        """
        self.detector = detector
        self.dataset_config = dataset_config
        self.classifier = classifier
        self.class_colors = dataset_config.get('colors', {})
        self.class_names = dataset_config.get('names', {})
        
        # Processing state
        self.is_processing = False
        self.progress = 0.0
        self.current_frame = 0
        self.total_frames = 0
        
        # Results
        self.results = None
        self.output_path = None
        
        # Tracking state
        self.track_dict = defaultdict(list)

        # Preserve the detector's exact output for every frame.
        # The renderer can therefore draw real detections instead of relying
        # exclusively on reconstructed/interpolated tracks.
        self.frame_detections = defaultdict(list)

        self.prev_positions = {}

    def _verify_with_classifier(self, frame, detections):
        """Drop detections the CNN classifier disagrees are vehicles.

        No-op when no classifier was supplied (self.classifier is None),
        so this pass is entirely opt-in.
        """
        if self.classifier is None or not detections:
            return detections

        verified = []
        for det in detections:
            bbox = det.get('bbox', [0, 0, 100, 100])
            try:
                # classify_bbox crops `bbox` out of `frame` itself and
                # classifies it -- classify_crop takes an already-cropped
                # image and has no (frame, bbox) signature.
                result = self.classifier.classify_bbox(frame, bbox)
            except Exception as e:
                print(f"⚠️ CNN verification failed for a box, keeping it: {e}")
                verified.append(det)
                continue

            # classify_bbox returns None for a degenerate/out-of-frame box,
            # or {"class": "unknown", ...} when confidence fell below the
            # classifier's threshold. Either way the CNN disagreed with the
            # primary detector, so drop the box; otherwise keep it.
            if result is None or result.get('class') == 'unknown':
                continue

            verified.append(det)

        return verified

    def normalize_box(self, box, frame_shape):
        """Convert (x1, y1, x2, y2) to normalized (xc, yc, w, h)"""
        h, w = frame_shape[:2]

        x1, y1, x2, y2 = box
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        return [xc, yc, bw, bh]
    
    def denormalize_box(self, xc, yc, bw, bh, frame_shape):
        """Convert normalized (xc, yc, w, h) to (x1, y1, x2, y2) pixel coords"""
        h, w = frame_shape[:2]
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        return [x1, y1, x2, y2]
    
    def draw_detection_boxes(self, frame, detections):
        """Draw bounding boxes with class names and confidence"""
        for det in detections:
            bbox = det.get('bbox', [0, 0, 100, 100])
            class_id = det.get('class', 0)
            confidence = det.get('confidence', 0)
            speed = det.get('speed', 0)
            track_id = det.get('track_id', None)
            class_name = self.class_names.get(class_id, f'Class {class_id}')
            
            # Get color for this class
            color = self.class_colors.get(class_id, [0, 255, 0])
            if isinstance(color, list):
                color = tuple(color)
            
            # Draw bounding box
            cv2.rectangle(frame, 
                        (bbox[0], bbox[1]), 
                        (bbox[2], bbox[3]), 
                        color, 2)
            
            # Create label
            label = f"{class_name} {confidence:.2f}"
            if speed > 0:
                label += f" {speed:.1f}km/h"
            if track_id is not None:
                label = f"{class_name} | Speed: {speed:.1f} km/h"
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(frame,
                        (bbox[0], bbox[1] - text_height - 10),
                        (bbox[0] + text_width, bbox[1] - 5),
                        color, -1)
            
            # Draw label text
            cv2.putText(frame, label,
                      (bbox[0], bbox[1] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                      (255, 255, 255), 2)
        
        return frame
    
    def interpolate_tracks(self, total_frames):
        """
        Fills in missing frames for each track_id by linear interpolation
        
        Returns:
            interp_results[frame_id] = list of (class_id, [xc, yc, w, h], track_id)
        """
        interp_results = defaultdict(list)
        
        for track_id, detections in self.track_dict.items():
            detections = sorted(detections, key=lambda x: x[0])  # Sort by frame
            for idx in range(len(detections) - 1):
                f1, bbox1, class_id = detections[idx]
                f2, bbox2, class_id2 = detections[idx + 1]
                # Add f1
                interp_results[f1].append((class_id, bbox1, track_id))
                # Linear interpolation for gap
                if f2 > f1 + 1:
                    for f in range(f1 + 1, f2):
                        alpha = (f - f1) / (f2 - f1)
                        interp_bbox = (1 - alpha) * np.array(bbox1) + alpha * np.array(bbox2)
                        interp_results[f].append((class_id, interp_bbox.tolist(), track_id))
            # Add last detection
            if detections:
                last_f, last_bbox, last_class = detections[-1]
                interp_results[last_f].append((last_class, last_bbox, track_id))
        
        # Make sure all frames are included
        for frame_id in range(total_frames):
            if frame_id not in interp_results:
                interp_results[frame_id] = []
        
        return interp_results
    
    def calculate_density(self, detections, frame_shape):
        """Calculate traffic density based on detections"""
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
    
    def process_video(self, video_path, confidence_threshold=0.25, 
                      class_id=-1, progress_callback=None, output_dir=None):
        """
        Process video with tracking and interpolation
        
        Args:
            video_path: Path to video file
            confidence_threshold: Confidence threshold
            class_id: Class ID to track (-1 for all)
            progress_callback: Progress callback function
            output_dir: Output directory for processed video
        
        Returns:
            output_path, results
        """
        if cv2 is None:
            raise ImportError("OpenCV is not available")
        
        self.is_processing = True
        self.progress = 0.0
        
        # Reset processing/tracking state
        self.track_dict = defaultdict(list)
        self.frame_detections = defaultdict(list)
        self.speed_history = {}
        self.prev_positions = {}

        # Reset the detector-owned tracker before every new video.
        if hasattr(self.detector, "reset_tracker"):
            try:
                self.detector.reset_tracker()
            except Exception as e:
                print(f"⚠️ Could not reset detector tracker: {e}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = total_frames
        
        print(f"📹 Video info - FPS: {fps}, Resolution: {width}x{height}, Frames: {total_frames}")
        
        # Create output directory
        if output_dir is None:
            output_dir = Path("output/processed_videos")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"processed_{timestamp}.mp4"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError("Could not create video writer")
        
        # Results storage
        results = {
            'frames': [],
            'detections': [],
            'speeds': [],
            'density': [],
            'total_vehicles': 0,
            'avg_speed': 0,
            'max_speed': 0,
            'min_speed': 0,
            'processing_time': 0,
            'frames_processed': 0,
            'output_path': str(output_path)
        }
        
        frame_id = 0
        processed_count = 0
        start_time = time.time()
        
        try:
            # First pass: Collect detections
            print("📊 First pass: Collecting detections...")
            while True:

                ret, frame = (
                    cap.read()
                )

                if not ret:

                    break
                
                frame_id += 1
                self.current_frame = frame_id
                
                # Update progress
                if progress_callback and total_frames > 0:
                    progress = frame_id / total_frames
                    self.progress = progress * 0.7
                    progress_callback(self.progress, frame_id, total_frames)
                
                # Update confidence threshold
                if hasattr(self.detector, 'conf_threshold'):
                    self.detector.conf_threshold = confidence_threshold
                
                # Get detections
                detections = self.detector.detect(frame)
                
                # Filter by class if specified
                if class_id >= 0:
                    detections = [d for d in detections if d.get('class', -1) == class_id]

                # Optional CNN verification pass - drops boxes the CNN
                # classifier disagrees are vehicles. No-op if no
                # classifier was supplied.
                detections = self._verify_with_classifier(frame, detections)
                
                # Preserve the exact model output for this frame.
                frame_key = frame_id - 1
                self.frame_detections[frame_key] = [
                    {
                        "bbox": list(d.get("bbox", [0, 0, 100, 100])),
                        "class": int(d.get("class", 0)),
                        "confidence": float(d.get("confidence", 0.0)),
                        "track_id": d.get("track_id"),
                        "class_name": d.get(
                            "class_name",
                            self.class_names.get(
                                int(d.get("class", 0)),
                                f"Class {d.get('class', 0)}"
                            )
                        ),
                    }
                    for d in detections
                ]

                # Store detections for interpolation as a fallback.
                if detections:
                    for det in detections:
                        bbox = det.get('bbox', [0, 0, 100, 100])
                        cls_id = det.get('class', 0)
                        track_id = det.get('track_id', frame_id * 100 + len(detections))
                        
                        # Normalize box
                        bbox_norm = self.normalize_box(bbox, frame.shape)
                        self.track_dict[int(track_id)].append((frame_id - 1, bbox_norm, cls_id))
                
                # Store results
                results['frames'].append(frame_id - 1)
                results['detections'].append(len(detections))
                results['total_vehicles'] += len(detections)
                
                # Memory cleanup
                if frame_id % 50 == 0:
                    gc.collect()
            
            # Second pass: render the EXACT detections collected in pass 1.
            # No re-inference and no interpolation is needed for drawing boxes.
            print("🎬 Second pass: Creating output video...")
            cap.release()
            cap = cv2.VideoCapture(video_path)

            frame_id = 0
            all_speeds = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if progress_callback and total_frames > 0:
                    progress = 0.7 + (frame_id / total_frames) * 0.3
                    self.progress = progress
                    progress_callback(progress, frame_id, total_frames)

                frame_detections = [
                    dict(det)
                    for det in self.frame_detections.get(frame_id, [])
                ]

                # Preserve the detector/tracker boxes exactly as returned.
                # Calculate a display-only speed without changing the bbox.
                for i, det in enumerate(frame_detections):
                    bbox = det.get("bbox", [0, 0, 100, 100])
                    track_id = det.get("track_id")

                    if track_id is None:
                        track_id = frame_id * 1000 + i
                        det["track_id"] = track_id

                    x1, y1, x2, y2 = map(int, bbox)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    speed = float(det.get("speed", 0.0) or 0.0)

                    if track_id in self.prev_positions:
                        prev_x, prev_y = self.prev_positions[track_id]
                        displacement = np.sqrt(
                            (center[0] - prev_x) ** 2 +
                            (center[1] - prev_y) ** 2
                        )
                        speed = displacement * 0.05 * fps * 3.6

                        history = self.speed_history.setdefault(track_id, [])
                        history.append(speed)
                        if len(history) > 5:
                            history.pop(0)
                        speed = float(np.mean(history))

                    self.prev_positions[track_id] = center
                    det["speed"] = round(max(0.0, speed), 2)

                    if speed > 0:
                        all_speeds.append(speed)

                self.draw_detection_boxes(frame, frame_detections)

                overlay_text = (
                    f"Vehicles: {len(frame_detections)} | "
                    f"Frame: {frame_id + 1}/{total_frames}"
                )
                cv2.putText(
                    frame,
                    overlay_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                density = self.calculate_density(
                    frame_detections, frame.shape
                )
                results["density"].append(density)

                frame_speeds = [
                    d["speed"] for d in frame_detections
                    if d.get("speed", 0) > 0
                ]
                results["speeds"].append(
                    float(np.mean(frame_speeds))
                    if frame_speeds else 0.0
                )

                out.write(frame)
                processed_count += 1
                frame_id += 1

                if frame_id % 50 == 0:
                    gc.collect()

            # Calculate statistics
            results['processing_time'] = time.time() - start_time
            results['frames_processed'] = processed_count
            
            if all_speeds:
                results['avg_speed'] = np.mean(all_speeds)
                results['max_speed'] = np.max(all_speeds)
                results['min_speed'] = np.min(all_speeds)
            
            print(f"✅ Processing complete! Output saved to: {output_path}")
            print(f"📊 Processed {processed_count} frames, {results['total_vehicles']} vehicles detected")
            
            self.results = results

            self.output_path = (
                str(output_path)
            )

            self.is_processing = False
            
            return str(output_path), results
            
        except Exception as e:

            self.is_processing = False

            print(
                f"❌ Processing error: "
                f"{e}"
            )

            raise

        finally:
            cap.release()
            out.release()
            gc.collect()
    
    def get_progress(self):
        """Get current processing progress"""
        return {
            'is_processing': self.is_processing,
            'progress': self.progress,
            'current_frame': self.current_frame,
            'total_frames': self.total_frames
        }
    
    def get_results(self):
        """Get processing results"""
        return self.results
    
    def get_output_path(self):
        """Get output video path"""
        return self.output_path
    
    def reset(self):
        """Reset the service state"""
        self.is_processing = False

        self.progress = 0.0

        self.current_frame = 0

        self.total_frames = 0

        self.results = None

        self.output_path = None
        self.track_dict = defaultdict(list)
        self.speed_history = {}
        self.prev_positions = {}
        gc.collect()