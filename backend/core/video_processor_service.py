"""
Video Processing Service - handles video I/O, detection, tracking,
and result aggregation. Optimized for performance and memory efficiency.
"""

import gc
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any, Tuple

import cv2
import numpy as np


class VideoProcessingService:
    """Processes a video end-to-end: detect -> track -> interpolate -> render."""

    _working_codec = None

    def __init__(self, detector, dataset_config: Dict[str, Any]):
        self.detector = detector
        self.dataset_config = dataset_config
        self.class_colors = dataset_config.get("colors", {})
        self.class_names = dataset_config.get("names", {})

        # Processing state
        self.is_processing = False
        self.progress = 0.0
        self.current_frame = 0
        self.total_frames = 0

        # Results
        self.results = None
        self.output_path = None

        # Tracking data
        self.track_dict = defaultdict(list)
        self.speed_history = {}
        self.prev_positions = {}

    def normalize_box(self, box: List[int], frame_shape: Tuple[int, int]) -> List[float]:
        """Convert pixel box to normalized coordinates."""
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        return [
            ((x1 + x2) / 2) / w,
            ((y1 + y2) / 2) / h,
            (x2 - x1) / w,
            (y2 - y1) / h
        ]

    def denormalize_box(self, xc: float, yc: float, bw: float, bh: float,
                        frame_shape: Tuple[int, int]) -> List[int]:
        """Convert normalized coordinates to pixel box."""
        h, w = frame_shape[:2]
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        return [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]

    def draw_detection_boxes(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection boxes with labels and speeds."""
        for det in detections:
            bbox = det.get("bbox", [0, 0, 100, 100])
            class_id = det.get("class", 0)
            confidence = det.get("confidence", 0)
            speed = det.get("speed", 0)
            track_id = det.get("track_id", None)
            class_name = self.class_names.get(class_id, f"Class {class_id}")

            color = self.class_colors.get(class_id, [0, 255, 0])
            color = tuple(color) if isinstance(color, list) else (0, 255, 0)

            # Draw rectangle
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Prepare label
            label = f"{class_name} {confidence:.2f}"
            if speed > 0:
                label += f" {speed:.1f}km/h"
            if track_id is not None:
                label = f"ID {int(track_id)} " + label

            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                frame,
                (bbox[0], bbox[1] - text_height - 10),
                (bbox[0] + text_width, bbox[1] - 5),
                color, -1
            )
            cv2.putText(
                frame, label, (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        return frame

    def interpolate_tracks(self, total_frames: int) -> Dict[int, List]:
        """Fill in missing frames per track_id via linear interpolation."""
        interp_results = defaultdict(list)

        for track_id, detections in self.track_dict.items():
            detections = sorted(detections, key=lambda x: x[0])

            for idx in range(len(detections) - 1):
                f1, bbox1, class_id = detections[idx]
                f2, bbox2, _ = detections[idx + 1]
                interp_results[f1].append((class_id, bbox1, track_id))

                if f2 > f1 + 1:
                    for f in range(f1 + 1, f2):
                        alpha = (f - f1) / (f2 - f1)
                        interp_bbox = (1 - alpha) * np.array(bbox1) + alpha * np.array(bbox2)
                        interp_results[f].append((class_id, interp_bbox.tolist(), track_id))

            if detections:
                last_f, last_bbox, last_class = detections[-1]
                interp_results[last_f].append((last_class, last_bbox, track_id))

        # Ensure all frames have entries
        for frame_id in range(total_frames):
            if frame_id not in interp_results:
                interp_results[frame_id] = []

        return interp_results

    def calculate_density(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> float:
        """Calculate traffic density as percentage of frame area occupied."""
        if not detections:
            return 0.0

        total_area = frame_shape[0] * frame_shape[1]
        vehicle_area = 0
        for det in detections:
            bbox = det.get("bbox", [0, 0, 100, 100])
            width = max(0, bbox[2] - bbox[0])
            height = max(0, bbox[3] - bbox[1])
            vehicle_area += width * height

        return min((vehicle_area / total_area) * 100, 100.0)

    def process_video(
        self,
        video_path: str,
        confidence_threshold: float = 0.25,
        class_id: int = -1,
        progress_callback: Optional[Callable] = None,
        output_dir: Optional[str] = None,
        frame_skip: int = 1
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process a video with detection, tracking, and speed estimation.
        """
        if cv2 is None:
            raise ImportError("OpenCV is not available")

        self.is_processing = True
        self.progress = 0.0
        self.track_dict.clear()
        self.speed_history.clear()
        self.prev_positions.clear()

        cap = None
        out = None

        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.total_frames = total_frames

            # Read first frame for dimensions
            ret, first_frame = cap.read()
            if not ret:
                cap.release()
                raise ValueError("Could not read any frames from video")

            height, width = first_frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            print(f"📹 Video: {width}x{height}, {fps} FPS, {total_frames} frames")

            # Setup output
            if output_dir is None:
                output_dir = Path("output/processed_videos")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"processed_{timestamp}.mp4"

            # Setup video writer
            out = self._create_video_writer(str(output_path), fps, (width, height))
            if out is None:
                cap.release()
                raise RuntimeError("Could not create video writer")

            # Initialize results
            results = {
                "frames": [],
                "detections": [],
                "speeds": [],
                "density": [],
                "total_vehicles": 0,
                "avg_speed": 0,
                "max_speed": 0,
                "min_speed": 0,
                "processing_time": 0,
                "frames_processed": 0,
                "output_path": str(output_path),
            }

            frame_id = 0
            processed_count = 0
            start_time = time.time()

            # First pass: Collect detections
            print("📊 First pass: Collecting detections...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_id += 1
                self.current_frame = frame_id

                if frame_id % frame_skip != 0:
                    continue

                if progress_callback and total_frames > 0:
                    self.progress = (frame_id / total_frames) * 0.7
                    progress_callback(self.progress, frame_id, total_frames)

                if hasattr(self.detector, "conf_threshold"):
                    self.detector.conf_threshold = confidence_threshold

                detections = self.detector.detect(frame)

                if class_id >= 0:
                    detections = [d for d in detections if d.get("class", -1) == class_id]

                for det in detections:
                    bbox = det.get("bbox", [0, 0, 100, 100])
                    cls_id = det.get("class", 0)
                    track_id = det.get("track_id", frame_id * 100 + len(detections))
                    bbox_norm = self.normalize_box(bbox, frame.shape)
                    self.track_dict[int(track_id)].append((frame_id - 1, bbox_norm, cls_id))

                results["frames"].append(frame_id - 1)
                results["detections"].append(len(detections))
                results["total_vehicles"] += len(detections)

                if frame_id % 50 == 0:
                    gc.collect()

            # Interpolate tracks
            print("🔄 Interpolating tracks...")
            interp_results = self.interpolate_tracks(total_frames)

            # Second pass: Create output video
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

                dets = interp_results.get(frame_id, [])
                frame_detections = []

                for cls_id, bbox_norm, track_id in dets:
                    x1, y1, x2, y2 = self.denormalize_box(*bbox_norm, frame.shape)

                    speed = self._calculate_speed(track_id, x1, y1, x2, y2, fps)

                    frame_detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "class": cls_id,
                        "confidence": 0.8,
                        "track_id": track_id,
                        "speed": round(max(0, speed), 2),
                        "class_name": self.class_names.get(cls_id, f"Class {cls_id}"),
                    })

                    if speed > 0:
                        all_speeds.append(speed)

                self.draw_detection_boxes(frame, frame_detections)

                overlay_text = f"Vehicles: {len(frame_detections)} | Frame: {frame_id}/{total_frames}"
                cv2.putText(
                    frame, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )

                density = self.calculate_density(frame_detections, frame.shape)
                results["density"].append(density)

                avg_speed = np.mean([d["speed"] for d in frame_detections]) if frame_detections else 0
                results["speeds"].append(round(avg_speed, 2))

                out.write(frame)

                processed_count += 1
                frame_id += 1
                if frame_id % 50 == 0:
                    gc.collect()

            # Finalize results
            results["processing_time"] = time.time() - start_time
            results["frames_processed"] = processed_count

            if all_speeds:
                results["avg_speed"] = float(np.mean(all_speeds))
                results["max_speed"] = float(np.max(all_speeds))
                results["min_speed"] = float(np.min(all_speeds))

            print(f"✅ Processing complete! Output: {output_path}")
            print(f"📊 Processed {processed_count} frames, {results['total_vehicles']} vehicles")

            self.results = results
            self.output_path = str(output_path)
            self.is_processing = False

            return str(output_path), results

        except Exception as e:
            self.is_processing = False
            raise e
        finally:
            # Cleanup resources
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
            
            # Safely destroy windows - handle headless OpenCV
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass  # Ignore if not available (headless OpenCV)
            
            self.track_dict.clear()
            self.prev_positions.clear()
            self.speed_history.clear()
            
            gc.collect()
            gc.collect()  # Double collect
            
            # Clear CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass

    def _calculate_speed(self, track_id: int, x1: int, y1: int, x2: int, y2: int, fps: int) -> float:
        """Calculate speed for a track using displacement."""
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        speed = 0
        if track_id in self.prev_positions:
            prev_x, prev_y = self.prev_positions[track_id]
            displacement = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)

            speed = displacement * 0.05 * fps * 3.6

            history = self.speed_history.setdefault(track_id, [])
            history.append(speed)
            if len(history) > 5:
                history.pop(0)
            speed = np.mean(history)

        self.prev_positions[track_id] = (center_x, center_y)
        return speed

    def _create_video_writer(self, output_path: str, fps: int, size: Tuple[int, int]):
        """Create video writer with codec fallback."""
        codec_options = [
            ("mp4v", cv2.VideoWriter_fourcc(*"mp4v")),
            ("avc1", cv2.VideoWriter_fourcc(*"avc1")),
            ("X264", cv2.VideoWriter_fourcc(*"X264")),
            ("MJPG", cv2.VideoWriter_fourcc(*"MJPG")),
        ]

        cached = VideoProcessingService._working_codec
        if cached is not None:
            codec_options = [cached] + [c for c in codec_options if c != cached]

        for name, fourcc in codec_options:
            try:
                out = cv2.VideoWriter(output_path, fourcc, fps, size)
                if out.isOpened():
                    print(f"✅ Using codec: {name}")
                    VideoProcessingService._working_codec = (name, fourcc)
                    return out
                out.release()
            except Exception:
                continue

        return None

    def get_progress(self) -> Dict[str, Any]:
        """Get current processing progress."""
        return {
            "is_processing": self.is_processing,
            "progress": self.progress,
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
        }

    def get_results(self) -> Optional[Dict[str, Any]]:
        """Get processing results."""
        return self.results

    def get_output_path(self) -> Optional[str]:
        """Get output video path."""
        return self.output_path

    def reset(self):
        """Reset service state."""
        self.is_processing = False
        self.progress = 0.0
        self.current_frame = 0
        self.total_frames = 0
        self.results = None
        self.output_path = None
        self.track_dict.clear()
        self.speed_history.clear()
        self.prev_positions.clear()
        gc.collect()