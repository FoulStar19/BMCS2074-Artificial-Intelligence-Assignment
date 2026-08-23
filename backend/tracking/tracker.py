"""
Vehicle tracking module using distance matching and Kalman filtering.

Designed for traffic video vehicle detection.
Provides stable track IDs and more reliable speed estimation.
"""

import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class TrackedVehicle:
    """Data class for a tracked vehicle."""

    track_id: int
    bbox: List[int]
    center: Tuple[int, int]
    class_id: int
    confidence: float

    speed: float = 0.0

    # Store recent center positions
    history: deque = field(default_factory=lambda: deque(maxlen=30))

    # Number of consecutive frames where this track was not detected
    lost_frames: int = 0

    # Total frames this track has existed
    age: int = 0

    active: bool = True

    # Smoothed speed
    smoothed_speed: float = 0.0


class VehicleTracker:
    """
    Vehicle tracker using:
        - Center-distance matching
        - IoU matching
        - Kalman filtering
        - Frame-based tracking
        - Smoothed speed estimation

    Important:
        Speed is calculated using video FPS rather than time.time().
        This prevents processing speed from affecting the estimated speed.
    """

    def __init__(
        self,
        max_lost_frames: int = 15,
        min_distance: float = 80.0,
        kalman: bool = True,
        optical_flow: bool = False,
        fps: float = 30.0,
        calibration_factor: float = 0.05,
        max_speed_kmh: float = 180.0,
        smoothing_alpha: float = 0.25,
    ):
        """
        Args:
            max_lost_frames:
                Number of consecutive frames a vehicle can disappear
                before its track is removed.

            min_distance:
                Maximum center distance allowed when matching detections.

            kalman:
                Enable Kalman filtering.

            optical_flow:
                Optional optical flow. Disabled by default because
                perspective traffic videos can produce unstable flow speeds.

            fps:
                FPS of the input video.

            calibration_factor:
                Approximate meters per pixel.

            max_speed_kmh:
                Maximum physically reasonable speed used to reject
                unrealistic speed estimates.

            smoothing_alpha:
                EMA smoothing factor for speed.
                Smaller = smoother.
        """

        self.max_lost_frames = max_lost_frames
        self.min_distance = min_distance

        self.kalman_enabled = kalman
        self.optical_flow_enabled = optical_flow

        self.fps = max(float(fps), 1.0)
        self.calibration_factor = float(calibration_factor)

        self.max_speed_kmh = float(max_speed_kmh)
        self.smoothing_alpha = float(smoothing_alpha)

        # Track storage
        self.next_track_id = 0
        self.tracks: Dict[int, TrackedVehicle] = {}

        # Frame counter
        self.frame_index = 0

        # Optional optical flow
        self.prev_gray = None
        self.flow = None

        # Kalman filters
        self.kalman_filters: Dict[int, cv2.KalmanFilter] = {}

    # ============================================================
    # KALMAN FILTER
    # ============================================================

    def _init_kalman_filter(
        self,
        center_x: float,
        center_y: float
    ) -> cv2.KalmanFilter:
        """Create a Kalman filter for one vehicle."""

        kf = cv2.KalmanFilter(4, 2)

        # State:
        # [x, y, vx, vy]

        kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ],
            dtype=np.float32
        )

        kf.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ],
            dtype=np.float32
        )

        kf.processNoiseCov = (
            np.eye(4, dtype=np.float32) * 0.03
        )

        kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * 0.5
        )

        kf.errorCovPost = (
            np.eye(4, dtype=np.float32)
        )

        kf.statePre = np.array(
            [
                [center_x],
                [center_y],
                [0],
                [0]
            ],
            dtype=np.float32
        )

        kf.statePost = np.array(
            [
                [center_x],
                [center_y],
                [0],
                [0]
            ],
            dtype=np.float32
        )

        return kf

    # ============================================================
    # OPTICAL FLOW
    # ============================================================

    def _compute_optical_flow(
        self,
        frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Compute optical flow.

        This is kept for compatibility but is NOT used as the
        primary speed calculation.
        """

        if frame is None:
            return None

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        self.flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        self.prev_gray = gray

        return self.flow

    # ============================================================
    # IOU
    # ============================================================

    def _calculate_iou(
        self,
        bbox1: List[int],
        bbox2: List[int]
    ) -> float:
        """Calculate Intersection over Union."""

        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])

        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection_width = max(
            0,
            x2 - x1
        )

        intersection_height = max(
            0,
            y2 - y1
        )

        intersection = (
            intersection_width *
            intersection_height
        )

        area1 = max(
            0,
            bbox1[2] - bbox1[0]
        ) * max(
            0,
            bbox1[3] - bbox1[1]
        )

        area2 = max(
            0,
            bbox2[2] - bbox2[0]
        ) * max(
            0,
            bbox2[3] - bbox2[1]
        )

        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    # ============================================================
    # CENTER
    # ============================================================

    @staticmethod
    def _get_center(
        bbox: List[int]
    ) -> Tuple[int, int]:
        """Get center point of bounding box."""

        x1, y1, x2, y2 = bbox

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )

        return center_x, center_y

    # ============================================================
    # MATCHING
    # ============================================================

    def _match_detections(
        self,
        detections: List[dict]
    ) -> Dict[int, int]:
        """
        Match current detections with existing tracks.

        Matching priority:
            1. Same class
            2. Distance
            3. IoU
            4. One detection can only be assigned once

        Returns:
            {track_id: detection_index}
        """

        if not self.tracks:
            return {}

        if not detections:
            return {}

        active_tracks = [
            (track_id, track)
            for track_id, track in self.tracks.items()
            if track.active
        ]

        if not active_tracks:
            return {}

        detection_centers = []

        for det in detections:
            bbox = det.get(
                "bbox",
                [0, 0, 100, 100]
            )

            detection_centers.append(
                self._get_center(bbox)
            )

        # Candidate matches:
        # (distance, -iou, track_id, det_index)

        candidates = []

        for track_id, track in active_tracks:

            # Predict next location if Kalman enabled
            predicted_center = track.center

            if (
                self.kalman_enabled
                and track_id in self.kalman_filters
            ):
                kf = self.kalman_filters[track_id]

                prediction = kf.predict()

                predicted_x = float(
                    prediction[0, 0]
                )

                predicted_y = float(
                    prediction[1, 0]
                )

                predicted_center = (
                    int(predicted_x),
                    int(predicted_y)
                )

            for det_index, det in enumerate(detections):

                # Don't match different classes when class information
                # is available.
                track_class = track.class_id
                detection_class = det.get(
                    "class",
                    track_class
                )

                if (
                    track_class is not None
                    and detection_class is not None
                    and track_class != detection_class
                ):
                    continue

                dx = (
                    predicted_center[0]
                    - detection_centers[det_index][0]
                )

                dy = (
                    predicted_center[1]
                    - detection_centers[det_index][1]
                )

                distance = float(
                    np.sqrt(
                        dx * dx +
                        dy * dy
                    )
                )

                bbox = det.get(
                    "bbox",
                    [0, 0, 100, 100]
                )

                iou = self._calculate_iou(
                    track.bbox,
                    bbox
                )

                # Allow a slightly larger distance for vehicles
                # that have overlapping bounding boxes.
                allowed_distance = self.min_distance

                if iou > 0.2:
                    allowed_distance *= 1.5

                if distance <= allowed_distance:

                    candidates.append(
                        (
                            distance,
                            -iou,
                            track_id,
                            det_index
                        )
                    )

        # Best matches first
        candidates.sort(
            key=lambda x: (
                x[0],
                x[1]
            )
        )

        matches = {}

        used_tracks = set()
        used_detections = set()

        for (
            distance,
            negative_iou,
            track_id,
            det_index
        ) in candidates:

            if track_id in used_tracks:
                continue

            if det_index in used_detections:
                continue

            matches[track_id] = det_index

            used_tracks.add(track_id)
            used_detections.add(det_index)

        return matches

    # ============================================================
    # UPDATE
    # ============================================================

    def update(
        self,
        frame: np.ndarray,
        detections: List[dict]
    ) -> List[dict]:
        """
        Update tracker.

        Args:
            frame:
                Current video frame.

            detections:
                Detection dictionaries.

        Returns:
            Detections with:
                track_id
                speed
                class_name
        """

        self.frame_index += 1

        if frame is None:
            return detections

        if detections is None:
            detections = []

        # Optional optical flow
        if self.optical_flow_enabled:
            self._compute_optical_flow(frame)

        # --------------------------------------------------------
        # Match detections to existing tracks
        # --------------------------------------------------------

        matches = self._match_detections(
            detections
        )

        matched_detection_indices = set(
            matches.values()
        )

        # --------------------------------------------------------
        # Update matched tracks
        # --------------------------------------------------------

        for track_id, det_index in matches.items():

            det = detections[det_index]

            bbox = det.get(
                "bbox",
                [0, 0, 100, 100]
            )

            center = self._get_center(
                bbox
            )

            track = self.tracks[track_id]

            # Update track
            track.bbox = list(bbox)

            track.center = center

            track.class_id = det.get(
                "class",
                track.class_id
            )

            track.confidence = float(
                det.get(
                    "confidence",
                    track.confidence
                )
            )

            track.lost_frames = 0

            track.age += 1

            # ----------------------------------------------------
            # Kalman correction
            # ----------------------------------------------------

            if (
                self.kalman_enabled
                and track_id in self.kalman_filters
            ):

                kf = self.kalman_filters[
                    track_id
                ]

                measurement = np.array(
                    [
                        [np.float32(center[0])],
                        [np.float32(center[1])]
                    ]
                )

                # Correct with actual detection
                corrected = kf.correct(
                    measurement
                )

                corrected_x = float(
                    corrected[0, 0]
                )

                corrected_y = float(
                    corrected[1, 0]
                )

                # Keep actual detection center
                # for more reliable speed calculation.
                track.center = (
                    int(center[0]),
                    int(center[1])
                )

            # Store position history
            track.history.append(
                (
                    center[0],
                    center[1]
                )
            )

            # Calculate speed
            speed = self._calculate_speed(
                track
            )

            track.speed = speed

            # Add information to detection
            det["track_id"] = track_id

            det["speed"] = speed

            det["speed_unit"] = "km/h"

            det["class_name"] = det.get(
                "class_name",
                f"Class {track.class_id}"
            )

            detections[det_index] = det

        # --------------------------------------------------------
        # Create new tracks
        # --------------------------------------------------------

        for det_index, det in enumerate(
            detections
        ):

            if det_index in matched_detection_indices:
                continue

            bbox = det.get(
                "bbox",
                [0, 0, 100, 100]
            )

            center = self._get_center(
                bbox
            )

            class_id = det.get(
                "class",
                0
            )

            confidence = float(
                det.get(
                    "confidence",
                    0.8
                )
            )

            track_id = self.next_track_id

            self.next_track_id += 1

            track = TrackedVehicle(
                track_id=track_id,
                bbox=list(bbox),
                center=center,
                class_id=class_id,
                confidence=confidence
            )

            track.history.append(
                (
                    center[0],
                    center[1]
                )
            )

            self.tracks[track_id] = track

            # Kalman filter
            if self.kalman_enabled:

                self.kalman_filters[
                    track_id
                ] = self._init_kalman_filter(
                    center[0],
                    center[1]
                )

            # New detection has no reliable speed yet
            det["track_id"] = track_id

            det["speed"] = 0.0

            det["speed_unit"] = "km/h"

            det["class_name"] = det.get(
                "class_name",
                f"Class {class_id}"
            )

            detections[det_index] = det

        # --------------------------------------------------------
        # Mark unmatched tracks as lost
        # --------------------------------------------------------

        matched_track_ids = set(
            matches.keys()
        )

        for track_id, track in list(
            self.tracks.items()
        ):

            if track_id in matched_track_ids:
                continue

            track.lost_frames += 1

            # Don't immediately delete.
            # This helps preserve IDs when a car is temporarily
            # hidden by another vehicle.
            if (
                track.lost_frames
                > self.max_lost_frames
            ):

                track.active = False

        # --------------------------------------------------------
        # Remove inactive tracks
        # --------------------------------------------------------

        tracks_to_delete = [
            track_id
            for track_id, track
            in self.tracks.items()
            if not track.active
        ]

        for track_id in tracks_to_delete:

            del self.tracks[track_id]

            if track_id in self.kalman_filters:
                del self.kalman_filters[
                    track_id
                ]

        return detections

    # ============================================================
    # SPEED
    # ============================================================

    def _calculate_speed(
        self,
        track: TrackedVehicle
    ) -> float:
        """
        Calculate vehicle speed.

        Uses frame-based timing:

            dt = 1 / FPS

        This is important because processing speed should NOT
        affect estimated vehicle speed.
        """

        if len(track.history) < 2:
            return 0.0

        # Use positions separated by several frames
        # instead of only adjacent frames.
        history = list(
            track.history
        )

        sample_gap = min(
            3,
            len(history) - 1
        )

        pos1 = history[
            -1 - sample_gap
        ]

        pos2 = history[-1]

        dx = (
            pos2[0] -
            pos1[0]
        )

        dy = (
            pos2[1] -
            pos1[1]
        )

        displacement_pixels = float(
            np.sqrt(
                dx * dx +
                dy * dy
            )
        )

        # Time represented by these frames
        dt = (
            sample_gap /
            self.fps
        )

        if dt <= 0:
            return 0.0

        # Pixel -> meter
        displacement_meters = (
            displacement_pixels *
            self.calibration_factor
        )

        # m/s
        speed_mps = (
            displacement_meters /
            dt
        )

        # km/h
        raw_speed = (
            speed_mps *
            3.6
        )

        # --------------------------------------------------------
        # Reject impossible values
        # --------------------------------------------------------

        if not np.isfinite(raw_speed):
            return 0.0

        raw_speed = max(
            0.0,
            raw_speed
        )

        # Hard physical sanity check
        if raw_speed > self.max_speed_kmh:
            raw_speed = self.max_speed_kmh

        # --------------------------------------------------------
        # Exponential moving average
        # --------------------------------------------------------

        if track.smoothed_speed <= 0:

            track.smoothed_speed = raw_speed

        else:

            alpha = self.smoothing_alpha

            track.smoothed_speed = (
                alpha * raw_speed
                +
                (1.0 - alpha)
                * track.smoothed_speed
            )

        smoothed = track.smoothed_speed

        # Final safety check
        smoothed = min(
            max(
                smoothed,
                0.0
            ),
            self.max_speed_kmh
        )

        return round(
            smoothed,
            2
        )

    # ============================================================
    # RESET
    # ============================================================

    def reset(self):
        """Reset all tracking state."""

        self.tracks.clear()

        self.kalman_filters.clear()

        self.next_track_id = 0

        self.frame_index = 0

        self.prev_gray = None

        self.flow = None

    # ============================================================
    # ACTIVE TRACKS
    # ============================================================

    def get_active_tracks(
        self
    ) -> List[TrackedVehicle]:
        """Return currently active tracks."""

        return [
            track
            for track in self.tracks.values()
            if track.active
        ]

    # ============================================================
    # DRAW TRAILS
    # ============================================================

    def draw_trails(
        self,
        frame: np.ndarray,
        trail_length: int = 20
    ) -> np.ndarray:
        """Draw vehicle tracking trails."""

        annotated_frame = frame.copy()

        for track in self.tracks.values():

            if not track.active:
                continue

            if len(track.history) < 2:
                continue

            points = list(
                track.history
            )[-trail_length:]

            # Draw trail
            for i in range(
                1,
                len(points)
            ):

                pt1 = (
                    int(points[i - 1][0]),
                    int(points[i - 1][1])
                )

                pt2 = (
                    int(points[i][0]),
                    int(points[i][1])
                )

                # Fade effect
                alpha = (
                    i /
                    max(len(points), 1)
                )

                color = (
                    0,
                    int(255 * alpha),
                    0
                )

                cv2.line(
                    annotated_frame,
                    pt1,
                    pt2,
                    color,
                    2
                )

            # Current position
            current = points[-1]

            cv2.circle(
                annotated_frame,
                (
                    int(current[0]),
                    int(current[1])
                ),
                5,
                (0, 255, 0),
                -1
            )

            # Track ID
            label = (
                f"ID {track.track_id}"
            )

            cv2.putText(
                annotated_frame,
                label,
                (
                    int(current[0]) + 8,
                    int(current[1]) - 8
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        return annotated_frame

    # ============================================================
    # CONFIGURATION
    # ============================================================

    def set_fps(
        self,
        fps: float
    ):
        """Update video FPS."""

        if fps > 0:
            self.fps = float(fps)

    def set_calibration(
        self,
        calibration_factor: float
    ):
        """Update meters-per-pixel calibration."""

        if calibration_factor > 0:
            self.calibration_factor = float(
                calibration_factor
            )

    def set_max_speed(
        self,
        max_speed_kmh: float
    ):
        """Update maximum allowed speed."""

        if max_speed_kmh > 0:
            self.max_speed_kmh = float(
                max_speed_kmh
            )