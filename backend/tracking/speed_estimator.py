"""
Vehicle speed estimation using tracking history.

Speed is calculated from video frame timing rather than wall-clock
processing time, making the result independent of CPU/GPU speed.
"""

import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class SpeedRecord:
    """Data class for speed records."""
    track_id: int
    speed: float
    timestamp: float
    position: Tuple[int, int]


class SpeedEstimator:
    """
    Estimate vehicle speed using tracked vehicle positions.

    The calculation uses:
        pixel displacement
        -> meters using calibration_factor
        -> meters/second using video FPS
        -> km/h

    Important:
        Video FPS is used instead of time.time() so that processing
        speed does not affect the estimated vehicle speed.
    """

    def __init__(
        self,
        fps: float = 30.0,
        calibration_factor: float = 0.02,
        smoothing_window: int = 5,
        min_samples: int = 3,
        max_reasonable_speed: float = 120.0
    ):
        self.fps = max(float(fps), 1.0)

        # Meters per pixel.
        # This should be adjusted according to the camera/view.
        self.calibration_factor = calibration_factor

        self.smoothing_window = smoothing_window
        self.min_samples = min_samples
        self.max_reasonable_speed = max_reasonable_speed

        # Track history:
        # track_id -> [(position, frame_number), ...]
        self.track_history: Dict[int, deque] = {}

        # Smoothed speed history:
        self.speed_history: Dict[int, deque] = {}

        # Statistics
        self.all_speeds: List[float] = []
        self.speed_records: List[SpeedRecord] = []

        # Video frame counter
        self.frame_number = 0

        # Optical flow is optional.
        self.prev_gray = None
        self.flow = None

    # ------------------------------------------------------------------
    # MAIN METHOD
    # ------------------------------------------------------------------

    def estimate(
        self,
        detections: List[dict],
        frame: np.ndarray,
        use_optical_flow: bool = False
    ) -> List[dict]:
        """
        Estimate speed for detected vehicles.

        Args:
            detections:
                Detection dictionaries containing:
                - track_id
                - bbox

            frame:
                Current video frame.

            use_optical_flow:
                Disabled by default because position-based tracking
                is faster and more stable for this application.

        Returns:
            Detections with speed and speed_unit fields.
        """

        # Increase frame counter once per processed video frame.
        self.frame_number += 1

        if frame is None:
            return detections

        # Optical flow is optional.
        # Keep disabled by default for much faster processing.
        if use_optical_flow:
            self._compute_optical_flow(frame)

        for det in detections:

            track_id = det.get("track_id")

            if track_id is None:
                det["speed"] = 0.0
                det["speed_unit"] = "km/h"
                continue

            bbox = det.get("bbox")

            if bbox is None or len(bbox) < 4:
                det["speed"] = 0.0
                det["speed_unit"] = "km/h"
                continue

            # ----------------------------------------------------------
            # Vehicle center
            # ----------------------------------------------------------

            x1, y1, x2, y2 = bbox[:4]

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            current_pos = (center_x, center_y)

            # ----------------------------------------------------------
            # Initialize track
            # ----------------------------------------------------------

            if track_id not in self.track_history:

                self.track_history[track_id] = deque(
                    maxlen=max(self.smoothing_window * 3, 15)
                )

                self.speed_history[track_id] = deque(
                    maxlen=self.smoothing_window
                )

            # ----------------------------------------------------------
            # Store current position
            # ----------------------------------------------------------

            self.track_history[track_id].append(
                (current_pos, self.frame_number)
            )

            # ----------------------------------------------------------
            # Calculate speed
            # ----------------------------------------------------------

            speed = self._estimate_speed_for_track(track_id)

            # Store speed history only when valid
            if speed > 0:

                self.speed_history[track_id].append(speed)

                # Keep statistics clean
                self.all_speeds.append(speed)

                self.speed_records.append(
                    SpeedRecord(
                        track_id=track_id,
                        speed=speed,
                        timestamp=self.frame_number / self.fps,
                        position=current_pos
                    )
                )

            det["speed"] = round(speed, 2)
            det["speed_unit"] = "km/h"

        return detections

    # ------------------------------------------------------------------
    # SPEED CALCULATION
    # ------------------------------------------------------------------

    def _estimate_speed_for_track(self, track_id: int) -> float:

        history = self.track_history.get(track_id)

        if history is None:
            return 0.0

        if len(history) < self.min_samples:
            return 0.0

        history_list = list(history)

        # --------------------------------------------------------------
        # Use a previous point several frames back instead of only
        # comparing consecutive frames.
        # --------------------------------------------------------------

        current_pos, current_frame = history_list[-1]

        # Prefer approximately smoothing_window frames ago.
        previous_index = max(
            0,
            len(history_list) - self.smoothing_window - 1
        )

        previous_pos, previous_frame = history_list[previous_index]

        frame_difference = current_frame - previous_frame

        if frame_difference <= 0:
            return 0.0

        # --------------------------------------------------------------
        # Pixel displacement
        # --------------------------------------------------------------

        dx = current_pos[0] - previous_pos[0]
        dy = current_pos[1] - previous_pos[1]

        displacement_pixels = float(
            np.sqrt(dx ** 2 + dy ** 2)
        )

        # Ignore tiny movements/noise
        if displacement_pixels < 0.5:
            return 0.0

        # --------------------------------------------------------------
        # Convert pixels -> meters
        # --------------------------------------------------------------

        displacement_meters = (
            displacement_pixels *
            self.calibration_factor
        )

        # --------------------------------------------------------------
        # Convert frames -> seconds
        # --------------------------------------------------------------

        elapsed_seconds = frame_difference / self.fps

        if elapsed_seconds <= 0:
            return 0.0

        # --------------------------------------------------------------
        # m/s -> km/h
        # --------------------------------------------------------------

        speed_mps = displacement_meters / elapsed_seconds

        speed_kmh = speed_mps * 3.6

        # --------------------------------------------------------------
        # Reject unrealistic values
        # --------------------------------------------------------------

        if speed_kmh < 0:
            return 0.0

        if speed_kmh > self.max_reasonable_speed:

            # Clamp extreme tracking noise instead of allowing
            # thousands of km/h.
            speed_kmh = self.max_reasonable_speed

        # --------------------------------------------------------------
        # Temporal smoothing
        # --------------------------------------------------------------

        previous_speeds = self.speed_history.get(track_id)

        if previous_speeds and len(previous_speeds) > 0:

            previous_average = float(
                np.mean(previous_speeds)
            )

            # 60% current value + 40% previous average
            speed_kmh = (
                0.6 * speed_kmh +
                0.4 * previous_average
            )

        return round(float(speed_kmh), 2)

    # ------------------------------------------------------------------
    # OPTIONAL OPTICAL FLOW
    # ------------------------------------------------------------------

    def _compute_optical_flow(
        self,
        frame: np.ndarray
    ) -> Optional[np.ndarray]:

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

    # ------------------------------------------------------------------
    # OPTICAL FLOW SPEED
    # ------------------------------------------------------------------

    def _get_optical_flow_speed(
        self,
        position: Tuple[int, int]
    ) -> float:

        if self.flow is None:
            return 0.0

        x, y = position

        h, w = self.flow.shape[:2]

        if not (0 <= x < w and 0 <= y < h):
            return 0.0

        flow_vector = self.flow[y, x]

        magnitude = float(
            np.sqrt(
                flow_vector[0] ** 2 +
                flow_vector[1] ** 2
            )
        )

        speed_kmh = (
            magnitude *
            self.calibration_factor *
            self.fps *
            3.6
        )

        return min(
            max(speed_kmh, 0.0),
            self.max_reasonable_speed
        )

    # ------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:

        if not self.all_speeds:

            return {
                "avg_speed": 0.0,
                "max_speed": 0.0,
                "min_speed": 0.0,
                "std_speed": 0.0,
                "total_samples": 0
            }

        speeds = np.array(
            self.all_speeds,
            dtype=float
        )

        return {
            "avg_speed": round(float(np.mean(speeds)), 2),
            "max_speed": round(float(np.max(speeds)), 2),
            "min_speed": round(float(np.min(speeds)), 2),
            "std_speed": round(float(np.std(speeds)), 2),
            "total_samples": len(speeds)
        }

    # ------------------------------------------------------------------
    # SPEED DISTRIBUTION
    # ------------------------------------------------------------------

    def get_speed_distribution(
        self,
        bins: int = 10
    ) -> dict:

        if not self.all_speeds:

            return {
                "bins": [],
                "counts": []
            }

        hist, bin_edges = np.histogram(
            self.all_speeds,
            bins=bins
        )

        return {
            "bins": bin_edges.tolist(),
            "counts": hist.tolist()
        }

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self):

        self.speed_history.clear()
        self.track_history.clear()

        self.all_speeds.clear()
        self.speed_records.clear()

        self.frame_number = 0

        self.prev_gray = None
        self.flow = None

    # ------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------

    def set_calibration(
        self,
        calibration_factor: float
    ):

        if calibration_factor <= 0:
            raise ValueError(
                "Calibration factor must be greater than 0."
            )

        self.calibration_factor = float(
            calibration_factor
        )

    def set_fps(self, fps: float):

        if fps <= 0:
            raise ValueError(
                "FPS must be greater than 0."
            )

        self.fps = float(fps)