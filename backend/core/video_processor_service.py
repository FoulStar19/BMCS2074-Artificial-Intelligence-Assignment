"""
Video Processing Service

Handles:
- Video I/O
- Vehicle detection
- Vehicle tracking
- Unique vehicle counting
- Speed estimation
- Traffic density estimation
- Processed video generation
- Result management

Optimized version:
- Runs detector only every N frames
- Uses interpolation for skipped frames
- Keeps existing detector unchanged
- Reduces CPU inference time
"""

import cv2
import os
import time
import gc
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
import numpy as np
import sys


# ============================================================
# PATH SETUP
# ============================================================

current_dir = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, current_dir)


class VideoProcessingService:
    """
    Service for processing traffic videos.

    Optimizations:
    - Detector is not executed on every frame.
    - Detection interval can be configured.
    - Missing detections are interpolated.
    - Speed is calculated using tracking history.
    """

    def __init__(
        self,
        detector,
        dataset_config
    ):

        self.detector = detector
        self.dataset_config = dataset_config

        self.class_colors = dataset_config.get(
            "colors",
            {}
        )

        self.class_names = dataset_config.get(
            "names",
            {}
        )

        # ========================================================
        # PROCESSING STATE
        # ========================================================

        self.is_processing = False
        self.progress = 0.0
        self.current_frame = 0
        self.total_frames = 0

        # ========================================================
        # RESULTS
        # ========================================================

        self.results = None
        self.output_path = None

        # ========================================================
        # TRACKING
        # ========================================================

        self.track_dict = defaultdict(list)

        self.prev_positions = {}

        self.position_history = defaultdict(
            lambda: deque(maxlen=10)
        )

        self.speed_history = defaultdict(
            lambda: deque(maxlen=5)
        )

        self.active_tracks = {}

        self.next_track_id = 1

        # ========================================================
        # SPEED CONFIGURATION
        # ========================================================

        # Approximate meters per pixel.
        #
        # IMPORTANT:
        # This is calibration-dependent.
        self.calibration_factor = 0.008

        # Maximum realistic road speed.
        self.max_reasonable_speed = 120.0

        # Number of frames used for speed calculation.
        self.speed_window = 10

        # Maximum centroid distance for tracking.
        #
        # Increased because detection is now performed
        # every few frames instead of every frame.
        self.max_tracking_distance = 150

        # ========================================================
        # PERFORMANCE CONFIGURATION
        # ========================================================

        # Detect every N frames.
        #
        # 1 = original behaviour
        # 3 = balanced
        # 5 = faster
        # 10 = very fast but less accurate
        #
        # Recommended for your assignment:
        # 5
        self.detection_interval = 5

    # ============================================================
    # BOUNDING BOX FUNCTIONS
    # ============================================================

    def normalize_box(
        self,
        box,
        frame_shape
    ):

        h, w = frame_shape[:2]

        x1, y1, x2, y2 = box

        xc = (
            (x1 + x2) / 2
        ) / w

        yc = (
            (y1 + y2) / 2
        ) / h

        bw = (
            x2 - x1
        ) / w

        bh = (
            y2 - y1
        ) / h

        return [
            xc,
            yc,
            bw,
            bh
        ]

    # ------------------------------------------------------------

    def denormalize_box(
        self,
        xc,
        yc,
        bw,
        bh,
        frame_shape
    ):

        h, w = frame_shape[:2]

        x1 = int(
            (xc - bw / 2) * w
        )

        y1 = int(
            (yc - bh / 2) * h
        )

        x2 = int(
            (xc + bw / 2) * w
        )

        y2 = int(
            (yc + bh / 2) * h
        )

        x1 = max(
            0,
            min(
                x1,
                w - 1
            )
        )

        y1 = max(
            0,
            min(
                y1,
                h - 1
            )
        )

        x2 = max(
            0,
            min(
                x2,
                w - 1
            )
        )

        y2 = max(
            0,
            min(
                y2,
                h - 1
            )
        )

        return [
            x1,
            y1,
            x2,
            y2
        ]

    # ============================================================
    # DRAW DETECTIONS
    # ============================================================

    def draw_detection_boxes(
        self,
        frame,
        detections
    ):

        for det in detections:

            bbox = det.get(
                "bbox",
                [0, 0, 100, 100]
            )

            class_id = det.get(
                "class",
                0
            )

            confidence = det.get(
                "confidence",
                0.0
            )

            speed = det.get(
                "speed",
                0.0
            )

            track_id = det.get(
                "track_id",
                None
            )

            class_name = self.class_names.get(
                class_id,
                f"Class {class_id}"
            )

            color = self.class_colors.get(
                class_id,
                [0, 255, 0]
            )

            if isinstance(
                color,
                list
            ):

                color = tuple(color)

            # ====================================================
            # BOUNDING BOX
            # ====================================================

            cv2.rectangle(
                frame,
                (
                    int(bbox[0]),
                    int(bbox[1])
                ),
                (
                    int(bbox[2]),
                    int(bbox[3])
                ),
                color,
                2
            )

            # ====================================================
            # LABEL
            # ====================================================

            if track_id is not None:

                label = (
                    f"ID {int(track_id)} "
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

            else:

                label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

            if speed > 0:

                label += (
                    f" {speed:.1f} km/h"
                )

            (
                text_width,
                text_height
            ), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                2
            )

            label_y1 = max(
                0,
                int(bbox[1])
                - text_height
                - 10
            )

            label_y2 = max(
                text_height + 5,
                int(bbox[1]) - 5
            )

            cv2.rectangle(
                frame,
                (
                    int(bbox[0]),
                    label_y1
                ),
                (
                    int(bbox[0])
                    + text_width,
                    label_y2
                ),
                color,
                -1
            )

            cv2.putText(
                frame,
                label,
                (
                    int(bbox[0]),
                    max(
                        text_height + 5,
                        int(bbox[1]) - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        return frame

    # ============================================================
    # AUTOMATIC TRACKING
    # ============================================================

    def assign_track_ids(
        self,
        detections,
        frame_id
    ):

        if not detections:

            return detections

        current_centers = []

        # ========================================================
        # CALCULATE CENTERS
        # ========================================================

        for det in detections:

            bbox = det.get(
                "bbox",
                [0, 0, 0, 0]
            )

            cx = (
                float(bbox[0])
                + float(bbox[2])
            ) / 2

            cy = (
                float(bbox[1])
                + float(bbox[3])
            ) / 2

            current_centers.append(
                (
                    cx,
                    cy
                )
            )

        used_previous_ids = set()

        # ========================================================
        # KEEP DETECTOR IDS
        # ========================================================

        for det in detections:

            existing_id = det.get(
                "track_id",
                None
            )

            if existing_id is not None:

                try:

                    det["track_id"] = int(
                        existing_id
                    )

                except Exception:

                    det["track_id"] = None

        # ========================================================
        # MATCH DETECTIONS
        # ========================================================

        for i, det in enumerate(
            detections
        ):

            if det.get(
                "track_id"
            ) is not None:

                continue

            cx, cy = current_centers[i]

            best_id = None

            best_distance = float(
                "inf"
            )

            current_class = det.get(
                "class",
                None
            )

            for (
                track_id,
                info
            ) in self.active_tracks.items():

                if (
                    track_id
                    in used_previous_ids
                ):

                    continue

                previous_center = (
                    info["center"]
                )

                previous_class = (
                    info.get(
                        "class",
                        None
                    )
                )

                # Prefer matching same class
                if (
                    current_class is not None
                    and previous_class is not None
                    and current_class != previous_class
                ):

                    continue

                distance = np.sqrt(
                    (
                        cx
                        - previous_center[0]
                    ) ** 2
                    +
                    (
                        cy
                        - previous_center[1]
                    ) ** 2
                )

                if (
                    distance
                    < best_distance
                    and
                    distance
                    <= self.max_tracking_distance
                ):

                    best_distance = (
                        distance
                    )

                    best_id = (
                        track_id
                    )

            if best_id is not None:

                det["track_id"] = (
                    best_id
                )

                used_previous_ids.add(
                    best_id
                )

            else:

                new_id = (
                    self.next_track_id
                )

                self.next_track_id += 1

                det["track_id"] = (
                    new_id
                )

        # ========================================================
        # UPDATE ACTIVE TRACKS
        # ========================================================

        self.active_tracks = {}

        for i, det in enumerate(
            detections
        ):

            track_id = det.get(
                "track_id"
            )

            if track_id is None:
                continue

            cx, cy = (
                current_centers[i]
            )

            self.active_tracks[
                int(track_id)
            ] = {

                "center": (
                    cx,
                    cy
                ),

                "frame": frame_id,

                "class": det.get(
                    "class",
                    None
                )
            }

        return detections

    # ============================================================
    # INTERPOLATION
    # ============================================================

    def interpolate_tracks(
        self,
        total_frames
    ):

        interp_results = (
            defaultdict(list)
        )

        for (
            track_id,
            detections
        ) in self.track_dict.items():

            detections = sorted(
                detections,
                key=lambda x: x[0]
            )

            for idx in range(
                len(detections) - 1
            ):

                (
                    f1,
                    bbox1,
                    class_id
                ) = detections[idx]

                (
                    f2,
                    bbox2,
                    class_id2
                ) = detections[
                    idx + 1
                ]

                # =================================================
                # ADD FIRST DETECTION
                # =================================================

                interp_results[
                    f1
                ].append(
                    (
                        class_id,
                        bbox1,
                        track_id
                    )
                )

                # =================================================
                # INTERPOLATE
                # =================================================

                if f2 > f1 + 1:

                    for f in range(
                        f1 + 1,
                        f2
                    ):

                        alpha = (
                            f - f1
                        ) / (
                            f2 - f1
                        )

                        interp_bbox = (
                            (
                                1 - alpha
                            )
                            * np.array(
                                bbox1
                            )
                            +
                            alpha
                            * np.array(
                                bbox2
                            )
                        )

                        interp_results[
                            f
                        ].append(
                            (
                                class_id,
                                interp_bbox.tolist(),
                                track_id
                            )
                        )

            # =====================================================
            # LAST DETECTION
            # =====================================================

            if detections:

                (
                    last_f,
                    last_bbox,
                    last_class
                ) = detections[-1]

                interp_results[
                    last_f
                ].append(
                    (
                        last_class,
                        last_bbox,
                        track_id
                    )
                )

        # ========================================================
        # ENSURE EVERY FRAME EXISTS
        # ========================================================

        for frame_id in range(
            total_frames
        ):

            if (
                frame_id
                not in interp_results
            ):

                interp_results[
                    frame_id
                ] = []

        return interp_results

    # ============================================================
    # SPEED ESTIMATION
    # ============================================================

    def calculate_speed(
        self,
        track_id,
        center,
        frame_id,
        fps
    ):

        history = (
            self.position_history[
                track_id
            ]
        )

        history.append(
            (
                frame_id,
                center
            )
        )

        if len(history) < 2:

            return 0.0

        # ========================================================
        # SPEED WINDOW
        # ========================================================

        if len(history) >= self.speed_window:

            old_frame, old_pos = (
                history[
                    -self.speed_window
                ]
            )

        else:

            old_frame, old_pos = (
                history[0]
            )

        new_frame, new_pos = (
            history[-1]
        )

        frame_difference = (
            new_frame
            - old_frame
        )

        if frame_difference <= 0:

            return 0.0

        # ========================================================
        # MOVEMENT
        # ========================================================

        dx = (
            float(new_pos[0])
            - float(old_pos[0])
        )

        dy = (
            float(new_pos[1])
            - float(old_pos[1])
        )

        displacement_pixels = float(
            np.sqrt(
                dx ** 2
                +
                dy ** 2
            )
        )

        if displacement_pixels < 1.0:

            return 0.0

        # ========================================================
        # FPS
        # ========================================================

        fps = float(fps)

        if fps <= 0:

            fps = 30.0

        elapsed_time = (
            frame_difference
            / fps
        )

        if elapsed_time <= 0:

            return 0.0

        # ========================================================
        # PIXELS → METERS
        # ========================================================

        displacement_meters = (
            displacement_pixels
            * self.calibration_factor
        )

        speed_mps = (
            displacement_meters
            / elapsed_time
        )

        speed_kmh = (
            speed_mps
            * 3.6
        )

        # ========================================================
        # VALIDATION
        # ========================================================

        if not np.isfinite(
            speed_kmh
        ):

            return 0.0

        if speed_kmh < 0:

            return 0.0

        # ========================================================
        # OUTLIER REJECTION
        # ========================================================

        if (
            speed_kmh
            > self.max_reasonable_speed
        ):

            previous_speeds = (
                self.speed_history[
                    track_id
                ]
            )

            if previous_speeds:

                return round(
                    float(
                        np.median(
                            previous_speeds
                        )
                    ),
                    2
                )

            return 0.0

        # ========================================================
        # SMOOTHING
        # ========================================================

        self.speed_history[
            track_id
        ].append(
            speed_kmh
        )

        smoothed_speed = float(
            np.median(
                self.speed_history[
                    track_id
                ]
            )
        )

        smoothed_speed = min(
            smoothed_speed,
            self.max_reasonable_speed
        )

        return round(
            max(
                0.0,
                smoothed_speed
            ),
            2
        )

    # ============================================================
    # TRAFFIC DENSITY
    # ============================================================

    def calculate_density(
        self,
        detections,
        frame_shape
    ):

        if not detections:

            return 0.0

        h, w = frame_shape[:2]

        scale = 0.25

        mask_h = max(
            1,
            int(h * scale)
        )

        mask_w = max(
            1,
            int(w * scale)
        )

        mask = np.zeros(
            (
                mask_h,
                mask_w
            ),
            dtype=np.uint8
        )

        for det in detections:

            bbox = det.get(
                "bbox",
                [0, 0, 0, 0]
            )

            x1, y1, x2, y2 = bbox

            x1 = int(
                x1 * scale
            )

            y1 = int(
                y1 * scale
            )

            x2 = int(
                x2 * scale
            )

            y2 = int(
                y2 * scale
            )

            x1 = max(
                0,
                min(
                    x1,
                    mask_w - 1
                )
            )

            y1 = max(
                0,
                min(
                    y1,
                    mask_h - 1
                )
            )

            x2 = max(
                0,
                min(
                    x2,
                    mask_w
                )
            )

            y2 = max(
                0,
                min(
                    y2,
                    mask_h
                )
            )

            if (
                x2 > x1
                and y2 > y1
            ):

                cv2.rectangle(
                    mask,
                    (
                        x1,
                        y1
                    ),
                    (
                        x2,
                        y2
                    ),
                    255,
                    -1
                )

        occupied_pixels = (
            np.count_nonzero(
                mask
            )
        )

        total_pixels = (
            mask_h
            * mask_w
        )

        density = (
            occupied_pixels
            / total_pixels
        ) * 100

        return round(
            min(
                max(
                    density,
                    0.0
                ),
                100.0
            ),
            2
        )

    # ============================================================
    # MAIN VIDEO PROCESSING
    # ============================================================

    def process_video(
        self,
        video_path,
        confidence_threshold=0.25,
        class_id=-1,
        progress_callback=None,
        output_dir=None
    ):

        if cv2 is None:

            raise ImportError(
                "OpenCV is not available"
            )

        self.is_processing = True
        self.progress = 0.0

        # ========================================================
        # RESET STATE
        # ========================================================

        self.track_dict = (
            defaultdict(list)
        )

        self.speed_history = (
            defaultdict(
                lambda: deque(maxlen=5)
            )
        )

        self.position_history = (
            defaultdict(
                lambda: deque(maxlen=10)
            )
        )

        self.prev_positions = {}

        self.active_tracks = {}

        self.next_track_id = 1

        # ========================================================
        # OPEN VIDEO
        # ========================================================

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            self.is_processing = False

            raise ValueError(
                f"Could not open video: "
                f"{video_path}"
            )

        # ========================================================
        # VIDEO PROPERTIES
        # ========================================================

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:

            fps = 30.0

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        self.total_frames = (
            total_frames
        )

        print(
            f"📹 Video info - "
            f"FPS: {fps:.2f}, "
            f"Resolution: "
            f"{width}x{height}, "
            f"Frames: {total_frames}"
        )

        print(
            f"⚡ Detection interval: "
            f"every {self.detection_interval} frames"
        )

        print(
            f"📏 Speed calibration: "
            f"{self.calibration_factor} "
            f"meters/pixel"
        )

        # ========================================================
        # OUTPUT DIRECTORY
        # ========================================================

        if output_dir is None:

            output_dir = Path(
                "output/processed_videos"
            )

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        timestamp = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        output_path = (
            output_dir
            / f"processed_{timestamp}.mp4"
        )

        # ========================================================
        # VIDEO WRITER
        # ========================================================

        fourcc = (
            cv2.VideoWriter_fourcc(
                *"mp4v"
            )
        )

        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (
                width,
                height
            )
        )

        if not out.isOpened():

            cap.release()

            self.is_processing = False

            raise RuntimeError(
                "Could not create "
                "video writer"
            )

        # ========================================================
        # RESULTS
        # ========================================================

        results = {

            "frames": [],

            "detections": [],

            "speeds": [],

            "density": [],

            "total_vehicles": 0,

            "avg_speed": 0.0,

            "max_speed": 0.0,

            "min_speed": 0.0,

            "avg_density": 0.0,

            "max_density": 0.0,

            "processing_time": 0.0,

            "frames_processed": 0,

            "output_path": str(
                output_path
            )
        }

        frame_id = 0

        processed_count = 0

        start_time = time.time()

        all_speeds = []

        try:

            # ====================================================
            # FIRST PASS
            # DETECTION
            # ====================================================

            print(
                "\n📊 First pass: "
                "Collecting detections..."
            )

            print(
                "⚡ Fast detection mode enabled"
            )

            while True:

                ret, frame = (
                    cap.read()
                )

                if not ret:

                    break

                current_frame = (
                    frame_id
                )

                self.current_frame = (
                    current_frame
                )

                # ------------------------------------------------
                # PROGRESS
                # ------------------------------------------------

                if (
                    progress_callback
                    and total_frames > 0
                ):

                    progress = (
                        current_frame
                        / total_frames
                    )

                    self.progress = (
                        progress * 0.7
                    )

                    progress_callback(
                        self.progress,
                        current_frame,
                        total_frames
                    )

                # ------------------------------------------------
                # ONLY RUN DETECTOR EVERY N FRAMES
                # ------------------------------------------------

                if (
                    frame_id
                    % self.detection_interval
                    != 0
                ):

                    frame_id += 1

                    continue

                # ------------------------------------------------
                # CONFIDENCE
                # ------------------------------------------------

                if hasattr(
                    self.detector,
                    "conf_threshold"
                ):

                    self.detector.conf_threshold = (
                        confidence_threshold
                    )

                # ------------------------------------------------
                # DETECTION
                # ------------------------------------------------

                detection_start = time.time()

                detections = (
                    self.detector.detect(
                        frame
                    )
                )

                detection_time = (
                    time.time()
                    - detection_start
                )

                print(
                    f"\r🔍 Detection frame "
                    f"{frame_id + 1}/"
                    f"{total_frames} "
                    f"| {detection_time:.2f}s",
                    end=""
                )

                if detections is None:

                    detections = []

                # ------------------------------------------------
                # CLASS FILTER
                # ------------------------------------------------

                if class_id >= 0:

                    detections = [

                        d

                        for d in detections

                        if d.get(
                            "class",
                            -1
                        ) == class_id
                    ]

                # ------------------------------------------------
                # TRACKING
                # ------------------------------------------------

                detections = (
                    self.assign_track_ids(
                        detections,
                        current_frame
                    )
                )

                # ------------------------------------------------
                # STORE TRACKS
                # ------------------------------------------------

                for det in detections:

                    bbox = det.get(
                        "bbox",
                        [
                            0,
                            0,
                            100,
                            100
                        ]
                    )

                    cls_id = det.get(
                        "class",
                        0
                    )

                    track_id = det.get(
                        "track_id"
                    )

                    if track_id is None:

                        continue

                    bbox_norm = (
                        self.normalize_box(
                            bbox,
                            frame.shape
                        )
                    )

                    self.track_dict[
                        int(track_id)
                    ].append(
                        (
                            current_frame,
                            bbox_norm,
                            cls_id
                        )
                    )

                # ------------------------------------------------
                # RESULTS
                # ------------------------------------------------

                # Analytics are collected in the second pass.
                # The detector runs only every N frames, but the
                # report generator needs one analytics row per frame.
                frame_id += 1

            print("\n")

            # ====================================================
            # UNIQUE VEHICLE COUNT
            # ====================================================

            unique_vehicle_ids = set(
                self.track_dict.keys()
            )

            results[
                "total_vehicles"
            ] = len(
                unique_vehicle_ids
            )

            print(
                f"🚗 Unique vehicles detected: "
                f"{results['total_vehicles']}"
            )

            # ====================================================
            # INTERPOLATION
            # ====================================================

            print(
                "🔄 Interpolating skipped frames..."
            )

            interp_results = (
                self.interpolate_tracks(
                    total_frames
                )
            )

            # ====================================================
            # SECOND PASS
            # OUTPUT VIDEO
            # ====================================================

            print(
                "🎬 Second pass: "
                "Creating output video..."
            )

            cap.release()

            cap = cv2.VideoCapture(
                video_path
            )

            frame_id = 0

            self.position_history = (
                defaultdict(
                    lambda: deque(
                        maxlen=10
                    )
                )
            )

            self.speed_history = (
                defaultdict(
                    lambda: deque(
                        maxlen=5
                    )
                )
            )

            while True:

                ret, frame = (
                    cap.read()
                )

                if not ret:

                    break

                # ------------------------------------------------
                # PROGRESS
                # ------------------------------------------------

                if (
                    progress_callback
                    and total_frames > 0
                ):

                    progress = (
                        0.7
                        +
                        (
                            frame_id
                            / total_frames
                        )
                        * 0.3
                    )

                    self.progress = (
                        progress
                    )

                    progress_callback(
                        progress,
                        frame_id,
                        total_frames
                    )

                # ------------------------------------------------
                # INTERPOLATED DETECTIONS
                # ------------------------------------------------

                dets = (
                    interp_results.get(
                        frame_id,
                        []
                    )
                )

                frame_detections = []

                # ------------------------------------------------
                # PROCESS VEHICLES
                # ------------------------------------------------

                for (
                    cls_id,
                    bbox_norm,
                    track_id
                ) in dets:

                    (
                        x1,
                        y1,
                        x2,
                        y2
                    ) = (
                        self.denormalize_box(
                            *bbox_norm,
                            frame.shape
                        )
                    )

                    center_x = (
                        x1 + x2
                    ) // 2

                    center_y = (
                        y1 + y2
                    ) // 2

                    center = (
                        center_x,
                        center_y
                    )

                    # ------------------------------------------------
                    # SPEED
                    # ------------------------------------------------

                    speed = (
                        self.calculate_speed(
                            int(track_id),
                            center,
                            frame_id,
                            fps
                        )
                    )

                    if speed > 0:

                        all_speeds.append(
                            speed
                        )

                    # ------------------------------------------------
                    # DETECTION OBJECT
                    # ------------------------------------------------

                    frame_detections.append(
                        {

                            "bbox": [
                                x1,
                                y1,
                                x2,
                                y2
                            ],

                            "class": cls_id,

                            "confidence": 0.8,

                            "track_id": int(
                                track_id
                            ),

                            "speed": round(
                                speed,
                                2
                            ),

                            "speed_unit":
                                "km/h",

                            "class_name":
                                self.class_names.get(
                                    cls_id,
                                    f"Class {cls_id}"
                                )
                        }
                    )

                # ------------------------------------------------
                # DRAW
                # ------------------------------------------------

                self.draw_detection_boxes(
                    frame,
                    frame_detections
                )

                # ------------------------------------------------
                # OVERLAY
                # ------------------------------------------------

                overlay_text = (
                    f"Vehicles: "
                    f"{len(frame_detections)}"
                    f" | Frame: "
                    f"{frame_id + 1}"
                    f"/{total_frames}"
                )

                cv2.putText(
                    frame,
                    overlay_text,
                    (
                        10,
                        30
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (
                        0,
                        255,
                        255
                    ),
                    2
                )

                # ------------------------------------------------
                # DENSITY
                # ------------------------------------------------

                density = (
                    self.calculate_density(
                        frame_detections,
                        frame.shape
                    )
                )

                # Keep analytics arrays aligned:
                # len(frames) == len(detections) == len(density)
                results["frames"].append(frame_id)
                results["detections"].append(len(frame_detections))
                results["density"].append(density)

                # ------------------------------------------------
                # WRITE
                # ------------------------------------------------

                out.write(
                    frame
                )

                processed_count += 1

                frame_id += 1

            # ====================================================
            # ANALYTICS ARRAY SAFETY CHECK
            # ====================================================

            analytics_lengths = {
                "frames": len(results["frames"]),
                "detections": len(results["detections"]),
                "density": len(results["density"])
            }

            if len(set(analytics_lengths.values())) != 1:
                raise RuntimeError(
                    f"Analytics arrays have different lengths: "
                    f"{analytics_lengths}"
                )

            # ====================================================
            # PROCESSING STATISTICS
            # ====================================================

            results[
                "processing_time"
            ] = (
                time.time()
                - start_time
            )

            results[
                "frames_processed"
            ] = processed_count

            # ====================================================
            # SPEED STATISTICS
            # ====================================================

            valid_speeds = [

                float(s)

                for s in all_speeds

                if (
                    np.isfinite(s)
                    and s > 0
                    and s
                    <= self.max_reasonable_speed
                )
            ]

            if valid_speeds:

                results[
                    "avg_speed"
                ] = round(
                    float(
                        np.mean(
                            valid_speeds
                        )
                    ),
                    2
                )

                results[
                    "max_speed"
                ] = round(
                    float(
                        np.max(
                            valid_speeds
                        )
                    ),
                    2
                )

                results[
                    "min_speed"
                ] = round(
                    float(
                        np.min(
                            valid_speeds
                        )
                    ),
                    2
                )

                results[
                    "speeds"
                ] = [

                    round(
                        float(s),
                        2
                    )

                    for s in valid_speeds
                ]

            # ====================================================
            # DENSITY STATISTICS
            # ====================================================

            if results[
                "density"
            ]:

                results[
                    "avg_density"
                ] = round(
                    float(
                        np.mean(
                            results[
                                "density"
                            ]
                        )
                    ),
                    2
                )

                results[
                    "max_density"
                ] = round(
                    float(
                        np.max(
                            results[
                                "density"
                            ]
                        )
                    ),
                    2
                )

            # ====================================================
            # COMPLETE
            # ====================================================

            print(
                "\n================================"
            )

            print(
                "✅ Processing complete!"
            )

            print(
                "================================"
            )

            print(
                f"🚗 Unique vehicles: "
                f"{results['total_vehicles']}"
            )

            print(
                f"🚀 Average speed: "
                f"{results['avg_speed']:.2f} km/h"
            )

            print(
                f"🚀 Maximum speed: "
                f"{results['max_speed']:.2f} km/h"
            )

            print(
                f"🚀 Minimum speed: "
                f"{results['min_speed']:.2f} km/h"
            )

            print(
                f"📊 Average density: "
                f"{results['avg_density']:.2f}%"
            )

            print(
                f"📊 Maximum density: "
                f"{results['max_density']:.2f}%"
            )

            print(
                f"🎞️ Frames processed: "
                f"{results['frames_processed']}"
            )

            print(
                f"⏱️ Processing time: "
                f"{results['processing_time']:.2f}s"
            )

            print(
                f"💾 Output: "
                f"{output_path}"
            )

            print(
                "================================\n"
            )

            self.results = results

            self.output_path = (
                str(output_path)
            )

            self.is_processing = False

            self.progress = 1.0

            return (
                str(output_path),
                results
            )

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

    # ============================================================
    # PROGRESS
    # ============================================================

    def get_progress(self):

        return {

            "is_processing":
                self.is_processing,

            "progress":
                self.progress,

            "current_frame":
                self.current_frame,

            "total_frames":
                self.total_frames
        }

    # ============================================================
    # RESULTS
    # ============================================================

    def get_results(self):

        return self.results

    # ============================================================
    # OUTPUT PATH
    # ============================================================

    def get_output_path(self):

        return self.output_path

    # ============================================================
    # RESET
    # ============================================================

    def reset(self):

        self.is_processing = False

        self.progress = 0.0

        self.current_frame = 0

        self.total_frames = 0

        self.results = None

        self.output_path = None

        self.track_dict = (
            defaultdict(list)
        )

        self.speed_history = (
            defaultdict(
                lambda: deque(maxlen=5)
            )
        )

        self.position_history = (
            defaultdict(
                lambda: deque(maxlen=10)
            )
        )

        self.prev_positions = {}

        self.active_tracks = {}

        self.next_track_id = 1

        gc.collect()