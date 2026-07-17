# ==================
#  IMPORTS & CONFIG
# ==================
import cv2
import os
from pathlib import Path
import numpy as np # 
from ultralytics import YOLO
from collections import defaultdict

# --- Configuration ---
VIDEO_PATH = r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\video1.mp4"
OUTPUT_FOLDER = r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\output\processed_videos"
DETECTION_MODEL_PATH = r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\model\yolo\runs\v1\train\weights\best.pt"
CONF_THRESHOLD = 0.25
MIN_WIDTH = 0
DEVICE = "cuda"
CLASS_ID = 0  # Person (assumes single-class model, adjust if needed)

print(f"Using device: {DEVICE}")

# ==================
#  UTILITY FUNCTIONS
# ==================

def normalize_box(box, frame_shape):
    """Convert (x1, y1, x2, y2) to normalized (xc, yc, w, h)"""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return [xc, yc, bw, bh]

def denormalize_box(xc, yc, bw, bh, frame_shape):
    """Convert normalized (xc, yc, w, h) to (x1, y1, x2, y2) pixel coords"""
    h, w = frame_shape[:2]
    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)
    return [x1, y1, x2, y2]

def draw_person_boxes(frame, bboxes_norm, ids):
    """Draw green boxes and track IDs at the top-left inside each bounding box."""
    for bbox, tid in zip(bboxes_norm, ids):
        x1, y1, x2, y2 = denormalize_box(*bbox, frame.shape)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tag_text = f"ID {int(tid)}"

        # Text params
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        pad = 3

        # Calculate text size and position
        text_size, _ = cv2.getTextSize(tag_text, font, font_scale, thickness)
        tx = x1 + pad
        ty = y1 + text_size[1] + pad

        # Make sure the label is inside the box vertically
        if ty + pad > y2:
            ty = y2 - pad

        # Draw text background
        cv2.rectangle(
            frame,
            (tx - pad, ty - text_size[1] - pad),
            (tx + text_size[0] + pad, ty + pad),
            color,
            -1
        )
        # Draw text (black for contrast)
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

def interpolate_tracks(track_dict, total_frames):
    """
    Fills in missing frames for each track_id by linear interpolation.
    Returns: interp_results[frame_id] = list of (class_id, [xc, yc, w, h], track_id)
    """
    interp_results = defaultdict(list)
    for track_id, detections in track_dict.items():
        detections = sorted(detections, key=lambda x: x[0])  # Sort by frame
        for idx in range(len(detections) - 1):
            f1, bbox1 = detections[idx]
            f2, bbox2 = detections[idx + 1]
            # Add f1
            interp_results[f1].append((CLASS_ID, bbox1, track_id))
            # Linear interpolation for gap
            # if f2 > f1 + 1:
            #     for f in range(f1 + 1, f2):
            #         alpha = (f - f1) / (f2 - f1)
            #         interp_bbox = (1 - alpha) * np.array(bbox1) + alpha * np.array(bbox2)
            #         interp_results[f].append((CLASS_ID, interp_bbox.tolist(), track_id))
        # Add last detection
        # last_f, last_bbox = detections[-1]
        # interp_results[last_f].append((CLASS_ID, last_bbox, track_id))
    # Make sure all frames included
    for frame_id in range(total_frames):
        if frame_id not in interp_results:
            interp_results[frame_id] = []
    return interp_results

def save_labels_txt(interp_results, labels_dir):
    """Save per-frame YOLO label txt files."""
    os.makedirs(labels_dir, exist_ok=True)
    for frame_id, dets in interp_results.items():
        label_path = labels_dir / f"{frame_id}.txt"
        lines = []
        for class_id, bbox, track_id in dets:
            xc, yc, bw, bh = bbox
            line = f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {int(track_id)}"
            lines.append(line)
        if lines:
            label_path.write_text('\n'.join(lines))
        else:
            label_path.write_text('')

# ==================
#  MAIN PROCESSING
# ==================

def process_video(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    labels_dir = Path(output_dir) / "labels"
    output_video_path = os.path.join(output_dir, f"{Path(video_path).stem}_with_id.mp4")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # --- Load Model ---
    detection_model = YOLO(DETECTION_MODEL_PATH)

    # --- Store all detections for interpolation ---
    track_dict = defaultdict(list)

    print(f"\nProcessing video: {Path(video_path).name}")
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        print(f"Processing frame {frame_id}/{total_frames}", end="\r")
        results = detection_model.track(frame, persist=True, classes=[CLASS_ID], conf=CONF_THRESHOLD, verbose=False)
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            # Filter boxes by width if needed
            valid = [i for i, box in enumerate(boxes_xyxy) if (box[2] - box[0]) >= MIN_WIDTH]
            if valid:
                boxes_xyxy = boxes_xyxy[valid]
                ids = ids[valid]
                for box, tid in zip(boxes_xyxy, ids):
                    bbox_norm = normalize_box(box, frame.shape)
                    track_dict[int(tid)].append((frame_id, bbox_norm))
        frame_id += 1

    cap.release()

    # --- Interpolate tracks ---
    interp_results = interpolate_tracks(track_dict, total_frames)

    # --- Save label files ---
    save_labels_txt(interp_results, labels_dir)

    # --- Create video with interpolated detections ---
    cap = cv2.VideoCapture(str(video_path))
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = interp_results[frame_id]
        if dets:
            bboxes_norm = [bbox for _, bbox, _ in dets]
            ids = [tid for _, _, tid in dets]
            draw_person_boxes(frame, bboxes_norm, ids)
        out_vid.write(frame)
        frame_id += 1
    cap.release()
    out_vid.release()

    print(f"\nDone! Interpolated results and video saved to: {output_dir}")

if __name__ == "__main__":
    if not Path(DETECTION_MODEL_PATH).exists():
        print(f"Error: YOLO detection model path does not exist: {DETECTION_MODEL_PATH}")
        exit()
    if not Path(VIDEO_PATH).exists():
        print(f"Error: Video path does not exist: {VIDEO_PATH}")
        exit()
    output_dir = Path(OUTPUT_FOLDER) / Path(VIDEO_PATH).stem
    process_video(VIDEO_PATH, output_dir)