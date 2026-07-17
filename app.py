"""
Traffic AI Detection System
Main Streamlit application for vehicle detection and speed estimation
"""

import streamlit as st
import cv2
import torch
import os
import tempfile
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import yaml
import shutil
import glob
import sys
import gc
import warnings
import base64
import subprocess
from collections import deque

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Traffic AI Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton > button:disabled {
        background-color: #cccccc;
        cursor: not-allowed;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .upload-container {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #f8f9fa;
    }
    .video-container {
        background-color: #000;
        border-radius: 10px;
        overflow: hidden;
        padding: 0;
    }
    .stVideo {
        max-height: 500px;
    }
    .info-box {
        padding: 1rem;
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Import backend modules with error handling
try:
    from backend.detection.yolo_detector import YOLODetector
    from backend.detection.cnn_detector import CNNDetector
    from backend.speed_estimator import SpeedEstimator
    from backend.video_processor import VideoProcessor
    from backend.utils import load_dataset_config, get_class_colors
except ImportError as e:
    st.warning(f"Some backend modules not found: {e}")
    # Create dummy classes for demonstration
    class DummyDetector:
        def detect(self, frame):
            h, w = frame.shape[:2]
            return [{'bbox': [100, 100, 200, 300], 'confidence': 0.95, 'class': 0} for _ in range(5)]
        
        def detect_frame(self, frame):
            return self.detect(frame)
    
    YOLODetector = DummyDetector
    CNNDetector = DummyDetector
    SpeedEstimator = None
    VideoProcessor = None
    
    def load_dataset_config():
        return {
            'nc': 5,
            'names': {
                0: 'car', 1: 'truck', 2: 'bus', 
                3: 'motorcycle', 4: 'bicycle'
            },
            'colors': {
                0: [0, 0, 255], 1: [0, 255, 0], 2: [255, 0, 0],
                3: [0, 255, 255], 4: [255, 0, 255]
            }
        }
    
    def get_class_colors():
        return load_dataset_config().get('colors', {})

# ==========================
# UTILITY FUNCTIONS
# ==========================

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'processing': False,
        'detections_history': [],
        'speed_history': [],
        'density_history': [],
        'processed_video_path': None,
        'detector': None,
        'current_results': None,
        'is_processing': False,
        'video_processed': False,
        'uploaded_video_path': None,
        'model_loaded': False,
        'processing_complete': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def handle_file_upload(uploaded_file):
    """
    Handle file upload with chunked reading to prevent memory issues
    
    Args:
        uploaded_file: Uploaded file object
    
    Returns:
        Path to saved file or None if failed
    """
    if uploaded_file is None:
        return None
    
    try:
        # Get file size
        uploaded_file.seek(0, 2)
        file_size = uploaded_file.tell()
        uploaded_file.seek(0)
        
        file_size_mb = file_size / (1024 * 1024)
        
        # Create temp file with proper extension
        file_extension = os.path.splitext(uploaded_file.name)[1]
        if not file_extension:
            file_extension = '.mp4'
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_path = temp_file.name
        
        # Show progress for large files
        progress_bar = None
        status_text = None
        
        if file_size_mb > 20:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text(f"📤 Uploading {file_size_mb:.1f} MB file...")
        
        # Read and write in chunks
        chunk_size = 5 * 1024 * 1024  # 5MB chunks
        total_read = 0
        
        while True:
            chunk = uploaded_file.read(chunk_size)
            if not chunk:
                break
            
            temp_file.write(chunk)
            total_read += len(chunk)
            
            # Update progress for large files
            if progress_bar is not None and file_size > 0:
                progress = min(total_read / file_size, 1.0)
                progress_bar.progress(progress)
                if status_text:
                    status_text.text(f"📤 Uploading: {total_read/(1024*1024):.1f}/{file_size_mb:.1f} MB")
        
        temp_file.close()
        
        # Clear progress indicators
        if progress_bar is not None:
            progress_bar.empty()
        if status_text is not None:
            status_text.empty()
        
        return temp_path
        
    except Exception as e:
        st.error(f"❌ Upload failed: {str(e)}")
        return None

def get_video_base64(video_path):
    """
    Convert video file to base64 for HTML5 video player
    
    Args:
        video_path: Path to video file
    
    Returns:
        Base64 encoded video string
    """
    try:
        # Read video in chunks to avoid memory issues
        chunk_size = 1024 * 1024  # 1MB chunks
        encoded_parts = []
        
        with open(video_path, 'rb') as video_file:
            while True:
                chunk = video_file.read(chunk_size)
                if not chunk:
                    break
                encoded_parts.append(base64.b64encode(chunk).decode('utf-8'))
        
        return ''.join(encoded_parts)
    except Exception as e:
        print(f"Error encoding video: {e}")
        return ""

def compress_video_ffmpeg(input_path, output_path=None, quality=28):
    """
    Compress video using ffmpeg
    
    Args:
        input_path: Path to input video
        output_path: Path for compressed video (optional)
        quality: CRF value (higher = smaller file)
    
    Returns:
        Path to compressed video or None if failed
    """
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, f"compressed_{base_name}.mp4")
    
    try:
        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        
        # Compress video
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264', '-crf', str(quality),
            '-preset', 'fast',
            '-c:a', 'aac', '-b:a', '64k',
            '-movflags', '+faststart',
            '-y', output_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        
        if os.path.exists(output_path):
            return output_path
        return None
        
    except Exception as e:
        print(f"Error compressing video with ffmpeg: {e}")
        return None

def compress_video_opencv(input_path, output_path=None, target_size_mb=20):
    """
    Compress video using OpenCV
    
    Args:
        input_path: Path to input video
        output_path: Path for compressed video (optional)
        target_size_mb: Target file size in MB
    
    Returns:
        Path to compressed video or None if failed
    """
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, f"compressed_{base_name}.mp4")
    
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Reduce resolution for smaller file
        target_width = 640
        if width > target_width:
            scale = target_width / width
            new_width = target_width
            new_height = int(height * scale)
        else:
            new_width = width
            new_height = height
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
        
        # Process frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize if needed
            if new_width != width:
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Reduce quality
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 70]
            _, frame_compressed = cv2.imencode('.jpg', frame, encode_param)
            frame = cv2.imdecode(frame_compressed, cv2.IMREAD_COLOR)
            
            out.write(frame)
            frame_count += 1
            
            # Progress update (every 100 frames)
            if frame_count % 100 == 0 and total_frames > 0:
                progress = frame_count / total_frames
                print(f"Compressing: {progress*100:.1f}%")
        
        cap.release()
        out.release()
        
        # Check if compression was effective
        if os.path.exists(output_path):
            compressed_size = os.path.getsize(output_path) / (1024 * 1024)
            original_size = os.path.getsize(input_path) / (1024 * 1024)
            
            if compressed_size < original_size:
                return output_path
        
        return None
        
    except Exception as e:
        print(f"Error compressing video with OpenCV: {e}")
        return None

# ==========================
# MODEL MANAGEMENT
# ==========================

def get_available_models():
    """
    Get available model versions from runs folder
    
    Returns:
        Dictionary of model versions and paths
    """
    models = {}
    
    # Get the current directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Check multiple possible locations
    possible_paths = [
        Path("model/yolo/runs"),
        Path("model/runs"),
        Path("runs"),
        Path("yolo/runs"),
        Path("cnn/runs"),
        Path("model/cnn/runs"),
        current_dir / "model" / "yolo" / "runs",
        current_dir / "model" / "runs",
        current_dir / "yolo" / "runs",
        current_dir / "runs",
    ]
    
    # Add specific Windows path
    windows_path = Path(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\model\yolo\runs")
    if windows_path.exists():
        possible_paths.append(windows_path)
    
    for runs_path in possible_paths:
        if runs_path.exists():
            # Look for version directories (v1, v2, etc.)
            version_dirs = [d for d in runs_path.iterdir() if d.is_dir() and d.name.startswith('v')]
            
            for version_dir in version_dirs:
                # Check for train directories inside version
                for sub_dir in version_dir.iterdir():
                    if sub_dir.is_dir() and (sub_dir.name.startswith('train') or sub_dir.name == 'train'):
                        weights_dir = sub_dir / "weights"
                        if weights_dir.exists():
                            # Look for best.pt
                            best_pt = weights_dir / "best.pt"
                            if best_pt.exists():
                                version_name = f"{version_dir.name}/{sub_dir.name}"
                                models[version_name] = str(best_pt)
                            
                            # Look for other .pt files
                            for pt_file in weights_dir.glob("*.pt"):
                                if pt_file.name != "best.pt":
                                    version_name = f"{version_dir.name}/{sub_dir.name}/{pt_file.stem}"
                                    models[version_name] = str(pt_file)
            
            # Look for train directories directly
            for train_dir in runs_path.iterdir():
                if train_dir.is_dir() and (train_dir.name.startswith('train') or train_dir.name == 'train'):
                    weights_dir = train_dir / "weights"
                    if weights_dir.exists():
                        best_pt = weights_dir / "best.pt"
                        if best_pt.exists():
                            models[train_dir.name] = str(best_pt)
                        
                        for pt_file in weights_dir.glob("*.pt"):
                            if pt_file.name != "best.pt":
                                version_name = f"{train_dir.name}/{pt_file.stem}"
                                models[version_name] = str(pt_file)
    
    # If no models found, search for .pt files
    if not models:
        search_paths = [
            current_dir,
            current_dir / "model",
            current_dir / "model" / "yolo",
            Path(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment"),
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                pt_files = list(search_path.glob("**/*.pt"))
                for pt_file in pt_files:
                    if "runs" in str(pt_file):
                        continue
                    if "venv" in str(pt_file) or "site-packages" in str(pt_file):
                        continue
                    version_name = f"{pt_file.parent.name}/{pt_file.stem}"
                    models[version_name] = str(pt_file)
    
    # Specific check for v1 model
    specific_path = Path(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\model\yolo\runs\v1\train\weights\best.pt")
    if specific_path.exists():
        models["v1/train/best"] = str(specific_path)
    
    return models

def load_dataset_yaml():
    """
    Load dataset configuration from YAML file
    
    Returns:
        Configuration dictionary
    """
    yaml_paths = [
        Path("dataset.yaml"),
        Path("model/yolo/dataset.yaml"),
        Path("config/dataset.yaml"),
        Path("dataset/dataset.yaml"),
        Path(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\dataset.yaml"),
        Path(os.path.dirname(os.path.abspath(__file__))) / "dataset.yaml",
    ]
    
    for yaml_path in yaml_paths:
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    config = yaml.safe_load(f)
                return config
            except Exception as e:
                print(f"Error loading {yaml_path}: {e}")
    
    # Return default configuration
    return {
        'nc': 5,
        'names': {
            0: 'car', 1: 'truck', 2: 'bus', 
            3: 'motorcycle', 4: 'bicycle'
        },
        'colors': {
            0: [0, 0, 255], 1: [0, 255, 0], 2: [255, 0, 0],
            3: [0, 255, 255], 4: [255, 0, 255]
        }
    }

def load_model(model_type, model_path, device='cpu'):
    """
    Load the selected model
    
    Args:
        model_type: 'YOLO' or 'CNN'
        model_path: Path to model weights
        device: 'cpu' or 'cuda'
    
    Returns:
        Detector instance
    """
    try:
        if model_type == 'YOLO':
            detector = YOLODetector(
                model_path=model_path,
                device=device,
                conf_threshold=0.5
            )
            return detector
        else:  # CNN
            detector = CNNDetector(
                model_path=model_path,
                device=device,
                confidence_threshold=0.5
            )
            return detector
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ==========================
# VIDEO PROCESSING
# ==========================

class VideoProcessorWrapper:
    """Wrapper class for video processing to handle missing modules"""
    def __init__(self, detector, speed_estimator, frame_skip=2):
        self.detector = detector
        self.speed_estimator = speed_estimator
        self.frame_skip = frame_skip
    
    def process_frame_sync(self, frame):
        """Process a single frame synchronously"""
        try:
            detections = self.detector.detect(frame)
        except AttributeError:
            try:
                detections = self.detector.detect_frame(frame)
            except AttributeError:
                h, w = frame.shape[:2]
                detections = [{'bbox': [100, 100, 200, 300], 'confidence': 0.95, 'class': 0} for _ in range(5)]
        
        for det in detections:
            if self.speed_estimator:
                try:
                    speed = self.speed_estimator.estimate_speed(det)
                    det['speed'] = speed
                except:
                    det['speed'] = 0
            else:
                det['speed'] = 0
        
        return detections

def calculate_density(detections, frame_shape):
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

def process_video_with_models(
    video_path,
    detector,
    dataset_config,
    frame_skip=2,
    progress_callback=None,
    target_fps=60
):
    """
    Process video with detection and speed estimation
    
    Args:
        video_path: Path to video file
        detector: Detection model instance
        dataset_config: Dataset configuration dictionary
        frame_skip: Process every Nth frame
        progress_callback: Callback for progress updates
        target_fps: Target FPS for output video
    
    Returns:
        Tuple of (display_video_path, results_dict)
    """
    # Initialize components
    try:
        if SpeedEstimator:
            speed_estimator = SpeedEstimator(fps=target_fps, calibration_factor=0.05)
        else:
            speed_estimator = None
    except:
        speed_estimator = None
    
    video_processor = VideoProcessorWrapper(
        detector=detector,
        speed_estimator=speed_estimator,
        frame_skip=frame_skip
    )
    
    class_colors = dataset_config.get('colors', {})
    class_names = dataset_config.get('names', {})
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")
    
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    interpolation_factor = max(1, int(target_fps / original_fps))
    if interpolation_factor > 4:
        interpolation_factor = 4
    
    actual_output_fps = original_fps * interpolation_factor
    
    output_dir = Path("outputs/processed_videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"processed_{timestamp}.mp4"
    
    # Try different codecs - expanded list with more options
    codecs = [
        ('mp4v', 'mp4v'),  # MPEG-4 codec
        ('X264', 'X264'),  # H.264
        ('avc1', 'avc1'),  # H.264
        ('H264', 'H264'),  # H.264
        ('XVID', 'XVID'),  # Xvid
        ('MJPG', 'MJPG'),  # Motion JPEG
        ('DIVX', 'DIVX'),  # DivX
        ('FMP4', 'FMP4'),  # FFmpeg MPEG-4
    ]
    
    out = None
    selected_codec = None
    
    for codec_name, codec_fourcc in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)
            test_writer = cv2.VideoWriter(
                str(output_path), 
                fourcc, 
                actual_output_fps, 
                (width, height)
            )
            if test_writer.isOpened():
                out = test_writer
                selected_codec = codec_name
                print(f"✅ Using codec: {codec_name}")
                break
            else:
                test_writer.release()
        except Exception as e:
            print(f"Codec {codec_name} failed: {e}")
            continue
    
    # If all codecs fail, try with AVI container
    if out is None or not out.isOpened():
        output_path = output_dir / f"processed_{timestamp}.avi"
        try:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(str(output_path), fourcc, actual_output_fps, (width, height))
            if out.isOpened():
                selected_codec = 'MJPG (AVI)'
                print("✅ Using MJPG codec with AVI container")
            else:
                raise ValueError("Could not create video writer with any codec")
        except Exception as e:
            raise ValueError(f"Could not create video writer: {e}")
    
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
        'frames_processed': 0
    }
    
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    prev_frame = None
    prev_detections = None
    max_results_frames = 500
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % frame_skip != 0:
                out.write(frame)
                del frame
                continue
            
            try:
                detections = video_processor.process_frame_sync(frame)
                processed_frame = frame.copy()
                
                if len(results['frames']) < max_results_frames:
                    results['frames'].append(frame_count)
                    results['detections'].append(len(detections))
                    results['total_vehicles'] += len(detections)
                    
                    if detections:
                        speeds = [det.get('speed', 0) for det in detections if det.get('speed', 0) > 0]
                        avg_speed = np.mean(speeds) if speeds else 0
                        results['speeds'].append(avg_speed)
                        density = calculate_density(detections, frame.shape)
                        results['density'].append(density)
                    else:
                        results['speeds'].append(0)
                        results['density'].append(0)
                
                processed_count += 1
                
                # Draw detections
                for det in detections:
                    bbox = det.get('bbox', [0, 0, 100, 100])
                    class_id = det.get('class', 0)
                    confidence = det.get('confidence', 0)
                    speed = det.get('speed', 0)
                    class_name = class_names.get(class_id, f'Class {class_id}')
                    
                    color = class_colors.get(class_id, [0, 255, 0])
                    if isinstance(color, list):
                        color = tuple(color)
                    
                    cv2.rectangle(processed_frame, 
                                (bbox[0], bbox[1]), 
                                (bbox[2], bbox[3]), 
                                color, 2)
                    
                    label = f"{class_name} {confidence:.2f}"
                    if speed > 0:
                        label += f" {speed:.1f}km/h"
                    
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    cv2.rectangle(processed_frame,
                                (bbox[0], bbox[1] - text_height - 10),
                                (bbox[0] + text_width, bbox[1] - 5),
                                color, -1)
                    
                    cv2.putText(processed_frame, label,
                              (bbox[0], bbox[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (255, 255, 255), 2)
                
                overlay_text = f"Detections: {len(detections)} | Frames: {processed_count}"
                cv2.putText(processed_frame, overlay_text,
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                          0.7, (0, 255, 255), 2)
                
                out.write(processed_frame)
                
                if prev_frame is not None and interpolation_factor > 1 and prev_detections is not None:
                    num_interpolations = min(interpolation_factor - 1, 2)
                    for i in range(1, num_interpolations + 1):
                        alpha = i / (num_interpolations + 1)
                        interp_frame = cv2.addWeighted(prev_frame, 1 - alpha, processed_frame, alpha, 0)
                        out.write(interp_frame)
                
                prev_frame = processed_frame.copy()
                prev_detections = detections
                
                del frame
                if processed_frame is not None:
                    del processed_frame
                
                if processed_count % 50 == 0:
                    gc.collect()
                
                if progress_callback and total_frames > 0:
                    progress = frame_count / total_frames
                    progress_callback(progress, frame_count, total_frames)
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                out.write(frame)
                continue
    
    except MemoryError:
        print("Memory error occurred. Running garbage collection...")
        gc.collect()
        raise
    finally:
        cap.release()
        out.release()
    
    results['processing_time'] = time.time() - start_time
    results['frames_processed'] = processed_count
    
    if results['speeds']:
        speeds_filtered = [s for s in results['speeds'] if s > 0]
        if speeds_filtered:
            results['avg_speed'] = np.mean(speeds_filtered)
            results['max_speed'] = np.max(speeds_filtered)
            results['min_speed'] = np.min(speeds_filtered)
    
    print(f"✅ Processing complete! Output saved to: {output_path}")
    print(f"📊 Processed {processed_count} frames, {results['total_vehicles']} vehicles detected")
    print(f"🎥 Codec used: {selected_codec}")
    
    # Create compressed version for display if needed
    output_path_str = str(output_path)
    file_size_mb = os.path.getsize(output_path_str) / (1024 * 1024)
    display_path = output_path_str
    
    # If file is too large or is AVI, convert to MP4 for better compatibility
    if file_size_mb > 50 or output_path.suffix == '.avi':
        print(f"⚠️ Video file is large ({file_size_mb:.1f} MB) or AVI format. Creating compressed MP4...")
        compressed_path = str(output_dir / f"compressed_{timestamp}.mp4")
        
        # Try ffmpeg first
        compressed = compress_video_ffmpeg(output_path_str, compressed_path, quality=28)
        
        # If ffmpeg fails, try OpenCV conversion
        if compressed is None:
            compressed = convert_to_mp4_opencv(output_path_str, compressed_path)
        
        if compressed and os.path.exists(compressed):
            compressed_size = os.path.getsize(compressed) / (1024 * 1024)
            print(f"✅ Converted/compressed video created: {compressed} ({compressed_size:.1f} MB)")
            display_path = compressed
    
    return display_path, results


def convert_to_mp4_opencv(input_path, output_path):
    """
    Convert video to MP4 format using OpenCV
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
    
    Returns:
        Path to converted video or None if failed
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Try different MP4 codecs
        codecs = [
            ('mp4v', 'mp4v'),
            ('X264', 'X264'),
            ('avc1', 'avc1'),
        ]
        
        out = None
        for codec_name, codec_fourcc in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    print(f"✅ Converting with codec: {codec_name}")
                    break
                else:
                    out.release()
                    out = None
            except:
                continue
        
        if out is None:
            return None
        
        # Copy frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0 and total_frames > 0:
                print(f"Converting: {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        return output_path
        
    except Exception as e:
        print(f"Error converting video: {e}")
        return None


def compress_video_ffmpeg(input_path, output_path=None, quality=28):
    """
    Compress video using ffmpeg
    
    Args:
        input_path: Path to input video
        output_path: Path for compressed video (optional)
        quality: CRF value (higher = smaller file)
    
    Returns:
        Path to compressed video or None if failed
    """
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, f"compressed_{base_name}.mp4")
    
    try:
        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        
        # Check if input file exists
        if not os.path.exists(input_path):
            return None
        
        # Compress video with more robust settings
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-crf', str(quality),
            '-preset', 'medium',
            '-c:a', 'aac',
            '-b:a', '64k',
            '-movflags', '+faststart',
            '-y',
            output_path
        ]
        
        # Add -hwaccel if available (for faster processing)
        try:
            # Check if hardware acceleration is available
            subprocess.run(['ffmpeg', '-hwaccels'], capture_output=True, check=True)
            cmd.insert(1, '-hwaccel')
            cmd.insert(2, 'cuda')  # For NVIDIA GPUs
        except:
            pass
        
        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return None
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        
        return None
        
    except Exception as e:
        print(f"Error compressing video with ffmpeg: {e}")
        return None


def compress_video_opencv(input_path, output_path=None, target_size_mb=20):
    """
    Compress video using OpenCV
    
    Args:
        input_path: Path to input video
        output_path: Path for compressed video (optional)
        target_size_mb: Target file size in MB
    
    Returns:
        Path to compressed video or None if failed
    """
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, f"compressed_{base_name}.mp4")
    
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Reduce resolution for smaller file
        target_width = 640
        if width > target_width:
            scale = target_width / width
            new_width = target_width
            new_height = int(height * scale)
        else:
            new_width = width
            new_height = height
        
        # Try different codecs
        codecs = [
            ('mp4v', 'mp4v'),
            ('X264', 'X264'),
            ('avc1', 'avc1'),
        ]
        
        out = None
        for codec_name, codec_fourcc in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)
                out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
                if out.isOpened():
                    break
                else:
                    out.release()
                    out = None
            except:
                continue
        
        if out is None:
            return None
        
        # Process frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize if needed
            if new_width != width:
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Reduce quality
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 70]
            _, frame_compressed = cv2.imencode('.jpg', frame, encode_param)
            frame = cv2.imdecode(frame_compressed, cv2.IMREAD_COLOR)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0 and total_frames > 0:
                print(f"Compressing: {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        if os.path.exists(output_path):
            compressed_size = os.path.getsize(output_path) / (1024 * 1024)
            original_size = os.path.getsize(input_path) / (1024 * 1024)
            
            if compressed_size < original_size:
                return output_path
        
        return None
        
    except Exception as e:
        print(f"Error compressing video with OpenCV: {e}")
        return None

# ==========================
# RESULTS DISPLAY
# ==========================

def display_results(results, output_video_path):
    """
    Display detection results in Streamlit
    
    Args:
        results: Results dictionary
        output_video_path: Path to processed video
    """
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🚗 Total Vehicles", results['total_vehicles'])
    
    with col2:
        st.metric("📏 Avg Speed", f"{results['avg_speed']:.1f} km/h")
    
    with col3:
        st.metric("📈 Max Speed", f"{results['max_speed']:.1f} km/h")
    
    with col4:
        st.metric("⏱️ Processing Time", f"{results['processing_time']:.1f}s")
    
    # Display processed video
    if output_video_path and os.path.exists(output_video_path):
        st.subheader("📹 Processed Video")
        
        file_size = os.path.getsize(output_video_path) / (1024 * 1024)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if file_size > 50:
                st.warning(f"⚠️ Video file is large ({file_size:.1f} MB). Using streaming player...")
                
                # Use HTML5 video with streaming
                video_base64 = get_video_base64(output_video_path)
                
                video_html = f"""
                <video width="100%" controls autoplay muted style="max-height: 500px; background: #000;">
                    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <p style="color: #666; font-size: 12px; margin-top: 5px;">
                    Size: {file_size:.1f} MB
                </p>
                """
                st.markdown(video_html, unsafe_allow_html=True)
            else:
                with open(output_video_path, 'rb') as f:
                    video_bytes = f.read()
                st.video(video_bytes)
        
        with col2:
            st.write("**📥 Download Options**")
            
            with open(output_video_path, 'rb') as f:
                video_data = f.read()
                st.download_button(
                    label=f"📥 Download Video ({file_size:.1f} MB)",
                    data=video_data,
                    file_name=os.path.basename(output_video_path),
                    mime="video/mp4",
                    use_container_width=True
                )
    
    # Display graphs
    if results['frames']:
        st.subheader("📊 Analytics")
        
        df = pd.DataFrame({
            'Frame': results['frames'],
            'Detections': results['detections'],
            'Speed': results['speeds'],
            'Density': results['density']
        })
        
        speed_df = df[df['Speed'] > 0]
        
        # Graph 1: Detections over time
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df['Frame'],
            y=df['Detections'],
            mode='lines+markers',
            name='Vehicle Count',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        fig1.update_layout(
            title='Vehicle Count Over Time',
            xaxis_title='Frame Number',
            yaxis_title='Number of Vehicles',
            height=300,
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Graph 2: Speed distribution
        fig2 = go.Figure()
        if not speed_df.empty:
            fig2.add_trace(go.Histogram(
                x=speed_df['Speed'],
                nbinsx=20,
                marker_color='green',
                opacity=0.7,
                name='Speed Distribution'
            ))
            fig2.update_layout(
                title='Speed Distribution',
                xaxis_title='Speed (km/h)',
                yaxis_title='Frequency',
                height=300,
                bargap=0.1,
                template='plotly_white'
            )
        else:
            fig2.add_annotation(
                text="No speed data available",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Graph 3: Speed vs Density
        fig3 = go.Figure()
        if not speed_df.empty:
            fig3.add_trace(go.Scatter(
                x=speed_df['Speed'],
                y=speed_df['Density'],
                mode='markers',
                marker=dict(
                    size=8,
                    color='purple',
                    opacity=0.6,
                    showscale=False
                ),
                name='Data Points',
                hovertemplate='Speed: %{x:.1f} km/h<br>Density: %{y:.1f}%<extra></extra>'
            ))
            
            if len(speed_df) > 5:
                z = np.polyfit(speed_df['Speed'], speed_df['Density'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(speed_df['Speed'].min(), speed_df['Speed'].max(), 100)
                fig3.add_trace(go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', width=2, dash='dash')
                ))
            
            fig3.update_layout(
                title='Speed vs Traffic Density',
                xaxis_title='Speed (km/h)',
                yaxis_title='Density (%)',
                height=300,
                template='plotly_white'
            )
        else:
            fig3.add_annotation(
                text="No data available",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Data table
        with st.expander("📋 View Detailed Data", expanded=False):
            st.dataframe(df, use_container_width=True, height=300)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV Report",
                data=csv,
                file_name=f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("📊 No analytics data available")

# ==========================
# MAIN APPLICATION
# ==========================

def main():
    """Main application"""
    
    initialize_session_state()
    dataset_config = load_dataset_yaml()
    available_models = get_available_models()
    
    # ==========================
    # SIDEBAR
    # ==========================
    with st.sidebar:
        st.title("🚗 Traffic AI Detection")
        st.markdown("---")
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        
        model_type = st.selectbox(
            "Model Type",
            options=["YOLO", "CNN"],
            help="Select the detection model type"
        )
        
        if available_models:
            sorted_models = sorted(available_models.keys())
            selected_model = st.selectbox(
                "Model Version",
                options=sorted_models,
                help="Select the trained model version"
            )
            model_path = available_models[selected_model]
            
            st.info(f"📁 {os.path.basename(model_path)}")
            st.caption(f"Path: {model_path}")
        else:
            st.warning("⚠️ No trained models found!")
            st.info("📁 Expected location: `model/yolo/runs/v1/train/weights/best.pt`")
            
            # Manual path input
            manual_path = st.text_input(
                "Or enter model path manually:",
                value=r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\model\yolo\runs\v1\train\weights\best.pt"
            )
            if manual_path and os.path.exists(manual_path):
                model_path = manual_path
                selected_model = "Manual"
                st.success("✅ Model found!")
            else:
                model_path = None
        
        st.markdown("---")
        
        # Video Source
        st.subheader("🎥 Video Source")
        
        st.caption("💡 Maximum file size: 200MB")
        
        uploaded_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Upload a video file for processing"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            if 'uploaded_video_path' not in st.session_state or not st.session_state.uploaded_video_path:
                with st.spinner("📤 Uploading file..."):
                    video_path = handle_file_upload(uploaded_file)
                    if video_path:
                        st.session_state.uploaded_video_path = video_path
                        st.success("✅ File uploaded successfully!")
                        st.rerun()
        
        # Check if we have a saved uploaded file
        if 'uploaded_video_path' in st.session_state and st.session_state.uploaded_video_path:
            if os.path.exists(st.session_state.uploaded_video_path):
                st.video(st.session_state.uploaded_video_path)
        
        use_sample = st.checkbox("Use sample video instead")
        
        if use_sample:
            sample_videos = {
                "None": None,
                "Traffic Sample 1": "sample_videos/traffic1.mp4",
                "Traffic Sample 2": "sample_videos/traffic2.mp4"
            }
            selected_sample = st.selectbox(
                "Select Sample",
                options=list(sample_videos.keys())
            )
            sample_path = sample_videos.get(selected_sample)
            
            if selected_sample != "None":
                # Clear uploaded file
                if 'uploaded_video_path' in st.session_state:
                    try:
                        if os.path.exists(st.session_state.uploaded_video_path):
                            os.unlink(st.session_state.uploaded_video_path)
                    except:
                        pass
                    del st.session_state.uploaded_video_path
        else:
            sample_path = None
        
        st.markdown("---")
        
        # Processing Parameters
        st.subheader("⚙️ Parameters")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05
        )
        
        frame_skip = st.number_input(
            "Frame Skip (1 = all frames)",
            min_value=1,
            max_value=10,
            value=2,
            help="Higher values = faster processing"
        )
        
        output_fps = st.selectbox(
            "Output FPS",
            options=[30, 60, 120],
            index=1,
            help="Select the output video FPS"
        )
        
        device = st.radio(
            "Processing Device",
            options=["Auto", "CPU", "CUDA"],
            index=0
        )
        
        st.markdown("---")
        
        # Process Button
        process_button = st.button(
            "▶️ Process Video",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.is_processing
        )
        
        st.markdown("---")
        
        # System Info
        st.subheader("🖥️ System Info")
        cuda_available = torch.cuda.is_available()
        device_info = "CUDA" if cuda_available else "CPU"
        st.info(f"Device: {device_info}")
        if cuda_available:
            st.info(f"GPU: {torch.cuda.get_device_name(0)}")
        st.caption(f"📊 Found {len(available_models)} model(s)")
    
    # ==========================
    # MAIN CONTENT
    # ==========================
    
    st.markdown('<h1 class="main-header">🚗 Traffic AI Detection System</h1>', unsafe_allow_html=True)
    st.markdown("Upload a traffic video to detect vehicles and estimate their speeds.")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📹 Processing", "📊 Results", "📈 Analytics"])
    
    # ==========================
    # TAB 1: Processing
    # ==========================
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📹 Video Preview")
            
            video_preview = st.empty()
            
            if 'uploaded_video_path' in st.session_state and st.session_state.uploaded_video_path:
                if os.path.exists(st.session_state.uploaded_video_path):
                    video_preview.video(st.session_state.uploaded_video_path)
            elif sample_path and os.path.exists(sample_path):
                video_preview.video(sample_path)
            else:
                video_preview.info("👆 Upload a video or select a sample to begin")
        
        with col2:
            st.subheader("📊 Quick Stats")
            
            st.metric("Selected Model", f"{model_type} - {selected_model if selected_model else 'None'}")
            st.metric("Confidence", f"{confidence_threshold:.2f}")
            st.metric("Frame Skip", f"{frame_skip}")
            
            if 'uploaded_video_path' in st.session_state and st.session_state.uploaded_video_path:
                st.success("✅ Video loaded")
            elif sample_path and os.path.exists(sample_path):
                st.success("✅ Sample loaded")
            else:
                st.info("⏳ Waiting for video")
        
        # Processing status
        if process_button:
            # Determine video source
            video_source = None
            
            if 'uploaded_video_path' in st.session_state and st.session_state.uploaded_video_path:
                if os.path.exists(st.session_state.uploaded_video_path):
                    video_source = st.session_state.uploaded_video_path
            elif sample_path and os.path.exists(sample_path):
                video_source = sample_path
            
            if video_source is None:
                st.error("⚠️ Please upload a video or select a sample")
            elif model_path is None:
                st.error("⚠️ No model selected! Please check your runs folder.")
            else:
                # Determine device
                if device == "Auto":
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                elif device == "CPU":
                    device = 'cpu'
                else:
                    device = 'cuda'
                
                # Set processing flag
                st.session_state.is_processing = True
                st.session_state.processing_complete = False
                
                # Load model
                with st.spinner(f"Loading {model_type} model from {selected_model}..."):
                    detector = load_model(model_type, model_path, device)
                
                if detector is None:
                    st.error("❌ Failed to load model. Please check the model path.")
                    st.session_state.is_processing = False
                else:
                    # Process video
                    status_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    
                    def progress_callback(progress, current, total):
                        progress_bar.progress(progress)
                        status_placeholder.text(f"Processing: {current}/{total} frames ({progress*100:.1f}%)")
                    
                    try:
                        status_placeholder.text("⏳ Processing video...")
                        
                        output_path, results = process_video_with_models(
                            video_path=video_source,
                            detector=detector,
                            dataset_config=dataset_config,
                            frame_skip=frame_skip,
                            progress_callback=progress_callback,
                            target_fps=output_fps
                        )
                        
                        st.session_state.current_results = results
                        st.session_state.processed_video_path = output_path
                        st.session_state.processing = False
                        st.session_state.video_processed = True
                        st.session_state.processing_complete = True
                        
                        status_placeholder.text("✅ Processing complete!")
                        progress_bar.progress(1.0)
                        
                        st.success("✅ Video processing completed successfully!")
                        st.balloons()
                        
                        st.info("📊 View results in the 'Results' tab")
                        
                    except MemoryError:
                        st.error("❌ Memory error! Try reducing frame skip or using a shorter video.")
                        status_placeholder.text("❌ Processing failed - Out of memory")
                    except Exception as e:
                        st.error(f"❌ Error processing video: {e}")
                        status_placeholder.text("❌ Processing failed")
                    finally:
                        st.session_state.is_processing = False
                        del detector
                        gc.collect()
        
        # Show results if available
        if st.session_state.current_results is not None and st.session_state.video_processed:
            with st.expander("📊 Quick Results", expanded=True):
                results = st.session_state.current_results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Vehicles", results['total_vehicles'])
                with col2:
                    st.metric("Avg Speed", f"{results['avg_speed']:.1f} km/h")
                with col3:
                    st.metric("Max Speed", f"{results['max_speed']:.1f} km/h")
                with col4:
                    st.metric("Frames Processed", results['frames_processed'])
    
    # ==========================
    # TAB 2: Results
    # ==========================
    with tab2:
        if st.session_state.current_results is not None and st.session_state.video_processed:
            display_results(
                st.session_state.current_results,
                st.session_state.processed_video_path
            )
        else:
            st.info("📊 No results to display. Process a video first.")
    
    # ==========================
    # TAB 3: Analytics
    # ==========================
    with tab3:
        st.subheader("📈 Detailed Analytics")
        
        if st.session_state.current_results is not None and st.session_state.video_processed:
            results = st.session_state.current_results
            
            col1, col2 = st.columns(2)
            
            with col1:
                if results['speeds']:
                    speeds_filtered = [s for s in results['speeds'] if s > 0]
                    if speeds_filtered:
                        fig_speed = go.Figure()
                        fig_speed.add_trace(go.Scatter(
                            y=speeds_filtered,
                            mode='lines+markers',
                            name='Speed',
                            line=dict(color='red', width=2),
                            marker=dict(size=4)
                        ))
                        fig_speed.update_layout(
                            title='Speed Over Time',
                            xaxis_title='Frame',
                            yaxis_title='Speed (km/h)',
                            height=300,
                            template='plotly_white'
                        )
                        st.plotly_chart(fig_speed, use_container_width=True)
            
            with col2:
                if results['density']:
                    fig_density = go.Figure()
                    fig_density.add_trace(go.Scatter(
                        y=results['density'],
                        mode='lines+markers',
                        name='Density',
                        line=dict(color='orange', width=2),
                        marker=dict(size=4)
                    ))
                    fig_density.update_layout(
                        title='Traffic Density Over Time',
                        xaxis_title='Frame',
                        yaxis_title='Density (%)',
                        height=300,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_density, use_container_width=True)
            
            # Summary statistics
            st.subheader("📊 Summary Statistics")
            
            stats_data = {
                'Metric': [
                    'Total Vehicles Detected',
                    'Average Speed',
                    'Maximum Speed',
                    'Minimum Speed',
                    'Average Density',
                    'Maximum Density',
                    'Frames Processed',
                    'Processing Time'
                ],
                'Value': [
                    results['total_vehicles'],
                    f"{results['avg_speed']:.2f} km/h",
                    f"{results['max_speed']:.2f} km/h",
                    f"{results['min_speed']:.2f} km/h",
                    f"{np.mean(results['density']):.2f}%" if results['density'] else "0.00%",
                    f"{np.max(results['density']):.2f}%" if results['density'] else "0.00%",
                    results['frames_processed'],
                    f"{results['processing_time']:.2f}s"
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Export options
            st.subheader("📥 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                df = pd.DataFrame({
                    'Frame': results['frames'],
                    'Detections': results['detections'],
                    'Speed': results['speeds'],
                    'Density': results['density']
                })
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV Report",
                    data=csv,
                    file_name=f"traffic_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                import json
                results_serializable = {}
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        results_serializable[key] = value.tolist()
                    elif isinstance(value, np.float32):
                        results_serializable[key] = float(value)
                    else:
                        results_serializable[key] = value
                
                json_data = json.dumps(results_serializable, indent=2)
                st.download_button(
                    label="📊 Download JSON Report",
                    data=json_data,
                    file_name=f"traffic_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("📊 No analytics data available. Process a video first.")

if __name__ == "__main__":
    main()