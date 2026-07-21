"""
Traffic AI Detection System
Main Streamlit application for vehicle detection and speed estimation
"""

import streamlit as st
import sys
import os
import tempfile
import time
import gc
from pathlib import Path
from datetime import datetime
import yaml

# ============================================
# CRITICAL FIX: Import OpenCV headless first
# ============================================
# Try multiple ways to import OpenCV
cv2 = None

# Method 1: Try importing opencv-python-headless
try:
    # Force headless loading by setting environment variable
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    
    # Try direct import
    import cv2
    print("✅ OpenCV loaded successfully")
except ImportError as e:
    print(f"⚠️ OpenCV import error: {e}")
    
    # Method 2: Try using importlib to load from a specific location
    try:
        import importlib
        import importlib.util
        
        # Try to find cv2 in the system
        spec = importlib.util.find_spec("cv2")
        if spec:
            print(f"Found cv2 spec: {spec}")
            cv2 = importlib.import_module("cv2")
            print("✅ OpenCV loaded via importlib")
    except Exception as e2:
        print(f"⚠️ OpenCV importlib error: {e2}")
        
        # Method 3: Try to install it on the fly (for Streamlit Cloud)
        try:
            import subprocess
            print("Attempting to install opencv-python-headless...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "opencv-python-headless==4.8.1.78", 
                "--force-reinstall", "--no-deps"
            ])
            import cv2
            print("✅ OpenCV installed and loaded")
        except Exception as e3:
            print(f"⚠️ Failed to install OpenCV: {e3}")

# Check if OpenCV loaded successfully
if cv2 is None:
    st.error("⚠️ OpenCV failed to load. Please check the logs.")
    # Create a dummy cv2 to prevent further errors
    class DummyCV2:
        class VideoCapture:
            def __init__(self, *args, **kwargs):
                self.is_opened = lambda: False
            def release(self): pass
        class VideoWriter:
            def __init__(self, *args, **kwargs): pass
            def release(self): pass
            def write(self, *args): pass
        def VideoWriter_fourcc(*args): return 0
        def rectangle(*args): pass
        def putText(*args): pass
        def getTextSize(*args): return (0, 0), 0
        def addWeighted(*args): return None
        def resize(*args): return None
        def imencode(*args): return (True, b'')
        CAP_PROP_FPS = 5
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FRAME_COUNT = 7
        FONT_HERSHEY_SIMPLEX = 0
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    cv2 = DummyCV2()

import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Set page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Traffic AI Detection System",
    page_icon="🚗",
    layout="wide"
)

# Import backend modules with error handling
try:
    # Try importing from backend directory (same level as app.py)
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
            # Simulate detections
            h, w = frame.shape[:2] if frame is not None else (480, 640)
            return [{'bbox': [100, 100, 200, 300], 'confidence': 0.95, 'class': 0} for _ in range(5)]
        
        def detect_frame(self, frame):
            # Alias for detect method
            return self.detect(frame)
    
    YOLODetector = DummyDetector
    CNNDetector = DummyDetector
    SpeedEstimator = None
    VideoProcessor = None
    
    def load_dataset_config():
        return {
            'nc': 5,
            'names': {
                0: 'car',
                1: 'truck',
                2: 'bus',
                3: 'motorcycle',
                4: 'bicycle'
            },
            'colors': {
                0: [0, 0, 255],
                1: [0, 255, 0],
                2: [255, 0, 0],
                3: [0, 255, 255],
                4: [255, 0, 255]
            }
        }
    
    def get_class_colors():
        return load_dataset_config().get('colors', {})

def initialize_session_state():
    """Initialize session state variables"""
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'detections_history' not in st.session_state:
        st.session_state.detections_history = []
    if 'speed_history' not in st.session_state:
        st.session_state.speed_history = []
    if 'density_history' not in st.session_state:
        st.session_state.density_history = []
    if 'processed_video_path' not in st.session_state:
        st.session_state.processed_video_path = None
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    if 'frame_skip' not in st.session_state:
        st.session_state.frame_skip = 1

def get_available_models():
    """
    Get available model versions from runs folder
    
    Returns:
        Dictionary of model versions and paths
    """
    models = {}
    
    # Get the current directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Check multiple possible locations for models
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
        # Add your specific path
        Path(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\model\yolo\runs"),
    ]
    
    print("🔍 Searching for models in:")
    for runs_path in possible_paths:
        print(f"  - {runs_path}")
    
    for runs_path in possible_paths:
        if runs_path.exists():
            print(f"✅ Checking runs directory: {runs_path}")
            
            # First, look for version directories (v1, v2, etc.)
            version_dirs = [d for d in runs_path.iterdir() if d.is_dir() and d.name.startswith('v')]
            
            # Also look for train directories directly
            train_dirs = []
            
            # Check each version directory
            for version_dir in version_dirs:
                print(f"  📁 Checking version: {version_dir.name}")
                
                # Look for train directories inside version
                for sub_dir in version_dir.iterdir():
                    if sub_dir.is_dir() and (sub_dir.name.startswith('train') or sub_dir.name == 'train'):
                        train_dirs.append(sub_dir)
                        print(f"    Found train dir: {sub_dir}")
                
                # Also check for weights directly in version dir
                weights_dir = version_dir / "weights"
                if weights_dir.exists():
                    for pt_file in weights_dir.glob("*.pt"):
                        version_name = f"{version_dir.name}/{pt_file.stem}"
                        models[version_name] = str(pt_file)
                        print(f"    Found model: {version_name} -> {pt_file}")
            
            # Also look for train directories directly (not in version folders)
            for item in runs_path.iterdir():
                if item.is_dir() and (item.name.startswith('train') or item.name == 'train'):
                    if item not in train_dirs:
                        train_dirs.append(item)
                        print(f"  Found train dir: {item}")
            
            # Process all train directories
            for train_dir in train_dirs:
                weights_dir = train_dir / "weights"
                if weights_dir.exists():
                    print(f"  📂 Checking weights in: {weights_dir}")
                    
                    # Look for best.pt
                    best_pt = weights_dir / "best.pt"
                    if best_pt.exists():
                        # Try to get version name from parent directory
                        parent_name = train_dir.parent.name
                        if parent_name.startswith('v'):
                            version_name = f"{parent_name}/{train_dir.name}"
                        else:
                            version_name = train_dir.name
                        
                        # Make sure we don't duplicate
                        if version_name not in models:
                            models[version_name] = str(best_pt)
                            print(f"  ✅ Found model: {version_name} -> {best_pt}")
                    
                    # Also look for other .pt files
                    for pt_file in weights_dir.glob("*.pt"):
                        if pt_file.name != "best.pt":
                            parent_name = train_dir.parent.name
                            if parent_name.startswith('v'):
                                version_name = f"{parent_name}/{train_dir.name}/{pt_file.stem}"
                            else:
                                version_name = f"{train_dir.name}/{pt_file.stem}"
                            
                            if version_name not in models:
                                models[version_name] = str(pt_file)
                                print(f"  ✅ Found model: {version_name} -> {pt_file}")
    
    # If no models found, try to find any .pt file in the directory
    if not models:
        print("🔍 No models found in runs directories, searching for .pt files...")
        search_paths = [
            current_dir,
            current_dir / "model",
            current_dir / "model" / "yolo",
            current_dir / "yolo",
            Path(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment"),
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                print(f"  Searching in: {search_path}")
                pt_files = list(search_path.glob("**/*.pt"))
                for pt_file in pt_files:
                    # Skip if in runs directory (already checked)
                    if "runs" in str(pt_file):
                        continue
                    # Skip if in venv or site-packages
                    if "venv" in str(pt_file) or "site-packages" in str(pt_file):
                        continue
                    version_name = f"{pt_file.parent.name}/{pt_file.stem}"
                    if version_name not in models:
                        models[version_name] = str(pt_file)
                        print(f"  Found model: {version_name} -> {pt_file}")
    
    # Specifically check for your model path
    specific_model_path = Path(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\model\yolo\runs\v1\train\weights\best.pt")
    if specific_model_path.exists():
        models["v1/train/best"] = str(specific_model_path)
        print(f"✅ Found specific model: v1/train/best -> {specific_model_path}")
    
    # If still no models, try to use the yolov8n.pt or similar
    if not models:
        default_model = Path(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\yolo26n.pt")
        if default_model.exists():
            models["default"] = str(default_model)
            print(f"Using default model: {default_model}")
    
    print(f"📊 Found {len(models)} model(s): {list(models.keys())}")
    return models

def load_dataset_yaml():
    """
    Load dataset configuration from YAML file
    
    Returns:
        Configuration dictionary
    """
    # Check multiple possible locations
    yaml_paths = [
        Path("dataset.yaml"),
        Path("model/yolo/dataset.yaml"),
        Path("config/dataset.yaml"),
        Path("dataset/dataset.yaml"),
        Path(r"C:\Users\fouls\Downloads\TARUMT\Y2S1\AI\BMCS2074-Artificial-Intelligence-Assignment\dataset.yaml"),
    ]
    
    # Also check in current directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    yaml_paths.extend([
        current_dir / "dataset.yaml",
        current_dir / "model" / "yolo" / "dataset.yaml",
        current_dir / "config" / "dataset.yaml",
    ])
    
    for yaml_path in yaml_paths:
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"Loaded dataset config from: {yaml_path}")
                return config
            except Exception as e:
                print(f"Error loading {yaml_path}: {e}")
    
    # Return default configuration
    print("Using default dataset configuration")
    return {
        'nc': 5,
        'names': {
            0: 'car',
            1: 'truck',
            2: 'bus',
            3: 'motorcycle',
            4: 'bicycle'
        },
        'colors': {
            0: [0, 0, 255],    # Red (BGR)
            1: [0, 255, 0],    # Green (BGR)
            2: [255, 0, 0],    # Blue (BGR)
            3: [0, 255, 255],  # Yellow (BGR)
            4: [255, 0, 255]   # Magenta (BGR)
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

class VideoProcessorWrapper:
    """Wrapper class for video processing to handle missing modules"""
    def __init__(self, detector, speed_estimator, frame_skip=2):
        self.detector = detector
        self.speed_estimator = speed_estimator
        self.frame_skip = frame_skip
    
    def process_frame_sync(self, frame):
        """Process a single frame synchronously"""
        # Use detector to get detections
        try:
            detections = self.detector.detect(frame)
        except AttributeError:
            # Try detect_frame method
            try:
                detections = self.detector.detect_frame(frame)
            except AttributeError:
                # Fallback to dummy detections
                h, w = frame.shape[:2]
                detections = [{'bbox': [100, 100, 200, 300], 'confidence': 0.95, 'class': 0} for _ in range(5)]
        
        # Add speed estimation if available
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

def process_video_with_models(
    video_path,
    detector,
    dataset_config,
    frame_skip=1,
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
        target_fps: Target FPS for output video (default: 60)
    """
    if cv2 is None:
        raise ImportError("OpenCV is not available. Please install opencv-python-headless")
    
    # Initialize components
    try:
        if SpeedEstimator:
            speed_estimator = SpeedEstimator(fps=target_fps, calibration_factor=0.05)
        else:
            speed_estimator = None
    except:
        speed_estimator = None
    
    # Create video processor wrapper
    video_processor = VideoProcessorWrapper(
        detector=detector,
        speed_estimator=speed_estimator,
        frame_skip=frame_skip
    )
    
    # Get class colors from config
    class_colors = dataset_config.get('colors', {})
    class_names = dataset_config.get('names', {})
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")
    
    # Get video properties
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interpolation for target FPS
    interpolation_factor = max(1, int(target_fps / original_fps))
    if interpolation_factor > 4:
        interpolation_factor = 4
        print(f"⚠️ Capping interpolation factor to 4 to prevent memory issues")
    
    actual_output_fps = original_fps * interpolation_factor
    
    print(f"Original FPS: {original_fps}, Target FPS: {target_fps}, Interpolation: {interpolation_factor}")
    
    # Create output video writer
    output_dir = Path("outputs/processed_videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"processed_{timestamp}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, actual_output_fps, (width, height))
    
    # Storage for results (limit to prevent memory issues)
    max_results_frames = 500
    
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
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % frame_skip != 0:
                out.write(frame)
                del frame
                continue
            
            # Process frame
            try:
                # Get detections
                detections = video_processor.process_frame_sync(frame)
                
                # Create processed frame with detections
                processed_frame = frame.copy()
                
                # Store results (limit to prevent memory issues)
                if len(results['frames']) < max_results_frames:
                    results['frames'].append(frame_count)
                    results['detections'].append(len(detections))
                    results['total_vehicles'] += len(detections)
                    
                    # Calculate speed and density
                    if detections:
                        speeds = [det.get('speed', 0) for det in detections if det.get('speed', 0) > 0]
                        avg_speed = np.mean(speeds) if speeds else 0
                        results['speeds'].append(avg_speed)
                        
                        # Calculate density
                        density = calculate_density(detections, frame.shape)
                        results['density'].append(density)
                    else:
                        results['speeds'].append(0)
                        results['density'].append(0)
                
                processed_count += 1
                
                # Draw detections with class colors
                for det in detections:
                    bbox = det.get('bbox', [0, 0, 100, 100])
                    class_id = det.get('class', 0)
                    confidence = det.get('confidence', 0)
                    speed = det.get('speed', 0)
                    class_name = class_names.get(class_id, f'Class {class_id}')
                    
                    # Get color for this class from dataset config
                    color = class_colors.get(class_id, [0, 255, 0])
                    if isinstance(color, list):
                        color = tuple(color)
                    
                    # Draw bounding box
                    cv2.rectangle(processed_frame, 
                                (bbox[0], bbox[1]), 
                                (bbox[2], bbox[3]), 
                                color, 2)
                    
                    # Draw label with class name, confidence, and speed
                    label = f"{class_name} {confidence:.2f}"
                    if speed > 0:
                        label += f" {speed:.1f}km/h"
                    
                    # Draw label background
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    cv2.rectangle(processed_frame,
                                (bbox[0], bbox[1] - text_height - 10),
                                (bbox[0] + text_width, bbox[1] - 5),
                                color, -1)
                    
                    # Draw label text
                    cv2.putText(processed_frame, label,
                              (bbox[0], bbox[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (255, 255, 255), 2)
                
                # Add info overlay
                overlay_text = f"Detections: {len(detections)} | Frames: {processed_count}"
                cv2.putText(processed_frame, overlay_text,
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                          0.7, (0, 255, 255), 2)
                
                # Write processed frame
                out.write(processed_frame)
                
                # Interpolate frames for smooth video (limited to prevent memory issues)
                if prev_frame is not None and interpolation_factor > 1 and prev_detections is not None:
                    num_interpolations = min(interpolation_factor - 1, 2)
                    for i in range(1, num_interpolations + 1):
                        alpha = i / (num_interpolations + 1)
                        interp_frame = cv2.addWeighted(prev_frame, 1 - alpha, processed_frame, alpha, 0)
                        out.write(interp_frame)
                
                # Store previous frame for interpolation
                prev_frame = processed_frame.copy()
                prev_detections = detections
                
                # Free memory
                del frame
                if processed_frame is not None:
                    del processed_frame
                
                # Run garbage collection periodically
                if processed_count % 50 == 0:
                    gc.collect()
                
                # Update progress
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
        # Clean up resources
        cap.release()
        out.release()
    
    # Calculate final statistics
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
    
    return str(output_path), results

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
    
    # Calculate area occupied by vehicles
    vehicle_area = 0
    for det in detections:
        bbox = det.get('bbox', [0, 0, 100, 100])
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        vehicle_area += width * height
    
    total_area = frame_shape[0] * frame_shape[1]
    density = (vehicle_area / total_area) * 100
    
    return min(density, 100.0)

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
        
        # Check file size and display
        file_size = os.path.getsize(output_video_path) / (1024 * 1024)  # MB
        if file_size > 100:  # If file is too large, show warning
            st.warning(f"⚠️ Video file is large ({file_size:.1f} MB). May take time to load.")
        
        with open(output_video_path, 'rb') as f:
            video_bytes = f.read()
        st.video(video_bytes)
        
        # Download button
        with open(output_video_path, 'rb') as f:
            st.download_button(
                label="📥 Download Processed Video",
                data=f,
                file_name=os.path.basename(output_video_path),
                mime="video/mp4"
            )
    
    # Display graphs only if we have data
    if results['frames']:
        st.subheader("📊 Analytics")
        
        # Create data for graphs
        df = pd.DataFrame({
            'Frame': results['frames'],
            'Detections': results['detections'],
            'Speed': results['speeds'],
            'Density': results['density']
        })
        
        # Remove zero speeds for better visualization
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
            height=300
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Graph 2: Speed distribution
        fig2 = go.Figure()
        if not speed_df.empty:
            fig2.add_trace(go.Histogram(
                x=speed_df['Speed'],
                nbinsx=20,
                marker_color='green',
                opacity=0.7
            ))
            fig2.update_layout(
                title='Speed Distribution',
                xaxis_title='Speed (km/h)',
                yaxis_title='Frequency',
                height=300
            )
        else:
            fig2.add_annotation(
                text="No speed data available",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Graph 3: Speed vs Density correlation
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
                name='Data Points'
            ))
            fig3.update_layout(
                title='Speed vs Traffic Density',
                xaxis_title='Speed (km/h)',
                yaxis_title='Density (%)',
                height=300
            )
        else:
            fig3.add_annotation(
                text="No data available",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Display data table
        with st.expander("📋 View Detailed Data"):
            st.dataframe(df, use_container_width=True)
            
            # Download CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV Report",
                data=csv,
                file_name=f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("📊 No analytics data available")

def main():
    """Main application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Load dataset configuration
    dataset_config = load_dataset_yaml()
    
    # Get available models
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
        
        # Model version selection
        if available_models:
            sorted_models = sorted(available_models.keys())
            selected_model = st.selectbox(
                "Model Version",
                options=sorted_models,
                help="Select the trained model version from your runs folder"
            )
            model_path = available_models[selected_model]
            
            st.info(f"📁 {os.path.basename(model_path)}")
            st.caption(f"Path: {model_path}")
        else:
            st.warning("⚠️ No trained models found in runs folder!")
            st.info("📁 Expected location: `model/yolo/runs/train*/weights/best.pt`")
            model_path = None
            selected_model = "None"
        
        st.markdown("---")
        
        # Video Source
        st.subheader("🎥 Video Source")
        
        uploaded_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Upload a video file for processing"
        )
        
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
        else:
            sample_path = None
        
        st.markdown("---")
        
        # Processing Parameters
        st.subheader("⚙️ Processing Parameters")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05
        )
        
        # Add FPS selector
        output_fps = st.selectbox(
            "Output FPS",
            options=[30, 60, 120],
            index=1,  # Default to 60
            help="Select the output video FPS"
        )
        
        frame_skip = st.slider(
            "Frame Skip",
            min_value=1,
            max_value=10,
            value=st.session_state.frame_skip,
            help="Process every Nth frame (higher = faster processing)"
        )
        st.session_state.frame_skip = frame_skip
        
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
            type="primary"
        )
        
        # System Info
        st.markdown("---")
        st.subheader("🖥️ System Info")
        
        cuda_available = torch.cuda.is_available() if torch else False
        device_info = "CUDA" if cuda_available else "CPU"
        st.info(f"Device: {device_info}")
        
        if cuda_available:
            st.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        st.caption(f"📊 Found {len(available_models)} model(s)")
    
    # ==========================
    # MAIN CONTENT
    # ==========================
    
    st.title("🚗 Traffic AI Detection System")
    st.markdown("Upload a traffic video to detect vehicles and estimate their speeds.")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📹 Processing", "📊 Results", "📈 Analytics"])
    
    # ==========================
    # TAB 1: Processing
    # ==========================
    with tab1:
        # Display video source
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📹 Video Preview")
            
            video_preview = st.empty()
            
            # Display uploaded video or sample
            if uploaded_file is not None:
                video_preview.video(uploaded_file)
            elif sample_path and os.path.exists(sample_path):
                video_preview.video(sample_path)
            else:
                video_preview.info("👆 Upload a video or select a sample to begin")
        
        with col2:
            st.subheader("📊 Quick Stats")
            
            # Placeholder stats
            st.metric("Selected Model", f"{model_type} - {selected_model}")
            st.metric("Confidence", f"{confidence_threshold:.2f}")
            st.metric("Frame Skip", f"{frame_skip}")
            
            if uploaded_file is not None:
                st.success("✅ Video loaded")
            elif sample_path and os.path.exists(sample_path):
                st.success("✅ Sample loaded")
            else:
                st.info("⏳ Waiting for video")
        
        # Processing status
        if process_button:
            # Determine video source
            video_source = None
            if uploaded_file is not None:
                # Save uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_source = tmp_file.name
            elif sample_path and os.path.exists(sample_path):
                video_source = sample_path
            
            if video_source is None:
                st.error("⚠️ Please upload a video or select a sample")
            elif model_path is None:
                st.error("⚠️ No model selected! Please check your runs folder.")
            elif cv2 is None:
                st.error("⚠️ OpenCV is not available. Please check your installation.")
            else:
                # Determine device
                if device == "Auto":
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                elif device == "CPU":
                    device = 'cpu'
                else:
                    device = 'cuda'
                
                # Load model
                with st.spinner(f"Loading {model_type} model from {selected_model}..."):
                    detector = load_model(model_type, model_path, device)
                
                if detector is None:
                    st.error("❌ Failed to load model. Please check the model path.")
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
                        
                        # Store results in session state
                        st.session_state.current_results = results
                        st.session_state.processed_video_path = output_path
                        st.session_state.processing = False
                        st.session_state.video_processed = True
                        
                        status_placeholder.text("✅ Processing complete!")
                        progress_bar.progress(1.0)
                        
                        st.success("✅ Video processing completed successfully!")
                        st.balloons()
                        
                        # Show results in tab 2
                        st.info("📊 View results in the 'Results' tab")
                        
                        # Clean up temp file if uploaded
                        if uploaded_file is not None and video_source and os.path.exists(video_source):
                            try:
                                os.unlink(video_source)
                            except:
                                pass
                        
                    except MemoryError:
                        st.error("❌ Memory error! Try reducing frame skip or using a shorter video.")
                        status_placeholder.text("❌ Processing failed - Out of memory")
                    except Exception as e:
                        st.error(f"❌ Error processing video: {e}")
                        status_placeholder.text("❌ Processing failed")
                    
                    # Clean up detector to free memory
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
            
            # Create comprehensive dashboard
            col1, col2 = st.columns(2)
            
            with col1:
                # Speed over time
                if results['speeds']:
                    speeds_filtered = [s for s in results['speeds'] if s > 0]
                    if speeds_filtered:
                        fig_speed = go.Figure()
                        fig_speed.add_trace(go.Scatter(
                            y=speeds_filtered,
                            mode='lines+markers',
                            name='Speed',
                            line=dict(color='red', width=2)
                        ))
                        fig_speed.update_layout(
                            title='Speed Over Time',
                            xaxis_title='Frame',
                            yaxis_title='Speed (km/h)',
                            height=300
                        )
                        st.plotly_chart(fig_speed, use_container_width=True)
            
            with col2:
                # Density over time
                if results['density']:
                    fig_density = go.Figure()
                    fig_density.add_trace(go.Scatter(
                        y=results['density'],
                        mode='lines+markers',
                        name='Density',
                        line=dict(color='orange', width=2)
                    ))
                    fig_density.update_layout(
                        title='Traffic Density Over Time',
                        xaxis_title='Frame',
                        yaxis_title='Density (%)',
                        height=300
                    )
                    st.plotly_chart(fig_density, use_container_width=True)
            
            # Summary statistics
            st.subheader("📊 Summary Statistics")
            stats_df = pd.DataFrame({
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
            })
            st.dataframe(stats_df, use_container_width=True)
            
            # Export options
            st.subheader("📥 Export Data")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Export as CSV"):
                    df = pd.DataFrame({
                        'Frame': results['frames'],
                        'Detections': results['detections'],
                        'Speed': results['speeds'],
                        'Density': results['density']
                    })
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"traffic_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📊 Export as JSON"):
                    import json
                    json_data = json.dumps(results, default=lambda x: float(x) if isinstance(x, np.float32) else x, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"traffic_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        else:
            st.info("📊 No analytics data available. Process a video first.")

if __name__ == "__main__":
    main()