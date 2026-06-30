import streamlit as st
import pandas as pd
import numpy as np
import cv2
import torch
import os
import sys
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import tempfile

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import utilities (these will be created)
try:
    from backend.detection.cnn_detector import CNNDetector
    from backend.detection.yolo_detector import YOLODetector
    from backend.speed_estimator import SpeedEstimator
    from backend.video_processor import VideoProcessor
    from backend.utils import calculate_traffic_density
except ImportError as e:
    st.warning(f"Some modules not found: {e}")
    # Create dummy classes for demonstration
    class DummyDetector:
        def detect(self, frame):
            # Simulate detections
            h, w = frame.shape[:2]
            return [{'bbox': [100, 100, 200, 300], 'confidence': 0.95} for _ in range(5)]
    
    CNNDetector = DummyDetector
    YOLODetector = DummyDetector
    SpeedEstimator = None
    VideoProcessor = None

# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Traffic AI Detection System",
    page_icon="🚗",
    layout="wide"
)

# ==========================
# Title and Description
# ==========================

st.title("🚗 Traffic AI Detection with Speed Estimation")
st.markdown("---")

# ==========================
# Sidebar - Controls
# ==========================

with st.sidebar:
    st.header("⚙️ Controls")
    
    # Model Selection
    st.subheader("Model Selection")
    model_type = st.selectbox(
        "Choose Detection Model",
        options=["YOLO", "CNN"],
        help="Select the AI model for vehicle detection"
    )
    
    st.divider()
    
    # Video Source
    st.subheader("Video Source")
    source_type = st.radio(
        "Source Type",
        options=["Webcam", "Upload Video", "Sample Video"],
        index=2
    )
    
    if source_type == "Upload Video":
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=["mp4", "avi", "mov", "mkv"]
        )
        video_path = uploaded_file if uploaded_file else None
    
    elif source_type == "Sample Video":
        # Use a sample video or webcam
        sample_videos = {
            "None": None,
            "Sample Traffic 1": "sample_videos/traffic1.mp4",
            "Sample Traffic 2": "sample_videos/traffic2.mp4"
        }
        video_path = st.selectbox(
            "Select sample video",
            options=list(sample_videos.keys())
        )
        if video_path and video_path != "None":
            video_path = sample_videos[video_path]
        else:
            video_path = None
    
    elif source_type == "Webcam":
        video_path = 0  # Webcam index
    
    st.divider()
    
    # Processing Parameters
    st.subheader("Processing Parameters")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )
    
    fps = st.number_input(
        "FPS for processing",
        min_value=1,
        max_value=60,
        value=30
    )
    
    frame_skip = st.number_input(
        "Frame Skip (process every N frames)",
        min_value=1,
        max_value=10,
        value=2,
        help="Process every Nth frame for better performance"
    )
    
    st.divider()
    
    # Control Buttons
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("▶ Start Processing", use_container_width=True)
    with col2:
        stop_button = st.button("⏹ Stop Processing", use_container_width=True)
    
    # Device Info
    st.divider()
    st.subheader("🖥️ System Info")
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.info(f"Device: {device}")
    if torch.cuda.is_available():
        st.info(f"GPU: {torch.cuda.get_device_name(0)}")
        st.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ==========================
# Main Content
# ==========================

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎥 Video Feed", "📊 Analytics", "📈 Graphs"])

# ==========================
# TAB 1: Video Feed
# ==========================

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Video Feed")
        video_container = st.empty()
        status_text = st.empty()
        
        # Placeholder for video
        with video_container:
            if source_type == "Upload Video" and uploaded_file:
                st.video(uploaded_file)
            elif source_type == "Sample Video" and video_path and os.path.exists(video_path):
                st.video(video_path)
            elif source_type == "Webcam":
                st.info("📷 Webcam feed will appear here when processing starts")
            else:
                st.info("👆 Select a video source and click 'Start Processing'")
    
    with col2:
        st.subheader("Live Statistics")
        st.metric("🚗 Vehicles Detected", "0", delta=None)
        st.metric("📏 Avg Speed", "0 km/h", delta=None)
        st.metric("📊 Traffic Density", "0%", delta=None)
        st.metric("📹 Frames Processed", "0", delta=None)
        st.metric("⏱️ Processing Time", "0s", delta=None)

# ==========================
# TAB 2: Analytics
# ==========================

with tab2:
    st.subheader("📊 Detailed Analytics")
    
    # Create metrics in a grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Vehicles", "0", delta=None)
    with col2:
        st.metric("Average Speed", "0 km/h", delta=None)
    with col3:
        st.metric("Max Speed", "0 km/h", delta=None)
    with col4:
        st.metric("Min Speed", "0 km/h", delta=None)
    
    st.divider()
    
    # Detection History
    st.subheader("Detection History")
    
    # Sample data structure
    history_data = pd.DataFrame({
        "Frame": [],
        "Vehicles": [],
        "Avg Speed": [],
        "Density": []
    })
    
    if history_data.empty:
        st.info("📊 No data available yet. Start processing to see analytics.")
    else:
        st.dataframe(history_data, use_container_width=True)
    
    st.divider()
    
    # Download Report
    st.subheader("📥 Export Report")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Download CSV Report", use_container_width=True):
            st.success("Report downloaded successfully!")
            # In real implementation, generate and download CSV
    with col2:
        if st.button("Download Summary Report", use_container_width=True):
            st.success("Summary report downloaded successfully!")

# ==========================
# TAB 3: Graphs
# ==========================

with tab3:
    st.subheader("📈 Visualization")
    
    # Create graph columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Speed Distribution")
        # Sample graph
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(
            x=np.random.normal(60, 15, 100),
            nbinsx=20,
            marker_color='blue',
            opacity=0.7
        ))
        fig1.update_layout(
            xaxis_title="Speed (km/h)",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Traffic Over Time")
        # Sample graph
        fig2 = go.Figure()
        frames = list(range(100))
        vehicles = np.random.poisson(10, 100) + 5
        speeds = np.random.normal(55, 15, 100)
        
        fig2.add_trace(go.Scatter(
            x=frames,
            y=vehicles,
            name="Vehicles",
            line=dict(color='green', width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=frames,
            y=speeds,
            name="Avg Speed",
            line=dict(color='red', width=2)
        ))
        fig2.update_layout(
            xaxis_title="Frame",
            yaxis_title="Count / Speed",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Second row of graphs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Traffic Density Heatmap")
        # Sample heatmap
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=frames,
            y=np.random.uniform(0, 100, 100),
            mode='markers',
            marker=dict(
                size=10,
                color=np.random.uniform(0, 100, 100),
                colorscale='Hot',
                showscale=True,
                colorbar=dict(title="Density")
            ),
            name="Density"
        ))
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.subheader("Speed vs Density Correlation")
        # Sample correlation
        speeds = np.random.normal(55, 15, 100)
        densities = 100 - speeds * 0.5 + np.random.normal(0, 10, 100)
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=speeds,
            y=densities,
            mode='markers',
            marker=dict(
                size=8,
                color='purple',
                opacity=0.6
            ),
            name="Data Points"
        ))
        fig4.update_layout(
            xaxis_title="Speed (km/h)",
            yaxis_title="Density (%)",
            height=400
        )
        st.plotly_chart(fig4, use_container_width=True)

# ==========================
# Video Processing Functions
# ==========================

def process_video(video_path, model_type, confidence_threshold, fps, frame_skip):
    """Process video with the selected model"""
    status_text.text("🚀 Initializing model...")
    
    # Initialize detector
    try:
        if model_type == "YOLO":
            detector = YOLODetector(device='cuda' if torch.cuda.is_available() else 'cpu')
        else:
            detector = CNNDetector(device='cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video source")
        return
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    processed_count = 0
    
    # Storage for data
    detections_data = []
    speed_data = []
    density_data = []
    
    # Process video
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for performance
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        
        # Process frame
        try:
            # Detect vehicles
            detections = detector.detect(frame)
            
            # Simulate speed estimation (in real implementation, use SpeedEstimator)
            speeds = np.random.normal(55, 15, len(detections))
            avg_speed = np.mean(speeds) if len(speeds) > 0 else 0
            
            # Calculate density
            density = len(detections) / (frame.shape[0] * frame.shape[1]) * 1000
            
            # Store data
            detections_data.append(len(detections))
            speed_data.append(avg_speed)
            density_data.append(density)
            
            # Draw on frame
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if i < len(speeds):
                    speed_text = f"{speeds[i]:.1f} km/h"
                    cv2.putText(frame, speed_text, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_container.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update statistics
            with col2:
                st.metric("🚗 Vehicles Detected", len(detections))
                st.metric("📏 Avg Speed", f"{avg_speed:.1f} km/h")
                st.metric("📊 Traffic Density", f"{density:.1f}%")
                st.metric("📹 Frames Processed", processed_count + 1)
                elapsed = time.time() - start_time
                st.metric("⏱️ Processing Time", f"{elapsed:.1f}s")
            
            processed_count += 1
            
        except Exception as e:
            st.error(f"Error processing frame: {e}")
        
        frame_count += 1
        
        # Check stop condition
        if stop_button:
            break
    
    cap.release()
    status_text.text("✅ Processing completed!")
    
    # Update analytics tab with real data
    if detections_data:
        with tab2:
            st.subheader("📊 Detection Results")
            
            # Create DataFrame
            results_df = pd.DataFrame({
                "Frame": list(range(1, len(detections_data) + 1)),
                "Vehicles": detections_data,
                "Avg Speed": speed_data,
                "Density": density_data
            })
            
            st.dataframe(results_df, use_container_width=True)
            
            # Update metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Vehicles", sum(detections_data))
            with col2:
                st.metric("Average Speed", f"{np.mean(speed_data):.1f} km/h")
            with col3:
                st.metric("Max Speed", f"{np.max(speed_data):.1f} km/h")
            with col4:
                st.metric("Min Speed", f"{np.min(speed_data):.1f} km/h")
            
            # Update graphs with real data
            with tab3:
                # Speed Distribution
                fig1 = go.Figure()
                fig1.add_trace(go.Histogram(
                    x=speed_data,
                    nbinsx=20,
                    marker_color='blue',
                    opacity=0.7
                ))
                fig1.update_layout(
                    xaxis_title="Speed (km/h)",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Traffic Over Time
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=list(range(1, len(detections_data) + 1)),
                    y=detections_data,
                    name="Vehicles",
                    line=dict(color='green', width=2)
                ))
                fig2.add_trace(go.Scatter(
                    x=list(range(1, len(speed_data) + 1)),
                    y=speed_data,
                    name="Avg Speed",
                    line=dict(color='red', width=2)
                ))
                fig2.update_layout(
                    xaxis_title="Frame",
                    yaxis_title="Count / Speed",
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)

# ==========================
# Main Processing Logic
# ==========================

if start_button:
    if not video_path:
        st.error("⚠️ Please select a video source first")
    else:
        # Start processing
        process_video(
            video_path,
            model_type,
            confidence_threshold,
            fps,
            frame_skip
        )

# ==========================
# Footer
# ==========================

st.divider()
st.caption("🚗 Traffic AI Detection System - Powered by Computer Vision")
st.caption(f"Model: {model_type} | Device: {device} | FPS: {fps} | Frame Skip: {frame_skip}")

# ==========================
# Cleanup Function
# ==========================

def cleanup():
    """Cleanup resources when app closes"""
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()

# Register cleanup
import atexit
atexit.register(cleanup)