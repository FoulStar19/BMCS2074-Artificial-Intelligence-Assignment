import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import sys
from PIL import Image
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import utilities
try:
    from backend.utils.model_loader import ModelLoader
    from backend.utils.image_processor import ImageProcessor
    from backend.utils.video_processor import VideoProcessor
except ImportError as e:
    st.error(f"Error importing backend modules: {e}")
    st.stop()

# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="AI Human Detection System",
    page_icon="👤",
    layout="wide"
)

# ==========================
# Custom CSS
# ==========================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .model-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    .model-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .model-type-cnn { background-color: #e3f2fd; border-left: 4px solid #1976d2; }
    .model-type-ml { background-color: #f3e5f5; border-left: 4px solid #7b1fa2; }
    .model-type-yolo { background-color: #e8f5e9; border-left: 4px solid #388e3c; }
</style>
""", unsafe_allow_html=True)

# ==========================
# Header
# ==========================

st.markdown('<div class="main-header">👤 AI Human Detection System</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    Computer Vision Demo - Compare CNN, Machine Learning, and YOLO Models
</div>
""", unsafe_allow_html=True)

st.divider()

# ==========================
# Model Selection
# ==========================

st.subheader("🎯 Select Detection Model")

# Model types and their descriptions
model_types = {
    "CNN": {
        "icon": "🧠",
        "description": "Custom Convolutional Neural Network built from scratch",
        "color": "blue",
        "features": ["Deep Learning", "Feature Extraction", "End-to-End Training"],
        "pros": "Good accuracy, learns features automatically",
        "cons": "Requires more data, longer training time"
    },
    "ML": {
        "icon": "📊",
        "description": "Traditional Machine Learning (SVM/Random Forest) with HOG features",
        "color": "purple",
        "features": ["Feature Engineering", "HOG Descriptors", "Classical ML"],
        "pros": "Faster training, interpretable, works with less data",
        "cons": "Requires manual feature extraction, lower accuracy"
    },
    "YOLO": {
        "icon": "🚀",
        "description": "YOLOv8 - State-of-the-art object detection",
        "color": "green",
        "features": ["Object Detection", "Bounding Boxes", "Real-Time"],
        "pros": "Highest accuracy, detects multiple objects, real-time",
        "cons": "Requires GPU, larger model size"
    }
}

col1, col2 = st.columns([2, 1])

with col1:
    selected_model_type = st.selectbox(
        "Choose Model Type",
        options=list(model_types.keys()),
        format_func=lambda x: f"{model_types[x]['icon']} {x}",
        help="Select the type of AI model to use for detection"
    )

with col2:
    model_info = model_types[selected_model_type]
    st.metric(
        label="Model Type",
        value=f"{model_info['icon']} {selected_model_type}",
        delta=model_info['color'].title()
    )

# Display model info
st.markdown(f"""
<div class="model-card model-type-{selected_model_type.lower()}">
    <h4>{model_info['icon']} {selected_model_type} Model</h4>
    <p><strong>Description:</strong> {model_info['description']}</p>
    <p><strong>Features:</strong> {', '.join(model_info['features'])}</p>
    <div style="display: flex; gap: 2rem; margin-top: 0.5rem;">
        <div><span style="color: green;">✅ Pros:</span> {model_info['pros']}</div>
        <div><span style="color: red;">⚠️ Cons:</span> {model_info['cons']}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ==========================
# Model Loader
# ==========================

@st.cache_resource
def load_model(model_type):
    """Load the selected model type"""
    model_loader = ModelLoader()
    return model_loader.load_model(model_type)

# Load the selected model
try:
    with st.spinner(f"Loading {selected_model_type} model..."):
        model = load_model(selected_model_type)
    st.success(f"✅ {selected_model_type} model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading {selected_model_type} model: {e}")
    st.info(f"""
    **How to get {selected_model_type} model:**
    
    **CNN Model:**
    - Run: `python models/cnn/train_cnn.py`
    
    **ML Model:**
    - Run: `python models/ml/train_ml.py`
    
    **YOLO Model:**
    - Download: `yolov8n.pt` or run `python models/yolo/train_yolo.py`
    """)
    model = None

st.divider()

# ==========================
# Mock Dataset for Dashboard
# ==========================

mock_data = pd.DataFrame({
    "Image ID": ["IMG001", "IMG002", "IMG003", "IMG004", "IMG005"],
    "Detected Humans": [2, 5, 1, 3, 4],
    "Confidence": [96, 91, 98, 89, 94],
    "Status": ["Detected", "Detected", "Detected", "Detected", "Detected"]
})

# ==========================
# Dashboard
# ==========================

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Images", len(mock_data))

with col2:
    st.metric("Total Humans", mock_data["Detected Humans"].sum())

with col3:
    st.metric("Average Confidence", f"{mock_data['Confidence'].mean():.2f}%")

with col4:
    st.metric("Model Active", selected_model_type)

st.dataframe(mock_data, use_container_width=True)

st.divider()

# ==========================
# Tabs for Image and Video
# ==========================

image_tab, video_tab = st.tabs(["📷 Image Detection", "🎥 Video Detection"])

# ==========================
# IMAGE DETECTION TAB
# ==========================

with image_tab:
    st.header(f"{model_types[selected_model_type]['icon']} Image Human Detection")
    
    uploaded_image = st.file_uploader(
        "Upload an image for detection",
        type=["jpg", "png", "jpeg"]
    )
    
    if uploaded_image:
        try:
            image = Image.open(uploaded_image)
            image_array = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Original Image")
                st.image(image, use_container_width=True)
                
                # Image info
                st.caption(f"Size: {image.size[0]}x{image.size[1]}")
                st.caption(f"Mode: {image.mode}")
            
            with col2:
                st.subheader(f"🎯 {selected_model_type} Detection Result")
                
                if st.button(f"🚀 Run {selected_model_type} Detection", use_container_width=True):
                    if model is None:
                        st.error("Please load a valid model first")
                    else:
                        with st.spinner(f"{selected_model_type} model analyzing..."):
                            start_time = time.time()
                            
                            # Process image with selected model type
                            result_img, detections = ImageProcessor.process_image(
                                image_array,
                                model,
                                model_type=selected_model_type,
                                confidence_threshold=0.5
                            )
                            
                            processing_time = time.time() - start_time
                            
                            # Display results
                            st.image(result_img, use_container_width=True)
                            st.success(f"✅ Detection Completed in {processing_time:.2f}s")
                            
                            # Display metrics
                            if detections:
                                cols = st.columns(3)
                                with cols[0]:
                                    st.metric("Humans Detected", len(detections))
                                with cols[1]:
                                    avg_conf = sum(d['confidence'] for d in detections) / len(detections)
                                    st.metric("Avg Confidence", f"{avg_conf:.1f}%")
                                with cols[2]:
                                    st.metric("Processing Time", f"{processing_time:.2f}s")
                                
                                # Show detection details
                                st.subheader("Detection Details")
                                for i, det in enumerate(detections[:5], 1):
                                    st.caption(f"Person {i}: {det['confidence']:.2f}% confidence")
                            else:
                                st.info("No humans detected in this image")
                            
        except Exception as e:
            st.error(f"Error processing image: {e}")

# ==========================
# VIDEO DETECTION TAB
# ==========================

with video_tab:
    st.header(f"{model_types[selected_model_type]['icon']} Video Human Detection")
    
    uploaded_video = st.file_uploader(
        "Upload a video for detection",
        type=["mp4", "avi", "mov", "mkv"]
    )
    
    if uploaded_video:
        st.video(uploaded_video, use_container_width=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05
            )
        
        with col2:
            frame_skip = st.number_input(
                "Frame Skip",
                min_value=1,
                max_value=10,
                value=2,
                step=1,
                help="Process every Nth frame for performance"
            )
        
        with col3:
            if st.button(f"▶ Run {selected_model_type} Detection", use_container_width=True):
                if model is None:
                    st.error("Please load a valid model first")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    result_container = st.empty()
                    metrics_container = st.empty()
                    
                    detection_counts = []
                    
                    # Save uploaded video temporarily
                    temp_video_path = "temp_video.mp4"
                    try:
                        with open(temp_video_path, "wb") as f:
                            f.write(uploaded_video.getbuffer())
                        
                        # Process video with selected model
                        video_processor = VideoProcessor(model, model_type=selected_model_type)
                        
                        # Get total frames
                        cap = cv2.VideoCapture(temp_video_path)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        
                        if total_frames == 0:
                            st.error("Video has no frames or is corrupted")
                        else:
                            # Process video frames
                            frame_idx = 0
                            processing_times = []
                            
                            for frame, detections in video_processor.process_video(
                                temp_video_path, confidence_threshold, frame_skip
                            ):
                                start_time = time.time()
                                
                                # Update progress
                                progress_value = min(frame_idx / (total_frames / frame_skip), 1.0)
                                progress_bar.progress(progress_value)
                                status_text.write(f"Processing frame {frame_idx + 1}/{int(total_frames / frame_skip)}")
                                
                                # Display frame with detections
                                result_container.image(frame, use_container_width=True)
                                
                                # Show detection info
                                if detections:
                                    detection_counts.append(len(detections))
                                    with metrics_container:
                                        st.caption(f"👤 {len(detections)} people detected")
                                        if len(detections) > 0:
                                            st.caption(f"Confidence: {detections[0]['confidence']:.1f}%")
                                else:
                                    with metrics_container:
                                        st.caption("No people detected")
                                
                                processing_times.append(time.time() - start_time)
                                frame_idx += 1
                            
                            # Final summary
                            st.success(f"✅ Video Detection Finished!")
                            
                            if detection_counts:
                                cols = st.columns(3)
                                with cols[0]:
                                    avg_detections = sum(detection_counts) / len(detection_counts)
                                    st.metric("Avg People/Frame", f"{avg_detections:.1f}")
                                with cols[1]:
                                    st.metric("Total Frames", len(detection_counts))
                                with cols[2]:
                                    avg_time = sum(processing_times) / len(processing_times)
                                    st.metric("Avg Process Time", f"{avg_time*1000:.0f}ms")
                    
                    except Exception as e:
                        st.error(f"Error processing video: {e}")
                    
                    finally:
                        if os.path.exists(temp_video_path):
                            try:
                                os.remove(temp_video_path)
                            except:
                                pass