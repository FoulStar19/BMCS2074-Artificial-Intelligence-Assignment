"""
Traffic AI Detection System
Main Streamlit application for vehicle detection and speed estimation
"""

import streamlit as st
import sys
import os
import tempfile
import gc
from pathlib import Path
import torch

# Set page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Traffic AI Detection System",
    page_icon="🚗",
    layout="wide"
)

# ============================================
# CRITICAL FIX: Set headless mode BEFORE importing OpenCV
# ============================================
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ":0"

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import backend modules
from backend.core.model_manager import ModelManager
from backend.core.video_processor_service import VideoProcessingService
from backend.core.session_manager import SessionManager
from backend.ui import components as ui
from backend.analytics.report_generator import ReportGenerator
from backend.utils.helpers import get_video_properties

# Initialize session state
SessionManager.initialize()

# ============================================
# OpenCV Headless Setup
# ============================================
cv2 = None
try:
    import cv2
    cv2.ocl.setUseOpenCL(False)
    print("✅ OpenCV loaded successfully in headless mode")
except ImportError as e:
    print(f"⚠️ OpenCV import error: {e}")
    
    # Try to install it on the fly
    try:
        import subprocess
        print("Attempting to install opencv-python-headless...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "opencv-python-headless==4.8.1.78", 
            "--force-reinstall"
        ])
        import cv2
        cv2.ocl.setUseOpenCL(False)
        print("✅ OpenCV installed and loaded")
    except Exception as e3:
        print(f"⚠️ Failed to install OpenCV: {e3}")
        
        # Create dummy cv2
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
            def destroyAllWindows(*args, **kwargs): pass
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
        cv2 = DummyCV2()

# ============================================
# Main Application
# ============================================

def main():
    """Main application"""
    
    # Initialize model manager if not already done
    model_manager = SessionManager.get(SessionManager.MODEL_MANAGER)
    if model_manager is None:
        model_manager = ModelManager()
        SessionManager.set(SessionManager.MODEL_MANAGER, model_manager)
    
    # Load dataset configuration
    dataset_config = ModelManager.load_dataset_config()
    
    # Discover available models
    available_models = model_manager.discover_models()
    
    # ==========================
    # SIDEBAR
    # ==========================
    sidebar_config = ui.display_sidebar(model_manager, available_models)
    
    # Extract config values
    model_type = sidebar_config['model_type']
    selected_model = sidebar_config['selected_model']
    model_path = sidebar_config['model_path']
    uploaded_file = sidebar_config['uploaded_file']
    confidence_threshold = sidebar_config['confidence_threshold']
    device_setting = sidebar_config['device']
    process_button = sidebar_config['process_button']
    
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
        ui.display_processing_tab(uploaded_file, sidebar_config)
        
        # Process button logic
        if process_button:
            if uploaded_file is None:
                st.error("⚠️ Please upload a video")
            elif model_path is None:
                st.error("⚠️ No model selected! Please check your runs folder.")
            elif cv2 is None:
                st.error("⚠️ OpenCV is not available. Please check your installation.")
            else:
                # Save uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_source = tmp_file.name
                    SessionManager.set(SessionManager.VIDEO_PATH, video_source)
                
                # Show video info
                video_info = get_video_properties(video_source)
                if 'error' not in video_info:
                    st.info(f"📹 Video: {video_info['width']}x{video_info['height']}, "
                           f"{video_info['fps']:.1f} FPS, {video_info['total_frames']} frames, "
                           f"{video_info['duration']}s")
                
                # Determine device
                if device_setting == "Auto":
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                elif device_setting == "CPU":
                    device = 'cpu'
                else:
                    device = 'cuda'
                
                # Load model
                with st.spinner(f"Loading {model_type} model from {selected_model}..."):
                    detector = model_manager.load_model(
                        model_type=model_type,
                        model_path=model_path,
                        device=device,
                        conf_threshold=confidence_threshold
                    )
                
                if detector is None:
                    st.error("❌ Failed to load model. Please check the model path.")
                else:
                    # Enable tracking
                    if hasattr(detector, 'enable_tracking'):
                        detector.enable_tracking = True
                    
                    # Create video processing service
                    video_service = VideoProcessingService(
                        detector=detector,
                        dataset_config=dataset_config
                    )
                    SessionManager.set(SessionManager.VIDEO_SERVICE, video_service)
                    
                    # Process video
                    status_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    
                    def progress_callback(progress, current, total):
                        ui.display_processing_status(
                            status_placeholder, progress_bar, 
                            progress, current, total
                        )
                    
                    try:
                        status_placeholder.text("⏳ Processing video...")
                        SessionManager.set(SessionManager.IS_PROCESSING, True)
                        
                        output_path, results = video_service.process_video(
                            video_path=video_source,
                            confidence_threshold=confidence_threshold,
                            class_id=-1,  # Track all vehicle classes
                            progress_callback=progress_callback
                        )
                        
                        # Store results - FIX: Access as dictionary
                        SessionManager.set(SessionManager.CURRENT_RESULTS, results)
                        SessionManager.set(SessionManager.PROCESSED_VIDEO_PATH, output_path)
                        SessionManager.set(SessionManager.VIDEO_PROCESSED, True)
                        SessionManager.set(SessionManager.IS_PROCESSING, False)
                        
                        status_placeholder.text("✅ Processing complete!")
                        progress_bar.progress(1.0)
                        
                        st.success("✅ Video processing completed successfully!")
                        st.balloons()
                        st.info("📊 View results in the 'Results' tab")
                        
                        # Clean up temp file
                        if video_source and os.path.exists(video_source):
                            try:
                                os.unlink(video_source)
                            except:
                                pass
                        
                    except MemoryError:
                        st.error("❌ Memory error! Try using a shorter video.")
                        status_placeholder.text("❌ Processing failed - Out of memory")
                        SessionManager.set(SessionManager.IS_PROCESSING, False)
                    except Exception as e:
                        st.error(f"❌ Error processing video: {e}")
                        status_placeholder.text("❌ Processing failed")
                        import traceback
                        st.code(traceback.format_exc())
                        SessionManager.set(SessionManager.IS_PROCESSING, False)
                    
                    # Clean up
                    model_manager.unload_model()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Show quick results if available
        # FIX: Use SessionManager.get() to retrieve results
        current_results = SessionManager.get(SessionManager.CURRENT_RESULTS)
        video_processed = SessionManager.get(SessionManager.VIDEO_PROCESSED, False)
        
        if current_results is not None and video_processed:
            with st.expander("📊 Quick Results", expanded=True):
                # FIX: Access results as dictionary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Vehicles", current_results.get('total_vehicles', 0))
                with col2:
                    st.metric("Avg Speed", f"{current_results.get('avg_speed', 0):.1f} km/h")
                with col3:
                    st.metric("Max Speed", f"{current_results.get('max_speed', 0):.1f} km/h")
                with col4:
                    st.metric("Frames Processed", current_results.get('frames_processed', 0))
    
    # ==========================
    # TAB 2: Results
    # ==========================
    with tab2:
        # FIX: Use SessionManager.get() to retrieve results
        current_results = SessionManager.get(SessionManager.CURRENT_RESULTS)
        processed_video_path = SessionManager.get(SessionManager.PROCESSED_VIDEO_PATH)
        video_processed = SessionManager.get(SessionManager.VIDEO_PROCESSED, False)
        
        if current_results is not None and video_processed:
            ui.display_results_tab(
                current_results,
                processed_video_path
            )
        else:
            st.info("📊 No results to display. Process a video first.")
    
    # ==========================
    # TAB 3: Analytics
    # ==========================
    with tab3:
        # FIX: Use SessionManager.get() to retrieve results
        current_results = SessionManager.get(SessionManager.CURRENT_RESULTS)
        ui.display_analytics_tab(current_results)


if __name__ == "__main__":
    main()