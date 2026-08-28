"""
Traffic AI Detection System
Main Streamlit application for vehicle detection and speed estimation.
Optimized with lazy loading, efficient memory management, and proper error handling.
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import torch

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Traffic AI Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Environment setup
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logs

# Add project root to path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Import backend modules
from backend.core.model_manager import ModelManager
from backend.core.video_processor_service import VideoProcessingService
from backend.core.session_manager import SessionManager
from backend.ui import components as ui
from backend.utils.helpers import get_video_properties

# Initialize session state
SessionManager.initialize()

# OpenCV setup with error handling
try:
    import cv2
    cv2.ocl.setUseOpenCL(False)
except ImportError:
    st.error(
        "❌ OpenCV is not installed. "
        "Run `pip install opencv-python-headless` and restart the app."
    )
    st.stop()


def main():
    """Main application entry point."""
    # Initialize model manager
    model_manager = _get_model_manager()
    
    # Get available models
    available_models = model_manager.discover_models()
    dataset_config = ModelManager.load_dataset_config()
    
    # Render sidebar
    sidebar_config = ui.display_sidebar(model_manager, available_models)
    
    # Extract config values
    model_type = sidebar_config["model_type"]
    selected_model = sidebar_config["selected_model"]
    model_path = sidebar_config["model_path"]
    uploaded_file = sidebar_config["uploaded_file"]
    confidence_threshold = sidebar_config["confidence_threshold"]
    device_setting = sidebar_config["device"]
    process_button = sidebar_config["process_button"]
    
    # Main title
    st.title("🚗 Traffic AI Detection System")
    st.markdown("Upload a traffic video to detect vehicles and estimate their speeds.")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📹 Processing", "📊 Results", "📈 Analytics"])
    
    # Processing tab
    with tab1:
        _handle_processing_tab(
            uploaded_file, sidebar_config, model_manager, model_type,
            model_path, selected_model, confidence_threshold, device_setting,
            process_button, dataset_config
        )
    
    # Results tab
    with tab2:
        _handle_results_tab()
    
    # Analytics tab
    with tab3:
        _handle_analytics_tab()


def _get_model_manager():
    """Get or create model manager instance."""
    model_manager = SessionManager.get(SessionManager.MODEL_MANAGER)
    if model_manager is None:
        model_manager = ModelManager()
        SessionManager.set(SessionManager.MODEL_MANAGER, model_manager)
    return model_manager


def _handle_processing_tab(uploaded_file, sidebar_config, model_manager,
                          model_type, model_path, selected_model,
                          confidence_threshold, device_setting,
                          process_button, dataset_config):
    """Handle the processing tab logic."""
    ui.display_processing_tab(uploaded_file, sidebar_config)
    
    is_processing = SessionManager.get(SessionManager.IS_PROCESSING, False)
    
    # Guard against concurrent processing
    if process_button and is_processing:
        st.warning("⏳ A video is already processing - please wait.")
        return
    
    if process_button:
        _start_processing(
            uploaded_file, model_manager, model_type, model_path,
            selected_model, confidence_threshold, device_setting,
            dataset_config, sidebar_config
        )
    
    # Display quick results if available
    _display_quick_results()


def _start_processing(uploaded_file, model_manager, model_type, model_path,
                     selected_model, confidence_threshold, device_setting,
                     dataset_config, sidebar_config):
    """Start video processing with proper error handling."""
    if uploaded_file is None:
        st.error("⚠️ Please upload a video")
        return
    
    if model_path is None:
        st.error("⚠️ No model selected! Please check your runs folder.")
        return
    
    # Save uploaded video to temp file
    video_source = _save_uploaded_video(uploaded_file)
    if video_source is None:
        return
    
    SessionManager.set(SessionManager.VIDEO_PATH, video_source)
    
    # Display video info
    _display_video_info(video_source)
    
    # Determine device
    device = _get_device(device_setting)
    
    # Load detector
    detector = _load_detector(model_manager, model_type, model_path, device, confidence_threshold)
    if detector is None:
        return
    
    # Enable tracking
    if hasattr(detector, "enable_tracking"):
        detector.enable_tracking = True

    # Optionally load the CNN verification classifier alongside the
    # primary detector (see components.py "Optional Verification").
    classifier = _load_classifier_if_requested(model_manager, sidebar_config, device)

    # Process video
    _process_video(video_source, detector, dataset_config, confidence_threshold, classifier)


def _load_classifier_if_requested(model_manager, sidebar_config, device):
    """Load the CNN verification classifier if the user enabled it in the
    sidebar. Returns None (no-op downstream) if not requested or not found."""
    if not sidebar_config.get("enable_cnn_verification"):
        return None

    cnn_model_path = sidebar_config.get("cnn_model_path")
    with st.spinner("Loading CNN verification classifier..."):
        classifier = model_manager.load_classifier(cnn_model_path, device=device)

    if classifier is None:
        st.warning("⚠️ Could not load the CNN classifier - continuing without verification.")
    return classifier


def _save_uploaded_video(uploaded_file):
    """Save uploaded video to temp file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            return tmp_file.name
    except Exception as e:
        st.error(f"❌ Failed to save uploaded video: {e}")
        return None


def _display_video_info(video_source):
    """Display video properties."""
    video_info = get_video_properties(video_source)
    if "error" not in video_info:
        st.info(
            f"📹 Video: {video_info['width']}x{video_info['height']}, "
            f"{video_info['fps']:.1f} FPS, {video_info['total_frames']} frames, "
            f"{video_info['duration']}s"
        )


def _get_device(device_setting):
    """Determine processing device."""
    if device_setting == "Auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device_setting == "CPU":
        return "cpu"
    else:
        return "cuda"


def _load_detector(model_manager, model_type, model_path, device, confidence_threshold):
    """Load the detector model."""
    with st.spinner(f"Loading {model_type} model..."):
        try:
            detector = model_manager.load_model(
                model_type=model_type,
                model_path=model_path,
                device=device,
                conf_threshold=confidence_threshold,
            )
            return detector
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            return None


def _process_video(video_source, detector, dataset_config, confidence_threshold, classifier=None):
    """Process the video with the loaded detector."""
    video_service = VideoProcessingService(
        detector=detector, dataset_config=dataset_config, classifier=classifier
    )
    SessionManager.set(SessionManager.VIDEO_SERVICE, video_service)
    
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    def progress_callback(progress, current, total):
        ui.display_processing_status(status_placeholder, progress_bar, progress, current, total)
    
    try:
        status_placeholder.text("⏳ Processing video...")
        SessionManager.set(SessionManager.IS_PROCESSING, True)
        
        output_path, results = video_service.process_video(
            video_path=video_source,
            confidence_threshold=confidence_threshold,
            class_id=-1,
            progress_callback=progress_callback,
        )
        
        # Store results
        SessionManager.set(SessionManager.CURRENT_RESULTS, results)
        SessionManager.set(SessionManager.PROCESSED_VIDEO_PATH, output_path)
        SessionManager.set(SessionManager.VIDEO_PROCESSED, True)
        SessionManager.set(SessionManager.VIDEO_BYTES_CACHE, None)
        
        status_placeholder.text("✅ Processing complete!")
        progress_bar.progress(1.0)
        st.success("✅ Video processing completed successfully!")
        st.info("📊 View results in the 'Results' tab")
        
    except MemoryError:
        st.error("❌ Memory error! Try using a shorter video.")
        status_placeholder.text("❌ Processing failed - Out of memory")
    except Exception as e:
        st.error(f"❌ Error processing video: {e}")
        status_placeholder.text("❌ Processing failed")
        import traceback
        st.code(traceback.format_exc())
    finally:
        SessionManager.set(SessionManager.IS_PROCESSING, False)
        SessionManager.release_heavy_runtime_objects()
        
        # Clean up temp file
        if video_source and os.path.exists(video_source):
            try:
                os.unlink(video_source)
            except OSError:
                pass


def _display_quick_results():
    """Display quick results summary."""
    current_results = SessionManager.get(SessionManager.CURRENT_RESULTS)
    video_processed = SessionManager.get(SessionManager.VIDEO_PROCESSED, False)
    
    if current_results is not None and video_processed:
        with st.expander("📊 Quick Results", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Vehicles", current_results.get("total_vehicles", 0))
            col2.metric("Avg Speed", f"{current_results.get('avg_speed', 0):.1f} km/h")
            col3.metric("Max Speed", f"{current_results.get('max_speed', 0):.1f} km/h")
            col4.metric("Frames Processed", current_results.get("frames_processed", 0))


def _handle_results_tab():
    """Handle the results tab."""
    current_results = SessionManager.get(SessionManager.CURRENT_RESULTS)
    processed_video_path = SessionManager.get(SessionManager.PROCESSED_VIDEO_PATH)
    video_processed = SessionManager.get(SessionManager.VIDEO_PROCESSED, False)
    
    if current_results is not None and video_processed:
        ui.display_results_tab(current_results, processed_video_path)
    else:
        st.info("📊 No results to display. Process a video first.")


def _handle_analytics_tab():
    """Handle the analytics tab."""
    current_results = SessionManager.get(SessionManager.CURRENT_RESULTS)
    ui.display_analytics_tab(current_results)


if __name__ == "__main__":
    main()