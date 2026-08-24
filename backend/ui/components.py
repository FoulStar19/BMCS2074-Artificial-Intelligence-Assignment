"""
UI Components - reusable Streamlit UI components.
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional

import streamlit as st

from backend.core.session_manager import SessionManager


def display_sidebar(model_manager, available_models: Dict[str, str]) -> Dict[str, Any]:
    """Display sidebar with model selection and parameters."""
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
                help="Select the trained model version from your runs folder",
            )
            model_path = available_models[selected_model]
            st.info(f"📁 {os.path.basename(model_path)}")
        else:
            st.warning("⚠️ No trained models found in runs folder!")
            selected_model = "None"
            model_path = None

        st.markdown("---")

        # Video Upload
        st.subheader("🎥 Video Source")
        uploaded_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Upload a video file for processing",
        )

        st.markdown("---")

        # Processing Parameters
        st.subheader("⚙️ Processing Parameters")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05,
            help="Minimum confidence score for detections",
        )

        device = st.radio(
            "Processing Device",
            options=["Auto", "CPU", "CUDA"],
            index=0
        )

        st.markdown("---")

        # Process Button
        is_processing = SessionManager.get(SessionManager.IS_PROCESSING, False)
        process_button = st.button(
            "⏳ Processing..." if is_processing else "▶️ Process Video",
            use_container_width=True,
            type="primary",
            disabled=is_processing,
        )

        st.markdown("---")

        # System Info
        st.subheader("🖥️ System Info")
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            st.info(f"Device: {'CUDA' if cuda_available else 'CPU'}")
            if cuda_available:
                st.info(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            st.info("Device: CPU")

        st.caption(f"📊 Found {len(available_models)} model(s)")

    return {
        "model_type": model_type,
        "selected_model": selected_model,
        "model_path": model_path,
        "uploaded_file": uploaded_file,
        "confidence_threshold": confidence_threshold,
        "device": device,
        "process_button": process_button,
    }


def display_processing_tab(uploaded_file, config: Dict[str, Any]):
    """Display the processing tab."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Video Preview")
        video_preview = st.empty()
        if uploaded_file is not None:
            video_preview.video(uploaded_file)
        else:
            video_preview.info("👆 Upload a video to begin")

    with col2:
        st.subheader("📊 Quick Stats")
        st.metric(
            "Selected Model",
            f"{config.get('model_type', 'N/A')} - {config.get('selected_model', 'None')}"
        )
        st.metric("Confidence", f"{config.get('confidence_threshold', 0.25):.2f}")

        if uploaded_file is not None:
            st.success("✅ Video loaded")
        else:
            st.info("⏳ Waiting for video")


def display_results_tab(results: Dict[str, Any], output_video_path: Optional[str]):
    """Display the results tab."""
    if results is None or not results:
        st.info("📊 No results to display. Process a video first.")
        return

    from backend.analytics.report_generator import display_metrics, ReportGenerator

    display_metrics(results)

    if output_video_path and os.path.exists(output_video_path):
        st.subheader("📥 Download Processed Video")
        
        file_size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
        st.info(f"📁 File: {os.path.basename(output_video_path)} ({file_size_mb:.1f} MB)")
        
        # Read file in chunks for better memory management
        try:
            with open(output_video_path, "rb") as f:
                video_bytes = f.read()
            
            # Use the data parameter with bytes
            st.download_button(
                label="📥 Download Video",
                data=video_bytes,
                file_name=os.path.basename(output_video_path),
                mime="video/mp4",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Error preparing video for download: {e}")
            
        st.caption(f"📍 File location: `{output_video_path}`")
                
    else:
        st.warning("⚠️ No processed video found. Please process a video first.")

    # Analytics
    st.subheader("📊 Analytics")
    generator = ReportGenerator(results)

    if generator.dataframe is not None and not generator.dataframe.empty:
        fig_detections = generator.create_detection_chart()
        if fig_detections:
            st.plotly_chart(fig_detections, use_container_width=True, key="detection_chart_results")

        fig_density = generator.create_density_chart()
        if fig_density:
            st.plotly_chart(fig_density, use_container_width=True, key="density_chart_results")

        fig_speed = generator.create_speed_chart()
        if fig_speed:
            st.plotly_chart(fig_speed, use_container_width=True, key="speed_chart_results")

        with st.expander("📋 View Detailed Data"):
            st.dataframe(generator.dataframe, use_container_width=True)
            csv_data = generator.export_csv()
            if csv_data:
                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv_data,
                    file_name=f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
    else:
        st.info("No detailed data available for analytics.")


def display_analytics_tab(results: Dict[str, Any]):
    """Display the analytics tab."""
    if results is None or not results:
        st.info("📊 No analytics data available. Process a video first.")
        return

    from backend.analytics.report_generator import ReportGenerator

    generator = ReportGenerator(results)

    st.subheader("📊 Summary Statistics")
    st.dataframe(generator.get_summary_statistics(), use_container_width=True)

    st.subheader("📈 Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        fig_detections = generator.create_detection_chart()
        if fig_detections:
            st.plotly_chart(fig_detections, use_container_width=True, key="detection_chart_analytics")

    with col2:
        fig_density = generator.create_density_chart()
        if fig_density:
            st.plotly_chart(fig_density, use_container_width=True, key="density_chart_analytics")

    fig_speed = generator.create_speed_chart()
    if fig_speed:
        st.plotly_chart(fig_speed, use_container_width=True, key="speed_chart_analytics")

    st.subheader("📥 Export Data")
    col1, col2 = st.columns(2)

    with col1:
        csv_data = generator.export_csv()
        if csv_data:
            st.download_button(
                label="📊 Download CSV Report",
                data=csv_data,
                file_name=f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with col2:
        json_data = generator.export_json()
        if json_data:
            st.download_button(
                label="📊 Download JSON Report",
                data=json_data,
                file_name=f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )


def display_processing_status(status_placeholder, progress_bar, progress: float,
                             current: int, total: int):
    """Display processing status updates."""
    progress_bar.progress(progress)
    status_placeholder.text(f"Processing: {current}/{total} frames ({progress * 100:.1f}%)")