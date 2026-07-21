"""
UI Components - Reusable Streamlit UI components
"""

import streamlit as st
import os
import base64
from datetime import datetime


def display_sidebar(model_manager, available_models):
    """
    Display the sidebar with configuration options
    
    Args:
        model_manager: ModelManager instance
        available_models: Dictionary of available models
        
    Returns:
        Dictionary of configuration values
    """
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
        else:
            st.warning("⚠️ No trained models found in runs folder!")
            selected_model = "None"
            model_path = None
        
        st.markdown("---")
        
        # Video Source
        st.subheader("🎥 Video Source")
        
        uploaded_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Upload a video file for processing"
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
            help="Minimum confidence score for detections"
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
            type="primary"
        )
        
        # System Info
        st.markdown("---")
        st.subheader("🖥️ System Info")
        
        try:
            import torch
            cuda_available = torch.cuda.is_available() if torch else False
            device_info = "CUDA" if cuda_available else "CPU"
            st.info(f"Device: {device_info}")
            
            if cuda_available:
                st.info(f"GPU: {torch.cuda.get_device_name(0)}")
        except:
            st.info("Device: CPU")
        
        st.caption(f"📊 Found {len(available_models)} model(s)")
    
    return {
        'model_type': model_type,
        'selected_model': selected_model,
        'model_path': model_path,
        'uploaded_file': uploaded_file,
        'confidence_threshold': confidence_threshold,
        'device': device,
        'process_button': process_button
    }


def display_processing_tab(uploaded_file, config):
    """
    Display the processing tab
    
    Args:
        uploaded_file: Uploaded video file
        config: Configuration dictionary
    """
    # Display video source
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
        st.metric("Selected Model", f"{config.get('model_type', 'N/A')} - {config.get('selected_model', 'None')}")
        st.metric("Confidence", f"{config.get('confidence_threshold', 0.25):.2f}")
        
        if uploaded_file is not None:
            st.success("✅ Video loaded")
        else:
            st.info("⏳ Waiting for video")


def display_results_tab(results, output_video_path):
    """
    Display the results tab
    
    Args:
        results: Results dictionary
        output_video_path: Path to processed video
    """
    if results is None or not results:
        st.info("📊 No results to display. Process a video first.")
        return
    
    # Import here to avoid circular imports
    from backend.analytics.report_generator import display_metrics, ReportGenerator
    
    # Display statistics
    display_metrics(results)
    
    # Display processed video
    if output_video_path and os.path.exists(output_video_path):
        st.subheader("📹 Processed Video")
        
        try:
            file_size = os.path.getsize(output_video_path)
            file_size_mb = file_size / (1024 * 1024)
            st.info(f"📁 File: {os.path.basename(output_video_path)} ({file_size_mb:.1f} MB)")
            
            st.video(output_video_path)
            
            # Download button
            with open(output_video_path, 'rb') as f:
                video_bytes = f.read()
                st.download_button(
                    label="📥 Download Processed Video",
                    data=video_bytes,
                    file_name=os.path.basename(output_video_path),
                    mime="video/mp4",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error displaying video: {e}")
            st.info(f"Video saved at: {output_video_path}")
    
    # Display analytics
    st.subheader("📊 Analytics")
    
    generator = ReportGenerator(results)
    
    if generator.dataframe is not None and not generator.dataframe.empty:
        # Display detection chart
        fig_detections = generator.create_detection_chart()
        if fig_detections:
            st.plotly_chart(fig_detections, use_container_width=True)
        
        # Display density chart
        fig_density = generator.create_density_chart()
        if fig_density:
            st.plotly_chart(fig_density, use_container_width=True)
        
        # Data table
        with st.expander("📋 View Detailed Data"):
            st.dataframe(generator.dataframe, use_container_width=True)
            
            csv_data = generator.export_csv()
            if csv_data:
                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv_data,
                    file_name=f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    else:
        st.info("No detailed data available for analytics.")


def display_analytics_tab(results):
    """Display the analytics tab"""
    if results is None or not results:
        st.info("📊 No analytics data available. Process a video first.")
        return
    
    # Import here to avoid circular imports
    from backend.analytics.report_generator import ReportGenerator
    
    generator = ReportGenerator(results)
    
    # Display summary statistics
    st.subheader("📊 Summary Statistics")
    stats_df = generator.get_summary_statistics()
    st.dataframe(stats_df, use_container_width=True)
    
    # Display charts
    st.subheader("📈 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_detections = generator.create_detection_chart()
        if fig_detections:
            st.plotly_chart(fig_detections, use_container_width=True)
    
    with col2:
        fig_density = generator.create_density_chart()
        if fig_density:
            st.plotly_chart(fig_density, use_container_width=True)
    
    # Export options
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
                use_container_width=True
            )
    
    with col2:
        json_data = generator.export_json()
        if json_data:
            st.download_button(
                label="📊 Download JSON Report",
                data=json_data,
                file_name=f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )


def display_processing_status(status_placeholder, progress_bar, progress, current, total):
    """
    Display processing status
    
    Args:
        status_placeholder: Streamlit placeholder for status text
        progress_bar: Streamlit progress bar
        progress: Progress value (0-1)
        current: Current frame
        total: Total frames
    """
    progress_bar.progress(progress)
    status_placeholder.text(f"Processing: {current}/{total} frames ({progress*100:.1f}%)")