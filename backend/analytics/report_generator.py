# backend/analytics/report_generator.py
"""
Analytics and Report Generation Service
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import streamlit as st


class ReportGenerator:
    """
    Generates analytics reports and visualizations from processing results
    """
    
    def __init__(self, results):
        """
        Initialize with processing results
        
        Args:
            results: Results dictionary from video processing
        """
        self.results = results
        self.dataframe = None
        
        if results and results.get('frames'):
            self.dataframe = pd.DataFrame({
                'Frame': results['frames'],
                'Detections': results['detections'],
                'Density': results['density']
            })
    
    def get_summary_statistics(self):
        """
        Get summary statistics from results
        
        Returns:
            DataFrame with summary statistics
        """
        results = self.results
        if not results:
            return pd.DataFrame()
        
        stats = {
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
                results.get('total_vehicles', 0),
                f"{results.get('avg_speed', 0):.2f} km/h",
                f"{results.get('max_speed', 0):.2f} km/h",
                f"{results.get('min_speed', 0):.2f} km/h",
                f"{np.mean(results.get('density', [0])):.2f}%" if results.get('density') else "0.00%",
                f"{np.max(results.get('density', [0])):.2f}%" if results.get('density') else "0.00%",
                results.get('frames_processed', 0),
                f"{results.get('processing_time', 0):.2f}s"
            ]
        }
        
        return pd.DataFrame(stats)
    
    def create_detection_chart(self):
        """
        Create vehicle detection over time chart
        
        Returns:
            Plotly figure
        """
        if self.dataframe is None or self.dataframe.empty:
            return None
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.dataframe['Frame'],
            y=self.dataframe['Detections'],
            mode='lines+markers',
            name='Vehicle Count',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        fig.update_layout(
            title='Vehicle Count Over Time',
            xaxis_title='Frame Number',
            yaxis_title='Number of Vehicles',
            height=300,
            hovermode='x unified'
        )
        return fig
    
    def create_density_chart(self):
        """
        Create traffic density over time chart
        
        Returns:
            Plotly figure
        """
        if self.dataframe is None or self.dataframe.empty:
            return None
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.dataframe['Frame'],
            y=self.dataframe['Density'],
            mode='lines+markers',
            name='Density',
            line=dict(color='orange', width=2),
            marker=dict(size=4)
        ))
        fig.update_layout(
            title='Traffic Density Over Time',
            xaxis_title='Frame Number',
            yaxis_title='Density (%)',
            height=300,
            hovermode='x unified'
        )
        return fig
    
    def create_combined_dashboard(self):
        """
        Create a combined dashboard with multiple charts
        
        Returns:
            Dictionary of figures
        """
        return {
            'detections': self.create_detection_chart(),
            'density': self.create_density_chart()
        }
    
    def export_csv(self):
        """Export data as CSV string"""
        if self.dataframe is None or self.dataframe.empty:
            return None
        return self.dataframe.to_csv(index=False)
    
    def export_json(self):
        """Export results as JSON string"""
        if not self.results:
            return None
        
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            else:
                return obj
        
        converted = convert(self.results)
        return json.dumps(converted, indent=2)
    
    def get_summary_metrics(self):
        """
        Get summary metrics for display
        
        Returns:
            Dictionary of metrics
        """
        results = self.results
        if not results:
            return {}
        
        return {
            'total_vehicles': results.get('total_vehicles', 0),
            'avg_speed': results.get('avg_speed', 0),
            'max_speed': results.get('max_speed', 0),
            'min_speed': results.get('min_speed', 0),
            'frames_processed': results.get('frames_processed', 0),
            'processing_time': results.get('processing_time', 0),
            'avg_density': np.mean(results.get('density', [0])) if results.get('density') else 0,
            'max_density': np.max(results.get('density', [0])) if results.get('density') else 0,
        }


def display_metrics(results):
    """
    Display summary metrics in Streamlit
    
    Args:
        results: Results dictionary
    """
    if not results:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🚗 Total Vehicles", results.get('total_vehicles', 0))
    
    with col2:
        st.metric("📏 Avg Speed", f"{results.get('avg_speed', 0):.1f} km/h")
    
    with col3:
        st.metric("📈 Max Speed", f"{results.get('max_speed', 0):.1f} km/h")
    
    with col4:
        st.metric("⏱️ Processing Time", f"{results.get('processing_time', 0):.1f}s")


def display_analytics_tab(results):
    """
    Display comprehensive analytics tab
    
    Args:
        results: Results dictionary
    """
    if not results:
        st.info("📊 No analytics data available. Process a video first.")
        return
    
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