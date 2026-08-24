"""
Report Generator for traffic analysis.
Creates visualizations, summaries, and exportable reports.
"""

import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional
import streamlit as st


class ReportGenerator:
    """Generate reports and visualizations from traffic analysis results."""

    def __init__(self, results: Dict[str, Any]):
        """
        Initialize report generator with results.

        Args:
            results: Dictionary containing analysis results
        """
        self.results = results
        self.dataframe = self._create_dataframe()

    def _create_dataframe(self) -> Optional[pd.DataFrame]:
        """Create a DataFrame from results for analysis."""
        if not self.results:
            return None

        frames = self.results.get("frames", [])
        detections = self.results.get("detections", [])
        speeds = self.results.get("speeds", [])
        density = self.results.get("density", [])

        if not frames:
            return None

        min_len = min(len(frames), len(detections), len(speeds), len(density))
        if min_len == 0:
            return None

        return pd.DataFrame({
            "Frame": frames[:min_len],
            "Vehicles": detections[:min_len],
            "Avg_Speed": speeds[:min_len],
            "Density": density[:min_len]
        })

    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics as a DataFrame."""
        if self.dataframe is None or self.dataframe.empty:
            return pd.DataFrame({"Metric": ["No Data"], "Value": [0]})

        stats = {
            "Total Frames": [len(self.dataframe)],
            "Total Vehicles Detected": [self.results.get("total_vehicles", 0)],
            "Average Vehicles per Frame": [self.dataframe["Vehicles"].mean()],
            "Max Vehicles in Frame": [self.dataframe["Vehicles"].max()],
            "Average Speed (km/h)": [self.results.get("avg_speed", 0)],
            "Max Speed (km/h)": [self.results.get("max_speed", 0)],
            "Min Speed (km/h)": [self.results.get("min_speed", 0)],
            "Average Density (%)": [self.dataframe["Density"].mean()],
            "Max Density (%)": [self.dataframe["Density"].max()],
        }

        summary_df = pd.DataFrame(stats).T.reset_index()
        summary_df.columns = ["Metric", "Value"]
        return summary_df

    def create_detection_chart(self) -> Optional[go.Figure]:
        """Create a chart showing vehicle detections over time."""
        if self.dataframe is None or self.dataframe.empty:
            return None

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.dataframe["Frame"],
            y=self.dataframe["Vehicles"],
            mode="lines+markers",
            name="Vehicles",
            line=dict(color="blue", width=2),
            marker=dict(size=4, color="blue"),
        ))

        fig.update_layout(
            title="Vehicle Count per Frame",
            xaxis_title="Frame Number",
            yaxis_title="Number of Vehicles",
            template="plotly_white",
            height=400,
            hovermode="x unified",
        )

        return fig

    def create_density_chart(self) -> Optional[go.Figure]:
        """Create a chart showing traffic density over time."""
        if self.dataframe is None or self.dataframe.empty:
            return None

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.dataframe["Frame"],
            y=self.dataframe["Density"],
            mode="lines+markers",
            name="Density",
            line=dict(color="orange", width=2),
            marker=dict(size=4, color="orange"),
            fill="tozeroy",
            fillcolor="rgba(255, 165, 0, 0.2)",
        ))

        fig.update_layout(
            title="Traffic Density per Frame",
            xaxis_title="Frame Number",
            yaxis_title="Density (%)",
            template="plotly_white",
            height=400,
            hovermode="x unified",
        )

        return fig

    def create_speed_chart(self) -> Optional[go.Figure]:
        """Create a chart showing average speed over time."""
        if self.dataframe is None or self.dataframe.empty:
            return None

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.dataframe["Frame"],
            y=self.dataframe["Avg_Speed"],
            mode="lines+markers",
            name="Avg Speed",
            line=dict(color="green", width=2),
            marker=dict(size=4, color="green"),
        ))

        # Add horizontal line for average
        avg_speed = self.results.get("avg_speed", 0)
        fig.add_hline(
            y=avg_speed,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Avg: {avg_speed:.1f} km/h",
            annotation_position="top right"
        )

        fig.update_layout(
            title="Average Speed per Frame",
            xaxis_title="Frame Number",
            yaxis_title="Speed (km/h)",
            template="plotly_white",
            height=400,
            hovermode="x unified",
        )

        return fig

    def create_combined_dashboard(self) -> Optional[go.Figure]:
        """Create a combined dashboard with all charts."""
        if self.dataframe is None or self.dataframe.empty:
            return None

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Vehicle Count", "Traffic Density", "Average Speed"),
            shared_xaxes=True,
            vertical_spacing=0.08,
        )

        # Vehicle count
        fig.add_trace(
            go.Scatter(
                x=self.dataframe["Frame"],
                y=self.dataframe["Vehicles"],
                mode="lines",
                name="Vehicles",
                line=dict(color="blue", width=2),
            ),
            row=1, col=1
        )

        # Density
        fig.add_trace(
            go.Scatter(
                x=self.dataframe["Frame"],
                y=self.dataframe["Density"],
                mode="lines",
                name="Density",
                line=dict(color="orange", width=2),
                fill="tozeroy",
                fillcolor="rgba(255, 165, 0, 0.2)",
            ),
            row=2, col=1
        )

        # Speed
        fig.add_trace(
            go.Scatter(
                x=self.dataframe["Frame"],
                y=self.dataframe["Avg_Speed"],
                mode="lines",
                name="Speed",
                line=dict(color="green", width=2),
            ),
            row=3, col=1
        )

        fig.update_layout(
            height=700,
            template="plotly_white",
            showlegend=True,
            hovermode="x unified",
        )

        fig.update_xaxes(title_text="Frame Number", row=3, col=1)

        return fig

    def export_csv(self) -> Optional[str]:
        """Export data as CSV string."""
        if self.dataframe is None or self.dataframe.empty:
            return None
        return self.dataframe.to_csv(index=False)

    def export_json(self) -> Optional[str]:
        """Export data as JSON string."""
        if not self.results:
            return None

        export_data = {
            "summary": {
                "total_vehicles": self.results.get("total_vehicles", 0),
                "avg_speed": self.results.get("avg_speed", 0),
                "max_speed": self.results.get("max_speed", 0),
                "min_speed": self.results.get("min_speed", 0),
                "frames_processed": self.results.get("frames_processed", 0),
                "processing_time": self.results.get("processing_time", 0),
            },
            "data": self.dataframe.to_dict(orient="records") if self.dataframe is not None else []
        }

        return json.dumps(export_data, indent=2)


def display_metrics(results: Dict[str, Any]):
    """Display quick metrics in a row."""
    if not results:
        st.info("No metrics available")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Vehicles", results.get("total_vehicles", 0))
    col2.metric("Avg Speed", f"{results.get('avg_speed', 0):.1f} km/h")
    col3.metric("Max Speed", f"{results.get('max_speed', 0):.1f} km/h")
    col4.metric("Frames Processed", results.get("frames_processed", 0))