"""
Utility functions for traffic analysis
"""

import numpy as np
import cv2
from collections import deque
import json
import os


def calculate_traffic_density(detections, frame_shape, road_mask=None):
    """
    Calculate traffic density based on detections
    
    Args:
        detections: List of detection dictionaries
        frame_shape: Shape of the frame (height, width)
        road_mask: Optional binary mask for road area
        
    Returns:
        Traffic density as percentage
    """
    if not detections:
        return 0.0
    
    # Calculate area of road
    if road_mask is not None:
        road_area = np.sum(road_mask > 0)
    else:
        road_area = frame_shape[0] * frame_shape[1]
    
    # Calculate total area occupied by vehicles
    vehicle_area = 0
    for det in detections:
        bbox = det['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        vehicle_area += width * height
    
    # Calculate density
    density = (vehicle_area / road_area) * 100
    
    return min(density, 100.0)  # Cap at 100%


def estimate_speed_from_flow(flow, bbox, calibration_factor=0.05):
    """
    Estimate speed using optical flow within a bounding box
    
    Args:
        flow: Optical flow field
        bbox: Bounding box [x1, y1, x2, y2]
        calibration_factor: Conversion factor (meters per pixel)
        
    Returns:
        Estimated speed in km/h
    """
    x1, y1, x2, y2 = bbox
    
    # Extract flow within bounding box
    flow_bbox = flow[y1:y2, x1:x2]
    
    if flow_bbox.size == 0:
        return 0.0
    
    # Calculate average flow magnitude
    avg_flow = np.mean(flow_bbox, axis=(0, 1))
    speed_pixels = np.sqrt(avg_flow[0]**2 + avg_flow[1]**2)
    
    # Convert to km/h (assuming 30 fps)
    speed_mps = speed_pixels * calibration_factor * 30
    speed_kmh = speed_mps * 3.6
    
    return speed_kmh


def draw_vehicle_tracks(frame, tracks, color=(0, 255, 0), trail_length=20):
    """
    Draw vehicle tracking trails on frame
    
    Args:
        frame: Input frame
        tracks: Dictionary of track histories
        color: Line color
        trail_length: Number of points to draw
        
    Returns:
        Annotated frame
    """
    annotated_frame = frame.copy()
    
    for track_id, history in tracks.items():
        if len(history) < 2:
            continue
        
        # Get recent points
        points = list(history)[-trail_length:]
        
        # Draw trail
        for i in range(1, len(points)):
            pt1 = (int(points[i-1][0]), int(points[i-1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            alpha = i / len(points)
            
            # Fade color based on position in trail
            color_fade = (
                int(color[0] * alpha),
                int(color[1] * alpha),
                int(color[2] * alpha)
            )
            
            cv2.line(annotated_frame, pt1, pt2, color_fade, 2)
        
        # Draw current position
        current_pos = points[-1]
        cv2.circle(annotated_frame, 
                  (int(current_pos[0]), int(current_pos[1])), 
                  5, color, -1)
        
        # Draw track ID
        cv2.putText(annotated_frame, f"ID:{track_id}",
                  (int(current_pos[0]) - 20, int(current_pos[1]) - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated_frame


def calculate_vehicle_count(detections):
    """Calculate total vehicle count from detections"""
    return len(detections)


def calculate_average_speed(detections):
    """Calculate average speed from detections"""
    speeds = [det.get('speed', 0) for det in detections if det.get('speed', 0) > 0]
    if not speeds:
        return 0.0
    return np.mean(speeds)


def calculate_speed_distribution(detections):
    """Calculate speed distribution statistics"""
    speeds = [det.get('speed', 0) for det in detections if det.get('speed', 0) > 0]
    
    if not speeds:
        return {
            'min': 0,
            'max': 0,
            'mean': 0,
            'median': 0,
            'std': 0,
            'count': 0
        }
    
    return {
        'min': np.min(speeds),
        'max': np.max(speeds),
        'mean': np.mean(speeds),
        'median': np.median(speeds),
        'std': np.std(speeds),
        'count': len(speeds)
    }


def export_results(results, output_path, format='json'):
    """
    Export detection results to file
    
    Args:
        results: List of results dictionaries
        output_path: Output file path
        format: Export format ('json' or 'csv')
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == 'csv':
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Results exported to {output_path}")


def generate_traffic_report(detections_history, output_path=None):
    """
    Generate traffic analysis report
    
    Args:
        detections_history: History of detections over time
        output_path: Optional output path for report
        
    Returns:
        Report dictionary
    """
    report = {
        'total_frames_processed': len(detections_history),
        'total_vehicles_detected': sum(d['vehicle_count'] for d in detections_history),
        'average_vehicles_per_frame': np.mean([d['vehicle_count'] for d in detections_history]),
        'peak_vehicles': max([d['vehicle_count'] for d in detections_history]),
        'average_speed': np.mean([d.get('avg_speed', 0) for d in detections_history if d.get('avg_speed', 0) > 0]),
        'max_speed': max([d.get('max_speed', 0) for d in detections_history if d.get('max_speed', 0) > 0]),
        'average_density': np.mean([d.get('density', 0) for d in detections_history]),
        'peak_density': max([d.get('density', 0) for d in detections_history]),
    }
    
    # Export if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_path}")
    
    return report