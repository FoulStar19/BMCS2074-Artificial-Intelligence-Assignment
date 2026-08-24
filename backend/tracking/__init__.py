"""
Tracking module for vehicle tracking and speed estimation.
"""

from backend.tracking.tracker import VehicleTracker, TrackedVehicle
from backend.tracking.speed_estimator import SpeedEstimator, SpeedRecord

__all__ = [
    'VehicleTracker',
    'TrackedVehicle',
    'SpeedEstimator',
    'SpeedRecord',
]