"""
SmartPark AI - Package Initialization
=====================================
AI-Powered Parking Management & Occupancy Detection System

Modules:
    utils: Core utilities for detection, tracking, and annotation
    app: Streamlit web application
    demo: Standalone CLI demo

Version: 1.0.0
Author: AI/ML Engineer
"""

__version__ = "1.0.0"
__author__ = "AI/ML Engineer"
__description__ = "AI-Powered Parking Management & Occupancy Detection System"

from .utils import (
    ParkingSlot,
    DetectionConfig,
    OccupancyStats,
    ParkingSlotManager,
    VehicleDetector,
    VideoProcessor,
    LicensePlateDetector,
    get_available_models,
    create_slots_interactively
)

__all__ = [
    'ParkingSlot',
    'DetectionConfig',
    'OccupancyStats',
    'ParkingSlotManager',
    'VehicleDetector',
    'VideoProcessor',
    'LicensePlateDetector',
    'get_available_models',
    'create_slots_interactively'
]
