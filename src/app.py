"""
SmartPark AI - Streamlit Web Application
========================================
Professional parking management dashboard with real-time
occupancy detection, video processing, and analytics.

Features:
- Real-time video processing with YOLO detection
- Interactive parking slot configuration
- Live occupancy metrics and charts
- Export annotated videos and CSV reports

Author: AI/ML Engineer
Date: 2026
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import time
from pathlib import Path
from datetime import datetime
import json

# Set page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="SmartPark AI - Parking Management System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom utilities
import sys
sys.path.append(str(Path(__file__).parent))
from utils import (
    ParkingSlotManager, VehicleDetector, VideoProcessor,
    DetectionConfig, get_available_models, LicensePlateDetector
)

# =============================================================================
# Custom CSS for Professional Styling
# =============================================================================

CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main > div {
        padding: 2rem 3rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    
    /* Metric cards */
    .metric-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
    }
    
    /* Status indicators */
    .status-occupied {
        color: #e74c3c;
        font-weight: bold;
    }
    
    .status-vacant {
        color: #27ae60;
        font-weight: bold;
    }
    
    /* Info boxes */
    .info-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 3px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress bar */
    .progress-container {
        background: #f0f0f0;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    /* Slot configuration preview */
    .slot-preview {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        background: #f8f9ff;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# Session State Management
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'slot_manager': None,
        'detector': None,
        'video_processor': None,
        'processed_video_path': None,
        'csv_report_path': None,
        'processing_stats': [],
        'is_processing': False,
        'slots_configured': False,
        'uploaded_video_path': None,
        'live_stats': None,
        'occupancy_history': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =============================================================================
# Sidebar Configuration Panel
# =============================================================================

def render_sidebar():
    """Render the configuration sidebar."""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration Panel")
        st.markdown("---")
        
        # Model Selection
        st.markdown("#### 🤖 YOLO Model")
        models = get_available_models()
        model_options = {f"{m['name']} - {m['speed']}": m['name'] for m in models}
        selected_model = st.selectbox(
            "Select Detection Model:",
            list(model_options.keys()),
            index=0,
            help="Choose a YOLO model. Nano models are faster but less accurate."
        )
        model_name = model_options[selected_model]
        
        # Confidence Threshold
        confidence = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.4,
            step=0.05,
            help="Minimum confidence score for vehicle detection"
        )
        
        # IoU Threshold
        iou_threshold = st.slider(
            "📐 IoU Threshold",
            min_value=0.1,
            max_value=0.8,
            value=0.45,
            step=0.05,
            help="Intersection over Union threshold for NMS"
        )
        
        st.markdown("---")
        
        # Parking Slot Configuration
        st.markdown("#### 🅿️ Parking Slots")
        
        slot_config_mode = st.radio(
            "Configuration Mode:",
            ["Auto Grid", "Manual JSON"],
            help="Auto Grid automatically generates slots. Manual JSON uses custom configuration."
        )
        
        num_slots = None
        if slot_config_mode == "Auto Grid":
            col1, col2 = st.columns(2)
            with col1:
                rows = st.number_input("Rows", min_value=1, max_value=10, value=3)
            with col2:
                cols = st.number_input("Columns", min_value=1, max_value=10, value=4)
            num_slots = rows * cols
        
        st.markdown("---")
        
        # Advanced Options
        with st.expander("🔧 Advanced Options"):
            enable_ocr = st.checkbox(
                "Enable License Plate Detection",
                value=False,
                help="Uses EasyOCR for license plate recognition (slower)"
            )
            
            overlap_threshold = st.slider(
                "Slot Overlap Threshold",
                min_value=0.1,
                max_value=0.8,
                value=0.3,
                step=0.05,
                help="Minimum vehicle-to-slot overlap to count as occupied"
            )
            
            show_slot_ids = st.checkbox("Show Slot IDs", value=True)
            show_vehicle_labels = st.checkbox("Show Vehicle Labels", value=True)
        
        st.markdown("---")
        
        # About Section
        with st.expander("ℹ️ About SmartPark AI"):
            st.markdown("""
            **SmartPark AI** v1.0
            
            AI-powered parking management system using:
            - YOLOv8/v11 for vehicle detection
            - Polygon-based slot occupancy tracking
            - Real-time analytics & reporting
            
            Built with Streamlit, OpenCV, and Ultralytics.
            """)
        
        return {
            'model_name': model_name,
            'confidence': confidence,
            'iou_threshold': iou_threshold,
            'slot_config_mode': slot_config_mode,
            'rows': rows if slot_config_mode == "Auto Grid" else None,
            'cols': cols if slot_config_mode == "Auto Grid" else None,
            'num_slots': num_slots,
            'enable_ocr': enable_ocr if 'enable_ocr' in locals() else False,
            'overlap_threshold': overlap_threshold if 'overlap_threshold' in locals() else 0.3,
            'show_slot_ids': show_slot_ids if 'show_slot_ids' in locals() else True,
            'show_vehicle_labels': show_vehicle_labels if 'show_vehicle_labels' in locals() else True
        }

# =============================================================================
# Main Application Header
# =============================================================================

def render_header():
    """Render the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🚗 SmartPark AI</h1>
        <p>AI-Powered Parking Management & Occupancy Detection System</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# Video Upload Section
# =============================================================================

def render_video_upload():
    """Render video upload interface."""
    st.markdown("### 📹 Video Input")
    
    tab1, tab2 = st.tabs(["📁 Upload Video", "📷 Webcam (Live)"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload parking lot video (MP4, AVI, MOV)",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file showing parking lot for analysis"
        )
        
        if uploaded_file is not None:
            # Save to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            st.session_state.uploaded_video_path = tfile.name
            
            # Show video info
            cap = cv2.VideoCapture(tfile.name)
            if cap.isOpened():
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frames / fps if fps > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Resolution", f"{width}x{height}")
                col2.metric("FPS", fps)
                col3.metric("Duration", f"{duration:.1f}s")
                col4.metric("Frames", frames)
            cap.release()
            
            st.success(f"✅ Video loaded: {uploaded_file.name}")
            return True
    
    with tab2:
        st.info("🎥 Webcam integration coming in next update!")
        st.markdown("""
        For now, please upload a video file. Live webcam support with 
        real-time processing will be available in version 1.1.
        """)
    
    return False

# =============================================================================
# Parking Slot Configuration
# =============================================================================

def configure_slots(config: dict, video_path: str) -> bool:
    """Configure parking slots based on user settings."""
    if st.session_state.slots_configured:
        return True
    
    st.markdown("### 🅿️ Parking Slot Configuration")
    
    # Get first frame for preview
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        st.error("❌ Could not read video frame")
        return False
    
    # Convert BGR to RGB for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if config['slot_config_mode'] == "Auto Grid":
        st.markdown("#### Auto-Generated Grid Layout")
        
        # Preview slot layout
        slot_manager = ParkingSlotManager()
        slot_manager.create_slots_from_grid(
            frame.shape,
            rows=config['rows'],
            cols=config['cols']
        )
        
        # Draw preview
        preview_frame = slot_manager.draw_slots(frame)
        preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
        
        st.image(preview_rgb, use_container_width=True, 
                caption=f"Preview: {len(slot_manager.slots)} parking slots")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ Accept Layout", type="primary", use_container_width=True):
                st.session_state.slot_manager = slot_manager
                st.session_state.slots_configured = True
                # Save slots for reference
                slot_manager.save_slots("data/parking_slots.json")
                st.success(f"✅ {len(slot_manager.slots)} slots configured!")
                return True
        
        with col2:
            if st.button("🔄 Regenerate", use_container_width=True):
                st.rerun()
    
    else:
        # Manual JSON upload
        st.markdown("#### Upload Slot Configuration JSON")
        
        # Show example format
        example_json = {
            "slots": [
                {
                    "slot_id": 1,
                    "polygon": [[100, 100], [200, 100], [200, 150], [100, 150]]
                }
            ]
        }
        
        with st.expander("📄 View JSON Format"):
            st.json(example_json)
        
        json_file = st.file_uploader("Upload slots JSON", type=['json'])
        
        if json_file:
            json_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
            with open(json_path, 'wb') as f:
                f.write(json_file.read())
            
            try:
                slot_manager = ParkingSlotManager(json_path)
                preview_frame = slot_manager.draw_slots(frame)
                preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                
                st.image(preview_rgb, use_container_width=True)
                
                if st.button("✅ Confirm Slots", type="primary"):
                    st.session_state.slot_manager = slot_manager
                    st.session_state.slots_configured = True
                    return True
            except Exception as e:
                st.error(f"❌ Error loading slots: {e}")
    
    return False

# =============================================================================
# Video Processing Section
# =============================================================================

def process_video(config: dict):
    """Main video processing pipeline."""
    video_path = st.session_state.uploaded_video_path
    
    if not video_path or not st.session_state.slots_configured:
        return
    
    st.markdown("### 🔄 Processing Video")
    
    # Initialize detector
    detection_config = DetectionConfig(
        model_name=config['model_name'],
        confidence_threshold=config['confidence'],
        iou_threshold=config['iou_threshold']
    )
    
    with st.spinner(f"🚀 Loading {config['model_name']} model..."):
        detector = VehicleDetector(detection_config)
        st.session_state.detector = detector
    
    # Initialize video processor
    processor = VideoProcessor(
        detector=detector,
        slot_manager=st.session_state.slot_manager,
        output_dir="outputs"
    )
    st.session_state.video_processor = processor
    
    # Processing UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Live metrics containers
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        occupied_metric = st.empty()
    with metrics_col2:
        available_metric = st.empty()
    with metrics_col3:
        occupancy_rate_metric = st.empty()
    with metrics_col4:
        frame_progress_metric = st.empty()
    
    # Progress callback
    def update_progress(progress, frame_num, total_frames, stats):
        progress_bar.progress(min(progress, 1.0))
        status_text.text(f"Processing frame {frame_num}/{total_frames}")
        
        occupied_metric.metric("🚗 Occupied", stats.occupied_slots)
        available_metric.metric("✅ Available", stats.available_slots)
        occupancy_rate_metric.metric("📊 Occupancy", f"{stats.occupancy_rate:.1f}%")
        frame_progress_metric.metric("🎞️ Frame", f"{frame_num}/{total_frames}")
        
        # Store for charts
        if frame_num % 10 == 0:  # Sample every 10 frames
            st.session_state.occupancy_history.append({
                'frame': frame_num,
                'occupancy_rate': stats.occupancy_rate,
                'occupied': stats.occupied_slots,
                'available': stats.available_slots
            })
    
    # Process video
    try:
        with st.spinner("🔍 Analyzing video..."):
            output_video, output_csv = processor.process_video(
                video_path,
                progress_callback=update_progress
            )
            
            st.session_state.processed_video_path = output_video
            st.session_state.csv_report_path = output_csv
            
            status_text.success("✅ Processing complete!")
            progress_bar.empty()
    
    except Exception as e:
        st.error(f"❌ Processing error: {e}")
        raise

# =============================================================================
# Results Dashboard
# =============================================================================

def render_results():
    """Render the results dashboard with analytics."""
    if not st.session_state.processed_video_path:
        return
    
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Summary metrics
    processor = st.session_state.video_processor
    
    if processor and processor.csv_data:
        df = pd.DataFrame(processor.csv_data)
        
        st.markdown("### 📈 Occupancy Summary")
        
        # Calculate aggregate statistics
        avg_occupancy = df['occupancy_rate'].mean()
        max_occupancy = df['occupancy_rate'].max()
        min_occupancy = df['occupancy_rate'].min()
        total_cars = df['car_count'].sum()
        total_bikes = df['motorcycle_count'].sum()
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Avg Occupancy", f"{avg_occupancy:.1f}%")
        with col2:
            st.metric("Peak Occupancy", f"{max_occupancy:.1f}%")
        with col3:
            st.metric("Min Occupancy", f"{min_occupancy:.1f}%")
        with col4:
            st.metric("Total Cars", int(total_cars))
        with col5:
            st.metric("Total Bikes", int(total_bikes))
        
        # Charts
        st.markdown("### 📉 Occupancy Trends")
        
        tab1, tab2, tab3 = st.tabs(["Occupancy Over Time", "Vehicle Distribution", "Peak Hours Analysis"])
        
        with tab1:
            fig = px.line(df, y='occupancy_rate', 
                         title='Parking Occupancy Rate Over Time',
                         labels={'occupancy_rate': 'Occupancy %', 'index': 'Time'})
            fig.add_hline(y=80, line_dash="dash", line_color="red", 
                         annotation_text="High Occupancy Threshold")
            fig.add_hline(y=20, line_dash="dash", line_color="green",
                         annotation_text="Low Occupancy Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Vehicle type distribution
            vehicle_totals = {
                'Cars': df['car_count'].sum(),
                'Motorcycles': df['motorcycle_count'].sum(),
                'Buses': df['bus_count'].sum(),
                'Trucks': df['truck_count'].sum()
            }
            
            fig2 = px.pie(values=list(vehicle_totals.values()),
                         names=list(vehicle_totals.keys()),
                         title='Vehicle Type Distribution')
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            # Occupancy heatmap
            df['frame_group'] = (df.index // (len(df) // 10)).astype(int)
            occupancy_by_group = df.groupby('frame_group')['occupancy_rate'].mean().reset_index()
            
            fig3 = px.bar(occupancy_by_group, x='frame_group', y='occupancy_rate',
                         title='Average Occupancy by Time Segment',
                         labels={'frame_group': 'Time Segment', 'occupancy_rate': 'Avg Occupancy %'})
            st.plotly_chart(fig3, use_container_width=True)
        
        # Data table
        with st.expander("📋 View Full Data Table"):
            st.dataframe(df, use_container_width=True)
    
    # Video preview and downloads
    st.markdown("### 🎬 Annotated Video Output")
    
    if Path(st.session_state.processed_video_path).exists():
        with open(st.session_state.processed_video_path, 'rb') as f:
            video_bytes = f.read()
        
        st.video(video_bytes)
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="⬇️ Download Annotated Video",
                data=video_bytes,
                file_name=f"smartpark_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                mime="video/mp4",
                use_container_width=True
            )
        
        with col2:
            if st.session_state.csv_report_path and Path(st.session_state.csv_report_path).exists():
                with open(st.session_state.csv_report_path, 'rb') as f:
                    csv_bytes = f.read()
                
                st.download_button(
                    label="⬇️ Download CSV Report",
                    data=csv_bytes,
                    file_name=f"smartpark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# =============================================================================
# Main Application Flow
# =============================================================================

def main():
    """Main application entry point."""
    render_header()
    
    # Get configuration from sidebar
    config = render_sidebar()
    
    # Video upload section
    video_loaded = render_video_upload()
    
    if video_loaded and st.session_state.uploaded_video_path:
        # Slot configuration
        slots_ready = configure_slots(config, st.session_state.uploaded_video_path)
        
        # Processing button
        if slots_ready:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                    process_video(config)
    
    # Show results if available
    render_results()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🚗 SmartPark AI v1.0 | Built with ❤️ using Streamlit, YOLO & OpenCV</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
