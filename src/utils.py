"""
SmartPark AI - Utility Functions
================================
Core utilities for parking slot management, vehicle detection,
occupancy tracking, and video annotation.

Author: AI/ML Engineer
Date: 2026
"""

import json
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import supervision as sv
from ultralytics import YOLO


# =============================================================================
# Data Classes for Parking Management
# =============================================================================

@dataclass
class ParkingSlot:
    """Represents a single parking slot with its geometry and state."""
    slot_id: int
    polygon: np.ndarray  # Shape: (N, 2) array of points
    is_occupied: bool = False
    occupied_by: Optional[str] = None
    confidence: float = 0.0
    center_point: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    
    def __post_init__(self):
        """Calculate center point from polygon."""
        if len(self.polygon) > 0:
            moments = cv2.moments(self.polygon.astype(np.float32))
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                self.center_point = (cx, cy)


@dataclass
class DetectionConfig:
    """Configuration for vehicle detection."""
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.4
    iou_threshold: float = 0.45
    vehicle_classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])  # COCO: car, motorcycle, bus, truck
    
    # Class mapping: COCO class ID -> vehicle type
    class_names: Dict[int, str] = field(default_factory=lambda: {
        2: "Car",
        3: "Motorcycle", 
        5: "Bus",
        7: "Truck"
    })


@dataclass
class OccupancyStats:
    """Real-time occupancy statistics."""
    timestamp: str = ""
    total_slots: int = 0
    occupied_slots: int = 0
    available_slots: int = 0
    occupancy_rate: float = 0.0
    vehicle_distribution: Dict[str, int] = field(default_factory=dict)
    violations: int = 0


# =============================================================================
# Parking Slot Manager
# =============================================================================

class ParkingSlotManager:
    """
    Manages parking slot definitions, occupancy detection, and state tracking.
    """
    
    def __init__(self, slots_file: Optional[str] = None):
        self.slots: List[ParkingSlot] = []
        self.slots_file = slots_file
        self.history: List[OccupancyStats] = []
        
        if slots_file and Path(slots_file).exists():
            self.load_slots(slots_file)
    
    def load_slots(self, filepath: str) -> None:
        """Load parking slot definitions from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.slots = []
        for slot_data in data.get('slots', []):
            polygon = np.array(slot_data['polygon'], dtype=np.int32)
            slot = ParkingSlot(
                slot_id=slot_data['slot_id'],
                polygon=polygon
            )
            self.slots.append(slot)
        
        print(f"[SmartPark] Loaded {len(self.slots)} parking slots from {filepath}")
    
    def save_slots(self, filepath: str) -> None:
        """Save parking slot definitions to JSON file."""
        data = {
            'slots': [
                {
                    'slot_id': slot.slot_id,
                    'polygon': slot.polygon.tolist()
                }
                for slot in self.slots
            ]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[SmartPark] Saved {len(self.slots)} parking slots to {filepath}")
    
    def create_slots_from_grid(self, 
                               frame_shape: Tuple[int, int, int],
                               rows: int = 3, 
                               cols: int = 4,
                               margin_x: int = 50,
                               margin_y: int = 50,
                               slot_width: int = 150,
                               slot_height: int = 80) -> None:
        """
        Automatically generate parking slots in a grid pattern.
        Useful for quick setup without manual drawing.
        """
        self.slots = []
        height, width = frame_shape[:2]
        
        # Calculate starting position to center the grid
        total_width = cols * slot_width + (cols - 1) * margin_x
        total_height = rows * slot_height + (rows - 1) * margin_y
        
        start_x = (width - total_width) // 2
        start_y = (height - total_height) // 2
        
        slot_id = 1
        for row in range(rows):
            for col in range(cols):
                x1 = start_x + col * (slot_width + margin_x)
                y1 = start_y + row * (slot_height + margin_y)
                x2 = x1 + slot_width
                y2 = y1 + slot_height
                
                # Create rectangular polygon
                polygon = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ], dtype=np.int32)
                
                self.slots.append(ParkingSlot(
                    slot_id=slot_id,
                    polygon=polygon
                ))
                slot_id += 1
        
        print(f"[SmartPark] Created {len(self.slots)} parking slots in grid layout")
    
    def check_occupancy(self, 
                       detections: sv.Detections,
                       config: DetectionConfig,
                       overlap_threshold: float = 0.3) -> OccupancyStats:
        """
        Check which parking slots are occupied based on vehicle detections.
        Uses polygon intersection for accurate overlap calculation.
        """
        stats = OccupancyStats(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_slots=len(self.slots)
        )
        
        # Reset occupancy state
        for slot in self.slots:
            slot.is_occupied = False
            slot.occupied_by = None
            slot.confidence = 0.0
        
        vehicle_count = {name: 0 for name in config.class_names.values()}
        
        # Check each detection against each slot
        if len(detections) > 0:
            for i, (bbox, conf, class_id) in enumerate(zip(
                detections.xyxy, 
                detections.confidence, 
                detections.class_id
            )):
                if class_id not in config.vehicle_classes:
                    continue
                
                # Create detection polygon from bbox
                x1, y1, x2, y2 = bbox.astype(int)
                det_polygon = np.array([
                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                ], dtype=np.int32)
                
                # Check overlap with each slot
                for slot in self.slots:
                    overlap = self._calculate_polygon_overlap(
                        slot.polygon, det_polygon
                    )
                    
                    if overlap > overlap_threshold and conf > config.confidence_threshold:
                        if not slot.is_occupied or conf > slot.confidence:
                            slot.is_occupied = True
                            slot.occupied_by = config.class_names.get(class_id, "Unknown")
                            slot.confidence = conf
                            vehicle_count[slot.occupied_by] += 1
        
        # Calculate statistics
        stats.occupied_slots = sum(1 for slot in self.slots if slot.is_occupied)
        stats.available_slots = stats.total_slots - stats.occupied_slots
        stats.occupancy_rate = (stats.occupied_slots / stats.total_slots * 100) if stats.total_slots > 0 else 0
        stats.vehicle_distribution = vehicle_count
        
        self.history.append(stats)
        return stats
    
    def _calculate_polygon_overlap(self, 
                                 polygon1: np.ndarray, 
                                 polygon2: np.ndarray) -> float:
        """
        Calculate IoU-like overlap between two polygons.
        """
        # Convert to contour format
        poly1 = polygon1.reshape(-1, 1, 2)
        poly2 = polygon2.reshape(-1, 1, 2)
        
        # Create mask images
        x_max = max(polygon1[:, 0].max(), polygon2[:, 0].max()) + 1
        y_max = max(polygon1[:, 1].max(), polygon2[:, 1].max()) + 1
        
        mask1 = np.zeros((y_max, x_max), dtype=np.uint8)
        mask2 = np.zeros((y_max, x_max), dtype=np.uint8)
        
        cv2.fillPoly(mask1, [poly1], 1)
        cv2.fillPoly(mask2, [poly2], 1)
        
        # Calculate intersection and union
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        return intersection / union if union > 0 else 0.0
    
    def draw_slots(self, 
                   frame: np.ndarray,
                   show_ids: bool = True,
                   show_status: bool = True) -> np.ndarray:
        """Draw parking slots on the frame with occupancy status."""
        annotated = frame.copy()
        
        for slot in self.slots:
            # Determine color based on occupancy
            if slot.is_occupied:
                color = (0, 0, 255)  # Red for occupied (BGR)
                status_text = "OCCUPIED"
            else:
                color = (0, 255, 0)  # Green for vacant
                status_text = "VACANT"
            
            # Draw polygon
            cv2.polylines(annotated, [slot.polygon], True, color, 2)
            
            # Fill with semi-transparent color
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [slot.polygon], color)
            annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
            
            if show_ids:
                # Draw slot ID
                cx, cy = slot.center_point
                cv2.putText(annotated, f"P{slot.slot_id}", 
                          (cx - 15, cy - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if show_status and slot.is_occupied:
                # Draw occupancy info
                cx, cy = slot.center_point
                label = f"{slot.occupied_by}"
                cv2.putText(annotated, label, 
                          (cx - 25, cy + 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return annotated


# =============================================================================
# Vehicle Detection Manager
# =============================================================================

class VehicleDetector:
    """
    Manages YOLO-based vehicle detection with supervision for tracking.
    """
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.model = None
        self.tracker = sv.ByteTrack()  # Tracking for consistent IDs
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            color_lookup=sv.ColorLookup.CLASS
        )
        self.label_annotator = sv.LabelAnnotator(
            text_thickness=1,
            text_scale=0.5,
            color_lookup=sv.ColorLookup.CLASS
        )
        self.load_model()
    
    def load_model(self) -> None:
        """Load the YOLO model."""
        try:
            self.model = YOLO(self.config.model_name)
            print(f"[SmartPark] Loaded model: {self.config.model_name}")
        except Exception as e:
            print(f"[SmartPark] Error loading model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run vehicle detection on a frame.
        Returns supervision Detections object.
        """
        # Run inference
        results = self.model(
            frame, 
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            verbose=False
        )[0]
        
        # Convert to supervision format
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter for vehicle classes only
        vehicle_mask = np.isin(detections.class_id, self.config.vehicle_classes)
        detections = detections[vehicle_mask]
        
        # Update tracker
        detections = self.tracker.update_with_detections(detections)
        
        return detections
    
    def draw_detections(self, 
                       frame: np.ndarray, 
                       detections: sv.Detections) -> np.ndarray:
        """Draw detection bounding boxes and labels on frame."""
        # Prepare labels with tracking IDs
        labels = []
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
            class_name = self.config.class_names.get(class_id, "Unknown")
            if tracker_id is not None:
                labels.append(f"#{tracker_id} {class_name}")
            else:
                labels.append(class_name)
        
        # Annotate
        annotated = self.box_annotator.annotate(frame.copy(), detections)
        annotated = self.label_annotator.annotate(annotated, detections, labels)
        
        return annotated


# =============================================================================
# Video Processor
# =============================================================================

class VideoProcessor:
    """
    Handles video processing pipeline: detection, occupancy tracking, 
    annotation, and CSV logging.
    """
    
    def __init__(self, 
                 detector: VehicleDetector,
                 slot_manager: ParkingSlotManager,
                 output_dir: str = "outputs"):
        self.detector = detector
        self.slot_manager = slot_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.csv_data: List[Dict] = []
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, OccupancyStats]:
        """Process a single frame: detect, track occupancy, annotate."""
        # Detect vehicles
        detections = self.detector.detect(frame)
        
        # Check parking slot occupancy
        stats = self.slot_manager.check_occupancy(
            detections, 
            self.detector.config
        )
        
        # Draw detections
        annotated = self.detector.draw_detections(frame, detections)
        
        # Draw parking slots
        annotated = self.slot_manager.draw_slots(annotated)
        
        # Add statistics overlay
        annotated = self._add_stats_overlay(annotated, stats)
        
        # Log to CSV data
        self._log_stats(stats)
        
        return annotated, stats
    
    def _add_stats_overlay(self, 
                          frame: np.ndarray, 
                          stats: OccupancyStats) -> np.ndarray:
        """Add statistics overlay to the frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Create semi-transparent overlay area
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (350, 140), (0, 0, 0), -1)
        annotated = cv2.addWeighted(annotated, 1.0, overlay, 0.7, 0)
        
        # Add text statistics
        y_pos = 35
        line_height = 22
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        texts = [
            ("SMARTPARK AI - LIVE MONITORING", (0, 255, 255), 0.6, 2),
            (f"Occupancy: {stats.occupied_slots}/{stats.total_slots} ({stats.occupancy_rate:.1f}%)", 
             (255, 255, 255), 0.5, 1),
            (f"Available: {stats.available_slots} spots", (0, 255, 0), 0.5, 1),
            (f"Vehicles: Car={stats.vehicle_distribution.get('Car', 0)} | "
             f"Bike={stats.vehicle_distribution.get('Motorcycle', 0)} | "
             f"Bus={stats.vehicle_distribution.get('Bus', 0)} | "
             f"Truck={stats.vehicle_distribution.get('Truck', 0)}", 
             (255, 255, 255), 0.45, 1),
        ]
        
        for text, color, scale, thickness in texts:
            cv2.putText(annotated, text, (20, y_pos), font, scale, color, thickness)
            y_pos += line_height
        
        return annotated
    
    def _log_stats(self, stats: OccupancyStats) -> None:
        """Log statistics for CSV export."""
        row = {
            'timestamp': stats.timestamp,
            'total_slots': stats.total_slots,
            'occupied_slots': stats.occupied_slots,
            'available_slots': stats.available_slots,
            'occupancy_rate': round(stats.occupancy_rate, 2),
            'car_count': stats.vehicle_distribution.get('Car', 0),
            'motorcycle_count': stats.vehicle_distribution.get('Motorcycle', 0),
            'bus_count': stats.vehicle_distribution.get('Bus', 0),
            'truck_count': stats.vehicle_distribution.get('Truck', 0),
            'violations': stats.violations
        }
        self.csv_data.append(row)
    
    def export_csv(self, filename: Optional[str] = None) -> str:
        """Export collected statistics to CSV file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"parking_report_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        if self.csv_data:
            df = pd.DataFrame(self.csv_data)
            df.to_csv(filepath, index=False)
            print(f"[SmartPark] CSV report saved: {filepath}")
            return str(filepath)
        
        return ""
    
    def process_video(self, 
                     video_path: str,
                     output_filename: Optional[str] = None,
                     progress_callback = None) -> Tuple[str, str]:
        """
        Process entire video file.
        Returns paths to annotated video and CSV report.
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video writer
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"annotated_{timestamp}.mp4"
        
        output_path = self.output_dir / output_filename
        
        # Use MP4V codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Reset tracking state
        self.detector.tracker = sv.ByteTrack()
        self.csv_data = []
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame, stats = self.process_frame(frame)
                
                # Write to output
                writer.write(annotated_frame)
                
                frame_count += 1
                
                # Update progress
                if progress_callback and total_frames > 0:
                    progress = frame_count / total_frames
                    progress_callback(progress, frame_count, total_frames, stats)
        
        finally:
            cap.release()
            writer.release()
        
        # Export CSV
        csv_path = self.export_csv()
        
        print(f"[SmartPark] Video processing complete: {frame_count} frames")
        return str(output_path), csv_path


# =============================================================================
# Helper Functions for Interactive Slot Drawing
# =============================================================================

def create_slots_interactively(frame: np.ndarray, 
                                num_slots: int) -> List[ParkingSlot]:
    """
    Create parking slots interactively by clicking on the frame.
    Returns list of ParkingSlot objects.
    
    Note: This is a placeholder for GUI-based slot creation.
    In Streamlit, we'll use canvas-based drawing instead.
    """
    slots = []
    
    # For CLI/script usage
    print(f"[SmartPark] Interactive slot creation mode")
    print(f"Click 4 points per slot to define polygon (top-left, top-right, bottom-right, bottom-left)")
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param['points'].append([x, y])
            print(f"  Point added: ({x}, {y}) - {len(param['points'])}/4")
            
            if len(param['points']) == 4:
                param['done'] = True
    
    temp_frame = frame.copy()
    
    for i in range(num_slots):
        print(f"\nDefine Slot {i + 1}:")
        param = {'points': [], 'done': False}
        
        cv2.namedWindow(f"Define Slot {i + 1}")
        cv2.setMouseCallback(f"Define Slot {i + 1}", mouse_callback, param)
        
        while not param['done']:
            display_frame = temp_frame.copy()
            
            # Draw already placed points
            for pt in param['points']:
                cv2.circle(display_frame, tuple(pt), 5, (0, 255, 0), -1)
            
            # Draw polygon if we have points
            if len(param['points']) > 1:
                pts = np.array(param['points'], np.int32)
                cv2.polylines(display_frame, [pts], False, (0, 255, 255), 2)
            
            cv2.imshow(f"Define Slot {i + 1}", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return slots
        
        cv2.destroyWindow(f"Define Slot {i + 1}")
        
        # Create slot
        polygon = np.array(param['points'], dtype=np.int32)
        slot = ParkingSlot(slot_id=i + 1, polygon=polygon)
        slots.append(slot)
        
        # Draw on temp frame for visual feedback
        cv2.polylines(temp_frame, [polygon], True, (0, 255, 0), 2)
        cv2.putText(temp_frame, f"P{i+1}", tuple(polygon[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.destroyAllWindows()
    return slots


def get_available_models() -> List[Dict[str, str]]:
    """Get list of available YOLO models with descriptions."""
    return [
        {"name": "yolov8n.pt", "description": "YOLOv8 Nano - Fastest, lowest accuracy", "speed": "Fastest"},
        {"name": "yolov8s.pt", "description": "YOLOv8 Small - Balanced speed/accuracy", "speed": "Balanced"},
        {"name": "yolo11n.pt", "description": "YOLOv11 Nano - Latest architecture", "speed": "Latest"},
    ]


# =============================================================================
# License Plate Detection (Optional Enhancement)
# =============================================================================

class LicensePlateDetector:
    """
    Optional license plate detection using EasyOCR.
    Can be integrated for advanced parking management.
    """
    
    def __init__(self):
        self.reader = None
        self.enabled = False
        
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False)
            self.enabled = True
            print("[SmartPark] License plate detection enabled (EasyOCR)")
        except Exception as e:
            print(f"[SmartPark] EasyOCR not available: {e}")
    
    def detect_plate(self, 
                     frame: np.ndarray, 
                     bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """Extract license plate text from vehicle ROI."""
        if not self.enabled or self.reader is None:
            return None
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding
        pad = 5
        h, w = frame.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        try:
            # Run OCR
            results = self.reader.readtext(roi)
            
            # Filter for license plate pattern (alphanumeric, 5-10 chars)
            for (bbox, text, conf) in results:
                text = text.replace(" ", "").upper()
                if 5 <= len(text) <= 10 and any(c.isdigit() for c in text):
                    return text
            
            return None
        except Exception as e:
            return None


# =============================================================================
# Main Entry Point for Testing
# =============================================================================

if __name__ == "__main__":
    print("SmartPark AI - Utility Module")
    print("Run 'streamlit run app.py' to start the web application.")
