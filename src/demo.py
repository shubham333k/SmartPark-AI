"""
SmartPark AI - Standalone Demo Script
======================================
Quick demonstration of parking detection without Streamlit.
Useful for testing and development.

Usage:
    python demo.py --video path/to/video.mp4 --slots data/parking_slots.json
"""

import argparse
import cv2
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from utils import (
    ParkingSlotManager, VehicleDetector, VideoProcessor,
    DetectionConfig
)


def main():
    parser = argparse.ArgumentParser(description='SmartPark AI Demo')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--slots', type=str, default=None, help='Path to slots JSON file')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolo11n.pt'],
                       help='YOLO model to use')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', help='Save output video')
    parser.add_argument('--display', action='store_true', default=True, help='Display video')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚗 SmartPark AI - Standalone Demo")
    print("=" * 60)
    
    # Verify video file
    if not Path(args.video).exists():
        print(f"❌ Error: Video file not found: {args.video}")
        return
    
    # Initialize slot manager
    slot_manager = ParkingSlotManager()
    
    if args.slots and Path(args.slots).exists():
        print(f"📂 Loading parking slots from: {args.slots}")
        slot_manager.load_slots(args.slots)
    else:
        # Auto-generate slots
        print("📐 Auto-generating parking slot grid...")
        cap = cv2.VideoCapture(args.video)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            slot_manager.create_slots_from_grid(frame.shape, rows=3, cols=4)
            slot_manager.save_slots("data/auto_slots.json")
    
    print(f"✅ {len(slot_manager.slots)} parking slots configured")
    
    # Initialize detector
    config = DetectionConfig(
        model_name=args.model,
        confidence_threshold=args.conf
    )
    
    print(f"🤖 Loading model: {args.model}")
    detector = VehicleDetector(config)
    
    # Process video
    if args.save:
        print("🔄 Processing video...")
        processor = VideoProcessor(detector, slot_manager, output_dir="outputs")
        
        def progress_callback(progress, frame_num, total, stats):
            if frame_num % 30 == 0:
                print(f"  Frame {frame_num}/{total} | "
                      f"Occupancy: {stats.occupancy_rate:.1f}% | "
                      f"Vehicles: {stats.occupied_slots}")
        
        start_time = time.time()
        video_path, csv_path = processor.process_video(
            args.video, 
            progress_callback=progress_callback
        )
        elapsed = time.time() - start_time
        
        print(f"\n✅ Processing complete!")
        print(f"   Annotated video: {video_path}")
        print(f"   CSV report: {csv_path}")
        print(f"   Processing time: {elapsed:.1f}s")
    
    # Real-time display
    if args.display:
        print("\n🎥 Starting real-time display (Press 'q' to quit)...")
        
        cap = cv2.VideoCapture(args.video)
        processor = VideoProcessor(detector, slot_manager)
        
        frame_count = 0
        fps_start = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                annotated, stats = processor.process_frame(frame)
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.time()
                    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, annotated.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.imshow("SmartPark AI", annotated)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            if args.save:
                csv_path = processor.export_csv()
                print(f"📊 CSV report saved: {csv_path}")
    
    print("\n👋 Demo complete!")


if __name__ == "__main__":
    main()
