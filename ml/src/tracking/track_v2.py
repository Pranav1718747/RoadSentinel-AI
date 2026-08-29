import os
import sys
import csv
import cv2
from pathlib import Path
from ultralytics import YOLO

def main():
    # Define paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    input_video_path = base_dir / "data" / "samples" / "traffic.mp4"
    output_video_path = base_dir / "outputs" / "tracking" / "tracked_traffic_v2.mp4"
    output_csv_path = base_dir / "outputs" / "tracking" / "tracks_v2.csv"

    # Ensure output directories exist
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_video_path.exists():
        print(f"Error: Input video not found at {input_video_path}")
        sys.exit(1)

    from ml.src.pipeline.config import YOLO_MODEL_NAME, DETECTION_CONF, DETECTION_IMGSZ, VEHICLE_CLASSES
    print(f"Loading {YOLO_MODEL_NAME} model for tracking...")
    try:
        model = YOLO(YOLO_MODEL_NAME)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print(f"Opening video: {input_video_path}")
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        print(f"Error: Failed to open video at {input_video_path}.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0 or width == 0 or height == 0:
        print("Error: Invalid video properties detected.")
        sys.exit(1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not out_video.isOpened():
        print(f"Error: Failed to open output video writer for {output_video_path}")
        cap.release()
        sys.exit(1)

    try:
        csv_file = open(output_csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame_id", "timestamp", "track_id", "class_id", "class_name", 
            "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"
        ])
    except Exception as e:
        print(f"Error: Failed to open CSV file for writing at {output_csv_path}: {e}")
        cap.release()
        out_video.release()
        sys.exit(1)

    print(f"Starting tracking on {total_frames} frames...")

    # Statistics variables
    stats_total_frames = 0
    stats_total_detections = 0
    stats_unique_track_ids = set()
    stats_detections_by_class = {}
    stats_sum_conf_by_class = {}
    stats_high_conf_count = 0

    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            stats_total_frames += 1
            timestamp = frame_id / fps

            # Run YOLO26 Small detection + ByteTrack tracking
            results = model.track(
                source=frame, 
                persist=True, 
                tracker="bytetrack.yaml", 
                verbose=False,
                imgsz=DETECTION_IMGSZ,
                conf=0.30,
                classes=VEHICLE_CLASSES
            )
            
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.int().cpu().tolist()
                else:
                    track_ids = [None] * len(result.boxes)

                for i, box in enumerate(result.boxes):
                    track_id = track_ids[i]
                    if track_id is None:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0

                    # Write to CSV
                    csv_writer.writerow([
                        frame_id, 
                        f"{timestamp:.3f}", 
                        track_id,
                        cls_id, 
                        cls_name, 
                        f"{conf:.3f}", 
                        f"{x1:.2f}", 
                        f"{y1:.2f}", 
                        f"{x2:.2f}", 
                        f"{y2:.2f}", 
                        f"{center_x:.2f}", 
                        f"{center_y:.2f}"
                    ])

                    # Collect stats
                    stats_total_detections += 1
                    stats_unique_track_ids.add(track_id)
                    stats_detections_by_class[cls_name] = stats_detections_by_class.get(cls_name, 0) + 1
                    stats_sum_conf_by_class[cls_name] = stats_sum_conf_by_class.get(cls_name, 0.0) + conf
                    if conf >= 0.45:
                        stats_high_conf_count += 1

                    # Custom Annotation Drawing
                    # Format: <class_name> ID:<track_id> <confidence>
                    label_text = f"{cls_name} ID:{track_id} {conf:.2f}"
                    
                    # Box colors can be varied; let's use a simple green
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Font setup
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    
                    # Get text size to draw background rectangle
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                    
                    # Background rectangle for text
                    cv2.rectangle(frame, (int(x1), int(y1) - text_height - 10), (int(x1) + text_width, int(y1)), color, -1)
                    
                    # Text (white)
                    cv2.putText(frame, label_text, (int(x1), int(y1) - 5), font, font_scale, (255, 255, 255), thickness)

            out_video.write(frame)
            
            if frame_id % 30 == 0:
                print(f"Processed frame {frame_id}/{total_frames}")

    except Exception as e:
        print(f"Error during tracking processing: {e}")
    finally:
        print("Cleaning up resources...")
        cap.release()
        out_video.release()
        csv_file.close()

    print(f"\nDone. Tracking outputs saved to:\n- {output_video_path}\n- {output_csv_path}")

    # Print statistics
    print("\n--- Processing Statistics ---")
    print(f"Total frames processed: {stats_total_frames}")
    print(f"Total detections: {stats_total_detections}")
    print(f"Unique track IDs: {len(stats_unique_track_ids)}")
    
    if stats_total_detections > 0:
        high_conf_pct = (stats_high_conf_count / stats_total_detections) * 100
        print(f"Percentage of detections with confidence >= 0.45: {high_conf_pct:.2f}%")
        print("\nDetections by class:")
        for cls_name, count in stats_detections_by_class.items():
            avg_conf = stats_sum_conf_by_class[cls_name] / count
            print(f"  - {cls_name}: {count} detections (Avg Conf: {avg_conf:.3f})")
    else:
        print("No detections were made during this run.")
    print("-----------------------------\n")

if __name__ == "__main__":
    main()
