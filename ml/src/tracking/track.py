import os
import sys
import csv
import cv2
from pathlib import Path
from ultralytics import YOLO

def main():
    # Define paths relative to the script's working directory or standard project structure
    base_dir = Path(__file__).resolve().parent.parent.parent
    input_video_path = base_dir / "data" / "samples" / "traffic.mp4"
    output_video_path = base_dir / "outputs" / "tracking" / "tracked_traffic.mp4"
    output_csv_path = base_dir / "outputs" / "tracking" / "tracks.csv"

    # Ensure output directories exist
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Error Handling: Missing video
    if not input_video_path.exists():
        print(f"Error: Input video not found at {input_video_path}")
        sys.exit(1)

    # Load YOLOv8n model (reused from detection)
    print("Loading YOLOv8n model for tracking...")
    try:
        model = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # 2. Error Handling: Failure to open video
    print(f"Opening video: {input_video_path}")
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        print(f"Error: Failed to open video at {input_video_path}. It might be invalid or corrupted.")
        sys.exit(1)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0 or width == 0 or height == 0:
        print("Error: Invalid video properties detected.")
        sys.exit(1)

    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not out_video.isOpened():
        print(f"Error: Failed to open output video writer for {output_video_path}")
        cap.release()
        sys.exit(1)

    # Open CSV for writing
    try:
        csv_file = open(output_csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        # Write CSV header as requested
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

    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            timestamp = frame_id / fps

            # Run YOLOv8 detection + ByteTrack tracking
            # verbose=False to keep stdout clean, persist=True is required for tracking
            results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            
            # The results is a list of Results objects (one per frame)
            result = results[0]

            # If there are any detections in this frame
            if result.boxes is not None and len(result.boxes) > 0:
                # Check if tracking IDs are available
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.int().cpu().tolist()
                else:
                    track_ids = [None] * len(result.boxes)

                # Iterate over each detection box
                for i, box in enumerate(result.boxes):
                    track_id = track_ids[i]
                    # Skip if track_id is None (tracker hasn't confidently assigned one yet)
                    if track_id is None:
                        continue

                    # Bounding box coordinates (xyxy)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Confidence and class ID
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Class name
                    cls_name = model.names[cls_id]
                    
                    # Center coordinates
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

            # Draw bounding boxes, labels, and track IDs on the frame
            # result.plot() automatically includes track IDs if available
            annotated_frame = result.plot()
            
            # Write the annotated frame to output video
            out_video.write(annotated_frame)
            
            if frame_id % 30 == 0:
                print(f"Processed frame {frame_id}/{total_frames}")

    except Exception as e:
        print(f"Error during tracking processing: {e}")
    finally:
        print("Cleaning up resources...")
        cap.release()
        out_video.release()
        csv_file.close()

    print(f"Done. Tracking outputs saved to:\n- {output_video_path}\n- {output_csv_path}")

if __name__ == "__main__":
    main()
