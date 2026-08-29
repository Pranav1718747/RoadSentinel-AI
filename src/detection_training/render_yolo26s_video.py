import os
import sys
import time
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path("/Users/pranav1718/Big One/RoadSentimentalAI")
VIDEO_INPUT = BASE_DIR / "ml" / "data" / "samples" / "traffic.mp4"
OUTPUT_DIR = BASE_DIR / "outputs" / "detection_improvement"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def render_yolo26s_tracked_video():
    print("==========================================")
    print(" RENDERING YOLO26 SMALL (yolo26s.pt) TRACKED VIDEO")
    print("==========================================")
    print(f" Source Video : {VIDEO_INPUT}")

    if not VIDEO_INPUT.exists():
        print(f"Error: Video not found at {VIDEO_INPUT}")
        sys.exit(1)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f" Loading yolo26s.pt on device: {device}...")
    model = YOLO("yolo26s.pt")

    cap = cv2.VideoCapture(str(VIDEO_INPUT))
    v_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video_path = OUTPUT_DIR / "yolo26s_tracked_traffic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_video_path), fourcc, v_fps, (w, h))

    total_detections = 0
    track_ids_seen = set()
    frame_id = 0

    print(f"Processing {total_frames} frames through YOLO26 Small at imgsz=1280, conf=0.30...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.30,
            imgsz=1280,
            classes=[0, 1, 2, 3, 5, 7], # person, bicycle, car, motorcycle, bus, truck
            device=device,
            verbose=False
        )[0]

        annotated_frame = results.plot()
        writer.write(annotated_frame)

        if results.boxes is not None:
            total_detections += len(results.boxes)
            if results.boxes.id is not None:
                for t_id in results.boxes.id.int().cpu().tolist():
                    track_ids_seen.add(t_id)

        if frame_id % 100 == 0:
            avg_det = total_detections / frame_id
            print(f"  Frame {frame_id}/{total_frames}: {len(results.boxes)} detections, avg {avg_det:.1f}/fr, unique tracks: {len(track_ids_seen)}")

    cap.release()
    writer.release()

    print("\n--- RENDERING SUMMARY ---")
    print(f"Total Detections     : {total_detections} ({total_detections / max(1, total_frames):.2f}/frame)")
    print(f"Total Unique Tracks  : {len(track_ids_seen)}")
    print(f"[OK] YOLO26s video saved to: {out_video_path}")

if __name__ == "__main__":
    render_yolo26s_tracked_video()
