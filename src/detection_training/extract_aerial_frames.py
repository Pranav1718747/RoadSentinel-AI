import os
import sys
import json
import cv2
from pathlib import Path

BASE_DIR = Path("/Users/pranav1718/Big One/RoadSentimentalAI")
VIDEO_PATH = BASE_DIR / "ml" / "data" / "samples" / "traffic.mp4"
DATASET_DIR = BASE_DIR / "datasets" / "aerial_vehicle"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

def extract_aerial_frames():
    print("==========================================")
    print(" PHASE 2: EXTRACT AERIAL FRAMES FROM traffic.mp4")
    print("==========================================")
    print(f" Source Video: {VIDEO_PATH}")

    if not VIDEO_PATH.exists():
        print(f"Error: Video not found at {VIDEO_PATH}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    v_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / v_fps if v_fps > 0 else 0.0

    print(f" Video FPS      : {v_fps}")
    print(f" Resolution     : {w}x{h}")
    print(f" Total Frames   : {total_frames}")
    print(f" Duration (sec) : {duration_sec:.2f}s\n")

    manifest = []
    frame_count = 0
    extracted_count = 0

    # Extract all 500 representative frames from traffic.mp4 (step=1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / v_fps
        img_name = f"frame_{frame_count:04d}.jpg"
        img_path = IMAGES_DIR / img_name

        cv2.imwrite(str(img_path), frame)
        extracted_count += 1

        manifest.append({
            "frame_number": frame_count,
            "timestamp": round(timestamp, 3),
            "image_name": img_name,
            "image_path": str(img_path)
        })

        if frame_count % 100 == 0:
            print(f"  Extracted frame {frame_count:04d}/{total_frames} -> {img_name}")

        frame_count += 1

    cap.release()

    manifest_path = DATASET_DIR / "frame_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "source_video": str(VIDEO_PATH),
            "fps": v_fps,
            "resolution": f"{w}x{h}",
            "total_frames_extracted": extracted_count,
            "manifest": manifest
        }, f, indent=2)

    print(f"\n[OK] Extracted {extracted_count} frames to: {IMAGES_DIR}")
    print(f"[OK] Saved frame manifest to: {manifest_path}")

if __name__ == "__main__":
    extract_aerial_frames()
