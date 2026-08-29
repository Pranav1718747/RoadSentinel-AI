import os
import sys
import json
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path("/Users/pranav1718/Big One/RoadSentimentalAI")
VIDEO_INPUT = BASE_DIR / "ml" / "data" / "samples" / "traffic.mp4"
FRAMES_DIR = BASE_DIR / "ml" / "outputs" / "detection_training" / "our_video_frames"
PRE_ANNOTATIONS_DIR = FRAMES_DIR / "pre_annotations"

FRAMES_DIR.mkdir(parents=True, exist_ok=True)
PRE_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

def extract_and_prepare_preannotations():
    print("==========================================")
    print(" PHASE 14 & 15: EXTRACT TRAFFIC FRAMES & PRE-ANNOTATIONS")
    print("==========================================")
    print(f" Source Video: {VIDEO_INPUT}")
    print(f" Target Frames Dir: {FRAMES_DIR}")

    if not VIDEO_INPUT.exists():
        print(f"Error: Video not found at {VIDEO_INPUT}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(VIDEO_INPUT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample ~30 diverse frames evenly distributed across the 500 frames of traffic.mp4
    # (e.g. frame 1, 15, 30, 45, 60, ... 490)
    frame_indices = sorted(list(set(range(0, total_frames, max(1, total_frames // 30)))))
    print(f"Extracting {len(frame_indices)} diverse frames from {total_frames} total frames...")

    extracted_info = []

    # Use pretrained yolo26n.pt to produce helper PRE-ANNOTATIONS (clearly marked as PRE-ANNOTATIONS)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    helper_model = YOLO("yolo26n.pt")

    coco_to_target = {2: 0, 7: 1, 5: 2} # 0: car, 1: truck, 2: bus

    current_idx = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count in frame_indices:
            frame_filename = f"traffic_frame_{frame_count:04d}.jpg"
            frame_path = FRAMES_DIR / frame_filename
            cv2.imwrite(str(frame_path), frame)

            # Generate helper pre-annotation
            res = helper_model.predict(source=frame, conf=0.35, imgsz=1280, classes=[2, 5, 7], device=device, verbose=False)[0]

            pre_annotation_lines = []
            pre_annotation_lines.append("# PRE-ANNOTATION ONLY - REQUIRES HUMAN VERIFICATION/CORRECTION")
            pre_annotation_lines.append("# Format: class_id x_center y_center width height")

            for box in res.boxes:
                c_id = int(box.cls[0])
                if c_id in coco_to_target:
                    tgt_id = coco_to_target[c_id]
                    xywhn = box.xywhn[0].tolist()
                    pre_annotation_lines.append(f"{tgt_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")

            lbl_filename = f"traffic_frame_{frame_count:04d}.txt"
            lbl_path = PRE_ANNOTATIONS_DIR / lbl_filename
            with open(lbl_path, "w") as f:
                f.write("\n".join(pre_annotation_lines) + "\n")

            extracted_info.append({
                "frame_id": frame_count,
                "image_path": str(frame_path),
                "pre_annotation_path": str(lbl_path),
                "pre_annotation_boxes_count": len(res.boxes),
                "status": "REQUIRES_HUMAN_VERIFICATION"
            })

            print(f"  Extracted frame {frame_count:04d} -> {len(res.boxes)} pre-annotations")

        frame_count += 1

    cap.release()

    manifest_path = FRAMES_DIR / "frame_extraction_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "source_video": str(VIDEO_INPUT),
            "total_extracted_frames": len(extracted_info),
            "frames": extracted_info,
            "instruction": "HUMAN-IN-THE-LOOP: Verify/correct bounding boxes in pre_annotations/ before round 2 training."
        }, f, indent=2)

    print(f"\n[OK] Extracted {len(extracted_info)} frames to {FRAMES_DIR}")
    print(f"[OK] Saved helper pre-annotations to {PRE_ANNOTATIONS_DIR}")
    print(f"[OK] Frame manifest saved to {manifest_path}")

if __name__ == "__main__":
    extract_and_prepare_preannotations()
