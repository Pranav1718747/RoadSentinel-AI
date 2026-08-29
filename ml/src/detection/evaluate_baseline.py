"""
evaluate_baseline.py — RoadSentimentalAI, Phase 2
Evaluates the baseline detector (yolov8n.pt) on traffic.mp4, records baseline detection
statistics, confidence distributions, and saves annotated sample frames to ml/outputs/detection/baseline/.
"""

import os
import sys
import csv
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

BASE_DIR       = Path(__file__).resolve().parent.parent.parent
VIDEO_PATH     = BASE_DIR / "data" / "samples" / "traffic.mp4"
BASELINE_DIR   = BASE_DIR / "outputs" / "detection" / "baseline"
STATS_JSON     = BASELINE_DIR / "baseline_stats.json"

def evaluate_baseline():
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    print("Evaluating Baseline Detector (yolov8n.pt)...")

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 50.0

    detections = []
    sample_frames_to_save = [1, 50, 100, 200, 300, 400, 500]

    fid = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        fid += 1

        results = model.predict(source=frame, verbose=False)
        result  = results[0]

        frame_det_count = len(result.boxes)

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            cname  = model.names[cls_id]

            detections.append({
                "frame_id": fid,
                "class_id": cls_id,
                "class_name": cname,
                "confidence": conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "bbox_area": (x2 - x1) * (y2 - y1)
            })

        if fid in sample_frames_to_save:
            annotated = result.plot()
            sample_img_path = BASELINE_DIR / f"frame_{fid:04d}.jpg"
            cv2.imwrite(str(sample_img_path), annotated)

    cap.release()

    df_det = pd.DataFrame(detections)

    # Class filter focus for vehicles
    vehicle_classes = ["car", "truck", "bus", "motorcycle", "person", "bicycle"]
    df_veh = df_det[df_det["class_name"].isin(vehicle_classes)].copy()

    conf_summary = {
        "total_detections_all_classes": len(df_det),
        "total_vehicle_detections": len(df_veh),
        "mean_confidence": round(float(df_veh["confidence"].mean()), 4) if not df_veh.empty else 0.0,
        "median_confidence": round(float(df_veh["confidence"].median()), 4) if not df_veh.empty else 0.0,
        "min_confidence": round(float(df_veh["confidence"].min()), 4) if not df_veh.empty else 0.0,
        "max_confidence": round(float(df_veh["confidence"].max()), 4) if not df_veh.empty else 0.0,
        "detections_per_frame_mean": round(len(df_veh) / total_frames, 2),
        "class_counts": df_veh["class_name"].value_counts().to_dict(),
        "confidence_bins": {
            "0.25-0.50": int(((df_veh["confidence"] >= 0.25) & (df_veh["confidence"] < 0.50)).sum()),
            "0.50-0.75": int(((df_veh["confidence"] >= 0.50) & (df_veh["confidence"] < 0.75)).sum()),
            "0.75-1.00": int((df_veh["confidence"] >= 0.75).sum()),
        }
    }

    with open(STATS_JSON, "w") as fh:
        json.dump(conf_summary, fh, indent=2)

    print("\nBaseline Evaluation Complete:")
    print(json.dumps(conf_summary, indent=2))
    return conf_summary

if __name__ == "__main__":
    evaluate_baseline()
