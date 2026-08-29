import os
import sys
import time
import json
import csv
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path("/Users/pranav1718/Big One/RoadSentimentalAI")
TEST_IMG_DIR = BASE_DIR / "datasets" / "aerial_vehicle" / "test" / "images"
TEST_LBL_DIR = BASE_DIR / "datasets" / "aerial_vehicle" / "test" / "labels"
OUTPUT_DIR = BASE_DIR / "ml" / "outputs" / "aerial_detection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_A_PATH = "yolo26n.pt"  # Pretrained COCO
MODEL_B_PATH = BASE_DIR / "ml" / "outputs" / "detection_training" / "models" / "best.pt"  # Kaggle Ground-Level
MODEL_C_PATH = BASE_DIR / "ml" / "outputs" / "detection_training" / "models" / "best_aerial.pt" # Aerial Fine-Tuned

COCO_MAP = {2: 0, 5: 1, 7: 2} # 0: car, 1: bus, 2: truck

def evaluate_model_on_test_set(model_path, is_coco_mapping=False, conf_thresh=0.25):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(model_path))

    test_images = sorted(list(TEST_IMG_DIR.glob("*.jpg")))
    
    tp = {0: 0, 1: 0, 2: 0}
    fp = {0: 0, 1: 0, 2: 0}
    fn = {0: 0, 1: 0, 2: 0}
    gt_totals = {0: 0, 1: 0, 2: 0}
    
    inference_times = []
    total_detections = 0

    for img_p in test_images:
        lbl_p = TEST_LBL_DIR / f"{img_p.stem}.txt"
        gt_boxes = []
        if lbl_p.exists():
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c, cx, cy, w, h = map(float, parts)
                        gt_boxes.append((int(c), cx, cy, w, h))
                        gt_totals[int(c)] += 1

        t0 = time.perf_counter()
        results = model.predict(source=str(img_p), conf=conf_thresh, imgsz=640, device=device, verbose=False)[0]
        t1 = time.perf_counter()
        inference_times.append(t1 - t0)

        det_boxes = []
        for box in results.boxes:
            raw_c = int(box.cls[0])
            conf = float(box.conf[0])
            xywhn = box.xywhn[0].tolist()

            if is_coco_mapping:
                if raw_c in COCO_MAP:
                    mapped_c = COCO_MAP[raw_c]
                    det_boxes.append((mapped_c, conf, xywhn[0], xywhn[1], xywhn[2], xywhn[3]))
            else:
                if raw_c in [0, 1, 2]:
                    det_boxes.append((raw_c, conf, xywhn[0], xywhn[1], xywhn[2], xywhn[3]))

        total_detections += len(det_boxes)

        matched_gt = set()
        for dcls, dconf, dcx, dcy, dw, dh in det_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, (gcls, gcx, gcy, gw, gh) in enumerate(gt_boxes):
                if gcls == dcls and g_idx not in matched_gt:
                    dx1, dy1, dx2, dy2 = dcx - dw/2, dcy - dh/2, dcx + dw/2, dcy + dh/2
                    gx1, gy1, gx2, gy2 = gcx - gw/2, gcy - gh/2, gcx + gw/2, gcy + gh/2

                    ix1, iy1 = max(dx1, gx1), max(dy1, gy1)
                    ix2, iy2 = min(dx2, gx2), min(dy2, gy2)
                    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                    inter = iw * ih
                    union = (dw * dh) + (gw * gh) - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

            if best_iou >= 0.50 and best_gt_idx != -1:
                tp[dcls] += 1
                matched_gt.add(best_gt_idx)
            else:
                fp[dcls] += 1

        for g_idx, (gcls, _, _, _, _) in enumerate(gt_boxes):
            if g_idx not in matched_gt:
                fn[gcls] += 1

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    car_rec = tp[0] / (tp[0] + fn[0]) if (tp[0] + fn[0]) > 0 else 0.0
    bus_rec = tp[1] / (tp[1] + fn[1]) if (tp[1] + fn[1]) > 0 else 0.0
    truck_rec = tp[2] / (tp[2] + fn[2]) if (tp[2] + fn[2]) > 0 else 0.0

    mean_inf = np.mean(inference_times) if inference_times else 0.0
    fps = 1.0 / mean_inf if mean_inf > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0,
        "car_recall": round(car_rec, 4),
        "bus_recall": round(bus_rec, 4),
        "truck_recall": round(truck_rec, 4),
        "avg_detections_per_frame": round(total_detections / len(test_images), 2),
        "fps": round(fps, 2)
    }

def benchmark_all_models():
    print("==========================================")
    print(" PHASE 8: 3-MODEL BENCHMARK COMPARISON ON UNSEEN AERIAL TEST SET")
    print("==========================================")

    res_a = evaluate_model_on_test_set(MODEL_A_PATH, is_coco_mapping=True, conf_thresh=0.25)
    res_b = evaluate_model_on_test_set(MODEL_B_PATH, is_coco_mapping=False, conf_thresh=0.25)
    res_c = evaluate_model_on_test_set(MODEL_C_PATH, is_coco_mapping=False, conf_thresh=0.25)

    comparison_data = [
        {"Model": "Model A (Pretrained COCO yolo26n)", **res_a},
        {"Model": "Model B (Kaggle Ground-Level Fine-Tuned)", **res_b},
        {"Model": "Model C (Aerial-Domain Fine-Tuned YOLO26)", **res_c}
    ]

    csv_path = OUTPUT_DIR / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comparison_data[0].keys())
        writer.writeheader()
        writer.writerows(comparison_data)

    json_path = OUTPUT_DIR / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(comparison_data, f, indent=2)

    print("\n--- 3-MODEL BENCHMARK TABLE ---")
    print(f"{'Model':<42} | {'Prec':<6} | {'Recall':<6} | {'Car Rec':<7} | {'Bus Rec':<7} | {'Trk Rec':<7} | {'FPS':<6}")
    print("-" * 90)
    for r in comparison_data:
        print(f"{r['Model']:<42} | {r['precision']:<6} | {r['recall']:<6} | {r['car_recall']:<7} | {r['bus_recall']:<7} | {r['truck_recall']:<7} | {r['fps']:<6}")

    print(f"\n[OK] Comparison CSV saved to: {csv_path}")
    print(f"[OK] Comparison JSON saved to: {json_path}")

if __name__ == "__main__":
    benchmark_all_models()
