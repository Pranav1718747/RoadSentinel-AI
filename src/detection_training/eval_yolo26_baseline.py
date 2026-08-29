import os
import sys
import time
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path("/Users/pranav1718/Big One/RoadSentimentalAI")
TEST_IMG_DIR = BASE_DIR / "ml" / "outputs" / "detection_training" / "dataset" / "test" / "images"
TEST_LBL_DIR = BASE_DIR / "ml" / "outputs" / "detection_training" / "dataset" / "test" / "labels"
VIDEO_INPUT = BASE_DIR / "ml" / "data" / "samples" / "traffic.mp4"
BASELINE_OUTPUT_DIR = BASE_DIR / "ml" / "outputs" / "detection_training" / "baseline"
BASELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# COCO mapping for pretrained yolo26n.pt -> target (0: car, 1: truck, 2: bus)
COCO_TO_TARGET = {
    2: 0, # car
    7: 1, # truck
    5: 2  # bus
}

def evaluate_baseline_yolo26():
    print("==========================================")
    print(" PHASE 9: PRETRAINED YOLO26 BASELINE EVALUATION")
    print("==========================================")

    model = YOLO("yolo26n.pt")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Evaluate on test set images
    test_images = sorted(list(TEST_IMG_DIR.glob("*")))
    print(f"Evaluating {len(test_images)} test set images using pretrained yolo26n.pt on {device}...")

    inference_times = []
    tp = {0: 0, 1: 0, 2: 0}
    fp = {0: 0, 1: 0, 2: 0}
    fn = {0: 0, 1: 0, 2: 0}
    total_gt = {0: 0, 1: 0, 2: 0}

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
                        total_gt[int(c)] += 1

        t0 = time.perf_counter()
        results = model.predict(source=str(img_p), conf=0.25, imgsz=640, device=device, verbose=False)
        t1 = time.perf_counter()
        inference_times.append(t1 - t0)

        det_boxes = []
        res = results[0]
        for box in res.boxes:
            coco_cls = int(box.cls[0])
            conf = float(box.conf[0])
            if coco_cls in COCO_TO_TARGET:
                target_cls = COCO_TO_TARGET[coco_cls]
                # normalized xywh
                xywhn = box.xywhn[0].tolist()
                det_boxes.append((target_cls, conf, xywhn[0], xywhn[1], xywhn[2], xywhn[3]))

        # Greedy match IoU > 0.5 for AP50
        matched_gt = set()
        for dcls, dconf, dcx, dcy, dw, dh in det_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, (gcls, gcx, gcy, gw, gh) in enumerate(gt_boxes):
                if gcls == dcls and g_idx not in matched_gt:
                    # Compute IoU
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

    mean_inf_time = np.mean(inference_times) if inference_times else 0.0
    fps = 1.0 / mean_inf_time if mean_inf_time > 0 else 0.0

    per_class_metrics = {}
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())

    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    class_names = {0: "car", 1: "truck", 2: "bus"}
    for c_id, c_name in class_names.items():
        c_tp = tp[c_id]
        c_fp = fp[c_id]
        c_fn = fn[c_id]
        p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        r = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class_metrics[c_name] = {
            "tp": c_tp, "fp": c_fp, "fn": c_fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4)
        }

    baseline_metrics = {
        "model": "yolo26n.pt (pretrained COCO)",
        "test_images_evaluated": len(test_images),
        "mean_inference_time_ms": round(mean_inf_time * 1000, 2),
        "fps": round(fps, 2),
        "overall_precision": round(overall_prec, 4),
        "overall_recall": round(overall_rec, 4),
        "per_class_metrics": per_class_metrics
    }

    metrics_path = BASELINE_OUTPUT_DIR / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    print("\n--- PRETRAINED BASELINE METRICS ---")
    print(f"FPS                  : {baseline_metrics['fps']}")
    print(f"Mean Latency (ms)    : {baseline_metrics['mean_inference_time_ms']} ms")
    print(f"Overall Precision    : {baseline_metrics['overall_precision']}")
    print(f"Overall Recall       : {baseline_metrics['overall_recall']}")
    print("Per Class:")
    for c_name, c_m in per_class_metrics.items():
        print(f"  {c_name:8s} -> Prec: {c_m['precision']:.4f}, Rec: {c_m['recall']:.4f}, F1: {c_m['f1_score']:.4f}")

    # Generate baseline_detection_video.mp4 on traffic.mp4
    print(f"\nProcessing {VIDEO_INPUT} to generate baseline_detection_video.mp4...")
    cap = cv2.VideoCapture(str(VIDEO_INPUT))
    if not cap.isOpened():
        print(f"Error opening video: {VIDEO_INPUT}")
        return

    v_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video_path = BASELINE_OUTPUT_DIR / "baseline_detection_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_video_path), fourcc, v_fps, (w, h))

    frame_id = 0
    t_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # Run inference at 1280 resolution with conf=0.45 and vehicle classes
        results = model.predict(source=frame, conf=0.45, imgsz=1280, classes=[2, 5, 7], device=device, verbose=False)
        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

        if frame_id % 100 == 0:
            print(f"  Processed frame {frame_id}/{total_frames}")

    cap.release()
    writer.release()
    t_end = time.perf_counter()

    print(f"[OK] Baseline video saved to: {out_video_path}")
    print(f"[OK] Baseline metrics saved to: {metrics_path}")

if __name__ == "__main__":
    evaluate_baseline_yolo26()
