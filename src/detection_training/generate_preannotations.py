import os
import sys
import json
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path("/Users/pranav1718/Big One/RoadSentimentalAI")
DATASET_DIR = BASE_DIR / "datasets" / "aerial_vehicle"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

LABELS_DIR.mkdir(parents=True, exist_ok=True)

# COCO to Aerial Target Taxonomy Mapping:
# 2: car -> 0
# 5: bus -> 1
# 7: truck -> 2
COCO_TO_AERIAL = {
    2: 0,  # car
    5: 1,  # bus
    7: 2   # truck
}

def generate_aerial_preannotations():
    print("==========================================")
    print(" PHASE 3: GENERATE AERIAL HELPER PRE-ANNOTATIONS")
    print("==========================================")
    print(f" Source Images: {IMAGES_DIR}")
    print(f" Target Labels: {LABELS_DIR}")

    images = sorted(list(IMAGES_DIR.glob("*.jpg")))
    print(f"Processing {len(images)} aerial frames using yolov8s.pt at imgsz=1280, conf=0.15...")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Use yolov8s.pt for high recall pre-annotation proposals
    model = YOLO("yolov8s.pt")

    stats = {
        "total_images_processed": len(images),
        "total_preannotations_generated": 0,
        "class_counts": {0: 0, 1: 0, 2: 0}, # 0: car, 1: bus, 2: truck
        "low_confidence_count": 0,
        "images_summary": []
    }

    for img_p in images:
        lbl_p = LABELS_DIR / f"{img_p.stem}.txt"

        # Predict with low confidence threshold to capture small overhead vehicles
        results = model.predict(source=str(img_p), conf=0.15, imgsz=1280, classes=[2, 5, 7], device=device, verbose=False)[0]

        lines = []
        img_annos_count = 0

        for box in results.boxes:
            coco_cls = int(box.cls[0])
            conf = float(box.conf[0])

            if coco_cls in COCO_TO_AERIAL:
                aerial_cls = COCO_TO_AERIAL[coco_cls]
                xywhn = box.xywhn[0].tolist()
                cx, cy, bw, bh = xywhn[0], xywhn[1], xywhn[2], xywhn[3]

                lines.append(f"{aerial_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                stats["class_counts"][aerial_cls] += 1
                stats["total_preannotations_generated"] += 1
                img_annos_count += 1

                if conf < 0.35:
                    stats["low_confidence_count"] += 1

        with open(lbl_p, "w") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

        stats["images_summary"].append({
            "image": img_p.name,
            "annotations_count": img_annos_count
        })

        if len(stats["images_summary"]) % 100 == 0:
            print(f"  Processed {len(stats['images_summary'])}/{len(images)} frames...")

    summary_path = DATASET_DIR / "preannotation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "taxonomy": {"0": "car", "1": "bus", "2": "truck"},
            "total_images": stats["total_images_processed"],
            "total_preannotations": stats["total_preannotations_generated"],
            "class_counts": {
                "car": stats["class_counts"][0],
                "bus": stats["class_counts"][1],
                "truck": stats["class_counts"][2]
            },
            "low_confidence_proposals_count": stats["low_confidence_count"],
            "avg_boxes_per_image": round(stats["total_preannotations_generated"] / max(1, len(images)), 2)
        }, f, indent=2)

    print("\n--- PRE-ANNOTATION SUMMARY ---")
    print(f"Total Images Processed        : {stats['total_images_processed']}")
    print(f"Total Pre-Annotations         : {stats['total_preannotations_generated']}")
    print(f"  - Car  (0)                  : {stats['class_counts'][0]}")
    print(f"  - Bus  (1)                  : {stats['class_counts'][1]}")
    print(f"  - Truck (2)                 : {stats['class_counts'][2]}")
    print(f"Low Confidence Proposals (<0.35): {stats['low_confidence_count']}")
    print(f"[OK] Pre-annotation labels written to: {LABELS_DIR}")
    print(f"[OK] Summary written to: {summary_path}")

if __name__ == "__main__":
    generate_aerial_preannotations()
