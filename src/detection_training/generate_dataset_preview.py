import os
import sys
import json
import random
from pathlib import Path
from collections import Counter
import cv2
import numpy as np

DATASET_DIR = Path("/Users/pranav1718/Big One/RoadSentimentalAI/ml/outputs/detection_training/dataset")
PREVIEW_DIR = Path("/Users/pranav1718/Big One/RoadSentimentalAI/ml/outputs/detection_training/dataset_preview")
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "car", 1: "truck", 2: "bus"}
CLASS_COLORS = {
    0: (0, 255, 0),     # Green for Car
    1: (255, 128, 0),   # Orange/Blue for Truck
    2: (0, 0, 255)      # Red for Bus
}

def analyze_and_generate_previews():
    print("==========================================")
    print(" PHASE 6: DATASET ANALYSIS & PREVIEW")
    print("==========================================")

    splits = ["train", "val", "test"]
    stats = {
        "total_images": 0,
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "total_car_annotations": 0,
        "total_truck_annotations": 0,
        "total_bus_annotations": 0,
        "annotations_per_image": 0.0,
        "average_bbox_area_norm": 0.0,
        "small_object_percentage": 0.0,
        "class_imbalance": {}
    }

    all_bbox_areas = []
    small_objects = 0
    total_annos = 0
    per_split_counts = {}

    image_label_pairs = []

    for split in splits:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"

        images = sorted(list(img_dir.glob("*")))
        stats[f"{split}_images"] = len(images)
        stats["total_images"] += len(images)

        split_annos = 0
        for img_p in images:
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            annos = []
            if lbl_p.exists():
                with open(lbl_p, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                for line in lines:
                    parts = line.split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        annos.append((cls_id, cx, cy, bw, bh))
                        
                        area = bw * bh
                        all_bbox_areas.append(area)
                        total_annos += 1
                        split_annos += 1

                        if cls_id == 0: stats["total_car_annotations"] += 1
                        elif cls_id == 1: stats["total_truck_annotations"] += 1
                        elif cls_id == 2: stats["total_bus_annotations"] += 1

                        # Small object threshold: normalized area < (32/640)^2 = 0.0025
                        if area < 0.0025:
                            small_objects += 1

            if len(annos) > 0:
                image_label_pairs.append((img_p, annos, split))

        per_split_counts[split] = split_annos

    if stats["total_images"] > 0:
        stats["annotations_per_image"] = round(total_annos / stats["total_images"], 3)

    if total_annos > 0:
        stats["average_bbox_area_norm"] = round(float(np.mean(all_bbox_areas)), 6)
        stats["small_object_percentage"] = round((small_objects / total_annos) * 100.0, 2)

    # Calculate class imbalance relative to car count
    car_cnt = max(1, stats["total_car_annotations"])
    stats["class_imbalance"] = {
        "car": "1.00x",
        "truck": f"{stats['total_truck_annotations'] / car_cnt:.3f}x",
        "bus": f"{stats['total_bus_annotations'] / car_cnt:.3f}x"
    }

    print("\n--- DATASET STATISTICAL SUMMARY ---")
    print(f"Total Images         : {stats['total_images']}")
    print(f"  - Train Images     : {stats['train_images']}")
    print(f"  - Val Images       : {stats['val_images']}")
    print(f"  - Test Images      : {stats['test_images']}")
    print(f"Total Annotations    : {total_annos}")
    print(f"  - Car Annotations  : {stats['total_car_annotations']}")
    print(f"  - Truck Annotations: {stats['total_truck_annotations']}")
    print(f"  - Bus Annotations  : {stats['total_bus_annotations']}")
    print(f"Annotations per Image: {stats['annotations_per_image']}")
    print(f"Average BBox Area    : {stats['average_bbox_area_norm']} (normalized)")
    print(f"Small Object %       : {stats['small_object_percentage']}% (area < 0.0025)")
    print(f"Class Imbalance      : {stats['class_imbalance']}")

    # Save summary JSON
    summary_path = Path("/Users/pranav1718/Big One/RoadSentimentalAI/ml/outputs/detection_training/dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")

    # Generate 20 representative preview images
    random.seed(42)
    # Ensure sample includes truck, bus, and car images
    truck_pairs = [p for p in image_label_pairs if any(a[0] == 1 for a in p[1])]
    bus_pairs = [p for p in image_label_pairs if any(a[0] == 2 for a in p[1])]
    car_pairs = [p for p in image_label_pairs if any(a[0] == 0 for a in p[1])]

    selected = []
    selected.extend(random.sample(truck_pairs, min(7, len(truck_pairs))))
    selected.extend(random.sample(bus_pairs, min(7, len(bus_pairs))))
    remaining_needed = 25 - len(selected)
    selected.extend(random.sample(car_pairs, min(remaining_needed, len(car_pairs))))
    random.shuffle(selected)

    print(f"\nGenerating {len(selected)} visual preview samples in '{PREVIEW_DIR}'...")

    for idx, (img_p, annos, split) in enumerate(selected[:25]):
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        for cls_id, cx, cy, bw, bh in annos:
            x1 = int((cx - bw / 2.0) * w)
            y1 = int((cy - bh / 2.0) * h)
            x2 = int((cx + bw / 2.0) * w)
            y2 = int((cy + bh / 2.0) * h)

            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            cls_name = CLASS_NAMES.get(cls_id, str(cls_id))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, cls_name, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_name = f"preview_{idx+1:02d}_{split}_{img_p.stem}.jpg"
        cv2.imwrite(str(PREVIEW_DIR / out_name), img)

    print(f"[OK] Generated {len(selected)} visual preview samples in {PREVIEW_DIR}")

if __name__ == "__main__":
    analyze_and_generate_previews()
