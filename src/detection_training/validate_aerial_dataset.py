import os
import sys
import json
import random
from pathlib import Path
from collections import Counter
import cv2
import numpy as np

BASE_DIR = Path("/Users/pranav1718/Big One/RoadSentimentalAI")
DATASET_DIR = BASE_DIR / "datasets" / "aerial_vehicle"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
OUTPUT_DIR = BASE_DIR / "ml" / "outputs"
VIS_DIR = DATASET_DIR / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "car", 1: "bus", 2: "truck"}
CLASS_COLORS = {
    0: (0, 255, 0),    # Green for Car
    1: (0, 0, 255),    # Red for Bus
    2: (255, 128, 0)   # Orange for Truck
}

def validate_aerial_dataset():
    print("==========================================")
    print(" PHASE 4: AERIAL DATASET STRUCTURAL VALIDATION")
    print("==========================================")
    print(f" Images Dir: {IMAGES_DIR}")
    print(f" Labels Dir: {LABELS_DIR}\n")

    images = sorted(list(IMAGES_DIR.glob("*.jpg")))
    labels = sorted(list(LABELS_DIR.glob("*.txt")))

    img_stems = {img.stem: img for img in images}
    lbl_stems = {lbl.stem: lbl for lbl in labels}

    report = {
        "total_images": len(images),
        "total_label_files": len(labels),
        "missing_label_files": 0,
        "corrupted_images": 0,
        "malformed_labels": 0,
        "out_of_bounds_bboxes": 0,
        "zero_area_bboxes": 0,
        "duplicate_bboxes": 0,
        "invalid_class_ids": 0,
        "extremely_small_bboxes": 0, # area < 0.00005
        "total_bboxes": 0,
        "class_counts": {"car": 0, "bus": 0, "truck": 0},
        "images_with_car": 0,
        "images_with_bus": 0,
        "images_with_truck": 0,
        "avg_boxes_per_image": 0.0,
        "min_box_dimensions_norm": [1.0, 1.0],
        "max_box_dimensions_norm": [0.0, 0.0],
        "issues": []
    }

    box_widths, box_heights = [], []

    for stem, img_p in img_stems.items():
        img = cv2.imread(str(img_p))
        if img is None:
            report["corrupted_images"] += 1
            report["issues"].append(f"Corrupted image: {img_p}")
            continue
        h, w = img.shape[:2]

        if stem not in lbl_stems:
            report["missing_label_files"] += 1
            continue

        lbl_p = lbl_stems[stem]
        with open(lbl_p, "r") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        seen_boxes = set()
        img_classes = set()

        for line_idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                report["malformed_labels"] += 1
                report["issues"].append(f"Line {line_idx} in {lbl_p.name} invalid format: '{line}'")
                continue

            try:
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
            except ValueError:
                report["malformed_labels"] += 1
                report["issues"].append(f"Line {line_idx} in {lbl_p.name} non-numeric")
                continue

            if cls_id not in CLASS_NAMES:
                report["invalid_class_ids"] += 1
                report["issues"].append(f"Invalid class ID {cls_id} in {lbl_p.name}")
                continue

            # Bounds check
            x1, y1 = cx - bw/2.0, cy - bh/2.0
            x2, y2 = cx + bw/2.0, cy + bh/2.0

            if x1 < -0.001 or y1 < -0.001 or x2 > 1.001 or y2 > 1.001 or cx < 0.0 or cx > 1.0 or cy < 0.0 or cy > 1.0:
                report["out_of_bounds_bboxes"] += 1

            if bw <= 0.0 or bh <= 0.0:
                report["zero_area_bboxes"] += 1
                report["issues"].append(f"Zero area bbox in {lbl_p.name}: {bw}x{bh}")
                continue

            area = bw * bh
            if area < 0.00005:
                report["extremely_small_bboxes"] += 1

            box_tuple = (cls_id, round(cx, 4), round(cy, 4), round(bw, 4), round(bh, 4))
            if box_tuple in seen_boxes:
                report["duplicate_bboxes"] += 1
            seen_boxes.add(box_tuple)

            report["total_bboxes"] += 1
            c_name = CLASS_NAMES[cls_id]
            report["class_counts"][c_name] += 1
            img_classes.add(c_name)

            box_widths.append(bw)
            box_heights.append(bh)

            report["min_box_dimensions_norm"][0] = min(report["min_box_dimensions_norm"][0], bw)
            report["min_box_dimensions_norm"][1] = min(report["min_box_dimensions_norm"][1], bh)
            report["max_box_dimensions_norm"][0] = max(report["max_box_dimensions_norm"][0], bw)
            report["max_box_dimensions_norm"][1] = max(report["max_box_dimensions_norm"][1], bh)

        if "car" in img_classes: report["images_with_car"] += 1
        if "bus" in img_classes: report["images_with_bus"] += 1
        if "truck" in img_classes: report["images_with_truck"] += 1

    report["avg_boxes_per_image"] = round(report["total_bboxes"] / max(1, report["total_images"]), 2)

    # Save JSON report
    json_path = OUTPUT_DIR / "aerial_dataset_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save TXT report
    txt_path = OUTPUT_DIR / "aerial_dataset_report.txt"
    txt_content = f"""==================================================
ROAD SENTINEL AI — AERIAL DATASET VALIDATION REPORT
==================================================

Total Images         : {report['total_images']}
Total Label Files    : {report['total_label_files']}
Missing Label Files  : {report['missing_label_files']}
Corrupted Images     : {report['corrupted_images']}
Malformed Labels     : {report['malformed_labels']}
Out of Bounds Bboxes : {report['out_of_bounds_bboxes']}
Zero-Area Bboxes     : {report['zero_area_bboxes']}
Duplicate Bboxes     : {report['duplicate_bboxes']}
Invalid Class IDs    : {report['invalid_class_ids']}
Extremely Small      : {report['extremely_small_bboxes']}

TOTAL ANNOTATIONS    : {report['total_bboxes']}
  - Car  (0)         : {report['class_counts']['car']} (in {report['images_with_car']} images)
  - Bus  (1)         : {report['class_counts']['bus']} (in {report['images_with_bus']} images)
  - Truck (2)        : {report['class_counts']['truck']} (in {report['images_with_truck']} images)

Average Boxes/Image  : {report['avg_boxes_per_image']}
Min Box W x H (norm) : {report['min_box_dimensions_norm'][0]:.6f} x {report['min_box_dimensions_norm'][1]:.6f}
Max Box W x H (norm) : {report['max_box_dimensions_norm'][0]:.6f} x {report['max_box_dimensions_norm'][1]:.6f}
==================================================
"""
    with open(txt_path, "w") as f:
        f.write(txt_content)

    print(txt_content)

    # Render 10 sample visualization frames with bounding boxes
    random.seed(42)
    sample_images = random.sample(images, min(10, len(images)))
    print(f"Generating {len(sample_images)} sample visualization images in: {VIS_DIR}")

    for img_p in sample_images:
        img = cv2.imread(str(img_p))
        h, w = img.shape[:2]
        lbl_p = LABELS_DIR / f"{img_p.stem}.txt"

        if lbl_p.exists():
            with open(lbl_p) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        x1 = int((cx - bw/2.0) * w)
                        y1 = int((cy - bh/2.0) * h)
                        x2 = int((cx + bw/2.0) * w)
                        y2 = int((cy + bh/2.0) * h)
                        color = CLASS_COLORS.get(c_id, (255, 255, 255))
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, CLASS_NAMES.get(c_id, str(c_id)), (x1, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite(str(VIS_DIR / f"vis_{img_p.name}"), img)

    print(f"[OK] Saved reports to:\n - {json_path}\n - {txt_path}")
    print(f"[OK] Saved visualization images to: {VIS_DIR}")

if __name__ == "__main__":
    validate_aerial_dataset()
