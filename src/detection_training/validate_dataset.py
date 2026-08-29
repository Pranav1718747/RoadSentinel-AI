import os
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict
import cv2
import matplotlib.pyplot as plt

RAW_DATASET_PATH = Path("/Users/pranav1718/.cache/kagglehub/datasets/daudshah/vehicle-detection-dataset/versions/1")
OUTPUT_DIR = Path("/Users/pranav1718/Big One/RoadSentimentalAI/ml/outputs/detection_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CLASS_NAMES = ['bicycle', 'bus', 'car', 'motorbike', 'rickshaw', 'truck', 'van']

def validate_kaggle_dataset():
    print("==========================================")
    print(" KAGGLE DATASET STRUCTURAL VALIDATION")
    print("==========================================")
    print(f" Dataset Path: {RAW_DATASET_PATH}\n")

    report = {
        "dataset_path": str(RAW_DATASET_PATH),
        "splits": {},
        "totals": {
            "total_images": 0,
            "total_labels": 0,
            "total_annotations": 0,
            "corrupted_images": 0,
            "missing_labels": 0,
            "orphaned_labels": 0,
            "invalid_annotations": 0,
            "out_of_bounds_bboxes": 0,
            "zero_neg_size_bboxes": 0,
            "extremely_tiny_bboxes": 0,
            "duplicate_annotations": 0,
            "invalid_class_ids": 0
        },
        "raw_class_counts": Counter(),
        "target_mapped_class_counts": Counter(),
        "excluded_class_counts": Counter(),
        "issues": []
    }

    splits = ["train", "valid", "test"]

    for split in splits:
        split_dir = RAW_DATASET_PATH / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        split_stats = {
            "images": 0,
            "labels": 0,
            "annotations": 0,
            "corrupted_images": 0,
            "missing_labels": 0,
            "orphaned_labels": 0,
            "class_counts": Counter()
        }

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"Warning: {split} directory structure incomplete.")
            continue

        images = sorted(list(img_dir.glob("*")))
        labels = sorted(list(lbl_dir.glob("*.txt")))

        img_stems = {img.stem: img for img in images}
        lbl_stems = {lbl.stem: lbl for lbl in labels}

        split_stats["images"] = len(images)
        split_stats["labels"] = len(labels)
        report["totals"]["total_images"] += len(images)
        report["totals"]["total_labels"] += len(labels)

        # Check missing labels
        missing_lbl_stems = set(img_stems.keys()) - set(lbl_stems.keys())
        split_stats["missing_labels"] = len(missing_lbl_stems)
        report["totals"]["missing_labels"] += len(missing_lbl_stems)

        # Check orphaned labels (labels without images)
        orphaned_lbl_stems = set(lbl_stems.keys()) - set(img_stems.keys())
        split_stats["orphaned_labels"] = len(orphaned_lbl_stems)
        report["totals"]["orphaned_labels"] += len(orphaned_lbl_stems)

        for stem, img_path in img_stems.items():
            # Test image opening
            img = cv2.imread(str(img_path))
            if img is None:
                split_stats["corrupted_images"] += 1
                report["totals"]["corrupted_images"] += 1
                report["issues"].append(f"Corrupted image: {img_path}")
                continue

            # Check label
            if stem in lbl_stems:
                lbl_path = lbl_stems[stem]
                with open(lbl_path, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]

                seen_boxes = set()
                for line_idx, line in enumerate(lines):
                    parts = line.split()
                    if len(parts) != 5:
                        report["totals"]["invalid_annotations"] += 1
                        report["issues"].append(f"Invalid format line {line_idx} in {lbl_path}: '{line}'")
                        continue

                    try:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                    except ValueError:
                        report["totals"]["invalid_annotations"] += 1
                        report["issues"].append(f"Non-numeric values line {line_idx} in {lbl_path}")
                        continue

                    if cls_id < 0 or cls_id >= len(RAW_CLASS_NAMES):
                        report["totals"]["invalid_class_ids"] += 1
                        report["issues"].append(f"Invalid class ID {cls_id} in {lbl_path}")
                        continue

                    # Check bounding box bounds
                    x1 = cx - bw / 2.0
                    y1 = cy - bh / 2.0
                    x2 = cx + bw / 2.0
                    y2 = cy + bh / 2.0

                    if x1 < -0.001 or y1 < -0.001 or x2 > 1.001 or y2 > 1.001 or cx < 0.0 or cx > 1.0 or cy < 0.0 or cy > 1.0:
                        report["totals"]["out_of_bounds_bboxes"] += 1

                    if bw <= 0 or bh <= 0:
                        report["totals"]["zero_neg_size_bboxes"] += 1
                        report["issues"].append(f"Zero/negative bbox size in {lbl_path}: {bw}x{bh}")

                    # Extremely tiny box check (< 0.0001 normalized area)
                    if bw * bh < 0.0001:
                        report["totals"]["extremely_tiny_bboxes"] += 1

                    box_tuple = (cls_id, round(cx, 4), round(cy, 4), round(bw, 4), round(bh, 4))
                    if box_tuple in seen_boxes:
                        report["totals"]["duplicate_annotations"] += 1
                    seen_boxes.add(box_tuple)

                    split_stats["annotations"] += 1
                    report["totals"]["total_annotations"] += 1
                    
                    cls_name = RAW_CLASS_NAMES[cls_id]
                    split_stats["class_counts"][cls_name] += 1
                    report["raw_class_counts"][cls_name] += 1

                    # Target mapping stats
                    if cls_name == "car":
                        report["target_mapped_class_counts"]["car"] += 1
                    elif cls_name == "truck":
                        report["target_mapped_class_counts"]["truck"] += 1
                    elif cls_name == "bus":
                        report["target_mapped_class_counts"]["bus"] += 1
                    else:
                        report["excluded_class_counts"][cls_name] += 1

        report["splits"][split] = {
            "images": split_stats["images"],
            "labels": split_stats["labels"],
            "annotations": split_stats["annotations"],
            "corrupted_images": split_stats["corrupted_images"],
            "missing_labels": split_stats["missing_labels"],
            "orphaned_labels": split_stats["orphaned_labels"],
            "class_counts": dict(split_stats["class_counts"])
        }

    # Save JSON reports
    report["raw_class_counts"] = dict(report["raw_class_counts"])
    report["target_mapped_class_counts"] = dict(report["target_mapped_class_counts"])
    report["excluded_class_counts"] = dict(report["excluded_class_counts"])

    with open(OUTPUT_DIR / "dataset_report.json", "w") as f:
        json.dump(report, f, indent=2)

    class_dist = {
        "raw_counts": report["raw_class_counts"],
        "target_mapped_counts": report["target_mapped_class_counts"],
        "excluded_counts": report["excluded_class_counts"],
        "splits": {s: report["splits"][s]["class_counts"] for s in report["splits"]}
    }

    with open(OUTPUT_DIR / "dataset_class_distribution.json", "w") as f:
        json.dump(class_dist, f, indent=2)

    # Plot class distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = list(report["raw_class_counts"].keys())
    counts = [report["raw_class_counts"][c] for c in classes]
    colors = ['#3498db' if c in ['car', 'truck', 'bus'] else '#95a5a6' for c in classes]

    bars = ax.bar(classes, counts, color=colors)
    ax.set_title("Kaggle Vehicle Dataset Class Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Class Name", fontsize=12)
    ax.set_ylabel("Number of Bounding Box Annotations", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dataset_class_distribution.png", dpi=300)
    plt.close()

    print("\n--- VALIDATION SUMMARY ---")
    print(f"Total Images         : {report['totals']['total_images']}")
    print(f"Total Annotations    : {report['totals']['total_annotations']}")
    print(f"Corrupted Images     : {report['totals']['corrupted_images']}")
    print(f"Missing Labels       : {report['totals']['missing_labels']}")
    print(f"Invalid Annotations  : {report['totals']['invalid_annotations']}")
    print(f"Out of Bounds Bboxes : {report['totals']['out_of_bounds_bboxes']}")
    print(f"Zero/Neg Bboxes      : {report['totals']['zero_neg_size_bboxes']}")
    print(f"Duplicate Annotations: {report['totals']['duplicate_annotations']}")
    print("\nRaw Class Counts:")
    for cls, cnt in report["raw_class_counts"].items():
        print(f"  {cls:12s}: {cnt}")

    print("\nTarget Mapped Counts (car, truck, bus):")
    for cls, cnt in report["target_mapped_class_counts"].items():
        print(f"  {cls:12s}: {cnt}")

    print("\nExcluded Class Counts:")
    for cls, cnt in report["excluded_class_counts"].items():
        print(f"  {cls:12s}: {cnt}")

    print(f"\n[OK] Validation report saved to: {OUTPUT_DIR / 'dataset_report.json'}")
    print(f"[OK] Class distribution saved to: {OUTPUT_DIR / 'dataset_class_distribution.png'}")

if __name__ == "__main__":
    validate_kaggle_dataset()
