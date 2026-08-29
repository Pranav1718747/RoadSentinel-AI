import os
import sys
import shutil
import json
from pathlib import Path

BASE_DIR = Path("/Users/pranav1718/Big One/RoadSentimentalAI")
DATASET_DIR = BASE_DIR / "datasets" / "aerial_vehicle"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

def split_aerial_dataset_temporally():
    print("==========================================")
    print(" PHASE 5: TEMPORAL-SEGMENT TRAIN/VAL/TEST SPLIT")
    print("==========================================")

    # Define temporal frame split boundaries to prevent contiguous data leakage
    # Frames 0000 - 0350: Train (70.2%)
    # Frames 0351 - 0450: Val   (20.0%)
    # Frames 0451 - 0499: Test  (9.8%)

    splits = {
        "train": (0, 350),
        "val": (351, 450),
        "test": (451, 499)
    }

    stats = {
        "splits": {},
        "total_images": 0
    }

    for split_name, (f_start, f_end) in splits.items():
        dst_img_dir = DATASET_DIR / split_name / "images"
        dst_lbl_dir = DATASET_DIR / split_name / "labels"

        if (DATASET_DIR / split_name).exists():
            shutil.rmtree(DATASET_DIR / split_name)

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        copied_cnt = 0
        for f_idx in range(f_start, f_end + 1):
            stem = f"frame_{f_idx:04d}"
            src_img = IMAGES_DIR / f"{stem}.jpg"
            src_lbl = LABELS_DIR / f"{stem}.txt"

            if src_img.exists():
                shutil.copy2(src_img, dst_img_dir / src_img.name)
                copied_cnt += 1

            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl_dir / src_lbl.name)

        stats["splits"][split_name] = {
            "frame_range": f"{f_start:04d}-{f_end:04d}",
            "count": copied_cnt,
            "percentage": round((copied_cnt / 500.0) * 100.0, 1)
        }
        stats["total_images"] += copied_cnt

    # Create data.yaml
    data_yaml_content = f"""path: {DATASET_DIR.resolve()}
train: train/images
val: val/images
test: test/images

nc: 3
names:
  0: car
  1: bus
  2: truck
"""
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(data_yaml_content)

    split_manifest = DATASET_DIR / "split_manifest.json"
    with open(split_manifest, "w") as f:
        json.dump(stats, f, indent=2)

    print("--- TEMPORAL SPLIT SUMMARY ---")
    print(f"Total Images Split : {stats['total_images']}")
    for s_name, s_info in stats["splits"].items():
        print(f"  Split '{s_name:5s}': Range {s_info['frame_range']} | Count: {s_info['count']} ({s_info['percentage']}%)")
    print(f"[OK] data.yaml created at: {yaml_path}")
    print(f"[OK] Split manifest written to: {split_manifest}")

if __name__ == "__main__":
    split_aerial_dataset_temporally()
