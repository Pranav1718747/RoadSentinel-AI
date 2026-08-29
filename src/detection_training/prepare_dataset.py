import os
import sys
import shutil
import json
from pathlib import Path
from collections import Counter

RAW_DATASET_PATH = Path("/Users/pranav1718/.cache/kagglehub/datasets/daudshah/vehicle-detection-dataset/versions/1")
OUTPUT_DATASET_DIR = Path("/Users/pranav1718/Big One/RoadSentimentalAI/ml/outputs/detection_training/dataset")

# Target Mapping:
# 2 (car) -> 0
# 5 (truck) -> 1
# 1 (bus) -> 2
CLASS_MAPPING = {
    2: 0,  # car
    5: 1,  # truck
    1: 2   # bus
}

TARGET_NAMES = {
    0: "car",
    1: "truck",
    2: "bus"
}

def prepare_yolo_dataset():
    print("==========================================")
    print(" PREPARING CLEAN 3-CLASS YOLO DATASET")
    print("==========================================")
    print(f" Source Path: {RAW_DATASET_PATH}")
    print(f" Target Path: {OUTPUT_DATASET_DIR}\n")

    if OUTPUT_DATASET_DIR.exists():
        print(f"Cleaning existing output directory: {OUTPUT_DATASET_DIR}")
        shutil.rmtree(OUTPUT_DATASET_DIR)

    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Split mapping: 'valid' in Kaggle dataset -> 'val' in target dataset
    split_map = {
        "train": "train",
        "valid": "val",
        "test": "test"
    }

    stats = {
        "splits": {},
        "target_class_counts": Counter(),
        "total_images": 0,
        "total_annotations": 0
    }

    for src_split, dst_split in split_map.items():
        src_img_dir = RAW_DATASET_PATH / src_split / "images"
        src_lbl_dir = RAW_DATASET_PATH / src_split / "labels"

        dst_split_dir = OUTPUT_DATASET_DIR / dst_split
        dst_img_dir = dst_split_dir / "images"
        dst_lbl_dir = dst_split_dir / "labels"

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(list(src_img_dir.glob("*")))
        
        split_img_cnt = 0
        split_ann_cnt = 0
        split_cls_counts = Counter()

        for img_path in images:
            stem = img_path.stem
            lbl_path = src_lbl_dir / f"{stem}.txt"

            new_label_lines = []
            if lbl_path.exists():
                with open(lbl_path, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]

                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    raw_cls_id = int(parts[0])

                    if raw_cls_id in CLASS_MAPPING:
                        new_cls_id = CLASS_MAPPING[raw_cls_id]
                        new_label_lines.append(f"{new_cls_id} {' '.join(parts[1:])}")
                        split_cls_counts[TARGET_NAMES[new_cls_id]] += 1
                        stats["target_class_counts"][TARGET_NAMES[new_cls_id]] += 1
                        split_ann_cnt += 1

            # Save copied image
            dst_img_path = dst_img_dir / img_path.name
            shutil.copy2(img_path, dst_img_path)

            # Write label file (even if empty, representing background/negative sample if no car/truck/bus)
            dst_lbl_path = dst_lbl_dir / f"{stem}.txt"
            with open(dst_lbl_path, "w") as f:
                if new_label_lines:
                    f.write("\n".join(new_label_lines) + "\n")

            split_img_cnt += 1

        stats["splits"][dst_split] = {
            "images": split_img_cnt,
            "annotations": split_ann_cnt,
            "class_counts": dict(split_cls_counts)
        }
        stats["total_images"] += split_img_cnt
        stats["total_annotations"] += split_ann_cnt

    # Create data.yaml
    data_yaml_content = f"""path: {OUTPUT_DATASET_DIR.resolve()}
train: train/images
val: val/images
test: test/images

nc: 3
names:
  0: car
  1: truck
  2: bus
"""

    yaml_path = OUTPUT_DATASET_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(data_yaml_content)

    print("--- CONVERSION COMPLETE ---")
    print(f"Data YAML created at: {yaml_path}")
    print(f"Total Images        : {stats['total_images']}")
    print(f"Total Annotations   : {stats['total_annotations']}")
    for split_name, sinfo in stats["splits"].items():
        print(f"\nSplit '{split_name}':")
        print(f"  Images     : {sinfo['images']}")
        print(f"  Annotations: {sinfo['annotations']}")
        print(f"  Classes    : {sinfo['class_counts']}")

    return stats

if __name__ == "__main__":
    prepare_yolo_dataset()
