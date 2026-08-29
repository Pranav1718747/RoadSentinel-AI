import os
import sys
import shutil
import json
from pathlib import Path

BASE_DIR = Path("/Users/pranav1718/Big One/RoadSentimentalAI")
PUBLIC_DATASET_DIR = BASE_DIR / "ml" / "outputs" / "detection_training" / "dataset"
TRAFFIC_FRAMES_DIR = BASE_DIR / "ml" / "outputs" / "detection_training" / "our_video_frames"
COMBINED_DATASET_DIR = BASE_DIR / "ml" / "outputs" / "detection_training" / "combined_dataset"

def build_combined_dataset():
    print("==========================================")
    print(" PHASE 16: COMBINED PUBLIC + OUR TRAFFIC DATASET (VIDEO SEGMENT SPLIT)")
    print("==========================================")

    if COMBINED_DATASET_DIR.exists():
        shutil.rmtree(COMBINED_DATASET_DIR)

    # Copy public dataset base
    shutil.copytree(PUBLIC_DATASET_DIR, COMBINED_DATASET_DIR)

    pre_annos_dir = TRAFFIC_FRAMES_DIR / "pre_annotations"
    traffic_frames = sorted(list(TRAFFIC_FRAMES_DIR.glob("*.jpg")))

    if not traffic_frames:
        print("No traffic frames found to merge.")
        return

    # VIDEO SEGMENT SPLIT (Prevent contiguous frame leakage)
    # Frames 0-350 -> Train, Frames 351-420 -> Val, Frames 421-500 -> Test
    train_added, val_added, test_added = 0, 0, 0

    for img_p in traffic_frames:
        stem = img_p.stem
        # Extract frame_id from filename traffic_frame_0123.jpg
        try:
            frame_id = int(stem.split("_")[-1])
        except ValueError:
            frame_id = 0

        if frame_id <= 350:
            target_split = "train"
            train_added += 1
        elif frame_id <= 420:
            target_split = "val"
            val_added += 1
        else:
            target_split = "test"
            test_added += 1

        dst_img = COMBINED_DATASET_DIR / target_split / "images" / img_p.name
        shutil.copy2(img_p, dst_img)

        # Copy annotation if available
        lbl_p = pre_annos_dir / f"{stem}.txt"
        dst_lbl = COMBINED_DATASET_DIR / target_split / "labels" / f"{stem}.txt"
        if lbl_p.exists():
            # Filter comments out of pre-annotations
            with open(lbl_p, "r") as f:
                valid_lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
            with open(dst_lbl, "w") as f:
                if valid_lines:
                    f.write("\n".join(valid_lines) + "\n")

    # Update data.yaml
    data_yaml_content = f"""path: {COMBINED_DATASET_DIR.resolve()}
train: train/images
val: val/images
test: test/images

nc: 3
names:
  0: car
  1: truck
  2: bus
"""
    with open(COMBINED_DATASET_DIR / "data.yaml", "w") as f:
        f.write(data_yaml_content)

    print("--- COMBINED DATASET SUMMARY ---")
    print(f"Total Traffic Frames Added: {len(traffic_frames)}")
    print(f"  - Train Split (Frames 0-350)  : {train_added}")
    print(f"  - Val Split   (Frames 351-420): {val_added}")
    print(f"  - Test Split  (Frames 421-500): {test_added}")
    print(f"[OK] Combined dataset ready at: {COMBINED_DATASET_DIR}")

if __name__ == "__main__":
    build_combined_dataset()
