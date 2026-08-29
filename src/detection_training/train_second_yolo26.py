import os
import sys
import shutil
import random
import torch
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path("/Users/pranav1718/Big One/RoadSentimentalAI")
COMBINED_DATA_DIR = BASE_DIR / "ml" / "outputs" / "detection_training" / "combined_dataset"
COMBINED_DATA_YAML = COMBINED_DATA_DIR / "data.yaml"
SECOND_TRAIN_DIR = BASE_DIR / "ml" / "outputs" / "detection_training" / "training_round2"
MODELS_OUTPUT_DIR = BASE_DIR / "ml" / "outputs" / "detection_training" / "models"

SECOND_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def train_second_round_yolo26(
    base_weights="yolo26n.pt",
    epochs=5,
    batch=16,
    imgsz=512,
    lr0=0.01,
    patience=3,
    subsample=1500
):
    print("==========================================")
    print(" PHASE 17: FAST SECOND YOLO26 TRAINING (PUBLIC + TRAFFIC DATA)")
    print("==========================================")
    print(f" Base Weights   : {base_weights}")
    print(f" Combined Config: {COMBINED_DATA_YAML}")
    print(f" Epochs         : {epochs}")
    print(f" Batch Size     : {batch}")
    print(f" Image Size     : {imgsz}")

    target_yaml = str(COMBINED_DATA_YAML)

    if subsample and subsample > 0:
        sub_img_dir = COMBINED_DATA_DIR / "train_sub" / "images"
        sub_lbl_dir = COMBINED_DATA_DIR / "train_sub" / "labels"
        if sub_img_dir.exists(): shutil.rmtree(sub_img_dir)
        if sub_lbl_dir.exists(): shutil.rmtree(sub_lbl_dir)
        sub_img_dir.mkdir(parents=True, exist_ok=True)
        sub_lbl_dir.mkdir(parents=True, exist_ok=True)

        orig_img_dir = COMBINED_DATA_DIR / "train" / "images"
        orig_lbl_dir = COMBINED_DATA_DIR / "train" / "labels"

        all_imgs = sorted(list(orig_img_dir.glob("*")))
        random.seed(42)
        selected = random.sample(all_imgs, min(subsample, len(all_imgs)))

        for img_f in selected:
            dst_img = sub_img_dir / img_f.name
            shutil.copy2(img_f, dst_img)
            lbl_f = orig_lbl_dir / f"{img_f.stem}.txt"
            if lbl_f.exists():
                dst_lbl = sub_lbl_dir / lbl_f.name
                shutil.copy2(lbl_f, dst_lbl)

        sub_yaml_path = COMBINED_DATA_DIR / "data_subsample.yaml"
        sub_yaml_content = f"""path: {COMBINED_DATA_DIR.resolve()}
train: train_sub/images
val: val/images
test: test/images

nc: 3
names:
  0: car
  1: truck
  2: bus
"""
        with open(sub_yaml_path, "w") as f: f.write(sub_yaml_content)
        target_yaml = str(sub_yaml_path)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(base_weights)

    results = model.train(
        data=target_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=lr0,
        patience=patience,
        device=device,
        workers=0,
        project=str(SECOND_TRAIN_DIR.parent),
        name="training_round2_run",
        exist_ok=True,
        seed=42,
        save=True,
        plots=True,
        verbose=True
    )

    run_dir = SECOND_TRAIN_DIR.parent / "training_round2_run"
    best_weights = run_dir / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, MODELS_OUTPUT_DIR / "best_round2.pt")
        print(f"[OK] Saved Round 2 best model to: {MODELS_OUTPUT_DIR / 'best_round2.pt'}")

    return results

if __name__ == "__main__":
    train_second_round_yolo26()
