import os
import sys
import shutil
import random
import torch
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path("/Users/pranav1718/Big One/RoadSentimentalAI")
DATASET_DIR = BASE_DIR / "ml" / "outputs" / "detection_training" / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
TRAIN_OUTPUT_DIR = BASE_DIR / "ml" / "outputs" / "detection_training" / "training"
MODELS_OUTPUT_DIR = BASE_DIR / "ml" / "outputs" / "detection_training" / "models"

TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def train_yolo26_model(
    model_weights="yolo26n.pt",
    epochs=5,
    batch=16,
    imgsz=512,
    lr0=0.01,
    patience=3,
    subsample=1500
):
    print("==========================================")
    print(" PHASE 10: OPTIMIZED TRANSFER LEARNING YOLO26 MODEL")
    print("==========================================")
    print(f" Initial Weights : {model_weights}")
    print(f" Dataset Config  : {DATA_YAML}")
    print(f" Epochs          : {epochs}")
    print(f" Batch Size      : {batch}")
    print(f" Image Size      : {imgsz}")
    print(f" Subsample Count : {subsample if subsample else 'ALL'}")

    target_data_yaml = str(DATA_YAML)

    if subsample and subsample > 0:
        sub_img_dir = DATASET_DIR / "train_sub" / "images"
        sub_lbl_dir = DATASET_DIR / "train_sub" / "labels"
        if sub_img_dir.exists():
            shutil.rmtree(sub_img_dir)
        if sub_lbl_dir.exists():
            shutil.rmtree(sub_lbl_dir)

        sub_img_dir.mkdir(parents=True, exist_ok=True)
        sub_lbl_dir.mkdir(parents=True, exist_ok=True)

        orig_img_dir = DATASET_DIR / "train" / "images"
        orig_lbl_dir = DATASET_DIR / "train" / "labels"

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

        sub_yaml_path = DATASET_DIR / "data_subsample.yaml"
        sub_yaml_content = f"""path: {DATASET_DIR.resolve()}
train: train_sub/images
val: val/images
test: test/images

nc: 3
names:
  0: car
  1: truck
  2: bus
"""
        with open(sub_yaml_path, "w") as f:
            f.write(sub_yaml_content)
        target_data_yaml = str(sub_yaml_path)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f" Target Device   : {device}\n")

    model = YOLO(model_weights)

    results = model.train(
        data=target_data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=lr0,
        patience=patience,
        device=device,
        workers=0,
        project=str(TRAIN_OUTPUT_DIR.parent),
        name="training_run",
        exist_ok=True,
        seed=42,
        save=True,
        plots=True,
        verbose=True
    )

    run_dir = TRAIN_OUTPUT_DIR.parent / "training_run"

    # Copy best and last weights to models directory
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    if best_weights.exists():
        shutil.copy2(best_weights, MODELS_OUTPUT_DIR / "best.pt")
        print(f"[OK] Saved best model to: {MODELS_OUTPUT_DIR / 'best.pt'}")

    if last_weights.exists():
        shutil.copy2(last_weights, MODELS_OUTPUT_DIR / "last.pt")
        print(f"[OK] Saved last model to: {MODELS_OUTPUT_DIR / 'last.pt'}")

    # Copy results.csv and plots to training directory
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        shutil.copy2(results_csv, TRAIN_OUTPUT_DIR / "training_results.csv")
        print(f"[OK] Saved training results to: {TRAIN_OUTPUT_DIR / 'training_results.csv'}")

    results_png = run_dir / "results.png"
    if results_png.exists():
        shutil.copy2(results_png, TRAIN_OUTPUT_DIR / "training_curves.png")
        print(f"[OK] Saved training curves to: {TRAIN_OUTPUT_DIR / 'training_curves.png'}")

    print("\n[OK] YOLO26 Fine-Tuning Complete!")
    return results

if __name__ == "__main__":
    train_yolo26_model()
