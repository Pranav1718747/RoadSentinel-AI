import os
import sys
import shutil
import torch
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path("/Users/pranav1718/Big One/RoadSentimentalAI")
AERIAL_DATA_YAML = BASE_DIR / "datasets" / "aerial_vehicle" / "data.yaml"
AERIAL_OUTPUT_DIR = BASE_DIR / "ml" / "outputs" / "aerial_detection"
MODELS_OUTPUT_DIR = BASE_DIR / "ml" / "outputs" / "detection_training" / "models"

AERIAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def train_aerial_yolo26_model(
    base_weights="yolo26n.pt",
    epochs=5,
    batch=16,
    imgsz=640,
    lr0=0.01,
    patience=3
):
    print("==========================================")
    print(" PHASE 6 & 7: FAST AERIAL YOLO26 FINE-TUNING")
    print("==========================================")
    print(f" Base Weights   : {base_weights}")
    print(f" Aerial Config  : {AERIAL_DATA_YAML}")
    print(f" Epochs         : {epochs}")
    print(f" Batch Size     : {batch}")
    print(f" Image Size     : {imgsz}")

    if not AERIAL_DATA_YAML.exists():
        print(f"Error: Dataset YAML not found at {AERIAL_DATA_YAML}")
        sys.exit(1)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f" Device         : {device}\n")

    model = YOLO(base_weights)

    results = model.train(
        data=str(AERIAL_DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=lr0,
        patience=patience,
        device=device,
        workers=0,
        mosaic=1.0,
        scale=0.7,
        degrees=15.0,
        translate=0.1,
        project=str(AERIAL_OUTPUT_DIR),
        name="best_aerial_model",
        exist_ok=True,
        seed=42,
        save=True,
        plots=True,
        verbose=True
    )

    run_dir = AERIAL_OUTPUT_DIR / "best_aerial_model"
    best_weights = run_dir / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, MODELS_OUTPUT_DIR / "best_aerial.pt")
        print(f"[OK] Saved best aerial model to: {MODELS_OUTPUT_DIR / 'best_aerial.pt'}")

    return results

if __name__ == "__main__":
    train_aerial_yolo26_model()
