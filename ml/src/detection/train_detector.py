"""
train_detector.py — RoadSentimentalAI, Phase 8
Reproducible YOLO Vehicle Detector Fine-Tuning Pipeline.

Performs transfer learning from pretrained weights (yolov8n.pt) on the
validated vehicle detection dataset (ml/data/vehicle_detection/data.yaml).

Saves best.pt, last.pt, training metrics, confusion matrix, and PR curves under
ml/outputs/detection/finetuned_model/.
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
DATA_YAML   = BASE_DIR / "data" / "vehicle_detection" / "data.yaml"
OUTPUT_DIR  = BASE_DIR / "outputs" / "detection"

def train_vehicle_detector(
    weights="yolov8n.pt",
    data_yaml=str(DATA_YAML),
    epochs=5,
    batch=32,
    imgsz=640,
    lr0=0.01,
    patience=5,
    device="mps",
    project=str(OUTPUT_DIR),
    name="finetuned_model",
    subsample=2000
):
    print("==========================================")
    print(" ROADSENTIMENTAL AI — DETECTOR FINE-TUNING")
    print("==========================================")
    print(f" Pretrained Weights : {weights}")
    print(f" Dataset YAML       : {data_yaml}")
    print(f" Epochs             : {epochs}")
    print(f" Batch Size         : {batch}")
    print(f" Image Size         : {imgsz}")
    print(f" Subsample Images   : {subsample if subsample else 'ALL'}")
    print(f" Learning Rate      : {lr0}")
    print(f" Hardware Device    : {device}")
    print(f" Output Directory   : {project}/{name}")
    print("==========================================\n")

    if not Path(data_yaml).exists():
        print(f"Error: Dataset YAML not found at {data_yaml}. Run prepare_and_validate_dataset.py first.")
        sys.exit(1)

    # Optional subsampling for rapid baseline fine-tuning
    if subsample and subsample > 0:
        sub_yaml_path = BASE_DIR / "data" / "vehicle_detection" / "data_subsample.yaml"
        sub_img_dir = BASE_DIR / "data" / "vehicle_detection" / "images" / "train_sub"
        sub_lbl_dir = BASE_DIR / "data" / "vehicle_detection" / "labels" / "train_sub"
        
        sub_img_dir.mkdir(parents=True, exist_ok=True)
        sub_lbl_dir.mkdir(parents=True, exist_ok=True)

        orig_img_dir = BASE_DIR / "data" / "vehicle_detection" / "images" / "train"
        orig_lbl_dir = BASE_DIR / "data" / "vehicle_detection" / "labels" / "train"

        import random, shutil
        random.seed(42)
        all_imgs = sorted(list(orig_img_dir.glob("*")))
        selected = random.sample(all_imgs, min(subsample, len(all_imgs)))

        for img_f in selected:
            dst_img = sub_img_dir / img_f.name
            if not dst_img.exists(): shutil.copy2(img_f, dst_img)
            lbl_f = orig_lbl_dir / f"{img_f.stem}.txt"
            if lbl_f.exists():
                dst_lbl = sub_lbl_dir / lbl_f.name
                if not dst_lbl.exists(): shutil.copy2(lbl_f, dst_lbl)

        sub_yaml_content = f"""path: {(BASE_DIR / "data" / "vehicle_detection").resolve()}
train: images/train_sub
val: images/val
test: images/test

nc: 7
names: ['bicycle', 'bus', 'car', 'motorbike', 'rickshaw', 'truck', 'van']
"""
        with open(sub_yaml_path, "w") as fh: fh.write(sub_yaml_content)
        data_yaml = str(sub_yaml_path)

    model = YOLO(weights)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=lr0,
        patience=patience,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        seed=42,
        save=True,
        plots=True,
        verbose=True
    )

    print(f"\n[OK] Training Complete! Best weights saved to: {project}/{name}/weights/best.pt")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoadSentimentalAI YOLO Vehicle Detector Fine-Tuning")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Pretrained weights")
    parser.add_argument("--data", type=str, default=str(DATA_YAML), help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="mps", help="Computation device (mps, cpu, cuda)")
    parser.add_argument("--project", type=str, default=str(OUTPUT_DIR), help="Output project folder")
    parser.add_argument("--name", type=str, default="finetuned_model", help="Run folder name")
    parser.add_argument("--subsample", type=int, default=2000, help="Subsample image count")
    args = parser.parse_args()

    train_vehicle_detector(
        weights=args.weights,
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        subsample=args.subsample
    )
