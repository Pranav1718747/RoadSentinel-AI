import os
import sys
import json
import shutil
import torch
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path("/Users/pranav1718/Big One/RoadSentimentalAI")
DATA_YAML = BASE_DIR / "ml" / "outputs" / "detection_training" / "dataset" / "data.yaml"
MODEL_PATH = BASE_DIR / "ml" / "outputs" / "detection_training" / "models" / "best.pt"
EVAL_OUTPUT_DIR = BASE_DIR / "ml" / "outputs" / "detection_training" / "evaluation"

EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_finetuned_yolo26():
    print("==========================================")
    print(" PHASE 11: EVALUATE FINE-TUNED YOLO26 ON TEST SET")
    print("==========================================")
    print(f" Model Path : {MODEL_PATH}")
    print(f" Dataset YAML: {DATA_YAML}\n")

    if not MODEL_PATH.exists():
        print(f"Error: Fine-tuned weights not found at {MODEL_PATH}")
        sys.exit(1)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(MODEL_PATH))

    # Evaluate on test split
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        batch=16,
        imgsz=640,
        device=device,
        project=str(EVAL_OUTPUT_DIR.parent),
        name="test_eval",
        exist_ok=True,
        plots=True,
        save_json=True
    )

    eval_run_dir = EVAL_OUTPUT_DIR.parent / "test_eval"

    # Copy plots to evaluation directory
    for plot_file in ["confusion_matrix.png", "PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png"]:
        src = eval_run_dir / plot_file
        if src.exists():
            dst_name = "precision_recall.png" if plot_file == "PR_curve.png" else plot_file
            shutil.copy2(src, EVAL_OUTPUT_DIR / dst_name)

    # Extract overall metrics
    test_results = {
        "model": "Fine-tuned YOLO26 (best.pt)",
        "split": "test",
        "precision": round(float(metrics.results_dict.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(metrics.results_dict.get("metrics/recall(B)", 0.0)), 4),
        "mAP50": round(float(metrics.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(metrics.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
    }

    class_names = ["car", "truck", "bus"]
    per_class_results = {}

    if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
        for idx, c_name in enumerate(class_names):
            p = float(metrics.box.p[idx]) if hasattr(metrics.box, 'p') and idx < len(metrics.box.p) else 0.0
            r = float(metrics.box.r[idx]) if hasattr(metrics.box, 'r') and idx < len(metrics.box.r) else 0.0
            map50 = float(metrics.box.ap50[idx]) if hasattr(metrics.box, 'ap50') and idx < len(metrics.box.ap50) else (float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0)
            map95 = float(metrics.box.maps[idx]) if idx < len(metrics.box.maps) else 0.0

            per_class_results[c_name] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "mAP50": round(map50, 4),
                "mAP50_95": round(map95, 4),
                "f1_score": round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0
            }

    with open(EVAL_OUTPUT_DIR / "test_metrics.json", "w") as f:
        json.dump(test_results, f, indent=2)

    with open(EVAL_OUTPUT_DIR / "per_class_metrics.json", "w") as f:
        json.dump(per_class_results, f, indent=2)

    print("\n--- HELD-OUT TEST METRICS ---")
    print(f"Precision  : {test_results['precision']}")
    print(f"Recall     : {test_results['recall']}")
    print(f"mAP50      : {test_results['mAP50']}")
    print(f"mAP50-95   : {test_results['mAP50_95']}")
    print("\nPer Class Breakdown:")
    for c_name, c_dict in per_class_results.items():
        print(f"  {c_name:8s} -> Prec: {c_dict['precision']:.4f}, Rec: {c_dict['recall']:.4f}, mAP50: {c_dict['mAP50']:.4f}, mAP50-95: {c_dict['mAP50_95']:.4f}")

    print(f"\n[OK] Test metrics saved to: {EVAL_OUTPUT_DIR / 'test_metrics.json'}")
    print(f"[OK] Per-class metrics saved to: {EVAL_OUTPUT_DIR / 'per_class_metrics.json'}")

if __name__ == "__main__":
    evaluate_finetuned_yolo26()
