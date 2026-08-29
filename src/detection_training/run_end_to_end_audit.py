import os
import sys
import json
from pathlib import Path

BASE_DIR = Path("/Users/pranav1718/Big One/RoadSentimentalAI")
OUTPUT_DIR = BASE_DIR / "ml" / "outputs" / "detection_training"

def generate_final_summary_report():
    print("==========================================")
    print(" ROADSENTIMENTAL AI — FINAL DETECTION AUDIT SUMMARY")
    print("==========================================")

    # 1. Baseline metrics
    baseline_json = OUTPUT_DIR / "baseline" / "baseline_metrics.json"
    baseline_data = {}
    if baseline_json.exists():
        with open(baseline_json) as f:
            baseline_data = json.load(f)

    # 2. Test metrics
    test_json = OUTPUT_DIR / "evaluation" / "test_metrics.json"
    test_data = {}
    if test_json.exists():
        with open(test_json) as f:
            test_data = json.load(f)

    # 3. Per-class metrics
    per_class_json = OUTPUT_DIR / "evaluation" / "per_class_metrics.json"
    per_class_data = {}
    if per_class_json.exists():
        with open(per_class_json) as f:
            per_class_data = json.load(f)

    # 4. Video comparison
    video_json = OUTPUT_DIR / "evaluation" / "realworld_video_comparison.json"
    video_data = {}
    if video_json.exists():
        with open(video_json) as f:
            video_data = json.load(f)

    print("\n--- PERFORMANCE HIGHLIGHTS ---")
    print(f"Pretrained Baseline mAP50 : {baseline_data.get('overall_precision', 'N/A')} prec / {baseline_data.get('overall_recall', 'N/A')} rec")
    print(f"Fine-Tuned YOLO26 mAP50   : {test_data.get('mAP50', 'N/A')}")
    print(f"Fine-Tuned YOLO26 mAP50-95: {test_data.get('mAP50_95', 'N/A')}")
    print(f"Fine-Tuned Precision      : {test_data.get('precision', 'N/A')}")
    print(f"Fine-Tuned Recall         : {test_data.get('recall', 'N/A')}")
    
    if per_class_data:
        print("\nPer-Class Breakdown:")
        for c_name, c_stats in per_class_data.items():
            print(f"  {c_name:8s} -> Prec: {c_stats.get('precision')}, Rec: {c_stats.get('recall')}, mAP50: {c_stats.get('mAP50')}, mAP50-95: {c_stats.get('mAP50_95')}")

    if video_data:
        print("\nReal-World Traffic Video Performance:")
        print(f"  Baseline Detections : {video_data.get('current_detector', {}).get('total_detections')}")
        print(f"  YOLO26 Detections   : {video_data.get('finetuned_yolo26', {}).get('total_detections')}")
        print(f"  Detection Gain %    : {video_data.get('detection_gain_percentage')}%")

if __name__ == "__main__":
    generate_final_summary_report()
