# Aerial Object Detection Baseline & Audit Report — Phase 1

## Executive Summary
This document records the exact state and configuration of the existing Object Detection pipeline in **RoadSentimentalAI** prior to building the dedicated **Aerial Object Detection Recovery & Fine-Tuning Pipeline**.

> [!CAUTION]
> **Audit Finding**: The previously fine-tuned YOLO26 model (`best.pt`) achieved high precision on ground-level vehicle photos, but produced **0 detections on high-angle 4K aerial footage (`traffic.mp4`) at confidence threshold 0.45**.
> This document details the 13 inspection points required to establish our technical baseline.

---

## 13 Inspection Points

### 1. Current YOLO26 Model Configuration
- Architecture: Ultralytics YOLO26 Nano (`yolo26n`)
- Layer Count: 260 layers (122 fused layers)
- Parameter Count: 2,504,970 parameters (2.37M fused parameters)
- GFLOPs: 5.9 GFLOPs at 640x640 resolution (5.3 GFLOPs fused)

### 2. Current Base Model / Checkpoint
- Fine-Tuned Model Weights: [best.pt](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/ml/outputs/detection_training/models/best.pt)
- Round 2 Fine-Tuned Weights: [best_round2.pt](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/ml/outputs/detection_training/models/best_round2.pt)
- Baseline COCO Weights: [yolo26n.pt](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/yolo26n.pt)

### 3. Training Scripts
- Primary YOLO26 Fine-Tuning Script: [train_yolo26.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/src/detection_training/train_yolo26.py)
- Round 2 Combined Dataset Fine-Tuning Script: [train_second_yolo26.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/src/detection_training/train_second_yolo26.py)
- Historical Baseline Fine-Tuning Script: [train_detector.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/ml/src/detection/train_detector.py)

### 4. Dataset Conversion Scripts
- Kaggle Dataset Preparation Script: [prepare_dataset.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/src/detection_training/prepare_dataset.py)
- Combined Public + Traffic Dataset Merger Script: [prepare_combined_dataset.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/src/detection_training/prepare_combined_dataset.py)

### 5. Dataset YAML Files
- Clean 3-Class Vehicle Dataset YAML: [data.yaml](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/ml/outputs/detection_training/dataset/data.yaml)
- Subsampled Dataset YAML: [data_subsample.yaml](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/ml/outputs/detection_training/dataset/data_subsample.yaml)
- Combined Dataset YAML: [data.yaml](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/ml/outputs/detection_training/combined_dataset/data.yaml)

### 6. Class Names and Class IDs
Taxonomy consists of 3 vehicle target classes:
- `0`: `car`
- `1`: `truck`
- `2`: `bus`

### 7. Validation Scripts
- Dataset Inspection & Structural Validation Script: [validate_dataset.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/src/detection_training/validate_dataset.py)
- Model Evaluation Script: [evaluate_finetuned.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/src/detection_training/evaluate_finetuned.py)
- Baseline Model Evaluation Script: [eval_yolo26_baseline.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/src/detection_training/eval_yolo26_baseline.py)

### 8. Inference Scripts
- Real-World Video Comparison Script: [compare_realworld_video.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/src/detection_training/compare_realworld_video.py)
- Pipeline Tracking & Detection Integration: [track_v2.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/ml/src/tracking/track_v2.py)
- Standalone Detector: [detect.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/ml/src/detection/detect.py)

### 9. Current Confidence and IoU Thresholds
- Confidence Threshold: `0.45` (`DETECTION_CONF = 0.45` in [config.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/ml/src/pipeline/config.py))
- NMS IoU Threshold: `0.70` (Ultralytics default)
- Tracker Match IoU Threshold: `0.50` (`IOU_DUPLICATE_THRESH = 0.50`)

### 10. Current Image Resolution
- Video Inference Image Size: `1280` (`DETECTION_IMGSZ = 1280` in `config.py` & `track_v2.py`)
- Dataset Training Image Size: `512` / `640`

### 11. Current Augmentation Settings
- `mosaic`: `1.0`
- `scale`: `0.5`
- `translate`: `0.1`
- `fliplr`: `0.5`
- `hsv_h`: `0.015`, `hsv_s`: `0.7`, `hsv_v`: `0.4`
- `erasing`: `0.4`

### 12. Existing Model Evaluation Metrics
- Ground-Level Test Split Metrics:
  - Precision: `0.6995`
  - Recall: `0.3321`
  - mAP50: `0.3264`
  - mAP50-95: `0.1978`
- Aerial Video (`traffic.mp4`) Performance:
  - Baseline `yolov8s.pt` (COCO): 22,457 total detections (~44.91 vehicles/frame)
  - Fine-Tuned YOLO26 (`best.pt`): **0 detections** at `conf=0.45`

### 13. Existing Output Directories
- Output Directory: `ml/outputs/detection_training/`
  - `baseline/`
  - `dataset/`
  - `training/`
  - `evaluation/`
  - `models/`
  - `our_video_frames/`
  - `visualizations/`
