# RoadSentinel AI — Current System Architecture Documentation

## Executive Summary
This document records the exact state, entry points, and component locations of the frozen ML/CV pipeline in **RoadSentinel AI** prior to building the Django + React application wrapper.

> [!IMPORTANT]
> **Frozen Pipeline Directive**: The ML/CV pipeline is 100% complete and frozen. No changes will be made to YOLO26 detection, ByteTrack tracking, trajectory extraction, physics risk rules, or XGBoost ML classifier logic during backend/frontend integration.

---

## 1. Frozen ML Pipeline Components & Entry Points

| Stage | Module Name | Primary File Location | Input / Output Schema |
| :--- | :--- | :--- | :--- |
| **Pipeline Master Entry** | `run_roadsentinel` | [run_roadsentinel.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/ml/src/pipeline/run_roadsentinel.py) | Full end-to-end pipeline orchestrator |
| **Hybrid ML Entry** | `run_hybrid` | [run_hybrid.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/ml/src/pipeline/run_hybrid.py) | Hybrid XGBoost + Rule Risk Orchestrator |
| **Central Config** | `config` | [config.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/ml/src/pipeline/config.py) | Pipeline hyperparameters (`YOLO_MODEL_NAME="yolo26s.pt"`, `imgsz=1280`, `conf=0.30`) |
| **YOLO26 Detector** | `YOLO26` Checkpoint | [yolo26s.pt](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/yolo26s.pt) | Official detector (`yolo26s.pt`, 19.5M params) |
| **ByteTrack Tracking** | `track_v2` | [track_v2.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/ml/src/tracking/track_v2.py) | Video → `tracks_v2.csv` (`frame_id`, `track_id`, `x1,y1,x2,y2`, `conf`) |
| **Stationary Filtering** | `extract_trajectories_v2` | [extract_trajectories_v2.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/ml/src/trajectories/extract_trajectories_v2.py) | Tracks → `validated_trajectories.csv` (Filters parked vehicles) |
| **Traffic Features** | `engineer_features` | [engineer_features.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/ml/src/features/engineer_features.py) | Trajectories → `traffic_features.csv` (TTC, PET, distance, closing speed, heading) |
| **Physics Risk Engine** | `risk_engine` | [risk_engine.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/ml/src/risk/risk_engine.py) | Features → `risk_events.csv` (Distance-gated sub-scoring) |
| **XGBoost ML Classifier** | `hybrid_risk_engine` | [hybrid_risk_engine.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/ml/src/risk/hybrid_risk_engine.py) | Features → `hybrid_risk_video.mp4` & `model_card.md` |
| **Event Grouper** | `event_grouper` | [event_grouper.py](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/ml/src/risk/event_grouper.py) | Groups contiguous high/medium risk interactions into events |

---

## 2. Model Checkpoints & Artifact Locations
- **Detector Checkpoint**: `/Users/pranav1718/Big One/RoadSentimentalAI/yolo26s.pt`
- **Aerial Checkpoint**: `/Users/pranav1718/Big One/RoadSentimentalAI/ml/outputs/detection_training/models/best_aerial.pt`
- **XGBoost Risk Model**: `/Users/pranav1718/Big One/RoadSentimentalAI/ml/outputs/hybrid/risk_model.pkl`
- **Sample Run Outputs**: `/Users/pranav1718/Big One/RoadSentimentalAI/ml/outputs/runs/run_20260824_214156/`

---

## 3. Core Architecture Target for Integration

```
                         [ React + TypeScript Frontend ]
                                       │
                                HTTP REST / SSE
                                       │
                         [ Django REST Framework API ]
                                       │
                        ┌──────────────┴──────────────┐
                        ▼                             ▼
              [ PostgreSQL Database ]         [ Celery + Redis ]
                                                      │
                                                      ▼
                                           [ PipelineService ]
                                                      │
                                                      ▼
                                           [ Frozen ML Pipeline ]
                                          (YOLO26 + XGBoost)
```
