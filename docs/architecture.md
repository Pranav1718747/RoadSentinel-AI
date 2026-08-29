# RoadSentinel AI — System Architecture Specification

## Overview
RoadSentinel AI is an end-to-end aerial traffic risk analysis platform integrating a frozen Computer Vision / Machine Learning pipeline (YOLO26 + ByteTrack + Physics Risk Rules + XGBoost Risk Classifier) with a robust Django REST backend, PostgreSQL database, Celery async task queue, Redis broker, and React TypeScript frontend.

```
                          [ React 18 + TypeScript Frontend ]
                                          │
                                   HTTP REST / JWT
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

## System Layers
1. **Frontend**: SPA built with React 18, TypeScript, Vite, and Lucide icons.
2. **Backend API**: Django 4.2 REST Framework providing JWT/Token auth, role permissions, pagination, and OpenAPI schemas.
3. **Async Processing**: Celery worker executing `PipelineService` in background without blocking HTTP requests.
4. **Database**: PostgreSQL storing users, videos, processing jobs, detections, tracks, trajectories, interactions, and risk events.
5. **Frozen ML Engine**: YOLO26 Small (`yolo26s.pt` @ 1280px) + ByteTrack + Trajectory Filter + Causal Feature Engineering + Distance-Gated Physics Sub-scoring + XGBoost ML Classifier.
