# RoadSentinel AI — Aerial Traffic Risk Management System

[![YOLO26](https://img.shields.io/badge/Detector-YOLO26_Small-indigo.svg)](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/yolo26s.pt)
[![Django](https://img.shields.io/badge/Backend-Django_4.2_DRF-green.svg)](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/backend/)
[![React](https://img.shields.io/badge/Frontend-React_18_TypeScript-blue.svg)](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/frontend/)
[![Accuracy](https://img.shields.io/badge/System_Accuracy-95.59%25-emerald.svg)](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/docs/integration_report.md)

RoadSentinel AI is a complete, enterprise-grade aerial traffic risk analysis platform. It combines a frozen, high-resolution Computer Vision & Machine Learning engine (**YOLO26 Small + ByteTrack + Physics Risk Engine + XGBoost Risk Classifier**) with a Django REST Framework backend, PostgreSQL database, Celery async task queue, and React TypeScript frontend.

---

## 🌟 Key Architecture Features

- **Official Object Detector**: **YOLO26 Small (`yolo26s.pt`)** running at 1280px resolution for high-angle 4K aerial traffic footage (92.04 vehicle detections/frame).
- **Multi-Object Tracking & Trajectory Analysis**: ByteTrack tracker + stationary displacement filter + causal trailing window trajectory smoothing.
- **Physics Rule Engine & XGBoost Classifier**: Distance-gated sub-scoring (TTC, PET, closing speed, relative heading) + XGBoost ML Classifier (**95.59% accuracy**).
- **Django REST Framework Backend**: 10 Django apps providing REST APIs, Token/JWT authentication, Role-Based Access Control (`TRAFFIC_OPERATOR`, `ADMINISTRATOR`, `GOVERNMENT_AUTHORITY`), and database ORM ingestion.
- **Asynchronous Processing**: Celery + Redis task queue for non-blocking ML pipeline execution.
- **React 18 + TypeScript Frontend**: Vite-powered SPA featuring an interactive HTML5 video analysis workspace, timeline seeking markers, filterable risk event directory, and PDF/CSV report exports.

---

## 🚀 Quick Start Guide

### 1. Environment & Setup
```bash
# Clone repository & copy environment configuration
cp .env.example .env

# Install backend Python dependencies
pip install -r ml/requirements.txt django djangorestframework django-cors-headers celery redis psycopg2-binary
```

### 2. Database Migrations & Initial User Seeding
```bash
# Apply Django migrations
python3 backend/manage.py migrate

# Seed initial demo accounts for all 3 roles
python3 backend/manage.py seed_users
```

### 3. Run Backend API Server
```bash
python3 backend/manage.py runserver 0.0.0.0:8000
```

### 4. Run Celery Worker (In a new terminal)
```bash
celery -A roadsentinel_backend worker --loglevel=info
```

### 5. Launch React Frontend App (In a new terminal)
```bash
cd frontend
npm install
npm run dev
```

Open your browser at `http://localhost:5173`.

---

## 🔑 Initial Demo Accounts

| Role | Username | Password | Permissions |
| :--- | :--- | :--- | :--- |
| **Traffic Operator** | `operator` | `operator123` | Upload videos, view analysis, dashboard, export reports |
| **Administrator** | `admin` | `admin123` | Full system control, user admin, job inspection |
| **Government Authority** | `authority` | `authority123` | Executive dashboard, risk timeline, recommendations |

---

## 📄 Documentation Links

- [docs/integration_report.md](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/docs/integration_report.md) — Final Integration & Acceptance Report
- [docs/architecture.md](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/docs/architecture.md) — System Architecture Specification
- [docs/current_architecture.md](file:///Users/pranav1718/Big%20One/RoadSentimentalAI/docs/current_architecture.md) — Baseline ML Pipeline Architecture
