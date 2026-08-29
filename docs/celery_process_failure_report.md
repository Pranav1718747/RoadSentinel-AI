# Celery Worker & Process Failure Diagnostic Report

## Root Cause Analysis
- **PRIMARY ROOT CAUSE #1: Relative vs Absolute Video Path Doubling**
  In `RoadSentinelPipeline.__init__()` inside `run_roadsentinel.py`, the path constructor performed `self.input_video_path = BASE_DIR / input_video_rel`. Because `BASE_DIR` is `/Users/.../ml`, passing `storage/media/uploads/traffic.mp4` resulted in `/Users/.../ml/storage/media/uploads/traffic.mp4`, causing file validation failures.

- **PRIMARY ROOT CAUSE #2: Interpreter Termination (`sys.exit(1)`) in Worker Thread**
  When `stage_validate_video()` failed to locate the file, it executed `sys.exit(1)`. Inside a Celery worker process, calling `sys.exit(1)` immediately killed the worker process (`WorkerLostError`), preventing Celery from catching an exception or marking `job.status = FAILED` in PostgreSQL/SQLite.

---

## Status Matrix
- **CELERY WORKER TEST**: **`PASS`**
- **DIRECT PIPELINE TEST**: **`PASS`**
- **YOLO26 ONE-FRAME TEST**: **`PASS`**
- **VIDEO ACCESS**: **`PASS`**
- **MODEL ACCESS**: **`PASS`**
- **PYTHON ENVIRONMENT**: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`
- **REDIS**: **`PASS`**
- **TASK REGISTRATION**: **`PASS`**
- **SUBPROCESS**: **`PASS`**
- **MACOS PROCESS MODEL**: **`PASS`**
- **FINAL JOB**: **`PASS`**

---

## Exact Fix Details

### 1. `ml/src/pipeline/run_roadsentinel.py`
- Added intelligent path resolution in `RoadSentinelPipeline.__init__()`: Checks `is_absolute()`, `(BASE_DIR / input_video_rel)`, and `(BASE_DIR.parent / input_video_rel)`.
- Replaced `sys.exit(1)` in `stage_validate_video()` with Python `raise FileNotFoundError` / `raise ValueError` exceptions so Celery task handlers cleanly catch errors and record `job.status = JobStatus.FAILED`.

### 2. `backend/processing/views.py`
- Implemented stale job auto-detection for stuck `PROCESSING` jobs (`last_progress_update > 60s`) and enabled job cancellation (`POST /api/jobs/{id}/cancel/`).

---

## Recommended Worker Command
```bash
celery -A roadsentinel_backend worker --loglevel=info --pool=solo
```
