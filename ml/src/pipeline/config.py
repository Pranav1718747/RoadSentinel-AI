"""
config.py — RoadSentinel AI, Phase 8
Central configuration file for the RoadSentinel end-to-end baseline pipeline.

SCIENTIFIC LIMITATION DISCLAIMER:
This pipeline implements the "RoadSentinel Rule-Based Traffic Risk Baseline".
It is NOT a trained predictive ML model.
Speed and distance are measured in image-plane quantities (pixels, px/s, seconds).
No camera calibration / homography to real-world metres or km/h has been applied.
Thresholds are initial baseline engineering assumptions and must be validated
against manually annotated ground-truth conflict data before making safety claims.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Default Inputs
# ──────────────────────────────────────────────────────────────
DEFAULT_INPUT_VIDEO = "data/samples/traffic.mp4"

# ──────────────────────────────────────────────────────────────
# Stage 1: Detection & Tracking Parameters
# ──────────────────────────────────────────────────────────────
YOLO_MODEL_NAME = "yolo26s.pt"
TRACKER_CONFIG  = "bytetrack.yaml"
DETECTION_CONF  = 0.45
DETECTION_IMGSZ = 1280
VEHICLE_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck

# ──────────────────────────────────────────────────────────────
# Stage 2: Trajectory & Stationary Filtering Parameters
# ──────────────────────────────────────────────────────────────
MIN_OBS               = 15       # Minimum observations to avoid SHORT track classification
MIN_AVG_DISPLACEMENT  = 1.5      # px/frame — minimum average displacement for MOVING
MIN_MOVEMENT_RATIO    = 0.35     # fraction of moves >= MOVE_STEP
MOVE_STEP             = 2.0      # px — per-frame displacement threshold
MIN_TOTAL_DISP        = 50.0     # px — minimum path length
MAX_GAPS_ALLOWED      = 20       # max missing frame gaps allowed
MAX_GAP_SIZE          = 30       # max single gap size in frames
MAX_JITTER_STD        = 8.0      # std dev threshold for NOISY
MAX_JUMP_PX           = 300.0    # single-frame jump threshold for QUESTIONABLE
IOU_DUPLICATE_THRESH  = 0.50     # IoU threshold for tracker ID-switch candidate detection
IOU_DUPLICATE_FRAMES  = 5        # minimum IoU > 0.50 frames to trigger duplicate filtering

# ──────────────────────────────────────────────────────────────
# Stage 3: Feature Engineering Parameters
# ──────────────────────────────────────────────────────────────
CAUSAL_SMOOTH_WINDOW  = 5        # Causal trailing window size for smoothing (past frames only)
SPEED_SPIKE_THRESHOLD = 500.0    # px/s — flag threshold for speed spikes
ACCEL_SPIKE_THRESHOLD = 500.0    # px/s² — flag threshold for acceleration spikes
TTC_CAP               = 30.0     # seconds — max valid TTC value
PROXIMITY_THRESHOLD   = 200.0    # px — proximity distance for interaction labeling
CONSISTENCY_WINDOW    = 10       # rolling window for movement consistency

# ──────────────────────────────────────────────────────────────
# Stage 4: Risk Engine & Event Grouping Parameters
# ──────────────────────────────────────────────────────────────
TTC_HIGH              = 3.5      # seconds — high TTC risk threshold
TTC_MEDIUM            = 5.0      # seconds — medium TTC risk threshold
DIST_HIGH             = 200.0    # px — high distance risk threshold
DIST_MEDIUM           = 450.0    # px — medium distance risk threshold
CS_HIGH               = 300.0    # px/s — high closing speed threshold
CS_MEDIUM             = 220.0    # px/s — medium closing speed threshold
HEADING_HIGH          = 120.0    # degrees — high heading divergence
HEADING_MEDIUM        = 60.0     # degrees — medium heading divergence

WEIGHT_TTC            = 0.35
WEIGHT_DISTANCE       = 0.25
WEIGHT_CLOSING        = 0.20
WEIGHT_TRAJECTORY     = 0.10
WEIGHT_HEADING        = 0.10

SCORE_HIGH            = 0.55
SCORE_MEDIUM          = 0.25

DISTANCE_GATING_THRESHOLD = 600.0  # px — gating threshold: f(dist)=max(0, 1-dist/600)
GROUP_GAP_FRAMES      = 15       # max frame gap for event grouping
MIN_EVENT_FRAMES      = 2        # min frames for valid event

# ──────────────────────────────────────────────────────────────
# Run Directory Path Generator
# ──────────────────────────────────────────────────────────────
def get_run_paths(run_id, base_dir=None):
    """
    Generate isolated output directories for a specific run_id.
    Prevents overwriting previous runs.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
    
    run_dir = base_dir / "outputs" / "runs" / run_id
    
    paths = {
        "run_dir"        : run_dir,
        "detection_dir"  : run_dir / "detection",
        "tracking_dir"   : run_dir / "tracking",
        "traj_dir"       : run_dir / "trajectories",
        "feat_dir"       : run_dir / "features",
        "risk_dir"       : run_dir / "risk",
        "vis_dir"        : run_dir / "visualization",
        "reports_dir"    : run_dir / "reports",

        # File outputs
        "tracks_csv"     : run_dir / "tracking"     / "tracks_v2.csv",
        "tracked_video"  : run_dir / "tracking"     / "tracked_traffic_v2.mp4",
        "traj_csv"       : run_dir / "trajectories" / "trajectories.csv",
        "val_traj_csv"   : run_dir / "trajectories" / "validated_trajectories.csv",
        "traj_report"    : run_dir / "trajectories" / "trajectory_quality_report.json",
        "feat_csv"       : run_dir / "features"     / "traffic_features.csv",
        "feat_summary"   : run_dir / "features"     / "feature_summary.json",
        "risk_csv"       : run_dir / "risk"         / "risk_events.csv",
        "risk_summary"   : run_dir / "risk"         / "risk_summary.json",
        "risk_json"      : run_dir / "risk"         / "risk_events.json",
        "final_video"    : run_dir / "visualization"/ "roadsentinel_final.mp4",
        "manifest_json"  : run_dir / "reports"      / "pipeline_manifest.json",
        "report_json"    : run_dir / "reports"      / "roadsentinel_report.json",
        "report_txt"     : run_dir / "reports"      / "roadsentinel_report.txt",
    }

    # Ensure all directories exist
    for k, p in paths.items():
        if k.endswith("_dir") or k == "run_dir":
            p.mkdir(parents=True, exist_ok=True)

    return paths
