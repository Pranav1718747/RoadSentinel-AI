"""
Phase 6 — Traffic Feature Engineering
RoadSentinel AI

Reads:
  ml/outputs/trajectories/validated_trajectories.csv
  ml/outputs/tracking/tracks_v2.csv        (raw bounding boxes for video)
  ml/data/samples/traffic.mp4             (original video for annotation)

Outputs:
  ml/outputs/features/traffic_features.csv
  ml/outputs/features/feature_summary.json
  ml/outputs/features/feature_visualization.mp4
  ml/outputs/features/speed_distribution.png
  ml/outputs/features/ttc_distribution.png
  ml/outputs/features/distance_distribution.png

TTC Assumptions (documented here and in output JSON):
  1. TTC is computed ONLY when closing_speed > 0 (vehicles are closing).
  2. If closing_speed <= 0 (separating or parallel), TTC = NaN — not a risk.
  3. TTC = distance / closing_speed — constant-velocity snapshot estimate.
  4. Distance and speed are image-plane quantities (pixels, px/s). Not km/h.
  5. TTC > TTC_CAP seconds is treated as NaN (not a meaningful near-miss).
  6. Zero closing speed → TTC = NaN (guard against division by zero).
"""

import csv
import json
import math
import sys
import itertools
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# Configurable Constants
# ─────────────────────────────────────────────────────────────
SMOOTH_WINDOW          = 5      # speed smoothing window (frames)
SPEED_SPIKE_THRESHOLD  = 500.0  # px/s  — flag as suspect speed
ACCEL_SPIKE_THRESHOLD  = 500.0  # px/s² — flag as suspect acceleration
TTC_CAP                = 30.0   # seconds — TTC beyond this = NaN
PROXIMITY_THRESHOLD    = 200.0  # px — threshold for APPROACHING interaction type
CONSISTENCY_WINDOW     = 10     # frames — rolling window for movement consistency
MIN_DT                 = 1e-6   # seconds — guard against zero time differences

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
IN_TRAJ    = BASE_DIR / "outputs" / "trajectories" / "validated_trajectories.csv"
IN_BBOXES  = BASE_DIR / "outputs" / "tracking"     / "tracks_v2.csv"
IN_VIDEO   = BASE_DIR / "data"    / "samples"      / "traffic.mp4"
OUT_DIR    = BASE_DIR / "outputs" / "features"

OUT_CSV    = OUT_DIR / "traffic_features.csv"
OUT_JSON   = OUT_DIR / "feature_summary.json"
OUT_VID    = OUT_DIR / "feature_visualization.mp4"
OUT_SPEED  = OUT_DIR / "speed_distribution.png"
OUT_TTC    = OUT_DIR / "ttc_distribution.png"
OUT_DIST   = OUT_DIR / "distance_distribution.png"

NaN = float("nan")

# Vivid per-track colour palette (BGR)
TRACK_PALETTE_BGR = [
    (86,  180, 233),
    (230, 159,   0),
    (0,   158, 115),
    (204, 121, 167),
    (0,   114, 178),
    (213,  94,   0),
    (0,   255, 200),
    (255, 100, 100),
    (50,  200, 255),
    (255, 200,  50),
]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def is_nan(v):
    return v != v

def safe_div(a, b, fallback=NaN):
    """Divide a/b, returning fallback if b is zero or NaN."""
    if is_nan(a) or is_nan(b) or abs(b) < MIN_DT:
        return fallback
    return a / b

def angle_deg(dx, dy):
    """Heading angle in degrees [0, 360). Returns NaN if zero vector."""
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return NaN
    return math.degrees(math.atan2(dy, dx)) % 360.0

def angle_diff(a1, a2):
    """Signed angular difference a1-a2 wrapped to [-180, 180]. Returns NaN if either is NaN."""
    if is_nan(a1) or is_nan(a2):
        return NaN
    diff = (a1 - a2 + 180.0) % 360.0 - 180.0
    return diff

def rolling_mean(values, window):
    """
    Causal trailing moving average (window size = window).
    At index i (time T), only uses past and current values: [max(0, i - window + 1), i + 1).
    Never uses future frames (T+1, T+2, etc.) to eliminate lookahead data leakage.
    For window=5 at frame T, uses frames T, T-1, T-2, T-3, T-4.
    """
    result = []
    n = len(values)
    for i in range(n):
        lo = max(0, i - window + 1)
        hi = i + 1
        segment = [v for v in values[lo:hi] if not is_nan(v)]
        result.append(sum(segment) / len(segment) if segment else NaN)
    return result

def std_dev(values):
    valid = [v for v in values if not is_nan(v)]
    if len(valid) < 2:
        return NaN
    m = sum(valid) / len(valid)
    return math.sqrt(sum((v - m) ** 2 for v in valid) / len(valid))

def segments_intersect(p1, p2, p3, p4):
    """
    Test whether line segment p1→p2 intersects p3→p4.
    Uses 2-D cross-product / parametric form.
    Returns True if they properly intersect (not just touch at endpoints).
    """
    def cross2d(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross2d(p3, p4, p1)
    d2 = cross2d(p3, p4, p2)
    d3 = cross2d(p1, p2, p3)
    d4 = cross2d(p1, p2, p4)

    if ((d1 > 0 < d2) or (d1 < 0 > d2)) and ((d3 > 0 < d4) or (d3 < 0 > d4)):
        return True
    return False


# ─────────────────────────────────────────────────────────────
# Step 1 — Load validated trajectories
# ─────────────────────────────────────────────────────────────
def load_trajectories(path):
    """Return dict: track_id → list of row dicts sorted by frame_id."""
    if not path.exists():
        print(f"Error: {path} not found.")
        sys.exit(1)
    raw = defaultdict(list)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tid = int(row["track_id"])
            raw[tid].append({
                "track_id"   : tid,
                "class_name" : row["class_name"].strip(),
                "frame_id"   : int(row["frame_id"]),
                "timestamp"  : float(row["timestamp"]),
                "center_x"   : float(row["center_x"]),
                "center_y"   : float(row["center_y"]),
                "smooth_x"   : float(row["smooth_x"]),
                "smooth_y"   : float(row["smooth_y"]),
                "confidence" : float(row["confidence"]),
            })
    for tid in raw:
        raw[tid].sort(key=lambda r: r["frame_id"])
    return dict(raw)


def load_raw_bboxes(path):
    """Return dict: (track_id, frame_id) → (x1, y1, x2, y2)."""
    if not path.exists():
        return {}
    bboxes = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (int(row["track_id"]), int(row["frame_id"]))
            bboxes[key] = (
                float(row["x1"]), float(row["y1"]),
                float(row["x2"]), float(row["y2"]),
            )
    return bboxes


# ─────────────────────────────────────────────────────────────
# Step 2 — Per-vehicle kinematics
# ─────────────────────────────────────────────────────────────
def compute_kinematics(tracks):
    """
    For each track, compute per-frame kinematic features.
    Returns dict: track_id → list of enriched dicts (one per observation).
    Uses smooth_x/smooth_y for velocity/heading, retains raw center_x/y.
    """
    enriched = {}
    sanity_errors = []

    for tid, obs in tracks.items():
        n = len(obs)

        # Raw displacement and dt lists (index i corresponds to the TRANSITION i→i+1)
        displacements = []
        dts           = []
        directions    = []
        speeds_raw    = []

        for i in range(n):
            if i == 0:
                displacements.append(NaN)
                dts.append(NaN)
                directions.append(NaN)
                speeds_raw.append(NaN)
            else:
                prev = obs[i - 1]
                curr = obs[i]
                dx   = curr["smooth_x"] - prev["smooth_x"]
                dy   = curr["smooth_y"] - prev["smooth_y"]
                dt   = curr["timestamp"] - prev["timestamp"]
                disp = math.sqrt(dx * dx + dy * dy)

                if dt <= 0:
                    sanity_errors.append(
                        f"ERROR: track {tid} frame {curr['frame_id']} — dt={dt:.6f}s <= 0"
                    )
                    displacements.append(NaN)
                    dts.append(NaN)
                    directions.append(NaN)
                    speeds_raw.append(NaN)
                else:
                    displacements.append(disp)
                    dts.append(dt)
                    directions.append(angle_deg(dx, dy))
                    speeds_raw.append(disp / dt)

        # Smoothed speed
        speeds_smooth = rolling_mean(speeds_raw, SMOOTH_WINDOW)

        # Acceleration (on smoothed speed)
        accels = [NaN] * n
        for i in range(1, n):
            dt = dts[i]
            if not is_nan(dt) and dt > MIN_DT:
                ds = speeds_smooth[i] - speeds_smooth[i - 1]
                accels[i] = ds / dt
            # else remains NaN

        # Heading change
        heading_changes = [NaN] * n
        for i in range(1, n):
            heading_changes[i] = angle_diff(directions[i], directions[i - 1])

        # Cumulative trajectory length
        traj_lengths = [0.0]
        for i in range(1, n):
            prev_len = traj_lengths[-1]
            d = displacements[i]
            traj_lengths.append(prev_len + (0.0 if is_nan(d) else d))

        # Net displacement from start
        sx0, sy0 = obs[0]["smooth_x"], obs[0]["smooth_y"]
        net_disps = []
        for o in obs:
            net_disps.append(math.sqrt(
                (o["smooth_x"] - sx0) ** 2 + (o["smooth_y"] - sy0) ** 2
            ))

        # Movement consistency — rolling std/mean of displacements
        consistency = [NaN] * n
        for i in range(n):
            lo = max(0, i - CONSISTENCY_WINDOW // 2)
            hi = min(n, i + CONSISTENCY_WINDOW // 2 + 1)
            window_disps = [displacements[j] for j in range(lo, hi) if not is_nan(displacements[j])]
            if len(window_disps) >= 2:
                m = sum(window_disps) / len(window_disps)
                if m > 1e-6:
                    consistency[i] = std_dev(window_disps) / m
                else:
                    consistency[i] = NaN

        # Speed validity flag
        speed_valid = [True] * n
        for i in range(n):
            sr = speeds_raw[i]
            if not is_nan(sr) and sr > SPEED_SPIKE_THRESHOLD:
                speed_valid[i] = False

        # Acceleration validity flag
        accel_valid = [True] * n
        for i in range(n):
            a = accels[i]
            if not is_nan(a) and abs(a) > ACCEL_SPIKE_THRESHOLD:
                accel_valid[i] = False

        rows = []
        for i, o in enumerate(obs):
            rows.append({
                # Identity
                "track_id"             : tid,
                "class_name"           : o["class_name"],
                "frame_id"             : o["frame_id"],
                "timestamp"            : o["timestamp"],
                # Position (raw and smooth)
                "center_x"             : o["center_x"],
                "center_y"             : o["center_y"],
                "smooth_x"             : o["smooth_x"],
                "smooth_y"             : o["smooth_y"],
                # Kinematics
                "displacement"         : displacements[i],
                "speed"                : speeds_raw[i],
                "smoothed_speed"       : speeds_smooth[i],
                "acceleration"         : accels[i],
                "direction"            : directions[i],
                "heading_change"       : heading_changes[i],
                "trajectory_length"    : traj_lengths[i],
                "net_displacement"     : net_disps[i],
                "movement_consistency" : consistency[i],
                # Quality flags
                "speed_valid"          : speed_valid[i],
                "accel_valid"          : accel_valid[i],
            })
        enriched[tid] = rows

    return enriched, sanity_errors


# ─────────────────────────────────────────────────────────────
# Step 3 — Build frame lookup for efficient interaction calc
# ─────────────────────────────────────────────────────────────
def build_frame_lookup(enriched):
    """Return dict: frame_id → {track_id: kinematic_row}."""
    lookup = defaultdict(dict)
    for tid, rows in enriched.items():
        for row in rows:
            lookup[row["frame_id"]][tid] = row
    return dict(lookup)


# ─────────────────────────────────────────────────────────────
# Step 4 — Trajectory path crossing detection
# ─────────────────────────────────────────────────────────────
def detect_path_crossings(enriched):
    """
    For each pair of tracks, test whether their smooth paths ever cross.
    Returns dict: (tid1, tid2) → True/False.
    Uses segment intersection over the full track paths.
    """
    crossings = {}
    tids = list(enriched.keys())
    for tid1, tid2 in itertools.combinations(tids, 2):
        path1 = [(r["smooth_x"], r["smooth_y"]) for r in enriched[tid1]]
        path2 = [(r["smooth_x"], r["smooth_y"]) for r in enriched[tid2]]
        crossed = False
        for i in range(len(path1) - 1):
            for j in range(len(path2) - 1):
                if segments_intersect(path1[i], path1[i+1], path2[j], path2[j+1]):
                    crossed = True
                    break
            if crossed:
                break
        key = (min(tid1, tid2), max(tid1, tid2))
        crossings[key] = crossed
    return crossings


# ─────────────────────────────────────────────────────────────
# Step 5 — Interaction features per frame pair
# ─────────────────────────────────────────────────────────────
def compute_interaction(r1, r2, crossing):
    """
    Compute all vehicle-to-vehicle interaction features for one frame.
    r1, r2: kinematic row dicts for the two vehicles.
    crossing: bool — do their full trajectories cross?
    Returns a dict of features.
    """
    # Positions (smooth)
    x1, y1 = r1["smooth_x"], r1["smooth_y"]
    x2, y2 = r2["smooth_x"], r2["smooth_y"]

    dx = x2 - x1
    dy = y2 - y1
    distance = math.sqrt(dx * dx + dy * dy)

    # Sanity check
    if distance < 0:
        # Should never happen with Euclidean distance, but guard anyway
        distance = NaN

    # Velocities in px/s (use smoothed_speed × direction unit vector)
    def vel_components(row):
        spd = row["smoothed_speed"]
        d   = row["direction"]
        if is_nan(spd) or is_nan(d):
            return NaN, NaN
        rad = math.radians(d)
        return spd * math.cos(rad), spd * math.sin(rad)

    vx1, vy1 = vel_components(r1)
    vx2, vy2 = vel_components(r2)

    rel_vx = NaN if (is_nan(vx1) or is_nan(vx2)) else vx1 - vx2
    rel_vy = NaN if (is_nan(vy1) or is_nan(vy2)) else vy1 - vy2

    relative_speed = NaN
    if not is_nan(rel_vx) and not is_nan(rel_vy):
        relative_speed = math.sqrt(rel_vx ** 2 + rel_vy ** 2)

    # Closing speed: project relative velocity onto unit vector from v2→v1
    closing_speed = NaN
    if not is_nan(distance) and distance > 1e-6 and not is_nan(rel_vx) and not is_nan(rel_vy):
        # unit vector from v2 to v1
        ux = (x1 - x2) / distance
        uy = (y1 - y2) / distance
        closing_speed = rel_vx * ux + rel_vy * uy

    approaching = False if is_nan(closing_speed) else closing_speed > 0

    # TTC — only if approaching
    ttc = NaN
    if approaching and not is_nan(closing_speed) and closing_speed > MIN_DT and not is_nan(distance):
        raw_ttc = distance / closing_speed
        ttc = raw_ttc if raw_ttc <= TTC_CAP else NaN

    # Heading difference
    d1 = r1["direction"]
    d2 = r2["direction"]
    heading_diff = abs(angle_diff(d1, d2)) if (not is_nan(d1) and not is_nan(d2)) else NaN

    # Lateral and longitudinal separation (relative to vehicle 1's heading)
    lat_sep = NaN
    lon_sep = NaN
    if not is_nan(d1) and not is_nan(distance):
        rad1 = math.radians(d1)
        fwd_x, fwd_y = math.cos(rad1), math.sin(rad1)
        # Vector from v1 to v2
        to_v2_x, to_v2_y = dx, dy
        lon_sep = to_v2_x * fwd_x + to_v2_y * fwd_y   # +ve = ahead
        lat_sep = to_v2_x * (-fwd_y) + to_v2_y * fwd_x  # +ve = left

    # Interaction type
    if is_nan(distance):
        interaction_type = "UNKNOWN"
    elif approaching and distance < PROXIMITY_THRESHOLD:
        interaction_type = "APPROACHING"
    elif crossing:
        interaction_type = "CROSSING"
    elif not is_nan(closing_speed) and closing_speed < -0.5:
        interaction_type = "DIVERGING"
    elif not is_nan(heading_diff) and heading_diff < 30.0:
        interaction_type = "PARALLEL"
    else:
        interaction_type = "NO_INTERACTION"

    return {
        "distance"             : distance,
        "rel_vx"               : rel_vx,
        "rel_vy"               : rel_vy,
        "relative_speed"       : relative_speed,
        "closing_speed"        : closing_speed,
        "ttc"                  : ttc,
        "approaching"          : approaching,
        "heading_difference"   : heading_diff,
        "lateral_sep"          : lat_sep,
        "longitudinal_sep"     : lon_sep,
        "trajectory_crossing"  : crossing,
        "interaction_type"     : interaction_type,
    }


# ─────────────────────────────────────────────────────────────
# Step 6 — Build the full feature table
# ─────────────────────────────────────────────────────────────
def build_feature_table(frame_lookup, crossings, enriched):
    """
    Iterate over every frame, every ordered pair of co-visible tracks.
    Returns list of feature row dicts.
    """
    feature_rows = []
    all_frames = sorted(frame_lookup.keys())

    for frame_id in all_frames:
        present = frame_lookup[frame_id]  # {track_id: kinematic_row}
        tids = sorted(present.keys())
        if len(tids) < 2:
            continue

        for tid1, tid2 in itertools.combinations(tids, 2):
            r1 = present[tid1]
            r2 = present[tid2]
            key = (min(tid1, tid2), max(tid1, tid2))
            crossing = crossings.get(key, False)
            inter = compute_interaction(r1, r2, crossing)

            def fmt(v):
                if is_nan(v):
                    return ""
                return v

            row = {
                "timestamp"            : r1["timestamp"],
                "frame_id"             : frame_id,
                # Vehicle 1
                "v1_id"                : tid1,
                "v1_class"             : r1["class_name"],
                "v1_x"                 : r1["smooth_x"],
                "v1_y"                 : r1["smooth_y"],
                "v1_speed"             : fmt(r1["smoothed_speed"]),
                "v1_acceleration"      : fmt(r1["acceleration"]),
                "v1_direction"         : fmt(r1["direction"]),
                "v1_speed_valid"       : r1["speed_valid"],
                # Vehicle 2
                "v2_id"                : tid2,
                "v2_class"             : r2["class_name"],
                "v2_x"                 : r2["smooth_x"],
                "v2_y"                 : r2["smooth_y"],
                "v2_speed"             : fmt(r2["smoothed_speed"]),
                "v2_acceleration"      : fmt(r2["acceleration"]),
                "v2_direction"         : fmt(r2["direction"]),
                "v2_speed_valid"       : r2["speed_valid"],
                # Interaction
                "distance"             : inter["distance"],
                "rel_vx"               : fmt(inter["rel_vx"]),
                "rel_vy"               : fmt(inter["rel_vy"]),
                "relative_speed"       : fmt(inter["relative_speed"]),
                "closing_speed"        : fmt(inter["closing_speed"]),
                "ttc"                  : fmt(inter["ttc"]),
                "approaching"          : inter["approaching"],
                "heading_difference"   : fmt(inter["heading_difference"]),
                "lateral_sep"          : fmt(inter["lateral_sep"]),
                "longitudinal_sep"     : fmt(inter["longitudinal_sep"]),
                "trajectory_crossing"  : inter["trajectory_crossing"],
                "interaction_type"     : inter["interaction_type"],
            }
            feature_rows.append(row)

    return feature_rows


# ─────────────────────────────────────────────────────────────
# Step 7 — Write CSV
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "timestamp", "frame_id",
    "v1_id", "v1_class", "v1_x", "v1_y", "v1_speed", "v1_acceleration", "v1_direction", "v1_speed_valid",
    "v2_id", "v2_class", "v2_x", "v2_y", "v2_speed", "v2_acceleration", "v2_direction", "v2_speed_valid",
    "distance", "rel_vx", "rel_vy", "relative_speed", "closing_speed", "ttc",
    "approaching", "heading_difference", "lateral_sep", "longitudinal_sep",
    "trajectory_crossing", "interaction_type",
]

def write_feature_csv(rows):
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FEATURE_COLS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in FEATURE_COLS})


# ─────────────────────────────────────────────────────────────
# Step 8 — Sanity checks
# ─────────────────────────────────────────────────────────────
def run_sanity_checks(feature_rows, pre_errors):
    print("\nRunning sanity checks...")
    errors = list(pre_errors)  # dt errors from kinematics phase
    warnings = []

    for i, row in enumerate(feature_rows):
        dist = row.get("distance", NaN)
        cs   = row.get("closing_speed", "")
        ttc  = row.get("ttc", "")
        app  = row.get("approaching", False)

        # Negative distance
        if not is_nan(dist) and dist < 0:
            errors.append(f"ERROR row {i}: negative distance={dist}")

        # TTC when separating
        cs_val = cs if isinstance(cs, float) else (float(cs) if cs != "" else NaN)
        ttc_val = ttc if isinstance(ttc, float) else (float(ttc) if ttc != "" else NaN)

        if not is_nan(ttc_val) and not app:
            errors.append(f"ERROR row {i}: TTC={ttc_val:.2f} computed but approaching=False")

        if not is_nan(ttc_val) and not is_nan(cs_val) and cs_val <= 0:
            errors.append(f"ERROR row {i}: TTC={ttc_val:.2f} but closing_speed={cs_val:.4f} <= 0")

        # Negative TTC
        if not is_nan(ttc_val) and ttc_val < 0:
            errors.append(f"ERROR row {i}: negative TTC={ttc_val:.4f}")

    if errors:
        print(f"  {len(errors)} ERROR(s) found:")
        for e in errors[:20]:
            print(f"    {e}")
        if len(errors) > 20:
            print(f"    ... and {len(errors)-20} more")
    else:
        print("  No errors detected.")

    if warnings:
        for w in warnings[:10]:
            print(f"  {w}")

    return errors, warnings


# ─────────────────────────────────────────────────────────────
# Step 9 — Feature summary JSON
# ─────────────────────────────────────────────────────────────
def build_summary(enriched, feature_rows, crossings, errors, warnings):
    # Collect numeric values for statistics
    def collect(field):
        out = []
        for row in feature_rows:
            v = row.get(field, "")
            if v == "" or v is None:
                continue
            try:
                fv = float(v)
                if not is_nan(fv):
                    out.append(fv)
            except (ValueError, TypeError):
                pass
        return out

    def stats(values):
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
        n = len(values)
        mn = min(values)
        mx = max(values)
        mean = sum(values) / n
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / n) if n > 1 else 0.0
        return {"count": n, "min": round(mn, 4), "max": round(mx, 4),
                "mean": round(mean, 4), "std": round(sd, 4)}

    distances   = collect("distance")
    speeds      = collect("v1_speed") + collect("v2_speed")
    accels      = collect("v1_acceleration") + collect("v2_acceleration")
    ttcs        = collect("ttc")
    closing_sps = collect("closing_speed")

    n_approaching   = sum(1 for r in feature_rows if r.get("approaching") is True)
    n_crossing_rows = sum(1 for r in feature_rows if r.get("trajectory_crossing") is True)
    n_unique_crossing_pairs = sum(1 for v in crossings.values() if v)
    unique_pairs = len(set(
        (min(r["v1_id"], r["v2_id"]), max(r["v1_id"], r["v2_id"]))
        for r in feature_rows
    ))

    summary = {
        "ttc_assumptions": {
            "method"          : "distance / closing_speed (constant-velocity snapshot)",
            "units"           : "image-plane pixels and px/s — NOT km/h",
            "ttc_cap_seconds" : TTC_CAP,
            "when_computed"   : "ONLY when closing_speed > 0 (vehicles approaching)",
            "when_nan"        : ["closing_speed <= 0", "closing_speed near zero",
                                 "distance is NaN", "raw TTC > TTC_CAP"],
        },
        "summary": {
            "vehicle_tracks_analyzed" : len(enriched),
            "unique_interaction_pairs": unique_pairs,
            "total_feature_rows"      : len(feature_rows),
            "valid_ttc_rows"          : len(ttcs),
            "approaching_interactions": n_approaching,
            "trajectory_crossings"    : n_unique_crossing_pairs,
            "sanity_errors"           : len(errors),
            "sanity_warnings"         : len(warnings),
        },
        "distance_stats"      : stats(distances),
        "speed_stats"         : stats(speeds),
        "acceleration_stats"  : stats(accels),
        "ttc_stats"           : stats(ttcs),
        "closing_speed_stats" : stats(closing_sps),
    }

    with open(OUT_JSON, "w") as fh:
        json.dump(summary, fh, indent=2)

    return summary


# ─────────────────────────────────────────────────────────────
# Step 10 — Annotated video
# ─────────────────────────────────────────────────────────────
def produce_video(enriched, frame_lookup, raw_bboxes):
    if not IN_VIDEO.exists():
        print(f"  Warning: video not found at {IN_VIDEO}. Skipping.")
        return

    cap = cv2.VideoCapture(str(IN_VIDEO))
    if not cap.isOpened():
        print("  Error: could not open video. Skipping.")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wtr = cv2.VideoWriter(str(OUT_VID), fourcc, fps, (width, height))
    if not wtr.isOpened():
        print("  Error: could not open video writer. Skipping.")
        cap.release()
        return

    tid_list  = sorted(enriched.keys())
    tid_color = {tid: TRACK_PALETTE_BGR[i % len(TRACK_PALETTE_BGR)]
                 for i, tid in enumerate(tid_list)}

    # Build smooth position history per track for trails
    smooth_hist = {tid: [] for tid in tid_list}

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.45
    THICK      = 1

    frame_id = 0
    print(f"  Rendering feature video ({total} frames)...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            present = frame_lookup.get(frame_id, {})

            # Draw each track
            for tid in tid_list:
                color = tid_color[tid]
                if tid not in present:
                    continue
                kr = present[tid]
                sx, sy = kr["smooth_x"], kr["smooth_y"]
                smooth_hist[tid].append((int(sx), int(sy)))
                # Trail (last 30 points)
                trail = smooth_hist[tid][-30:]
                for i in range(1, len(trail)):
                    t = max(1, int((i / len(trail)) * 3))
                    cv2.line(frame, trail[i-1], trail[i], color, t)

                # Bounding box
                bbox = raw_bboxes.get((tid, frame_id))
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    spd = kr["smoothed_speed"]
                    spd_str = f"{spd:.1f}px/s" if not is_nan(spd) else "N/A"
                    label = f"ID{tid} {kr['class_name']} {spd_str}"
                    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, THICK)
                    ly = max(int(y1) - 4, th + 4)
                    cv2.rectangle(frame, (int(x1), ly - th - 3), (int(x1) + tw, ly + 2), color, -1)
                    cv2.putText(frame, label, (int(x1), ly), FONT, FONT_SCALE, (255, 255, 255), THICK)

            # Draw lines between approaching pairs with distance/TTC
            tids_here = sorted(present.keys())
            for tid1, tid2 in itertools.combinations(tids_here, 2):
                r1 = present[tid1]
                r2 = present[tid2]
                key = (min(tid1, tid2), max(tid1, tid2))
                inter = compute_interaction(r1, r2, False)  # crossings already separate
                if inter["approaching"] and not is_nan(inter["distance"]) and inter["distance"] < PROXIMITY_THRESHOLD * 3:
                    pt1 = (int(r1["smooth_x"]), int(r1["smooth_y"]))
                    pt2 = (int(r2["smooth_x"]), int(r2["smooth_y"]))
                    cv2.line(frame, pt1, pt2, (0, 0, 220), 1)
                    mid_x = (pt1[0] + pt2[0]) // 2
                    mid_y = (pt1[1] + pt2[1]) // 2
                    ttc_val = inter["ttc"]
                    dist_val = inter["distance"]
                    if not is_nan(ttc_val):
                        info = f"D:{dist_val:.0f}px TTC:{ttc_val:.1f}s"
                    else:
                        info = f"D:{dist_val:.0f}px"
                    cv2.putText(frame, info, (mid_x, mid_y),
                                FONT, 0.4, (0, 0, 220), 1)

            wtr.write(frame)
            if frame_id % 50 == 0:
                print(f"    Frame {frame_id}/{total}")
    finally:
        cap.release()
        wtr.release()


# ─────────────────────────────────────────────────────────────
# Step 11 — Distribution plots
# ─────────────────────────────────────────────────────────────
def make_hist(values, title, xlabel, color, out_path, vline=None):
    valid = [v for v in values if not is_nan(v)]
    if not valid:
        print(f"  Warning: no valid values for {title}. Skipping plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.hist(valid, bins=40, color=color, edgecolor="#222", alpha=0.85)
    if vline is not None:
        ax.axvline(vline, color="red", linewidth=1.5, linestyle="--", label=f"mean={vline:.2f}")
        ax.legend(facecolor="#161b22", labelcolor="white", fontsize=9)
    ax.set_title(title, color="white", fontsize=13)
    ax.set_xlabel(xlabel, color="#aaa")
    ax.set_ylabel("Count", color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


def produce_plots(feature_rows):
    def collect(field):
        out = []
        for row in feature_rows:
            v = row.get(field, "")
            if v == "" or v is None:
                continue
            try:
                fv = float(v)
                if not is_nan(fv):
                    out.append(fv)
            except (ValueError, TypeError):
                pass
        return out

    speeds = collect("v1_speed") + collect("v2_speed")
    ttcs   = collect("ttc")
    dists  = collect("distance")

    mean_spd  = sum(speeds) / len(speeds) if speeds else None
    mean_ttc  = sum(ttcs) / len(ttcs) if ttcs else None
    mean_dist = sum(dists) / len(dists) if dists else None

    make_hist(speeds, "RoadSentinel — Speed Distribution (image-plane px/s)",
              "Speed (px/s)", "#5b9cf6", OUT_SPEED, vline=mean_spd)
    make_hist(ttcs, "RoadSentinel — TTC Distribution (approaching pairs only)",
              "TTC (seconds)", "#f97316", OUT_TTC, vline=mean_ttc)
    make_hist(dists, "RoadSentinel — Inter-Vehicle Distance Distribution",
              "Distance (pixels)", "#22c55e", OUT_DIST, vline=mean_dist)


# ─────────────────────────────────────────────────────────────
# Step 12 — Terminal Report
# ─────────────────────────────────────────────────────────────
def print_terminal_report(summary, feature_rows):
    s = summary["summary"]
    ds = summary["distance_stats"]
    sp = summary["speed_stats"]
    tt = summary["ttc_stats"]

    W = 44
    print()
    print("=" * W)
    print("  ROAD SENTINEL — FEATURE ENGINEERING")
    print("=" * W)
    print(f"  Vehicle tracks analyzed   : {s['vehicle_tracks_analyzed']}")
    print(f"  Interaction pairs         : {s['unique_interaction_pairs']}")
    print(f"  Total feature rows        : {s['total_feature_rows']}")
    print()
    print(f"  Valid TTC rows            : {s['valid_ttc_rows']}")
    print(f"  Approaching interactions  : {s['approaching_interactions']}")
    print(f"  Trajectory crossings      : {s['trajectory_crossings']} pairs")
    print()
    if sp["count"] > 0:
        print(f"  Average speed             : {sp['mean']:.2f} px/s")
        print(f"  Maximum speed             : {sp['max']:.2f} px/s")
    if ds["count"] > 0:
        print(f"  Average distance          : {ds['mean']:.2f} px")
        print(f"  Minimum distance          : {ds['min']:.2f} px")
    if tt["count"] > 0:
        print(f"  Average TTC               : {tt['mean']:.2f} s")
        print(f"  Minimum TTC               : {tt['min']:.2f} s")
    print()
    print(f"  Sanity errors             : {s['sanity_errors']}")
    print(f"  Sanity warnings           : {s['sanity_warnings']}")
    print("=" * W)
    print()

    # Interaction type breakdown
    type_counts = defaultdict(int)
    for row in feature_rows:
        type_counts[row.get("interaction_type", "UNKNOWN")] += 1
    print("  Interaction type breakdown:")
    for itype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {itype:<20}: {cnt:>6}")
    print()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load
    print("Loading validated_trajectories.csv ...")
    tracks = load_trajectories(IN_TRAJ)
    print(f"  Loaded {len(tracks)} VALID_MOVING tracks.")

    print("Loading raw bounding boxes ...")
    raw_bboxes = load_raw_bboxes(IN_BBOXES)
    print(f"  Loaded {len(raw_bboxes)} bbox records.")

    # 2. Per-vehicle kinematics
    print("Computing per-vehicle kinematics ...")
    enriched, pre_errors = compute_kinematics(tracks)
    if pre_errors:
        for e in pre_errors:
            print(f"  {e}")

    # 3. Frame lookup
    print("Building frame lookup ...")
    frame_lookup = build_frame_lookup(enriched)
    print(f"  {len(frame_lookup)} frames with at least one track.")

    # 4. Path crossing detection
    print("Detecting trajectory path crossings ...")
    crossings = detect_path_crossings(enriched)
    n_cross = sum(1 for v in crossings.values() if v)
    print(f"  {n_cross} crossing pairs found among {len(crossings)} unique pairs.")

    # 5. Build interaction feature table
    print("Building interaction feature table ...")
    feature_rows = build_feature_table(frame_lookup, crossings, enriched)
    print(f"  {len(feature_rows)} feature rows generated.")

    # 6. Write CSV
    print("Writing traffic_features.csv ...")
    write_feature_csv(feature_rows)
    print(f"  Saved: {OUT_CSV}")

    # 7. Sanity checks
    errors, warnings = run_sanity_checks(feature_rows, pre_errors)

    # 8. Feature summary JSON
    print("Writing feature_summary.json ...")
    summary = build_summary(enriched, feature_rows, crossings, errors, warnings)
    print(f"  Saved: {OUT_JSON}")

    # 9. Video
    print("Rendering feature_visualization.mp4 ...")
    produce_video(enriched, frame_lookup, raw_bboxes)
    print(f"  Saved: {OUT_VID}")

    # 10. Plots
    print("Generating distribution plots ...")
    produce_plots(feature_rows)
    print(f"  Saved: {OUT_SPEED}, {OUT_TTC}, {OUT_DIST}")

    # 11. Terminal report
    print_terminal_report(summary, feature_rows)

    # 12. File manifest
    print("Output files:")
    for p in [OUT_CSV, OUT_JSON, OUT_VID, OUT_SPEED, OUT_TTC, OUT_DIST]:
        sz = p.stat().st_size if p.exists() else 0
        print(f"  {p}  ({sz:,} bytes)")

    # 13. Show first 10 rows of CSV
    print("\nFirst 10 rows of traffic_features.csv:")
    with open(OUT_CSV, newline="") as fh:
        reader = csv.DictReader(fh)
        cols_to_show = ["frame_id", "v1_id", "v2_id", "distance", "closing_speed",
                        "ttc", "approaching", "interaction_type"]
        header = "  " + " | ".join(f"{c:>20}" for c in cols_to_show)
        print(header)
        print("  " + "-" * len(header))
        for i, row in enumerate(reader):
            if i >= 10:
                break
            vals = " | ".join(f"{row.get(c, ''):>20}" for c in cols_to_show)
            print(f"  {vals}")


if __name__ == "__main__":
    main()
