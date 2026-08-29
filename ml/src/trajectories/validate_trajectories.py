"""
Phase 4 — Trajectory Quality Validation
RoadSentinel AI

Reads:
  ml/outputs/trajectories/trajectories.csv   (enriched by Phase 3)
  ml/outputs/tracking/tracks_v2.csv          (raw bounding boxes for video)
  ml/data/samples/traffic.mp4               (original video for annotation)

Outputs:
  ml/outputs/trajectories/validated_trajectories.csv
  ml/outputs/trajectories/trajectory_quality_report.json
  ml/outputs/trajectories/validated_trajectory_visualization.mp4
  ml/outputs/trajectories/trajectory_quality_plot.png
"""

import csv
import json
import math
import sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# Configurable Thresholds  ← tune these without touching logic
# ─────────────────────────────────────────────────────────────
MIN_OBS               = 15      # fewer observations → SHORT
MIN_AVG_DISPLACEMENT  = 1.5     # px/frame — minimum average motion to be considered moving
MIN_MOVEMENT_RATIO    = 0.35    # fraction of consecutive moves that must exceed MOVE_STEP
MOVE_STEP             = 2.0     # px — per-frame displacement counted as "a real move"
MIN_TOTAL_DISP        = 50.0    # px — total path length below this → STATIONARY
MAX_GAPS_ALLOWED      = 20      # frame-gap count above this → FRAGMENTED
MAX_GAP_SIZE          = 30      # single gap (in frames) above this → FRAGMENTED
MAX_JITTER_STD        = 8.0     # std-dev of displacements above this → NOISY
MAX_JUMP_PX           = 300.0   # single jump above this → QUESTIONABLE

TRAJECTORY_HISTORY    = 30      # trail length drawn in video
SMOOTH_WINDOW         = 5       # moving-average window for smoothed coords


# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
TRAJ_CSV   = BASE_DIR / "outputs" / "trajectories" / "trajectories.csv"
RAW_CSV    = BASE_DIR / "outputs" / "tracking"     / "tracks_v2.csv"
INPUT_VID  = BASE_DIR / "data"    / "samples"      / "traffic.mp4"
OUT_DIR    = BASE_DIR / "outputs" / "trajectories"

OUT_VALID_CSV  = OUT_DIR / "validated_trajectories.csv"
OUT_REPORT     = OUT_DIR / "trajectory_quality_report.json"
OUT_VID        = OUT_DIR / "validated_trajectory_visualization.mp4"
OUT_PLOT       = OUT_DIR / "trajectory_quality_plot.png"

# Colour key for quality labels (BGR for OpenCV, RGB-normalised for matplotlib)
LABEL_COLORS_BGR = {
    "VALID_MOVING" : (0,   210,  0),
    "STATIONARY"   : (160, 160, 160),
    "SHORT"        : (0,   165, 255),   # orange
    "NOISY"        : (0,   0,   220),   # red
    "FRAGMENTED"   : (213, 0,   213),   # purple
    "QUESTIONABLE" : (0,   215, 255),   # yellow
}

def bgr_to_rgb_norm(bgr):
    b, g, r = bgr
    return (r / 255.0, g / 255.0, b / 255.0)

# Per-track colour for VALID_MOVING tracks (cycle through a vivid palette)
TRACK_PALETTE_BGR = [
    (86,  180, 233),
    (230, 159,   0),
    (0,   158, 115),
    (204, 121,  167),
    (0,   114, 178),
    (213,  94,   0),
    (0,   255, 200),
    (255, 100, 100),
]


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────
def euclidean(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def std_dev(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

def moving_average(values, window):
    """
    Causal trailing moving average.
    At index i (frame T), only uses current and previous observations [max(0, i - window + 1), i + 1).
    Never uses future frames (T+1, T+2) to avoid lookahead data leakage.
    """
    result = []
    n = len(values)
    for i in range(n):
        lo = max(0, i - window + 1)
        hi = i + 1
        segment = values[lo:hi]
        result.append(sum(segment) / len(segment))
    return result

def direction_degrees(obs):
    """Angle (degrees, 0=right, 90=down) from first to last centroid."""
    if len(obs) < 2:
        return None
    dx = obs[-1]["center_x"] - obs[0]["center_x"]
    dy = obs[-1]["center_y"] - obs[0]["center_y"]
    return round(math.degrees(math.atan2(dy, dx)) % 360, 2)


# ─────────────────────────────────────────────────────────────
# Step 1 — Load trajectories.csv  (Phase 3 output)
# ─────────────────────────────────────────────────────────────
def load_trajectories(path):
    """Return dict: track_id → sorted list of row dicts."""
    if not path.exists():
        print(f"Error: trajectories.csv not found at {path}")
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
        print(f"Warning: tracks_v2.csv not found at {path}. Video will be skipped.")
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
# Step 2 — Per-track metrics
# ─────────────────────────────────────────────────────────────
def compute_metrics(tid, obs):
    """Return a dict of all computed metrics for this track."""
    n         = len(obs)
    fids      = [o["frame_id"] for o in obs]
    start_f   = fids[0]
    end_f     = fids[-1]

    # Frame gaps
    gaps      = [fids[i+1] - fids[i] - 1 for i in range(len(fids) - 1)]
    n_gaps    = sum(1 for g in gaps if g > 0)
    max_gap   = max(gaps) if gaps else 0
    # Continuity: fraction of expected frames actually present
    expected_frames = end_f - start_f + 1
    continuity      = round(n / expected_frames, 4) if expected_frames > 0 else 1.0

    # Displacements
    disps = []
    has_jump = False
    for i in range(1, n):
        d = euclidean(obs[i-1]["center_x"], obs[i-1]["center_y"],
                      obs[i]["center_x"],   obs[i]["center_y"])
        disps.append(d)
        if d > MAX_JUMP_PX:
            has_jump = True

    total_disp = sum(disps)
    avg_disp   = total_disp / len(disps) if disps else 0.0
    max_disp   = max(disps)             if disps else 0.0
    disp_std   = std_dev(disps)

    above          = sum(1 for d in disps if d >= MOVE_STEP)
    movement_ratio = above / len(disps) if disps else 0.0
    move_pct       = round(movement_ratio * 100, 2)

    direction  = direction_degrees(obs)

    return {
        "track_id"             : tid,
        "class_name"           : obs[0]["class_name"],
        "n_obs"                : n,
        "start_frame"          : start_f,
        "end_frame"            : end_f,
        "total_displacement"   : round(total_disp, 3),
        "avg_displacement"     : round(avg_disp,   4),
        "max_displacement"     : round(max_disp,   3),
        "displacement_std"     : round(disp_std,   4),
        "movement_ratio"       : round(movement_ratio, 4),
        "movement_percentage"  : move_pct,
        "n_gaps"               : n_gaps,
        "max_gap_size"         : max_gap,
        "trajectory_continuity": continuity,
        "estimated_direction"  : direction,
        "has_jump"             : has_jump,
    }


def bbox_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0: return 0.0
    return inter_area / union_area

def detect_id_switch_duplicates(raw_bboxes, traj_by_track):
    """
    Conservative Tracker ID-Switch Filtering using Bounding-Box IoU.
    Identifies candidate pairs with bounding box IoU > 0.50 in co-observed frames.
    Evaluates temporal and spatial consistency to separate duplicate fragments
    spawned by tracker flicker vs legitimate nearby vehicles.
    Returns:
      candidate_pairs: list of (t1, t2, high_iou_count, total_overlap, max_iou)
      filtered_duplicates: set of track_ids determined to be duplicate fragments
      questionable_tracks: set of track_ids marked questionable due to ambiguity
    """
    frame_bboxes = defaultdict(dict)
    for (tid, fid), box in raw_bboxes.items():
        if tid in traj_by_track:
            frame_bboxes[fid][tid] = box

    pair_overlap_frames  = defaultdict(int)
    pair_high_iou_frames = defaultdict(int)
    pair_max_iou         = defaultdict(float)

    for fid, tids_dict in frame_bboxes.items():
        tids = sorted(tids_dict.keys())
        for i in range(len(tids)):
            for j in range(i+1, len(tids)):
                t1, t2 = tids[i], tids[j]
                pair = (t1, t2)
                pair_overlap_frames[pair] += 1
                iou = bbox_iou(tids_dict[t1], tids_dict[t2])
                if iou > pair_max_iou[pair]:
                    pair_max_iou[pair] = iou
                if iou > 0.50:
                    pair_high_iou_frames[pair] += 1

    candidate_pairs = []
    filtered_duplicates = set()
    questionable_tracks = set()

    for (t1, t2), high_cnt in pair_high_iou_frames.items():
        total_ov = pair_overlap_frames[(t1, t2)]
        max_iou  = pair_max_iou[(t1, t2)]
        ratio    = high_cnt / total_ov if total_ov > 0 else 0
        candidate_pairs.append((t1, t2, high_cnt, total_ov, round(max_iou, 4)))

        # Strong temporal evidence of duplicate fragment (IoU > 0.50 in >=5 frames or >=30% of overlap)
        if high_cnt >= 5 or ratio >= 0.30:
            # Reclassify the shorter / secondary track as a duplicate fragment
            len1 = len(traj_by_track[t1])
            len2 = len(traj_by_track[t2])
            shorter_tid = t2 if len2 <= len1 else t1
            filtered_duplicates.add(shorter_tid)
        else:
            # Isolated 1-2 frame overlap -> mark as QUESTIONABLE for caution
            questionable_tracks.add(t1)
            questionable_tracks.add(t2)

    return candidate_pairs, filtered_duplicates, questionable_tracks


# ─────────────────────────────────────────────────────────────
# Step 3 — 6-label priority cascade classifier
# ─────────────────────────────────────────────────────────────
def classify(m, is_duplicate=False, is_questionable=False):
    """Priority cascade — return one of 6 quality labels."""
    if is_duplicate or is_questionable:
        return "QUESTIONABLE"   # Filter out duplicate ID-switch fragments
    if m["n_obs"] < MIN_OBS:
        return "SHORT"
    if m["has_jump"]:
        return "QUESTIONABLE"
    if m["n_gaps"] > MAX_GAPS_ALLOWED or m["max_gap_size"] > MAX_GAP_SIZE:
        return "FRAGMENTED"
    if (m["avg_displacement"] < MIN_AVG_DISPLACEMENT
            or m["movement_ratio"] < MIN_MOVEMENT_RATIO
            or m["total_displacement"] < MIN_TOTAL_DISP):
        return "STATIONARY"
    if m["displacement_std"] > MAX_JITTER_STD:
        return "NOISY"
    return "VALID_MOVING"


# ─────────────────────────────────────────────────────────────
# Step 4 — Write validated_trajectories.csv  (VALID_MOVING only)
# ─────────────────────────────────────────────────────────────
def write_validated_csv(traj_by_track, labels):
    with open(OUT_VALID_CSV, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "track_id", "class_name", "frame_id", "timestamp",
            "center_x", "center_y", "smooth_x", "smooth_y",
            "confidence", "quality_label",
        ])
        for tid, obs in traj_by_track.items():
            if labels[tid] != "VALID_MOVING":
                continue
            for o in obs:
                writer.writerow([
                    tid,
                    o["class_name"],
                    o["frame_id"],
                    f"{o['timestamp']:.3f}",
                    f"{o['center_x']:.2f}",
                    f"{o['center_y']:.2f}",
                    f"{o['smooth_x']:.2f}",
                    f"{o['smooth_y']:.2f}",
                    f"{o['confidence']:.3f}",
                    "VALID_MOVING",
                ])


# ─────────────────────────────────────────────────────────────
# Step 5 — Quality report JSON
# ─────────────────────────────────────────────────────────────
def write_report(all_metrics, labels):
    label_order = ["VALID_MOVING", "STATIONARY", "SHORT", "NOISY", "FRAGMENTED", "QUESTIONABLE"]
    counts = {lbl: 0 for lbl in label_order}
    for lbl in labels.values():
        counts[lbl] = counts.get(lbl, 0) + 1
    total = len(labels)

    # Longest VALID_MOVING track
    valid_tracks = [m for m in all_metrics.values() if labels[m["track_id"]] == "VALID_MOVING"]
    longest_valid = max(valid_tracks, key=lambda m: m["n_obs"]) if valid_tracks else None

    # Aggregate stats for VALID_MOVING tracks
    valid_lengths = [m["n_obs"] for m in all_metrics.values() if labels[m["track_id"]] == "VALID_MOVING"]
    valid_avg_d   = [m["avg_displacement"] for m in all_metrics.values() if labels[m["track_id"]] == "VALID_MOVING"]
    avg_valid_len = round(sum(valid_lengths) / len(valid_lengths), 2) if valid_lengths else 0.0
    avg_disp_all  = round(sum(m["avg_displacement"] for m in all_metrics.values()) / len(all_metrics), 4) if all_metrics else 0.0
    avg_move_valid = round(sum(valid_avg_d) / len(valid_avg_d), 4) if valid_avg_d else 0.0

    report = {
        "thresholds": {
            "MIN_OBS"              : MIN_OBS,
            "MIN_AVG_DISPLACEMENT" : MIN_AVG_DISPLACEMENT,
            "MIN_MOVEMENT_RATIO"   : MIN_MOVEMENT_RATIO,
            "MIN_TOTAL_DISP"       : MIN_TOTAL_DISP,
            "MAX_GAPS_ALLOWED"     : MAX_GAPS_ALLOWED,
            "MAX_GAP_SIZE"         : MAX_GAP_SIZE,
            "MAX_JITTER_STD"       : MAX_JITTER_STD,
            "MAX_JUMP_PX"          : MAX_JUMP_PX,
        },
        "summary": {
            "total_tracks"              : total,
            "valid_moving_pct"          : round(counts["VALID_MOVING"] / total * 100, 2) if total else 0,
            **{lbl + "_count": counts[lbl] for lbl in label_order},
            "longest_valid_track"       : (
                {"track_id": longest_valid["track_id"], "n_obs": longest_valid["n_obs"]}
                if longest_valid else None
            ),
            "avg_valid_trajectory_length" : avg_valid_len,
            "avg_displacement_all_tracks" : avg_disp_all,
            "avg_movement_per_frame_valid": avg_move_valid,
        },
        "per_track": {
            str(tid): {**m, "quality_label": labels[tid]}
            for tid, m in all_metrics.items()
        },
    }
    with open(OUT_REPORT, "w") as fh:
        json.dump(report, fh, indent=2)


# ─────────────────────────────────────────────────────────────
# Step 6 — Annotated video  (VALID_MOVING only, colour by track ID)
# ─────────────────────────────────────────────────────────────
def produce_video(traj_by_track, labels, raw_bboxes):
    if not INPUT_VID.exists():
        print(f"  Warning: video not found at {INPUT_VID}. Skipping.")
        return

    cap = cv2.VideoCapture(str(INPUT_VID))
    if not cap.isOpened():
        print("  Error: Could not open input video. Skipping.")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wtr    = cv2.VideoWriter(str(OUT_VID), fourcc, fps, (width, height))
    if not wtr.isOpened():
        print("  Error: Could not open video writer. Skipping.")
        cap.release()
        return

    # Only keep VALID_MOVING tracks; assign each a distinct palette colour
    valid_ids = sorted(tid for tid, lbl in labels.items() if lbl == "VALID_MOVING")
    tid_color = {
        tid: TRACK_PALETTE_BGR[i % len(TRACK_PALETTE_BGR)]
        for i, tid in enumerate(valid_ids)
    }
    # Build lookup: track_id → observation list (already sorted)
    valid_obs = {tid: traj_by_track[tid] for tid in valid_ids}

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    THICK      = 1

    frame_id = 0
    print(f"  Rendering validation video ({total} frames)...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            for tid in valid_ids:
                color = tid_color[tid]
                obs   = valid_obs[tid]
                cls_name = obs[0]["class_name"]

                # Bounding box from raw track data
                bbox = raw_bboxes.get((tid, frame_id))
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox

                # Draw box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Label
                label = f"ID {tid} | {cls_name} | VALID_MOVING"
                (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, THICK)
                ly = max(int(y1) - 5, th + 5)
                cv2.rectangle(frame, (int(x1), ly - th - 4), (int(x1) + tw, ly + 2), color, -1)
                cv2.putText(frame, label, (int(x1), ly), FONT, FONT_SCALE, (255, 255, 255), THICK)

                # Trajectory trail (last TRAJECTORY_HISTORY smooth positions up to this frame)
                trail = [o for o in obs if o["frame_id"] <= frame_id][-TRAJECTORY_HISTORY:]
                for i in range(1, len(trail)):
                    pt1 = (int(trail[i-1]["smooth_x"]), int(trail[i-1]["smooth_y"]))
                    pt2 = (int(trail[i]["smooth_x"]),   int(trail[i]["smooth_y"]))
                    seg_thickness = max(1, int((i / len(trail)) * 3))
                    cv2.line(frame, pt1, pt2, color, seg_thickness)

            wtr.write(frame)
            if frame_id % 50 == 0:
                print(f"    Frame {frame_id}/{total}")
    finally:
        cap.release()
        wtr.release()


# ─────────────────────────────────────────────────────────────
# Step 7 — Quality scatter / path plot
# ─────────────────────────────────────────────────────────────
def produce_plot(traj_by_track, labels):
    label_order = ["VALID_MOVING", "FRAGMENTED", "NOISY", "STATIONARY", "SHORT", "QUESTIONABLE"]
    mpl_colors  = {lbl: bgr_to_rgb_norm(LABEL_COLORS_BGR[lbl]) for lbl in label_order}

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.set_title("RoadSentinel AI — Trajectory Quality Map", color="white", fontsize=15, pad=12)
    ax.set_xlabel("X (pixels)", color="#aaa")
    ax.set_ylabel("Y (pixels)", color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plotted_labels = set()

    for tid, obs in traj_by_track.items():
        lbl   = labels[tid]
        color = mpl_colors[lbl]
        xs    = [o["smooth_x"] for o in obs]
        ys    = [o["smooth_y"] for o in obs]
        mx    = sum(xs) / len(xs)
        my    = sum(ys) / len(ys)

        if lbl == "VALID_MOVING":
            # Draw full path
            ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.85,
                    label=lbl if lbl not in plotted_labels else "")
            ax.plot(xs[-1], ys[-1], "o", color=color, markersize=4)
        else:
            # Draw dot at mean position
            ax.plot(mx, my, marker="s", color=color, markersize=5, alpha=0.55,
                    label=lbl if lbl not in plotted_labels else "")

        plotted_labels.add(lbl)

    # Legend
    handles = [
        mpatches.Patch(color=mpl_colors[lbl], label=lbl)
        for lbl in label_order if lbl in plotted_labels
    ]
    ax.legend(handles=handles, loc="upper right",
              facecolor="#161b22", labelcolor="white", fontsize=9, framealpha=0.9)

    ax.invert_yaxis()   # match image coordinate system
    plt.tight_layout()
    plt.savefig(str(OUT_PLOT), dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────
# Terminal Quality Report
# ─────────────────────────────────────────────────────────────
def print_report(all_metrics, labels):
    label_order = ["VALID_MOVING", "STATIONARY", "SHORT", "NOISY", "FRAGMENTED", "QUESTIONABLE"]
    counts = {lbl: 0 for lbl in label_order}
    for lbl in labels.values():
        counts[lbl] = counts.get(lbl, 0) + 1
    total = len(labels)

    # Aggregate stats
    valid_ms = [m for m in all_metrics.values() if labels[m["track_id"]] == "VALID_MOVING"]
    valid_pct         = counts["VALID_MOVING"] / total * 100 if total else 0.0
    avg_valid_len     = (sum(m["n_obs"] for m in valid_ms) / len(valid_ms)) if valid_ms else 0.0
    avg_disp_all      = (sum(m["avg_displacement"] for m in all_metrics.values()) / len(all_metrics)) if all_metrics else 0.0
    avg_move_valid    = (sum(m["avg_displacement"] for m in valid_ms) / len(valid_ms)) if valid_ms else 0.0

    W = 44
    print()
    print("=" * W)
    print("  ROAD SENTINEL — TRAJECTORY QUALITY")
    print("=" * W)
    print(f"  Total tracks           : {total}")
    print()
    for lbl in label_order:
        pct = counts[lbl] / total * 100 if total else 0
        print(f"  {lbl:<20}   : {counts[lbl]:>4}  ({pct:5.1f}%)")
    print()
    print(f"  Valid moving %         : {valid_pct:.1f}%")
    print()
    print(f"  Avg valid traj length  : {avg_valid_len:.1f} frames")
    print(f"  Avg displacement       : {avg_disp_all:.3f} px/frame (all tracks)")
    print(f"  Avg movement/frame     : {avg_move_valid:.3f} px/frame (valid only)")
    print("=" * W)
    print()

    # Per-track detail for VALID_MOVING tracks
    if valid_ms:
        print("  VALID_MOVING track details:")
        print(f"    {'tid':>5}  {'class':<9}  {'n':>5}  {'avg_d':>6}  "
              f"{'total_d':>8}  {'ratio':>6}  {'gaps':>5}  {'cont':>5}  {'dir°':>7}")
        for m in sorted(valid_ms, key=lambda x: -x["avg_displacement"]):
            print(f"    {m['track_id']:>5}  {m['class_name']:<9}  {m['n_obs']:>5}  "
                  f"{m['avg_displacement']:>6.2f}  {m['total_displacement']:>8.1f}  "
                  f"{m['movement_ratio']:>6.2f}  {m['n_gaps']:>5}  "
                  f"{m['trajectory_continuity']:>5.2f}  "
                  f"{str(m['estimated_direction']):>7}")
        print()
    print("=" * W)
    print()


# ─────────────────────────────────────────────────────────────
# Data integrity checks
# ─────────────────────────────────────────────────────────────
def run_integrity_checks(traj_by_track):
    print("Running data integrity checks...")
    issues = 0

    for tid, obs in traj_by_track.items():
        fids = [o["frame_id"] for o in obs]

        # Frame ordering
        if fids != sorted(fids):
            print(f"  [WARN] Frames out of order — track {tid}")
            issues += 1

        # Duplicate frame
        if len(fids) != len(set(fids)):
            print(f"  [WARN] Duplicate frame_id — track {tid}")
            issues += 1

        # NaN coordinates
        for o in obs:
            for field in ("center_x", "center_y", "smooth_x", "smooth_y"):
                v = o[field]
                if v != v:
                    print(f"  [WARN] NaN in {field} — track {tid} frame {o['frame_id']}")
                    issues += 1

        # Coordinate range sanity (allow large video resolutions)
        for o in obs:
            if not (0 <= o["center_x"] <= 20000 and 0 <= o["center_y"] <= 20000):
                print(f"  [WARN] Out-of-range coordinates — track {tid} frame {o['frame_id']}")
                issues += 1

    if issues == 0:
        print("  All integrity checks passed — no issues found.")
    else:
        print(f"  {issues} issue(s) detected. Review warnings above.")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("Loading trajectories.csv ...")
    traj_by_track = load_trajectories(TRAJ_CSV)
    print(f"  Loaded {len(traj_by_track)} tracks.")

    print("Loading raw bounding boxes from tracks_v2.csv ...")
    raw_bboxes = load_raw_bboxes(RAW_CSV)
    print(f"  Loaded {len(raw_bboxes)} bbox records.")

    # 2. Compute per-track metrics
    print("Computing per-track metrics ...")
    all_metrics = {}
    for tid, obs in traj_by_track.items():
        all_metrics[tid] = compute_metrics(tid, obs)

    # 3. Detect candidate ID-switch duplicates via IoU
    print("Evaluating bounding box IoU for ID-switch duplicate detection ...")
    candidates, filtered_dups, quest_tracks = detect_id_switch_duplicates(raw_bboxes, traj_by_track)
    print(f"  Candidate ID-switch pairs (IoU > 0.50): {len(candidates)}")
    print(f"  Filtered duplicate tracks (reclassified): {sorted(filtered_dups)}")

    # 4. Classify every track
    labels = {
        tid: classify(m, is_duplicate=(tid in filtered_dups), is_questionable=(tid in quest_tracks))
        for tid, m in all_metrics.items()
    }
    valid_moving_tids = [tid for tid, lbl in labels.items() if lbl == "VALID_MOVING"]
    print(f"  Remaining VALID_MOVING tracks: {len(valid_moving_tids)} (IDs: {sorted(valid_moving_tids)})")

    # 4. Integrity checks
    run_integrity_checks(traj_by_track)

    # 5. Write validated CSV (VALID_MOVING only)
    print("Writing validated_trajectories.csv ...")
    write_validated_csv(traj_by_track, labels)
    print(f"  Saved: {OUT_VALID_CSV}")

    # 6. Write quality report JSON
    print("Writing trajectory_quality_report.json ...")
    write_report(all_metrics, labels)
    print(f"  Saved: {OUT_REPORT}")

    # 7. Produce annotated video
    print("Rendering validated_trajectory_visualization.mp4 ...")
    produce_video(traj_by_track, labels, raw_bboxes)
    print(f"  Saved: {OUT_VID}")

    # 8. Produce quality plot
    print("Generating trajectory_quality_plot.png ...")
    produce_plot(traj_by_track, labels)
    print(f"  Saved: {OUT_PLOT}")

    # 9. Terminal report
    print_report(all_metrics, labels)

    # 10. File manifest
    print("Output files:")
    for p in [OUT_VALID_CSV, OUT_REPORT, OUT_VID, OUT_PLOT]:
        size = p.stat().st_size if p.exists() else 0
        print(f"  {p}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
