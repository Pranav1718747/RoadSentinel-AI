"""
Phase 3 — Trajectory Extraction with Stationary-Object Filtering
RoadSentinel AI

Reads: ml/outputs/tracking/tracks_v2.csv
Reads: ml/data/samples/traffic.mp4  (for video annotation)

Outputs:
  ml/outputs/trajectories/trajectories.csv
  ml/outputs/trajectories/trajectory_visualization.mp4
  ml/outputs/trajectories/moving_trajectories.mp4
  ml/outputs/trajectories/trajectory_plot.png
  ml/outputs/trajectories/trajectory_summary.json
"""

import csv
import json
import math
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────
# Configurable Parameters
# ─────────────────────────────────────────────
MIN_TRACK_LENGTH         = 10       # Minimum observations to be considered a full track
MOVEMENT_THRESHOLD       = 2.0      # Pixels – frame-to-frame displacement counts as "moved"
MOVEMENT_RATIO_THRESHOLD = 0.30     # At least 30 % of moves must exceed threshold → MOVING
MAX_JUMP_THRESHOLD       = 300.0    # Pixels – single jump larger than this → QUESTIONABLE
SMOOTH_WINDOW            = 5        # Rolling-average window for trajectory smoothing
TRAJECTORY_HISTORY       = 30       # Number of past positions to draw as trail


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR           = Path(__file__).resolve().parent.parent.parent
INPUT_CSV          = BASE_DIR / "outputs" / "tracking" / "tracks_v2.csv"
INPUT_VIDEO        = BASE_DIR / "data" / "samples" / "traffic.mp4"
OUTPUT_DIR         = BASE_DIR / "outputs" / "trajectories"
OUT_CSV            = OUTPUT_DIR / "trajectories.csv"
OUT_VID_FULL       = OUTPUT_DIR / "trajectory_visualization.mp4"
OUT_VID_MOVING     = OUTPUT_DIR / "moving_trajectories.mp4"
OUT_PLOT           = OUTPUT_DIR / "trajectory_plot.png"
OUT_SUMMARY        = OUTPUT_DIR / "trajectory_summary.json"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def euclidean(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def moving_average(values, window):
    """
    Causal trailing moving average (window size = window).
    At index i (frame T), only uses current and previous observations:
    indices [max(0, i - window + 1), i + 1).
    Never uses future frames (T+1, T+2, etc.) to prevent lookahead data leakage.
    """
    result = []
    n = len(values)
    for i in range(n):
        lo = max(0, i - window + 1)
        hi = i + 1
        segment = values[lo:hi]
        result.append(sum(segment) / len(segment))
    return result


def make_color_map(class_names):
    """Assign a distinct BGR colour to each class name."""
    palette = [
        (86, 180, 233),
        (230, 159, 0),
        (0, 158, 115),
        (240, 228, 66),
        (0, 114, 178),
        (213, 94, 0),
    ]
    out = {}
    for i, name in enumerate(sorted(set(class_names))):
        out[name] = palette[i % len(palette)]
    return out


# ─────────────────────────────────────────────
# Step 1 — Load CSV
# ─────────────────────────────────────────────
def load_tracks(csv_path):
    """Return dict:  track_id → sorted list of row dicts."""
    if not csv_path.exists():
        print(f"Error: Input CSV not found at {csv_path}")
        sys.exit(1)

    raw = defaultdict(list)
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tid = int(row["track_id"])
            raw[tid].append({
                "frame_id"  : int(row["frame_id"]),
                "timestamp" : float(row["timestamp"]),
                "class_name": row["class_name"].strip(),
                "confidence": float(row["confidence"]),
                "x1"        : float(row["x1"]),
                "y1"        : float(row["y1"]),
                "x2"        : float(row["x2"]),
                "y2"        : float(row["y2"]),
                "center_x"  : float(row["center_x"]),
                "center_y"  : float(row["center_y"]),
            })

    for tid in raw:
        raw[tid].sort(key=lambda r: r["frame_id"])

    return dict(raw)


# ─────────────────────────────────────────────
# Step 2 & 3 — Movement metrics + classification
# ─────────────────────────────────────────────
def analyse_track(observations):
    """Compute movement metrics and return a classification dict."""
    n = len(observations)
    if n < 2:
        return {
            "total_displacement": 0.0,
            "avg_displacement"  : 0.0,
            "max_displacement"  : 0.0,
            "n_obs"             : n,
            "movement_ratio"    : 0.0,
            "movement_status"   : "STATIONARY",
            "has_jump"          : False,
        }

    displacements = []
    has_jump = False
    for i in range(1, n):
        prev = observations[i - 1]
        curr = observations[i]
        d = euclidean(prev["center_x"], prev["center_y"],
                      curr["center_x"], curr["center_y"])
        displacements.append(d)
        if d > MAX_JUMP_THRESHOLD:
            has_jump = True

    total_disp  = sum(displacements)
    avg_disp    = total_disp / len(displacements)
    max_disp    = max(displacements)
    above       = sum(1 for d in displacements if d >= MOVEMENT_THRESHOLD)
    movement_ratio = above / len(displacements)

    status = "MOVING" if movement_ratio >= MOVEMENT_RATIO_THRESHOLD else "STATIONARY"

    return {
        "total_displacement": total_disp,
        "avg_displacement"  : avg_disp,
        "max_displacement"  : max_disp,
        "n_obs"             : n,
        "movement_ratio"    : movement_ratio,
        "movement_status"   : status,
        "has_jump"          : has_jump,
    }


# ─────────────────────────────────────────────
# Step 4 — Quality flag
# ─────────────────────────────────────────────
def quality_flag(n_obs, has_jump):
    if n_obs < MIN_TRACK_LENGTH or has_jump:
        return "QUESTIONABLE"
    return "GOOD"


# ─────────────────────────────────────────────
# Step 5 — Smooth
# ─────────────────────────────────────────────
def smooth_track(observations, status):
    xs = [o["center_x"] for o in observations]
    ys = [o["center_y"] for o in observations]
    if status == "MOVING":
        sx = moving_average(xs, SMOOTH_WINDOW)
        sy = moving_average(ys, SMOOTH_WINDOW)
    else:
        sx, sy = xs[:], ys[:]
    return sx, sy


# ─────────────────────────────────────────────
# Step 7 — Write trajectory CSV
# ─────────────────────────────────────────────
def write_csv(track_data, output_path):
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "track_id", "class_name", "frame_id", "timestamp",
            "center_x", "center_y", "smooth_x", "smooth_y",
            "confidence", "movement_status", "trajectory_quality"
        ])
        for row in track_data:
            writer.writerow([
                row["track_id"],
                row["class_name"],
                row["frame_id"],
                f"{row['timestamp']:.3f}",
                f"{row['center_x']:.2f}",
                f"{row['center_y']:.2f}",
                f"{row['smooth_x']:.2f}",
                f"{row['smooth_y']:.2f}",
                f"{row['confidence']:.3f}",
                row["movement_status"],
                row["trajectory_quality"],
            ])


# ─────────────────────────────────────────────
# Step 8 & 9 — Annotate video frames
# ─────────────────────────────────────────────
def draw_annotations(frame, frame_id, track_meta, traj_rows_by_track, color_map, moving_only=False):
    """
    Draw bounding boxes and trajectory trails on a frame.
    track_meta: dict of track_id → {status, quality, class_name, bboxes_by_frame}
    traj_rows_by_track: dict of track_id → list of trajectory row dicts sorted by frame_id
    """
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    THICKNESS  = 1

    for tid, meta in track_meta.items():
        status    = meta["status"]
        cls_name  = meta["class_name"]

        if moving_only and status != "MOVING":
            continue

        # Bounding box for the current frame (if any)
        bbox = meta["bboxes"].get(frame_id)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        color = color_map.get(cls_name, (0, 255, 0))

        if status == "MOVING":
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"ID {tid} | {cls_name} | MOVING"
            # Draw trajectory trail
            trail = [r for r in traj_rows_by_track.get(tid, [])
                     if r["frame_id"] <= frame_id][-TRAJECTORY_HISTORY:]
            for i in range(1, len(trail)):
                pt1 = (int(trail[i - 1]["smooth_x"]), int(trail[i - 1]["smooth_y"]))
                pt2 = (int(trail[i]["smooth_x"]),     int(trail[i]["smooth_y"]))
                # Fade older trail points by reducing thickness
                alpha_idx = i / len(trail)
                t = max(1, int(alpha_idx * 3))
                cv2.line(frame, pt1, pt2, color, t)
        else:
            # Stationary — grey thin box
            grey = (160, 160, 160)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), grey, 1)
            label = f"ID {tid} | {cls_name} | STATIONARY"
            color = grey

        # Draw label background + text
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
        ly = max(int(y1) - 5, th + 5)
        cv2.rectangle(frame,
                      (int(x1), ly - th - 4),
                      (int(x1) + tw, ly + 2),
                      color, -1)
        cv2.putText(frame, label, (int(x1), ly),
                    FONT, FONT_SCALE, (255, 255, 255), THICKNESS)

    return frame


def produce_videos(tracks_raw, track_metrics, traj_rows_by_track, color_map):
    if not INPUT_VIDEO.exists():
        print(f"Warning: Original video not found at {INPUT_VIDEO}. Skipping video outputs.")
        return

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        print("Error: Could not open input video for annotation.")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wtr_full   = cv2.VideoWriter(str(OUT_VID_FULL),   fourcc, fps, (width, height))
    wtr_moving = cv2.VideoWriter(str(OUT_VID_MOVING), fourcc, fps, (width, height))

    if not wtr_full.isOpened() or not wtr_moving.isOpened():
        print("Error: Could not open one or more video writers.")
        cap.release()
        return

    # Build: track_id → {status, class_name, bboxes: {frame_id: (x1,y1,x2,y2)}}
    track_meta = {}
    for tid, rows in tracks_raw.items():
        m = track_metrics[tid]
        bboxes = {r["frame_id"]: (r["x1"], r["y1"], r["x2"], r["y2"]) for r in rows}
        track_meta[tid] = {
            "status"   : m["movement_status"],
            "class_name": rows[0]["class_name"],
            "bboxes"   : bboxes,
        }

    frame_id = 0
    print(f"Rendering annotation videos ({total} frames)...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            full_frame   = draw_annotations(frame.copy(), frame_id, track_meta, traj_rows_by_track, color_map, moving_only=False)
            moving_frame = draw_annotations(frame.copy(), frame_id, track_meta, traj_rows_by_track, color_map, moving_only=True)

            wtr_full.write(full_frame)
            wtr_moving.write(moving_frame)

            if frame_id % 50 == 0:
                print(f"  Annotated frame {frame_id}/{total}")
    finally:
        cap.release()
        wtr_full.release()
        wtr_moving.release()

    print(f"  Saved: {OUT_VID_FULL}")
    print(f"  Saved: {OUT_VID_MOVING}")


# ─────────────────────────────────────────────
# Step 10 — Static trajectory plot
# ─────────────────────────────────────────────
def produce_plot(traj_rows_by_track, track_metrics, color_map):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_title("RoadSentinel AI — Trajectory Map", color="white", fontsize=16, pad=12)
    ax.set_xlabel("X (pixels)", color="white")
    ax.set_ylabel("Y (pixels)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # Moving tracks
    for tid, rows in traj_rows_by_track.items():
        m = track_metrics[tid]
        if m["movement_status"] != "MOVING":
            continue
        xs = [r["smooth_x"] for r in rows]
        ys = [r["smooth_y"] for r in rows]
        cls_name = rows[0]["class_name"]
        c = tuple(v / 255.0 for v in color_map.get(cls_name, (0, 200, 100)))
        ax.plot(xs, ys, color=c, linewidth=1.0, alpha=0.7)
        ax.plot(xs[-1], ys[-1], "o", color=c, markersize=4)

    # Stationary tracks — grey dots at mean position
    stat_plotted = set()
    for tid, rows in traj_rows_by_track.items():
        m = track_metrics[tid]
        if m["movement_status"] != "STATIONARY":
            continue
        mx = sum(r["center_x"] for r in rows) / len(rows)
        my = sum(r["center_y"] for r in rows) / len(rows)
        cls_name = rows[0]["class_name"]
        if cls_name not in stat_plotted:
            ax.plot(mx, my, "s", color="#888888", markersize=5,
                    label="stationary vehicle", alpha=0.5)
            stat_plotted.add(cls_name)
        else:
            ax.plot(mx, my, "s", color="#888888", markersize=5, alpha=0.5)

    # Legend (class colours for moving tracks)
    legend_handles = []
    for cls_name, bgr in color_map.items():
        c = tuple(v / 255.0 for v in bgr)
        legend_handles.append(
            plt.Line2D([0], [0], color=c, linewidth=2, label=f"{cls_name} (moving)")
        )
    legend_handles.append(
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#888888",
                   markersize=8, label="stationary vehicle")
    )
    ax.legend(handles=legend_handles, loc="upper right",
              facecolor="#2a2a4a", labelcolor="white", fontsize=9)

    # Invert Y so it matches image coordinate system
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(str(OUT_PLOT), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_PLOT}")


# ─────────────────────────────────────────────
# Step 11 — Summary JSON
# ─────────────────────────────────────────────
def produce_summary(tracks_raw, track_metrics):
    all_ids      = list(track_metrics.keys())
    moving       = [t for t in all_ids if track_metrics[t]["movement_status"] == "MOVING"]
    stationary   = [t for t in all_ids if track_metrics[t]["movement_status"] == "STATIONARY"]
    short_tracks = [t for t in all_ids if track_metrics[t]["n_obs"] < MIN_TRACK_LENGTH]
    quest_tracks = [t for t in all_ids
                    if (track_metrics[t]["n_obs"] < MIN_TRACK_LENGTH or track_metrics[t]["has_jump"])]
    total = len(all_ids)

    # Track lengths
    track_lengths = {t: track_metrics[t]["n_obs"] for t in all_ids}
    longest_tid   = max(track_lengths, key=track_lengths.get)

    # Class distribution
    tracks_by_class = defaultdict(int)
    for tid, rows in tracks_raw.items():
        tracks_by_class[rows[0]["class_name"]] += 1

    # Displacement across all tracks
    all_avg  = [track_metrics[t]["avg_displacement"] for t in all_ids if track_metrics[t]["n_obs"] >= 2]
    all_max  = [track_metrics[t]["max_displacement"] for t in all_ids if track_metrics[t]["n_obs"] >= 2]

    summary = {
        "total_tracks"         : total,
        "moving_tracks"        : len(moving),
        "stationary_tracks"    : len(stationary),
        "short_tracks"         : len(short_tracks),
        "questionable_tracks"  : len(quest_tracks),
        "moving_percentage"    : round(len(moving) / total * 100, 2) if total else 0,
        "stationary_percentage": round(len(stationary) / total * 100, 2) if total else 0,
        "average_track_length" : round(sum(track_lengths.values()) / total, 2) if total else 0,
        "longest_track"        : {"track_id": longest_tid, "length": track_lengths[longest_tid]},
        "tracks_by_class"      : dict(tracks_by_class),
        "average_displacement" : round(sum(all_avg) / len(all_avg), 4) if all_avg else 0,
        "maximum_displacement" : round(max(all_max), 4) if all_max else 0,
    }

    with open(OUT_SUMMARY, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved: {OUT_SUMMARY}")
    return summary


# ─────────────────────────────────────────────
# Step 12 — Quality check & console report
# ─────────────────────────────────────────────
def quality_checks(tracks_raw, traj_rows_flat):
    print("\nRunning data quality checks...")
    errors = 0

    # Duplicate track_id + frame_id check
    seen = set()
    for row in traj_rows_flat:
        key = (row["track_id"], row["frame_id"])
        if key in seen:
            print(f"  [WARN] Duplicate track_id+frame_id: {key}")
            errors += 1
        seen.add(key)

    # NaN / invalid coordinates
    for row in traj_rows_flat:
        for field in ("center_x", "center_y", "smooth_x", "smooth_y"):
            v = row[field]
            if v != v:  # NaN check
                print(f"  [WARN] NaN in {field} for track {row['track_id']} frame {row['frame_id']}")
                errors += 1

    # Frame order per track
    for tid, rows in tracks_raw.items():
        fids = [r["frame_id"] for r in rows]
        if fids != sorted(fids):
            print(f"  [WARN] Frames out of order for track {tid}")
            errors += 1

    if errors == 0:
        print("  All checks passed — no issues found.")


def print_summary_report(summary):
    print()
    print("=" * 44)
    print("  TRAJECTORY ANALYSIS")
    print("=" * 44)
    print(f"  Total tracks        : {summary['total_tracks']}")
    print(f"  Moving tracks       : {summary['moving_tracks']}")
    print(f"  Stationary tracks   : {summary['stationary_tracks']}")
    print(f"  Short tracks        : {summary['short_tracks']}")
    print(f"  Questionable tracks : {summary['questionable_tracks']}")
    print()
    print(f"  Moving %            : {summary['moving_percentage']:.1f}%")
    print(f"  Stationary %        : {summary['stationary_percentage']:.1f}%")
    print()
    print(f"  Avg track length    : {summary['average_track_length']:.1f} frames")
    print(f"  Longest track       : ID {summary['longest_track']['track_id']} "
          f"({summary['longest_track']['length']} frames)")
    print()
    print("  Tracks by class:")
    for cls, cnt in summary["tracks_by_class"].items():
        print(f"    {cls:<12}: {cnt}")
    print()
    print(f"  Avg displacement    : {summary['average_displacement']:.2f} px/frame")
    print(f"  Max displacement    : {summary['maximum_displacement']:.2f} px")
    print("=" * 44)
    print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load
    print("Loading tracking CSV...")
    tracks_raw = load_tracks(INPUT_CSV)
    print(f"  Loaded {len(tracks_raw)} unique tracks.")

    # 2. Analyse + classify each track
    print("Analysing track movement...")
    track_metrics = {}
    for tid, obs in tracks_raw.items():
        track_metrics[tid] = analyse_track(obs)

    # Build flat trajectory rows with all enriched fields
    traj_rows_by_track = {}
    traj_rows_flat     = []

    for tid, obs in tracks_raw.items():
        m    = track_metrics[tid]
        qual = quality_flag(m["n_obs"], m["has_jump"])
        sx, sy = smooth_track(obs, m["movement_status"])

        rows = []
        for i, o in enumerate(obs):
            row = {
                "track_id"          : tid,
                "class_name"        : o["class_name"],
                "frame_id"          : o["frame_id"],
                "timestamp"         : o["timestamp"],
                "center_x"          : o["center_x"],
                "center_y"          : o["center_y"],
                "smooth_x"          : sx[i],
                "smooth_y"          : sy[i],
                "confidence"        : o["confidence"],
                "movement_status"   : m["movement_status"],
                "trajectory_quality": qual,
            }
            rows.append(row)
            traj_rows_flat.append(row)

        traj_rows_by_track[tid] = rows

    # 7. Write CSV
    print("Writing trajectories.csv...")
    write_csv(traj_rows_flat, OUT_CSV)
    print(f"  Saved: {OUT_CSV}")

    # Colour map for classes
    all_classes = [rows[0]["class_name"] for rows in tracks_raw.values()]
    color_map   = make_color_map(all_classes)

    # 8 & 9. Produce annotated videos
    produce_videos(tracks_raw, track_metrics, traj_rows_by_track, color_map)

    # 10. Static trajectory plot
    print("Generating trajectory plot...")
    produce_plot(traj_rows_by_track, track_metrics, color_map)

    # 11. Summary JSON
    print("Writing trajectory_summary.json...")
    summary = produce_summary(tracks_raw, track_metrics)

    # 12. Quality checks
    quality_checks(tracks_raw, traj_rows_flat)

    # Console report
    print_summary_report(summary)


if __name__ == "__main__":
    main()
