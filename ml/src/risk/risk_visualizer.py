"""
risk_visualizer.py — RoadSentinel AI, Phase 7
Annotates the original video with risk level, scores, and TTC for each
vehicle pair. Uses colour coding: green=LOW, orange=MEDIUM, red=HIGH.
"""

import cv2
import math
import itertools
from pathlib import Path
from collections import defaultdict


# Colour palette (BGR)
COLORS = {
    "LOW"    : (0,   200,  0),
    "MEDIUM" : (0,   165, 255),   # orange
    "HIGH"   : (0,   0,   220),   # red
}

TRACK_PALETTE_BGR = [
    (86,  180, 233), (230, 159,  0), (0,  158, 115),
    (204, 121, 167), (0,  114, 178), (213,  94,   0),
    (0,  255, 200),  (255, 100, 100),(50,  200, 255),
    (255, 200,  50),
]

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.4
FONT_MED   = 0.5
THICK      = 1
NaN        = float("nan")


def is_nan(v):
    try: return v != v
    except: return True

def sf(v):
    if v is None or str(v).strip() == "": return NaN
    try: return float(v)
    except: return NaN


def render_risk_video(
    input_video_path,
    output_video_path,
    raw_bboxes,          # dict: (track_id, frame_id) → (x1,y1,x2,y2)
    frame_scored_rows,   # dict: frame_id → list of scored interaction rows
    valid_track_ids,     # list of VALID_MOVING track IDs
    smooth_positions,    # dict: track_id → list of (frame_id, sx, sy)
):
    """
    Render the risk-annotated video.

    Per-frame:
    - Draw bounding box for each VALID_MOVING vehicle (colour by risk level that frame)
    - Draw trajectory trail
    - For MEDIUM/HIGH pairs: draw a connecting line, show risk level + TTC/distance
    - Overlay top-right legend
    """
    in_p  = Path(input_video_path)
    out_p = Path(output_video_path)

    if not in_p.exists():
        print(f"  Warning: input video not found at {in_p}. Skipping.")
        return

    cap = cv2.VideoCapture(str(in_p))
    if not cap.isOpened():
        print("  Error: could not open video. Skipping.")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wtr    = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))
    if not wtr.isOpened():
        print("  Error: could not open video writer. Skipping.")
        cap.release()
        return

    # Build smooth position lookup: track_id → {frame_id: (sx, sy)}
    sp_lookup = defaultdict(dict)
    for tid, hist in smooth_positions.items():
        for fid, sx, sy in hist:
            sp_lookup[tid][fid] = (sx, sy)

    # Track colour assignment
    tid_color = {tid: TRACK_PALETTE_BGR[i % len(TRACK_PALETTE_BGR)]
                 for i, tid in enumerate(sorted(valid_track_ids))}

    # Trail history
    trail_hist = {tid: [] for tid in valid_track_ids}

    frame_id = 0
    print(f"  Rendering risk video ({total} frames)...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            # Find highest risk level per vehicle for this frame (for bbox colour)
            scored = frame_scored_rows.get(frame_id, [])
            vehicle_risk = {}   # track_id → highest risk level this frame
            for r in scored:
                for vid in [r["v1_id"], r["v2_id"]]:
                    cur = vehicle_risk.get(vid, "LOW")
                    if r["risk_level"] == "HIGH":
                        vehicle_risk[vid] = "HIGH"
                    elif r["risk_level"] == "MEDIUM" and cur != "HIGH":
                        vehicle_risk[vid] = "MEDIUM"
                    else:
                        vehicle_risk.setdefault(vid, "LOW")

            # Draw trails
            for tid in valid_track_ids:
                sp = sp_lookup[tid].get(frame_id)
                if sp:
                    trail_hist[tid].append((int(sp[0]), int(sp[1])))
                trail = trail_hist[tid][-30:]
                level = vehicle_risk.get(tid, "LOW")
                color = COLORS[level]
                for i in range(1, len(trail)):
                    t = max(1, int((i / len(trail)) * 3))
                    cv2.line(frame, trail[i-1], trail[i], color, t)

            # Draw bounding boxes
            for tid in valid_track_ids:
                bbox = raw_bboxes.get((tid, frame_id))
                if not bbox:
                    continue
                x1, y1, x2, y2 = bbox
                level = vehicle_risk.get(tid, "LOW")
                color = COLORS[level]
                lw = 3 if level == "HIGH" else 2
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, lw)

                # Label
                label = f"ID{tid} [{level}]"
                (tw, th), _ = cv2.getTextSize(label, FONT, FONT_MED, THICK)
                ly = max(int(y1) - 4, th + 4)
                cv2.rectangle(frame, (int(x1), ly - th - 3), (int(x1) + tw, ly + 2), color, -1)
                cv2.putText(frame, label, (int(x1), ly), FONT, FONT_MED, (255,255,255), THICK)

            # Draw pair lines for MEDIUM/HIGH interactions
            for r in scored:
                if r["risk_level"] not in ("MEDIUM", "HIGH"):
                    continue
                sp1 = sp_lookup[r["v1_id"]].get(frame_id)
                sp2 = sp_lookup[r["v2_id"]].get(frame_id)
                if not sp1 or not sp2:
                    continue
                pt1 = (int(sp1[0]), int(sp1[1]))
                pt2 = (int(sp2[0]), int(sp2[1]))
                color = COLORS[r["risk_level"]]
                cv2.line(frame, pt1, pt2, color, 2)

                # Mid-point label: level + score + TTC
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                ttc_v = sf(r.get("ttc"))
                dist_v = sf(r.get("distance"))
                score_v = sf(r.get("risk_score"))
                line1 = f"{r['risk_level']} s={score_v:.2f}"
                line2 = f"TTC:{ttc_v:.1f}s" if not is_nan(ttc_v) else "TTC:n/a"
                line3 = f"D:{dist_v:.0f}px"  if not is_nan(dist_v) else ""
                cv2.putText(frame, line1, (mid_x, mid_y - 20),
                            FONT, FONT_SMALL, color, THICK)
                cv2.putText(frame, line2, (mid_x, mid_y - 6),
                            FONT, FONT_SMALL, color, THICK)
                if line3:
                    cv2.putText(frame, line3, (mid_x, mid_y + 8),
                                FONT, FONT_SMALL, color, THICK)

            # Top-right legend
            _draw_legend(frame, width)

            wtr.write(frame)
            if frame_id % 50 == 0:
                print(f"    Frame {frame_id}/{total}")
    finally:
        cap.release()
        wtr.release()


def _draw_legend(frame, width):
    """Draw a small colour legend in the top-right corner."""
    pad = 8
    item_h = 18
    items = [("HIGH", (0,0,220)), ("MEDIUM", (0,165,255)), ("LOW", (0,200,0))]
    box_w = 120
    box_h = pad * 2 + len(items) * item_h
    x0 = width - box_w - pad
    y0 = pad
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "RISK ENGINE", (x0 + 4, y0 + 13),
                FONT, 0.35, (200, 200, 200), 1)
    for i, (lbl, color) in enumerate(items):
        y = y0 + pad + 16 + i * item_h
        cv2.circle(frame, (x0 + 10, y), 5, color, -1)
        cv2.putText(frame, lbl, (x0 + 22, y + 4), FONT, 0.38, (220, 220, 220), 1)
