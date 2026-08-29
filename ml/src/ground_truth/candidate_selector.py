"""
candidate_selector.py — RoadSentinel AI, Phase 9
Candidate Interaction Selection & Video Clip Generator.

Reads latest run outputs:
  ml/outputs/runs/<latest_run>/risk/risk_events.csv
  ml/outputs/runs/<latest_run>/risk/risk_events.json
  ml/outputs/runs/<latest_run>/tracking/tracks_v2.csv
  ml/outputs/runs/<latest_run>/trajectories/validated_trajectories.csv
  ml/data/samples/traffic.mp4

Outputs:
  ml/outputs/ground_truth/candidates.csv
  ml/outputs/ground_truth/clips/CAND_XXX.mp4

Candidate Sampling Strategy:
  HIGH    ≈ 25%
  MEDIUM  ≈ 25%
  LOW     ≈ 50% (including randomly sampled normal traffic interactions)
"""

import csv
import json
import math
import sys
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

NaN = float("nan")

def is_nan(v):
    try: return v != v
    except: return True

def sf(v):
    if v is None or str(v).strip() == "": return NaN
    try: return float(v)
    except: return NaN


# Paths
BASE_DIR       = Path(__file__).resolve().parent.parent.parent
GT_DIR         = BASE_DIR / "outputs" / "ground_truth"
CLIPS_DIR      = GT_DIR / "clips"
CANDIDATES_CSV = GT_DIR / "candidates.csv"
CONTEXT_FRAMES = 30   # Configurable padding frames before & after candidate segment


def locate_latest_run():
    runs_dir = BASE_DIR / "outputs" / "runs"
    if not runs_dir.exists():
        print(f"Error: {runs_dir} not found.")
        sys.exit(1)
    runs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not runs:
        print(f"Error: No run directories found in {runs_dir}")
        sys.exit(1)
    return runs[-1]


def build_candidate_dataset():
    GT_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    latest_run = locate_latest_run()
    print(f"Loading pipeline outputs from latest run: {latest_run.name}")

    risk_csv   = latest_run / "risk" / "risk_events.csv"
    risk_json  = latest_run / "risk" / "risk_events.json"
    bboxes_csv = latest_run / "tracking" / "tracks_v2.csv"
    traj_csv   = latest_run / "trajectories" / "validated_trajectories.csv"
    feat_csv   = latest_run / "features" / "traffic_features.csv"
    video_path = BASE_DIR / "data" / "samples" / "traffic.mp4"

    # 1. Load interaction rows
    rows = []
    with open(risk_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    # Load events JSON
    with open(risk_json) as fh:
        events = json.load(fh)

    # Load raw bboxes
    raw_bboxes = {}
    with open(bboxes_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            raw_bboxes[(int(r["track_id"]), int(r["frame_id"]))] = (
                float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
            )

    # Load smooth positions
    smooth_pos = defaultdict(dict)
    with open(traj_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            smooth_pos[int(r["track_id"])][int(r["frame_id"])] = (
                float(r["smooth_x"]), float(r["smooth_y"])
            )

    # 2. Extract balanced candidate interaction episodes
    # Target: 5 HIGH, 5 MEDIUM, 10 LOW (20 candidates total)
    candidate_episodes = []

    # A. HIGH candidates
    high_rows = [r for r in rows if r["risk_level"] == "HIGH"]
    high_events = [e for e in events if e["peak_risk_level"] == "HIGH"]

    if high_rows:
        # Group HIGH rows by consecutive frame blocks
        high_rows_by_frame = defaultdict(list)
        for r in high_rows:
            high_rows_by_frame[int(r["frame_id"])].append(r)

        frames_sorted = sorted(high_rows_by_frame.keys())
        blocks = []
        curr_b = [frames_sorted[0]]
        for f in frames_sorted[1:]:
            if f - curr_b[-1] <= 5:
                curr_b.append(f)
            else:
                blocks.append(curr_b)
                curr_b = [f]
        blocks.append(curr_b)

        for b_idx, block in enumerate(blocks[:5]):
            block_rows = [r for f in block for r in high_rows_by_frame[f]]
            peak_r = max(block_rows, key=lambda x: sf(x["risk_score"]))
            candidate_episodes.append({
                "event_id": int(peak_r.get("event_id", 2)),
                "v1_id": int(peak_r["v1_id"]), "v2_id": int(peak_r["v2_id"]),
                "v1_class": peak_r["v1_class"], "v2_class": peak_r["v2_class"],
                "start_frame": max(1, min(block) - 5),
                "end_frame": min(500, max(block) + 5),
                "peak_row": peak_r, "target_risk": "HIGH",
            })

    # B. MEDIUM candidates
    med_events = [e for e in events if e["peak_risk_level"] == "MEDIUM"]
    med_rows = [r for r in rows if r["risk_level"] == "MEDIUM"]
    med_pairs = defaultdict(list)
    for r in med_rows:
        pair = (min(int(r["v1_id"]), int(r["v2_id"])), max(int(r["v1_id"]), int(r["v2_id"])))
        med_pairs[pair].append(r)

    for pair, prs in list(med_pairs.items())[:5]:
        peak_r = max(prs, key=lambda x: sf(x["risk_score"]))
        pf = int(peak_r["frame_id"])
        candidate_episodes.append({
            "event_id": int(peak_r.get("event_id", 0)),
            "v1_id": pair[0], "v2_id": pair[1],
            "v1_class": peak_r["v1_class"], "v2_class": peak_r["v2_class"],
            "start_frame": max(1, pf - 15), "end_frame": min(500, pf + 15),
            "peak_row": peak_r, "target_risk": "MEDIUM",
        })

    # C. LOW candidates (normal traffic, distant passing, parallel travel)
    low_rows = [r for r in rows if r["risk_level"] == "LOW"]
    low_pairs = defaultdict(list)
    for r in low_rows:
        pair = (min(int(r["v1_id"]), int(r["v2_id"])), max(int(r["v1_id"]), int(r["v2_id"])))
        low_pairs[pair].append(r)

    for pair, prs in list(low_pairs.items())[:10]:
        mid_idx = len(prs) // 2
        peak_r = prs[mid_idx]
        pf = int(peak_r["frame_id"])
        candidate_episodes.append({
            "event_id": 0,
            "v1_id": pair[0], "v2_id": pair[1],
            "v1_class": peak_r["v1_class"], "v2_class": peak_r["v2_class"],
            "start_frame": max(1, pf - 15), "end_frame": min(500, pf + 15),
            "peak_row": peak_r, "target_risk": "LOW",
        })

    # Deduplicate candidate episodes
    unique_candidates = []
    seen = set()
    for cand in candidate_episodes:
        key = (cand["v1_id"], cand["v2_id"], cand["start_frame"] // 10)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)

    print(f"Extracted {len(unique_candidates)} unique candidate episodes.")

    # Render video clips & build candidates.csv
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cand_rows = []

    for idx, cand in enumerate(unique_candidates, 1):
        cand_id = f"CAND_{idx:03d}"
        v1, v2  = cand["v1_id"], cand["v2_id"]
        sf_start, sf_end = cand["start_frame"], cand["end_frame"]
        peak = cand["peak_row"]
        clip_rel_path = f"clips/{cand_id}.mp4"
        clip_abs_path = GT_DIR / clip_rel_path

        # Context window: CONTEXT_FRAMES (30) before & after
        clip_start = max(1, sf_start - CONTEXT_FRAMES)
        clip_end   = min(500, sf_end + CONTEXT_FRAMES)

        # Render clip
        cap = cv2.VideoCapture(str(video_path))
        wtr = cv2.VideoWriter(str(clip_abs_path), fourcc, fps, (w, h))

        fid = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            fid += 1
            if clip_start <= fid <= clip_end:
                b1 = raw_bboxes.get((v1, fid))
                b2 = raw_bboxes.get((v2, fid))

                if b1:
                    cv2.rectangle(frame, (int(b1[0]), int(b1[1])), (int(b1[2]), int(b1[3])), (0, 215, 255), 2)
                    cv2.putText(frame, f"v1:{v1} ({cand['v1_class']})", (int(b1[0]), max(int(b1[1])-5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)

                if b2:
                    cv2.rectangle(frame, (int(b2[0]), int(b2[1])), (int(b2[2]), int(b2[3])), (255, 165, 0), 2)
                    cv2.putText(frame, f"v2:{v2} ({cand['v2_class']})", (int(b2[0]), max(int(b2[1])-5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)

                if b1 and b2:
                    cx1, cy1 = (b1[0]+b1[2])/2, (b1[1]+b1[3])/2
                    cx2, cy2 = (b2[0]+b2[2])/2, (b2[1]+b2[3])/2
                    cv2.line(frame, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (0, 255, 255), 2)

                # Overlay Candidate Card (UNBIASED: NO LOW/MEDIUM/HIGH RULE LABEL SHOWN)
                cv2.rectangle(frame, (10, 10), (480, 65), (20, 20, 20), -1)
                cv2.rectangle(frame, (10, 10), (480, 65), (200, 200, 200), 1)
                cv2.putText(frame, f"CANDIDATE ID: {cand_id} (Frames {sf_start}-{sf_end})", (20, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(frame, f"Vehicle {v1} ({cand['v1_class']}) <-> Vehicle {v2} ({cand['v2_class']})", (20, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

                wtr.write(frame)

        cap.release()
        wtr.release()

        # Build candidate record
        cand_rows.append({
            "candidate_id"          : cand_id,
            "event_id"              : cand["event_id"],
            "vehicle_1_id"          : v1,
            "vehicle_2_id"          : v2,
            "vehicle_1_class"       : cand["v1_class"],
            "vehicle_2_class"       : cand["v2_class"],
            "start_frame"           : sf_start,
            "end_frame"             : sf_end,
            "start_time"            : round(sf_start / fps, 2),
            "end_time"              : round(sf_end / fps, 2),
            "risk_score"            : peak.get("risk_score", ""),
            "rule_based_risk"       : peak.get("risk_level", "LOW"),
            "TTC"                   : peak.get("ttc", ""),
            "distance"              : peak.get("distance", ""),
            "closing_speed"         : peak.get("closing_speed", ""),
            "relative_speed"        : peak.get("relative_speed", ""),
            "acceleration"          : peak.get("v1_acceleration", ""),
            "heading_difference"    : peak.get("heading_difference", ""),
            "trajectory_intersection": peak.get("trajectory_crossing", ""),
            "approaching"           : peak.get("approaching", ""),
        })

    # Write candidates.csv
    cand_cols = [
        "candidate_id", "event_id", "vehicle_1_id", "vehicle_2_id",
        "vehicle_1_class", "vehicle_2_class", "start_frame", "end_frame",
        "start_time", "end_time", "risk_score", "rule_based_risk",
        "TTC", "distance", "closing_speed", "relative_speed",
        "acceleration", "heading_difference", "trajectory_intersection", "approaching"
    ]
    with open(CANDIDATES_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cand_cols)
        writer.writeheader()
        for r in cand_rows:
            writer.writerow({c: r.get(c, "") for c in cand_cols})

    print(f"Saved {len(cand_rows)} candidate interactions to {CANDIDATES_CSV}")
    return cand_rows

if __name__ == "__main__":
    build_candidate_dataset()
