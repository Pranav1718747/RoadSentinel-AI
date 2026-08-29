"""
candidate_builder.py — RoadSentinel AI, Phase 9
Builds the balanced candidate dataset (candidates.csv) and renders individual
video clips (clips/cand_XXX.mp4) for human ground-truth annotation.

Candidate Sampling Strategy:
  ~20–25% HIGH-risk rule-based candidates
  ~25% MEDIUM-risk candidates
  ~50–55% LOW-risk / normal traffic interactions
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
FRAMES_DIR     = GT_DIR / "frames"
CANDIDATES_CSV = GT_DIR / "candidates.csv"
SCHEMA_JSON    = GT_DIR / "annotation_schema.json"


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


def build_candidates():
    GT_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    latest_run = locate_latest_run()
    print(f"Using pipeline output from: {latest_run}")

    risk_csv   = latest_run / "risk" / "risk_events.csv"
    risk_json  = latest_run / "risk" / "risk_events.json"
    bboxes_csv = latest_run / "tracking" / "tracks_v2.csv"
    traj_csv   = latest_run / "trajectories" / "validated_trajectories.csv"
    video_path = BASE_DIR / "data" / "samples" / "traffic.mp4"

    if not risk_csv.exists():
        print(f"Error: {risk_csv} not found.")
        sys.exit(1)

    # 1. Load interaction rows
    rows = []
    with open(risk_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    # Load events JSON
    with open(risk_json) as fh:
        events = json.load(fh)

    # Load bboxes
    raw_bboxes = {}
    with open(bboxes_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            raw_bboxes[(int(r["track_id"]), int(r["frame_id"]))] = (
                float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
            )

    # Load trajectories
    smooth_pos = defaultdict(dict)
    with open(traj_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            smooth_pos[int(r["track_id"])][int(r["frame_id"])] = (
                float(r["smooth_x"]), float(r["smooth_y"])
            )

    # 2. Extract Candidate Episodes
    # We want ~20 total candidates: 4 HIGH, 5 MEDIUM, 11 LOW
    candidate_episodes = []

    # HIGH candidates
    high_rows = [r for r in rows if r["risk_level"] == "HIGH"]
    high_events = [e for e in events if e["peak_risk_level"] == "HIGH"]

    # Candidate 1: Peak frame interval of HIGH Event (Pair 80, 83)
    if high_rows:
        r0 = high_rows[0]
        candidate_episodes.append({
            "event_id": int(r0.get("event_id", 14)),
            "v1_id": int(r0["v1_id"]), "v2_id": int(r0["v2_id"]),
            "v1_class": r0["v1_class"], "v2_class": r0["v2_class"],
            "start_frame": max(1, int(r0["frame_id"]) - 10),
            "end_frame": min(500, int(r0["frame_id"]) + 10),
            "peak_row": r0, "target_risk": "HIGH",
        })

    # Additional HIGH candidates around peak conflict moments in intersection
    for ev in high_events:
        v1, v2 = ev["v1_id"], ev["v2_id"]
        # Find peak score frame
        pair_rows = [r for r in rows if (int(r["v1_id"])==v1 and int(r["v2_id"])==v2) or (int(r["v1_id"])==v2 and int(r["v1_id"])==v1)]
        if pair_rows:
            peak = max(pair_rows, key=lambda x: sf(x["risk_score"]))
            pf = int(peak["frame_id"])
            candidate_episodes.append({
                "event_id": ev["event_id"],
                "v1_id": v1, "v2_id": v2,
                "v1_class": ev["v1_class"], "v2_class": ev["v2_class"],
                "start_frame": max(1, pf - 15), "end_frame": min(500, pf + 15),
                "peak_row": peak, "target_risk": "HIGH",
            })

    # MEDIUM candidates (from MEDIUM events)
    med_events = [e for e in events if e["peak_risk_level"] == "MEDIUM"]
    for ev in med_events[:5]:
        v1, v2 = ev["v1_id"], ev["v2_id"]
        pair_rows = [r for r in rows if (int(r["v1_id"])==v1 and int(r["v2_id"])==v2) or (int(r["v1_id"])==v2 and int(r["v1_id"])==v1)]
        if pair_rows:
            peak = max(pair_rows, key=lambda x: sf(x["risk_score"]))
            pf = int(peak["frame_id"])
            candidate_episodes.append({
                "event_id": ev["event_id"],
                "v1_id": v1, "v2_id": v2,
                "v1_class": ev["v1_class"], "v2_class": ev["v2_class"],
                "start_frame": max(1, pf - 20), "end_frame": min(500, pf + 20),
                "peak_row": peak, "target_risk": "MEDIUM",
            })

    # LOW candidates (from LOW interactions across diverse pairs)
    low_rows = [r for r in rows if r["risk_level"] == "LOW"]
    low_pairs = defaultdict(list)
    for r in low_rows:
        pair = (min(int(r["v1_id"]), int(r["v2_id"])), max(int(r["v1_id"]), int(r["v2_id"])))
        low_pairs[pair].append(r)

    # Sample representative frames from low pairs
    for pair, prs in list(low_pairs.items())[:12]:
        mid_idx = len(prs) // 2
        r = prs[mid_idx]
        pf = int(r["frame_id"])
        candidate_episodes.append({
            "event_id": 0,
            "v1_id": pair[0], "v2_id": pair[1],
            "v1_class": r["v1_class"], "v2_class": r["v2_class"],
            "start_frame": max(1, pf - 20), "end_frame": min(500, pf + 20),
            "peak_row": r, "target_risk": "LOW",
        })

    # Deduplicate candidate episodes by (v1, v2, start_frame)
    unique_candidates = []
    seen = set()
    for cand in candidate_episodes:
        key = (cand["v1_id"], cand["v2_id"], cand["start_frame"] // 10)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)

    print(f"Total unique candidates generated: {len(unique_candidates)}")

    # Write candidates.csv & render clips
    cand_rows = []
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for idx, cand in enumerate(unique_candidates, 1):
        cand_id = f"CAND_{idx:03d}"
        v1, v2  = cand["v1_id"], cand["v2_id"]
        sf_start, sf_end = cand["start_frame"], cand["end_frame"]
        peak = cand["peak_row"]
        clip_rel_path = f"clips/{cand_id}.mp4"
        clip_abs_path = GT_DIR / clip_rel_path

        # Render clip (30 frames context before & after)
        clip_start = max(1, sf_start - 30)
        clip_end   = min(500, sf_end + 30)

        cap = cv2.VideoCapture(str(video_path))
        wtr = cv2.VideoWriter(str(clip_abs_path), fourcc, fps, (w, h))

        fid = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            fid += 1
            if clip_start <= fid <= clip_end:
                # Draw bboxes for v1 and v2
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

                # Overlay candidate header (UNBIASED: NO RULE-BASED LABEL DISPLAYED)
                cv2.rectangle(frame, (10, 10), (450, 65), (20, 20, 20), -1)
                cv2.rectangle(frame, (10, 10), (450, 65), (200, 200, 200), 1)
                cv2.putText(frame, f"CANDIDATE ID: {cand_id}", (20, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(frame, f"Interaction: Vehicle {v1} ({cand['v1_class']}) <-> Vehicle {v2} ({cand['v2_class']})", (20, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

                wtr.write(frame)

        cap.release()
        wtr.release()

        # Save candidate row
        cand_rows.append({
            "candidate_id"          : cand_id,
            "event_id"              : cand["event_id"],
            "vehicle_1_id"          : v1,
            "vehicle_2_id"          : v2,
            "vehicle_1_class"       : cand["v1_class"],
            "vehicle_2_class"       : cand["v2_class"],
            "start_frame"           : sf_start,
            "end_frame"             : sf_end,
            "timestamp"             : peak.get("timestamp", ""),
            "risk_score"            : peak.get("risk_score", ""),
            "rule_based_risk"       : peak.get("risk_level", "LOW"),
            "TTC"                   : peak.get("ttc", ""),
            "distance"              : peak.get("distance", ""),
            "closing_speed"         : peak.get("closing_speed", ""),
            "relative_speed"        : peak.get("relative_speed", ""),
            "trajectory_intersection": peak.get("trajectory_crossing", ""),
            "approaching"           : peak.get("approaching", ""),
            "heading_difference"    : peak.get("heading_difference", ""),
            "clip_path"             : clip_rel_path,
        })

    # Write candidates.csv
    cand_cols = [
        "candidate_id", "event_id", "vehicle_1_id", "vehicle_2_id",
        "vehicle_1_class", "vehicle_2_class", "start_frame", "end_frame", "timestamp",
        "risk_score", "rule_based_risk", "TTC", "distance", "closing_speed", "relative_speed",
        "trajectory_intersection", "approaching", "heading_difference", "clip_path"
    ]
    with open(CANDIDATES_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cand_cols)
        writer.writeheader()
        for r in cand_rows:
            writer.writerow({c: r.get(c, "") for c in cand_cols})

    print(f"Saved {len(cand_rows)} candidates to {CANDIDATES_CSV}")

    # Write annotation_schema.json
    schema = {
        "dataset_name": "RoadSentinel Ground-Truth Risk Dataset",
        "phase": 9,
        "allowed_human_labels": ["SAFE", "MEDIUM_RISK", "HIGH_RISK", "UNCERTAIN"],
        "label_definitions": {
            "SAFE": "Normal traffic interaction with no meaningful indication of imminent conflict or dangerous behavior.",
            "MEDIUM_RISK": "Potentially dangerous interaction (aggressive approach, unsafe lane interaction, decreasing separation).",
            "HIGH_RISK": "Strong, imminent traffic conflict or near-miss visually supported (extremely small separation, evasive action).",
            "UNCERTAIN": "Video evidence inconclusive for reliable judgment."
        },
        "confidence_levels": ["HIGH", "MEDIUM", "LOW"],
        "keyboard_shortcuts": {
            "1": "SAFE",
            "2": "MEDIUM_RISK",
            "3": "HIGH_RISK",
            "4": "UNCERTAIN",
            "N": "NEXT",
            "P": "PREVIOUS",
            "S": "SAVE"
        },
        "disclaimer": "Human-annotated ground-truth dataset for RoadSentinel AI risk model training and evaluation."
    }
    with open(SCHEMA_JSON, "w") as fh:
        json.dump(schema, fh, indent=2)

    return cand_rows

if __name__ == "__main__":
    build_candidates()
