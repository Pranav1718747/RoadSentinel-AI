"""
run_risk_engine.py — RoadSentinel AI, Phase 7
Main orchestrator: loads features, scores interactions, groups events,
writes all outputs, and prints the terminal report.

SCIENTIFIC LIMITATION (reiterated at runtime):
This is a BASELINE RULE-BASED RISK ENGINE.
It is NOT a trained predictive ML model.
TTC and distance are image-plane quantities (pixels, px/s).
Thresholds are initial engineering assumptions — not validated against
ground-truth traffic-conflict data.
Do NOT claim this system predicts actual collisions with proven accuracy.

Run from the ml/ directory:
    python3 src/risk/run_risk_engine.py
"""

import csv
import json
import sys
import math
from pathlib import Path
from collections import defaultdict

# Add risk module to path (run from ml/)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import risk_config as cfg
from risk_engine   import evaluate_interaction
from event_grouper import group_events
from risk_visualizer import render_risk_video

NaN = float("nan")

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent.parent
IN_FEATURES   = BASE_DIR / "outputs" / "features"    / "traffic_features.csv"
IN_TRAJ       = BASE_DIR / "outputs" / "trajectories" / "validated_trajectories.csv"
IN_BBOXES     = BASE_DIR / "outputs" / "tracking"    / "tracks_v2.csv"
IN_VIDEO      = BASE_DIR / "data"    / "samples"     / "traffic.mp4"
OUT_DIR       = BASE_DIR / "outputs" / "risk"

OUT_EVENTS_CSV  = OUT_DIR / "risk_events.csv"
OUT_SUMMARY_JSON= OUT_DIR / "risk_summary.json"
OUT_VID         = OUT_DIR / "risk_visualization.mp4"
OUT_EVENTS_JSON = OUT_DIR / "risk_events.json"


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def sf(v):
    if v is None or str(v).strip() == "": return NaN
    try: return float(v)
    except: return NaN

def is_nan(v):
    try: return v != v
    except: return True

def safe_bool(v):
    return str(v).strip().lower() == "true"


# ──────────────────────────────────────────────────────────────
# 1. Load feature rows
# ──────────────────────────────────────────────────────────────
def load_features(path):
    if not path.exists():
        print(f"Error: {path} not found."); sys.exit(1)
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def load_trajectories(path):
    """Load smooth positions for visualizer: track_id → [(frame_id, sx, sy)]."""
    traj = defaultdict(list)
    if not path.exists():
        return traj
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            traj[int(r["track_id"])].append(
                (int(r["frame_id"]), float(r["smooth_x"]), float(r["smooth_y"]))
            )
    for tid in traj:
        traj[tid].sort(key=lambda x: x[0])
    return dict(traj)


def load_raw_bboxes(path):
    bboxes = {}
    if not path.exists():
        return bboxes
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (int(r["track_id"]), int(r["frame_id"]))
            bboxes[key] = (float(r["x1"]), float(r["y1"]),
                           float(r["x2"]), float(r["y2"]))
    return bboxes


# ──────────────────────────────────────────────────────────────
# 2. Score every interaction row
# ──────────────────────────────────────────────────────────────
def score_all(feature_rows):
    scored = []
    for r in feature_rows:
        result = evaluate_interaction(r)
        scored_row = dict(r)           # copy original columns
        scored_row.update(result)
        scored.append(scored_row)
    return scored


# ──────────────────────────────────────────────────────────────
# 3. Write risk_events.csv
# ──────────────────────────────────────────────────────────────
EVENTS_COLS = [
    "timestamp", "frame_id",
    "v1_id", "v1_class",
    "v2_id", "v2_class",
    "distance", "relative_speed", "closing_speed", "ttc",
    "trajectory_crossing", "approaching", "heading_difference",
    "risk_score", "risk_level",
    "ttc_score", "distance_score", "closing_speed_score",
    "trajectory_score", "heading_score",
    "risk_reasons",
]

def write_events_csv(scored_rows):
    with open(OUT_EVENTS_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=EVENTS_COLS)
        writer.writeheader()
        for r in scored_rows:
            writer.writerow({col: r.get(col, "") for col in EVENTS_COLS})


# ──────────────────────────────────────────────────────────────
# 4. Sanity checks
# ──────────────────────────────────────────────────────────────
def run_sanity_checks(scored_rows):
    errors = []
    warnings = []

    for i, r in enumerate(scored_rows):
        dist  = sf(r.get("distance"))
        ttc   = sf(r.get("ttc"))
        cs    = sf(r.get("closing_speed"))
        score = sf(r.get("risk_score"))
        app   = safe_bool(r.get("approaching"))
        level = r.get("risk_level")
        ttc_s = sf(r.get("ttc_score"))

        # Negative distance
        if not is_nan(dist) and dist < 0:
            errors.append(f"[ERROR] Row {i}: negative distance={dist:.3f}")

        # TTC risk on non-approaching
        if not app and not is_nan(ttc_s) and ttc_s > 0:
            errors.append(f"[ERROR] Row {i}: ttc_score={ttc_s} but approaching=False")

        # TTC risk when closing_speed <= 0
        if not is_nan(cs) and cs <= 0 and not is_nan(ttc_s) and ttc_s > 0:
            errors.append(f"[ERROR] Row {i}: ttc_score>0 but closing_speed={cs:.3f}<=0")

        # Risk score range
        if not is_nan(score) and not (0.0 <= score <= 1.0):
            errors.append(f"[ERROR] Row {i}: risk_score={score} out of [0,1]")

        # HIGH requires evidence
        if level == "HIGH" and not is_nan(score) and score < cfg.SCORE_HIGH:
            errors.append(f"[ERROR] Row {i}: risk_level=HIGH but score={score}<{cfg.SCORE_HIGH}")

        # Duplicate alert check (same frame, same pair)
        # (structural check — CSV write is unique by design)

        # Speed spikes — warnings only
        for col in ["v1_speed", "v2_speed"]:
            v = sf(r.get(col))
            if not is_nan(v) and v > 500:
                warnings.append(f"[WARN] Row {i}: {col}={v:.1f}px/s (spike)")

    print(f"\nSanity checks: {len(errors)} error(s), {len(warnings)} warning(s)")
    for e in errors[:20]:
        print(f"  {e}")
    for w in warnings[:5]:
        print(f"  {w}")
    if len(errors) == 0:
        print("  All sanity checks passed.")
    return errors, warnings


# ──────────────────────────────────────────────────────────────
# 5. Build risk_summary.json
# ──────────────────────────────────────────────────────────────
def build_summary(scored_rows, events, errors, warnings):
    total = len(scored_rows)
    level_counts = defaultdict(int)
    for r in scored_rows:
        level_counts[r["risk_level"]] += 1

    high_events  = [e for e in events if e["peak_risk_level"] == "HIGH"]
    med_events   = [e for e in events if e["peak_risk_level"] == "MEDIUM"]

    def evt_block(evts):
        if not evts: return []
        return [{
            "event_id"          : e["event_id"],
            "v1_id"             : e["v1_id"],
            "v2_id"             : e["v2_id"],
            "v1_class"          : e["v1_class"],
            "v2_class"          : e["v2_class"],
            "duration_frames"   : e["event_duration_frames"],
            "duration_seconds"  : e["duration_seconds"],
            "min_ttc"           : e["min_ttc"],
            "min_distance"      : e["min_distance"],
            "max_closing_speed" : e["max_closing_speed"],
            "max_risk_score"    : e["max_risk_score"],
            "top_reasons"       : e["top_reasons"],
        } for e in evts]

    # Overall stats
    all_ttcs  = [sf(r["ttc"])           for r in scored_rows if not is_nan(sf(r["ttc"]))]
    all_dists = [sf(r["distance"])      for r in scored_rows if not is_nan(sf(r["distance"]))]
    all_cs    = [sf(r["closing_speed"]) for r in scored_rows if not is_nan(sf(r["closing_speed"]))]

    summary = {
        "scientific_limitation": (
            "BASELINE RULE-BASED RISK ENGINE — NOT an ML model. "
            "TTC and distance are image-plane measurements (px, px/s). "
            "Thresholds are initial engineering assumptions and have NOT been "
            "validated against ground-truth traffic-conflict data."
        ),
        "thresholds_used": {
            "TTC_HIGH_s"          : cfg.TTC_HIGH,
            "TTC_MEDIUM_s"        : cfg.TTC_MEDIUM,
            "DIST_HIGH_px"        : cfg.DIST_HIGH,
            "DIST_MEDIUM_px"      : cfg.DIST_MEDIUM,
            "CS_HIGH_px_s"        : cfg.CS_HIGH,
            "CS_MEDIUM_px_s"      : cfg.CS_MEDIUM,
            "HEADING_HIGH_deg"    : cfg.HEADING_HIGH,
            "HEADING_MEDIUM_deg"  : cfg.HEADING_MEDIUM,
            "SCORE_HIGH"          : cfg.SCORE_HIGH,
            "SCORE_MEDIUM"        : cfg.SCORE_MEDIUM,
        },
        "totals": {
            "total_interactions"  : total,
            "low_risk"            : level_counts.get("LOW", 0),
            "medium_risk"         : level_counts.get("MEDIUM", 0),
            "high_risk"           : level_counts.get("HIGH", 0),
            "unique_risk_events"  : len(events),
            "high_risk_events"    : len(high_events),
            "medium_risk_events"  : len(med_events),
        },
        "statistics": {
            "min_ttc_valid"       : round(min(all_ttcs), 4)   if all_ttcs  else None,
            "min_distance"        : round(min(all_dists), 2)  if all_dists else None,
            "max_closing_speed"   : round(max(all_cs), 2)     if all_cs    else None,
            "mean_closing_speed"  : round(sum(all_cs)/len(all_cs), 2) if all_cs else None,
        },
        "sanity": {
            "errors"   : len(errors),
            "warnings" : len(warnings),
        },
        "high_risk_events"   : evt_block(high_events),
        "medium_risk_events" : evt_block(med_events[:10]),  # top 10
    }

    with open(OUT_SUMMARY_JSON, "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


# ──────────────────────────────────────────────────────────────
# 6. Write risk_events.json (per-event detailed records)
# ──────────────────────────────────────────────────────────────
def write_events_json(events):
    with open(OUT_EVENTS_JSON, "w") as fh:
        json.dump(events, fh, indent=2)


# ──────────────────────────────────────────────────────────────
# 7. Terminal report
# ──────────────────────────────────────────────────────────────
def print_terminal_report(scored_rows, events, summary):
    W = 44
    level_counts = summary["totals"]

    # Top reasons across all HIGH+MEDIUM events
    reason_freq = defaultdict(int)
    for r in scored_rows:
        if r["risk_level"] in ("HIGH", "MEDIUM"):
            for rr in r.get("risk_reasons", "").split(";"):
                rs = rr.strip()
                if rs and rs != "No significant risk indicators":
                    reason_freq[rs] += 1
    top_reasons = sorted(reason_freq.items(), key=lambda x: -x[1])[:5]

    print()
    print("=" * W)
    print("  ROAD SENTINEL — RISK ENGINE")
    print("=" * W)
    print(f"  Total interactions   : {level_counts['total_interactions']}")
    print(f"  LOW                  : {level_counts['low_risk']}")
    print(f"  MEDIUM               : {level_counts['medium_risk']}")
    print(f"  HIGH                 : {level_counts['high_risk']}")
    print()
    print(f"  Unique risk events   : {level_counts['unique_risk_events']}")
    print(f"    HIGH events        : {level_counts['high_risk_events']}")
    print(f"    MEDIUM events      : {level_counts['medium_risk_events']}")
    print()
    st = summary["statistics"]
    print(f"  Highest risk score   : {max(sf(r['risk_score']) for r in scored_rows if not is_nan(sf(r['risk_score']))):.4f}")
    if st["min_ttc_valid"] is not None:
        print(f"  Lowest valid TTC     : {st['min_ttc_valid']:.3f}s")
    if st["min_distance"] is not None:
        print(f"  Minimum distance     : {st['min_distance']:.1f}px")
    if st["max_closing_speed"] is not None:
        print(f"  Max closing speed    : {st['max_closing_speed']:.1f}px/s")
    print()
    print("  Top risk reasons:")
    for reason, cnt in top_reasons:
        print(f"    [{cnt:>4}x] {reason}")
    print("=" * W)
    print()

    # Top 10 events
    print("  Top 10 highest-risk events:")
    print(f"  {'evt':>4}  {'v1':>4}–{'v2':<4}  {'lvl':<6}  {'score':>6}  {'TTC':>6}  {'dist':>6}  {'dur':>5}")
    for e in events[:10]:
        ttc_s  = f"{e['min_ttc']:.2f}s"    if e["min_ttc"]    is not None else "  n/a"
        dist_s = f"{e['min_distance']:.0f}px" if e["min_distance"] is not None else " n/a"
        print(f"  {e['event_id']:>4}  {e['v1_id']:>4}–{e['v2_id']:<4}  {e['peak_risk_level']:<6}"
              f"  {e['max_risk_score']:>6.4f}  {ttc_s:>6}  {dist_s:>6}  {e['event_duration_frames']:>4}f")
    print()
    print("  NOTE: This is a BASELINE RULE-BASED system. TTC/distance are")
    print("  image-plane quantities. Thresholds require ground-truth calibration.")
    print("=" * W)
    print()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("ROAD SENTINEL — PHASE 7: RULE-BASED RISK ENGINE")
    print("─" * 48)
    print("SCIENTIFIC LIMITATION: BASELINE RULE-BASED system.")
    print("Image-plane quantities only. Thresholds not yet validated.")
    print("─" * 48)

    # 1. Load
    print("\nLoading traffic_features.csv ...")
    feature_rows = load_features(IN_FEATURES)
    print(f"  {len(feature_rows)} interaction rows loaded.")

    print("Loading trajectory positions ...")
    smooth_positions = load_trajectories(IN_TRAJ)
    valid_track_ids = list(smooth_positions.keys())
    print(f"  {len(valid_track_ids)} valid tracks loaded.")

    print("Loading raw bounding boxes ...")
    raw_bboxes = load_raw_bboxes(IN_BBOXES)
    print(f"  {len(raw_bboxes)} bbox records loaded.")

    # 2. Score
    print("\nScoring interactions ...")
    scored_rows = score_all(feature_rows)
    level_counts = defaultdict(int)
    for r in scored_rows:
        level_counts[r["risk_level"]] += 1
    print(f"  LOW={level_counts['LOW']}  MEDIUM={level_counts['MEDIUM']}  HIGH={level_counts['HIGH']}")

    # 3. Write events CSV
    print("\nWriting risk_events.csv ...")
    write_events_csv(scored_rows)
    print(f"  Saved: {OUT_EVENTS_CSV}")

    # 4. Sanity checks
    errors, warnings = run_sanity_checks(scored_rows)

    # 5. Group events
    print("\nGrouping consecutive risk events ...")
    events = group_events(scored_rows)
    print(f"  {len(events)} unique risk events identified.")

    # 6. Summary JSON
    print("Writing risk_summary.json ...")
    summary = build_summary(scored_rows, events, errors, warnings)
    print(f"  Saved: {OUT_SUMMARY_JSON}")

    # 7. Events JSON
    print("Writing risk_events.json ...")
    write_events_json(events)
    print(f"  Saved: {OUT_EVENTS_JSON}")

    # 8. Build frame→scored_rows lookup for visualiser
    frame_scored = defaultdict(list)
    for r in scored_rows:
        frame_scored[int(r["frame_id"])].append(r)

    # 9. Video
    print("\nRendering risk_visualization.mp4 ...")
    render_risk_video(
        input_video_path   = IN_VIDEO,
        output_video_path  = OUT_VID,
        raw_bboxes         = raw_bboxes,
        frame_scored_rows  = dict(frame_scored),
        valid_track_ids    = valid_track_ids,
        smooth_positions   = smooth_positions,
    )
    print(f"  Saved: {OUT_VID}")

    # 10. Terminal report
    print_terminal_report(scored_rows, events, summary)

    # 11. File manifest
    print("Output files:")
    for p in [OUT_EVENTS_CSV, OUT_SUMMARY_JSON, OUT_VID, OUT_EVENTS_JSON]:
        sz = p.stat().st_size if p.exists() else 0
        print(f"  {p}  ({sz:,} bytes)")

    # 12. Show first 10 rows of risk_events.csv
    print("\nFirst 10 rows of risk_events.csv (selected columns):")
    with open(OUT_EVENTS_CSV, newline="") as fh:
        reader = csv.DictReader(fh)
        show = ["frame_id","v1_id","v2_id","distance","ttc",
                "closing_speed","risk_score","risk_level","risk_reasons"]
        hdr = " | ".join(f"{c:>14}" for c in show[:8])
        print("  " + hdr)
        print("  " + "-"*len(hdr))
        for i, row in enumerate(reader):
            if i >= 10: break
            vals = " | ".join(f"{row.get(c,''):>14}" for c in show[:8])
            print(f"  {vals}")
            reason_short = row.get("risk_reasons","")[:60]
            print(f"    reasons: {reason_short}")


if __name__ == "__main__":
    main()
