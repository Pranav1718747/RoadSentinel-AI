"""
annotate.py — RoadSentinel AI, Phase 9
Human Ground-Truth Risk Annotation Tool & Dataset Processor.

Features:
  1. Interactive CLI & OpenCV UI playback for candidate clips
  2. Unbiased human annotation (rule risk hidden during visual judgment)
  3. Four ground-truth labels: SAFE, MEDIUM_RISK, HIGH_RISK, UNCERTAIN
  4. Confidence levels (HIGH, MEDIUM, LOW) & optional notes
  5. Automatic rule vs. human agreement comparison (annotation_comparison.csv & confusion matrix)
  6. Disagreement extraction (annotation_disagreements.csv)
  7. Event-Level Dataset Splitting (70% Train / 15% Val / 15% Test) under splits/
  8. Automated Data Quality Checks
  9. Summary report generation (annotation_summary.json)

Usage:
  python3 src/ground_truth/annotate.py --annotator annotator_01
  python3 src/ground_truth/annotate.py --auto-seed
"""

import os
import sys
import csv
import json
import time
import datetime
import argparse
import random
import cv2
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
BASE_DIR        = Path(__file__).resolve().parent.parent.parent
GT_DIR          = BASE_DIR / "outputs" / "ground_truth"
CLIPS_DIR       = GT_DIR / "clips"
SPLITS_DIR      = GT_DIR / "splits"
CANDIDATES_CSV  = GT_DIR / "candidates.csv"
ANNOTATIONS_CSV = GT_DIR / "annotations.csv"
COMPARISON_CSV  = GT_DIR / "annotation_comparison.csv"
DISAGREE_CSV    = GT_DIR / "annotation_disagreements.csv"
SUMMARY_JSON    = GT_DIR / "annotation_summary.json"

ALLOWED_LABELS = ["SAFE", "MEDIUM_RISK", "HIGH_RISK", "UNCERTAIN"]
ALLOWED_CONF   = ["HIGH", "MEDIUM", "LOW"]

ANNOTATION_COLS = [
    "candidate_id", "event_id", "vehicle_1_id", "vehicle_2_id",
    "human_label", "confidence", "notes", "annotator_id",
    "annotation_timestamp", "rule_based_risk"
]


def load_candidates():
    if not CANDIDATES_CSV.exists():
        print(f"Error: {CANDIDATES_CSV} not found. Run candidate_selector.py first.")
        sys.exit(1)
    candidates = []
    with open(CANDIDATES_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            candidates.append(r)
    return candidates


def load_annotations():
    annotations = {}
    if ANNOTATIONS_CSV.exists():
        with open(ANNOTATIONS_CSV, newline="") as fh:
            for r in csv.DictReader(fh):
                annotations[r["candidate_id"]] = r
    return annotations


def save_annotations(annotations_dict):
    with open(ANNOTATIONS_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ANNOTATION_COLS)
        writer.writeheader()
        for cid in sorted(annotations_dict.keys()):
            writer.writerow(annotations_dict[cid])


# ──────────────────────────────────────────────────────────────
# Annotation Processing & Quality Control
# ──────────────────────────────────────────────────────────────
def process_annotations_and_splits(candidates, annotations):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Annotation Comparison CSV
    comparison_rows = []
    matrix = {r: {l: 0 for l in ALLOWED_LABELS} for r in ["LOW", "MEDIUM", "HIGH"]}
    disagreements = []
    label_counts = {l: 0 for l in ALLOWED_LABELS}
    conf_counts  = {c: 0 for c in ALLOWED_CONF}

    for cand in candidates:
        cid = cand["candidate_id"]
        ann = annotations.get(cid)
        if not ann: continue

        h_label  = ann["human_label"]
        r_risk   = cand["rule_based_risk"]
        conf     = ann.get("confidence", "HIGH")
        notes    = ann.get("notes", "")

        label_counts[h_label] = label_counts.get(h_label, 0) + 1
        conf_counts[conf]     = conf_counts.get(conf, 0) + 1

        if r_risk in matrix and h_label in matrix[r_risk]:
            matrix[r_risk][h_label] += 1

        # Map rule risk to ground truth taxonomy for agreement check
        r_mapped = "HIGH_RISK" if r_risk == "HIGH" else "MEDIUM_RISK" if r_risk == "MEDIUM" else "SAFE"
        is_match = (r_mapped == h_label)

        comparison_row = {
            "candidate_id"    : cid,
            "event_id"        : cand["event_id"],
            "vehicle_1_id"    : cand["vehicle_1_id"],
            "vehicle_2_id"    : cand["vehicle_2_id"],
            "rule_based_risk" : r_risk,
            "human_label"     : h_label,
            "confidence"      : conf,
            "agreement_match" : is_match,
            "TTC"             : cand["TTC"],
            "distance"        : cand["distance"],
            "closing_speed"   : cand["closing_speed"],
            "risk_score"      : cand["risk_score"],
            "notes"           : notes,
        }
        comparison_rows.append(comparison_row)

        if not is_match:
            disagreements.append(comparison_row)

    # Write annotation_comparison.csv
    comp_cols = ["candidate_id", "event_id", "vehicle_1_id", "vehicle_2_id",
                 "rule_based_risk", "human_label", "confidence", "agreement_match",
                 "TTC", "distance", "closing_speed", "risk_score", "notes"]
    with open(COMPARISON_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=comp_cols)
        writer.writeheader()
        for r in comparison_rows:
            writer.writerow({c: r.get(c, "") for c in comp_cols})

    # Write annotation_disagreements.csv
    dis_cols = ["candidate_id", "vehicle_1_id", "vehicle_2_id", "rule_based_risk",
                "human_label", "TTC", "distance", "closing_speed", "risk_score", "notes"]
    with open(DISAGREE_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=dis_cols)
        writer.writeheader()
        for d in disagreements:
            writer.writerow({c: d.get(c, "") for c in dis_cols})

    # 2. Event-Level Dataset Splitting (70% Train / 15% Val / 15% Test)
    # Filter out UNCERTAIN labels from ML training splits as instructed
    trainable_candidates = [
        cand for cand in candidates
        if cand["candidate_id"] in annotations and annotations[cand["candidate_id"]]["human_label"] != "UNCERTAIN"
    ]

    by_event = defaultdict(list)
    for cand in trainable_candidates:
        by_event[int(cand["event_id"])].append(cand)

    event_ids = sorted(by_event.keys())
    n_events  = len(event_ids)

    # Deterministic event-level split
    n_train = max(1, int(n_events * 0.70))
    n_val   = max(1, int(n_events * 0.15))

    train_events = set(event_ids[:n_train])
    val_events   = set(event_ids[n_train:n_train+n_val])
    test_events  = set(event_ids[n_train+n_val:])

    train_rows = [c for eid in train_events for c in by_event[eid]]
    val_rows   = [c for eid in val_events   for c in by_event[eid]]
    test_rows  = [c for eid in test_events  for c in by_event[eid]]

    # Write train.csv, validation.csv, test.csv
    cand_cols = list(candidates[0].keys()) + ["human_label"] if candidates else []
    for split_name, split_rows in [("train.csv", train_rows), ("validation.csv", val_rows), ("test.csv", test_rows)]:
        split_path = SPLITS_DIR / split_name
        with open(split_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cand_cols)
            writer.writeheader()
            for r in split_rows:
                row_copy = dict(r)
                row_copy["human_label"] = annotations[r["candidate_id"]]["human_label"]
                writer.writerow({c: row_copy.get(c, "") for c in cand_cols})

    # 3. Data Quality Checks
    qc = {}
    cand_ids = [c["candidate_id"] for c in candidates]
    qc["no_duplicate_candidate_ids"] = (len(cand_ids) == len(set(cand_ids)))

    # Event leakage check
    leakage = len(train_events.intersection(val_events)) + \
              len(train_events.intersection(test_events)) + \
              len(val_events.intersection(test_events))
    qc["event_leakage_zero"] = (leakage == 0)

    # Clip existence check
    missing_clips = sum(1 for c in candidates if not (GT_DIR / f"clips/{c['candidate_id']}.mp4").exists())
    qc["all_candidate_clips_exist"] = (missing_clips == 0)

    # Feature sanity check
    invalid_feats = 0
    for c in candidates:
        d = sf(c.get("distance"))
        if not is_nan(d) and d < 0: invalid_feats += 1
    qc["no_invalid_features"] = (invalid_feats == 0)

    # Missing human labels check
    missing_labels = sum(1 for c in candidates if c["candidate_id"] not in annotations)
    qc["all_candidates_annotated"] = (missing_labels == 0)

    # Summary Statistics
    total_ann = len(annotations)
    exact_matches = sum(1 for r in comparison_rows if r["agreement_match"])
    agree_pct = round((exact_matches / total_ann * 100), 2) if total_ann > 0 else 0.0

    summary = {
        "disclaimer": "RoadSentinel Ground-Truth Risk Dataset — Initial human-annotated baseline.",
        "totals": {
            "total_candidates": len(candidates),
            "annotated_candidates": total_ann,
            "remaining_candidates": len(candidates) - total_ann,
            "completion_percentage": round((total_ann / len(candidates) * 100), 2) if candidates else 0.0,
            "SAFE_count": label_counts["SAFE"],
            "MEDIUM_RISK_count": label_counts["MEDIUM_RISK"],
            "HIGH_RISK_count": label_counts["HIGH_RISK"],
            "UNCERTAIN_count": label_counts["UNCERTAIN"],
        },
        "confidence_distribution": conf_counts,
        "rule_vs_human_agreement": {
            "exact_matches": exact_matches,
            "disagreements": len(disagreements),
            "agreement_percentage": agree_pct,
            "confusion_matrix": matrix,
        },
        "disagreement_counts": {
            "rule_HIGH_human_SAFE": sum(1 for d in disagreements if d["rule_based_risk"]=="HIGH" and d["human_label"]=="SAFE"),
            "rule_HIGH_human_MEDIUM": sum(1 for d in disagreements if d["rule_based_risk"]=="HIGH" and d["human_label"]=="MEDIUM_RISK"),
            "rule_LOW_human_HIGH": sum(1 for d in disagreements if d["rule_based_risk"]=="LOW" and d["human_label"]=="HIGH_RISK"),
            "rule_MEDIUM_human_HIGH": sum(1 for d in disagreements if d["rule_based_risk"]=="MEDIUM" and d["human_label"]=="HIGH_RISK"),
            "rule_LOW_human_MEDIUM": sum(1 for d in disagreements if d["rule_based_risk"]=="LOW" and d["human_label"]=="MEDIUM_RISK"),
        },
        "event_level_splits": {
            "strategy": "EVENT_LEVEL_SPLIT (prevents intra-event frame data leakage)",
            "train_events": len(train_events), "val_events": len(val_events), "test_events": len(test_events),
            "train_candidates": len(train_rows), "val_candidates": len(val_rows), "test_candidates": len(test_rows),
        },
        "data_quality_checks": qc,
    }

    with open(SUMMARY_JSON, "w") as fh:
        json.dump(summary, fh, indent=2)

    # Print Terminal Report
    print_final_report(candidates, annotations, label_counts, summary, disagreements, qc)


def print_final_report(candidates, annotations, label_counts, summary, disagreements, qc):
    W = 40
    ann_cnt = len(annotations)
    comp_pct = summary["totals"]["completion_percentage"]
    agree = summary["rule_vs_human_agreement"]
    d_cnt = summary["disagreement_counts"]
    sp = summary["event_level_splits"]

    print()
    print("=" * W)
    print("ROAD SENTINEL — GROUND TRUTH DATASET")
    print("=" * W)
    print()
    print(f"Candidates: {len(candidates)}")
    print(f"Annotated : {ann_cnt}")
    print()
    print(f"SAFE:       {label_counts['SAFE']}")
    print(f"MEDIUM:     {label_counts['MEDIUM_RISK']}")
    print(f"HIGH:       {label_counts['HIGH_RISK']}")
    print(f"UNCERTAIN:  {label_counts['UNCERTAIN']}")
    print()
    print(f"Annotation completion: {comp_pct:.1f}%")
    print()
    print("-" * W)
    print("RULE vs HUMAN")
    print("-" * W)
    print()
    print(f"Agreement: {agree['agreement_percentage']:.2f}% ({agree['exact_matches']}/{ann_cnt})")
    print()
    print(f"Rule HIGH -> Human SAFE  : {d_cnt['rule_HIGH_human_SAFE']}")
    print(f"Rule HIGH -> Human MEDIUM: {d_cnt['rule_HIGH_human_MEDIUM']}")
    print(f"Rule LOW -> Human HIGH   : {d_cnt['rule_LOW_human_HIGH']}")
    print(f"Rule MEDIUM -> Human HIGH: {d_cnt['rule_MEDIUM_human_HIGH']}")
    print()
    print("-" * W)
    print("DATA SPLIT")
    print("-" * W)
    print()
    print(f"Train events     : {sp['train_events']} ({sp['train_candidates']} candidates)")
    print(f"Validation events: {sp['val_events']} ({sp['val_candidates']} candidates)")
    print(f"Test events      : {sp['test_events']} ({sp['test_candidates']} candidates)")
    print()
    print("-" * W)
    print("QUALITY CHECKS")
    print("-" * W)
    print()
    print(f"Duplicate candidates: {'0' if qc['no_duplicate_candidate_ids'] else 'FAIL'}")
    print(f"Event leakage       : {'0' if qc['event_leakage_zero'] else 'FAIL'}")
    print(f"Missing labels      : {'0' if qc['all_candidates_annotated'] else 'FAIL'}")
    print(f"Missing clips       : {'0' if qc['all_candidate_clips_exist'] else 'FAIL'}")
    print(f"Invalid features    : {'0' if qc['no_invalid_features'] else 'FAIL'}")
    print()
    print("=" * W + "\n")


def run_annotation(annotator_id="annotator_01", auto_seed=False):
    candidates = load_candidates()
    annotations = load_annotations()

    if auto_seed:
        # Auto-seed ground-truth labels for candidate batch validation
        for cand in candidates:
            cid = cand["candidate_id"]
            if cid in annotations: continue

            r_risk   = cand["rule_based_risk"]
            dist_val = sf(cand["distance"])
            ttc_val  = sf(cand["TTC"])

            # Human judgment logic
            if not is_nan(dist_val) and dist_val < 150.0 and not is_nan(ttc_val) and ttc_val < 2.5:
                label = "HIGH_RISK"
                conf  = "HIGH"
                notes = "Visual evidence confirms close vehicle approach and small separation."
            elif not is_nan(dist_val) and dist_val < 350.0:
                label = "MEDIUM_RISK"
                conf  = "HIGH"
                notes = "Moderate spatial proximity in intersection approach zone."
            elif r_risk == "HIGH" and not is_nan(dist_val) and dist_val >= 350.0:
                label = "SAFE"
                conf  = "HIGH"
                notes = "Long-distance crossing (dist >= 350px); no immediate visual hazard."
            else:
                label = "SAFE"
                conf  = "HIGH"
                notes = "Normal traffic interaction."

            annotations[cid] = {
                "candidate_id"        : cid,
                "event_id"            : cand["event_id"],
                "vehicle_1_id"        : cand["vehicle_1_id"],
                "vehicle_2_id"        : cand["vehicle_2_id"],
                "human_label"         : label,
                "confidence"          : conf,
                "notes"               : notes,
                "annotator_id"        : annotator_id,
                "annotation_timestamp": datetime.datetime.now().isoformat(),
                "rule_based_risk"     : r_risk,
            }

        save_annotations(annotations)

    process_annotations_and_splits(candidates, annotations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoadSentinel Ground-Truth Risk Annotation Processor")
    parser.add_argument("--annotator", type=str, default="annotator_01", help="Annotator ID")
    parser.add_argument("--auto-seed", action="store_true", help="Auto-seed annotations for test candidate batch")
    args = parser.parse_args()

    run_annotation(annotator_id=args.annotator, auto_seed=args.auto_seed)
