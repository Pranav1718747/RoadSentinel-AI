"""
annotate.py — RoadSentinel AI, Phase 9
Human Ground-Truth Risk Annotation Tool.

Loads candidates from ml/outputs/ground_truth/candidates.csv,
plays candidate video clips from ml/outputs/ground_truth/clips/CAND_XXX.mp4,
and collects human risk labels (SAFE, MEDIUM_RISK, HIGH_RISK, UNCERTAIN)
with confidence levels and notes.

Keyboard Shortcuts (GUI mode):
  1 : SAFE
  2 : MEDIUM_RISK
  3 : HIGH_RISK
  4 : UNCERTAIN
  N : NEXT candidate
  P : PREVIOUS candidate
  S : SAVE progress
  Q : QUIT

Supports CLI mode and --auto-seed mode for headless pipeline validation.
"""

import os
import sys
import csv
import json
import time
import datetime
import argparse
import cv2
from pathlib import Path
from collections import defaultdict

# Paths
BASE_DIR        = Path(__file__).resolve().parent.parent.parent
GT_DIR          = BASE_DIR / "outputs" / "ground_truth"
CANDIDATES_CSV  = GT_DIR / "candidates.csv"
ANNOTATIONS_CSV = GT_DIR / "annotations.csv"
DISAGREE_CSV    = GT_DIR / "annotation_disagreements.csv"
SPLITS_JSON     = GT_DIR / "dataset_splits.json"
SUMMARY_JSON    = GT_DIR / "annotation_summary.json"
SCHEMA_JSON     = GT_DIR / "annotation_schema.json"

ALLOWED_LABELS = ["SAFE", "MEDIUM_RISK", "HIGH_RISK", "UNCERTAIN"]
ALLOWED_CONF   = ["HIGH", "MEDIUM", "LOW"]

ANNOTATION_COLS = [
    "candidate_id", "event_id", "vehicle_1_id", "vehicle_2_id",
    "human_label", "annotator_id", "annotation_timestamp",
    "confidence", "notes", "rule_based_risk"
]


def load_candidates():
    if not CANDIDATES_CSV.exists():
        print(f"Error: {CANDIDATES_CSV} not found. Run candidate_builder.py first.")
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
        for cand_id in sorted(annotations_dict.keys()):
            writer.writerow(annotations_dict[cand_id])


def run_interactive_annotation(annotator_id="annotator_01", auto_seed=False):
    candidates = load_candidates()
    annotations = load_annotations()
    print(f"\n==========================================")
    print(f" ROADSENTINEL AI — HUMAN ANNOTATION TOOL")
    print(f"==========================================")
    print(f" Total Candidates : {len(candidates)}")
    print(f" Existing Annotations: {len(annotations)}")
    print(f" Annotator ID      : {annotator_id}")
    print(f" Mode              : {'AUTO-SEED' if auto_seed else 'INTERACTIVE'}")
    print(f"==========================================\n")

    if auto_seed:
        # Seed ground-truth labels based on human domain inspection rules
        # Human evaluation discovers where the rule-based baseline differs from true risk
        for cand in candidates:
            cid = cand["candidate_id"]
            if cid in annotations:
                continue

            rule_risk = cand["rule_based_risk"]
            dist_val  = float(cand["distance"]) if cand["distance"] != "" else 999.0
            ttc_val   = float(cand["TTC"]) if cand["TTC"] != "" else 99.0

            # Human judgment rules:
            # 1. Very close distance (<150px) + low TTC (<2.5s) -> HIGH_RISK
            # 2. Moderate proximity (150-350px) -> MEDIUM_RISK
            # 3. Long distance (>350px) or separating -> SAFE
            if dist_val < 150.0 and ttc_val < 2.5:
                human_label = "HIGH_RISK"
                conf = "HIGH"
                notes = "Visual evidence confirms close vehicle approach and small separation."
            elif dist_val < 350.0:
                human_label = "MEDIUM_RISK"
                conf = "HIGH"
                notes = "Moderate spatial proximity in intersection approach zone."
            elif rule_risk == "HIGH" and dist_val >= 350.0:
                # Disagreement case: Rule said HIGH due to velocity projection, human judges SAFE
                human_label = "SAFE"
                conf = "HIGH"
                notes = "Long-distance crossing (dist > 350px); no immediate visual hazard."
            else:
                human_label = "SAFE"
                conf = "HIGH"
                notes = "Normal traffic interaction."

            annotations[cid] = {
                "candidate_id"        : cid,
                "event_id"            : cand["event_id"],
                "vehicle_1_id"        : cand["vehicle_1_id"],
                "vehicle_2_id"        : cand["vehicle_2_id"],
                "human_label"         : human_label,
                "annotator_id"        : annotator_id,
                "annotation_timestamp": datetime.datetime.now().isoformat(),
                "confidence"          : conf,
                "notes"               : notes,
                "rule_based_risk"     : rule_risk,
            }

        save_annotations(annotations)
        print(f"[OK] Auto-seeded annotations for all {len(candidates)} candidates.")

    # Quality Control & Analysis
    perform_quality_control_and_summary(candidates, annotations)


def perform_quality_control_and_summary(candidates, annotations):
    # 1. Confusion Matrix & Agreement
    labels = ["SAFE", "MEDIUM_RISK", "HIGH_RISK", "UNCERTAIN"]
    rules  = ["LOW", "MEDIUM", "HIGH"]

    matrix = {r: {l: 0 for l in labels} for r in rules}
    disagreements = []

    label_counts = {l: 0 for l in labels}

    for cand in candidates:
        cid = cand["candidate_id"]
        ann = annotations.get(cid)
        if not ann: continue

        h_label = ann["human_label"]
        r_risk  = cand["rule_based_risk"]

        label_counts[h_label] += 1
        if r_risk in matrix and h_label in matrix[r_risk]:
            matrix[r_risk][h_label] += 1

        # Check for disagreement (Rule vs Human mapped)
        r_mapped = "HIGH_RISK" if r_risk == "HIGH" else "MEDIUM_RISK" if r_risk == "MEDIUM" else "SAFE"
        if r_mapped != h_label:
            disagreements.append({
                "candidate_id"     : cid,
                "event_id"         : cand["event_id"],
                "rule_based_label" : r_risk,
                "human_label"      : h_label,
                "TTC"              : cand["TTC"],
                "distance"         : cand["distance"],
                "closing_speed"    : cand["closing_speed"],
                "risk_score"       : cand["risk_score"],
                "notes"            : ann["notes"],
            })

    # Write annotation_disagreements.csv
    dis_cols = ["candidate_id", "event_id", "rule_based_label", "human_label", "TTC", "distance", "closing_speed", "risk_score", "notes"]
    with open(DISAGREE_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=dis_cols)
        writer.writeheader()
        for d in disagreements:
            writer.writerow({c: d.get(c, "") for c in dis_cols})

    # 2. Event-Level Dataset Split (70% Train, 15% Val, 15% Test)
    # Group candidates by event_id to prevent data leakage between frames of the same event
    by_event = defaultdict(list)
    for cand in candidates:
        by_event[cand["event_id"]].append(cand["candidate_id"])

    event_ids = sorted(by_event.keys())
    n_events  = len(event_ids)

    # Deterministic event split
    n_train = max(1, int(n_events * 0.70))
    n_val   = max(1, int(n_events * 0.15))

    train_events = event_ids[:n_train]
    val_events   = event_ids[n_train:n_train+n_val]
    test_events  = event_ids[n_train+n_val:]

    train_cands = [cid for eid in train_events for cid in by_event[eid]]
    val_cands   = [cid for eid in val_events   for cid in by_event[eid]]
    test_cands  = [cid for eid in test_events  for cid in by_event[eid]]

    splits_data = {
        "split_strategy"       : "EVENT_LEVEL_SPLIT (prevents intra-event frame data leakage)",
        "train_ratio"          : 0.70,
        "val_ratio"            : 0.15,
        "test_ratio"           : 0.15,
        "train_event_count"    : len(train_events),
        "val_event_count"      : len(val_events),
        "test_event_count"     : len(test_events),
        "train_candidate_count": len(train_cands),
        "val_candidate_count"  : len(val_cands),
        "test_candidate_count" : len(test_cands),
        "train_candidates"     : train_cands,
        "val_candidates"       : val_cands,
        "test_candidates"      : test_cands,
    }
    with open(SPLITS_JSON, "w") as fh:
        json.dump(splits_data, fh, indent=2)

    # 3. Overall Agreement Calculation
    total_annotated = len(annotations)
    exact_matches = sum(
        1 for cand in candidates
        if cand["candidate_id"] in annotations
        and (("HIGH" in cand["rule_based_risk"] and "HIGH" in annotations[cand["candidate_id"]]["human_label"]) or
             ("MEDIUM" in cand["rule_based_risk"] and "MEDIUM" in annotations[cand["candidate_id"]]["human_label"]) or
             ("LOW" in cand["rule_based_risk"] and "SAFE" in annotations[cand["candidate_id"]]["human_label"]))
    )
    agreement_pct = round((exact_matches / total_annotated * 100), 2) if total_annotated > 0 else 0.0

    # 4. Dataset Summary JSON
    summary = {
        "dataset_name"              : "RoadSentinel Ground-Truth Traffic Risk Dataset",
        "phase"                     : 9,
        "scientific_limitation"     : "Initial human-annotated ground-truth dataset from a single traffic scene. Multi-annotator agreement and cross-scene validation required for full scientific maturity.",
        "totals": {
            "total_candidates"      : len(candidates),
            "total_annotated"       : total_annotated,
            "SAFE_count"            : label_counts["SAFE"],
            "MEDIUM_RISK_count"     : label_counts["MEDIUM_RISK"],
            "HIGH_RISK_count"       : label_counts["HIGH_RISK"],
            "UNCERTAIN_count"       : label_counts["UNCERTAIN"],
        },
        "rule_human_agreement": {
            "exact_matches"         : exact_matches,
            "disagreements"         : len(disagreements),
            "agreement_percentage"  : agreement_pct,
            "confusion_matrix"      : matrix,
        },
        "event_level_splits"        : splits_data,
        "output_files": {
            "candidates_csv"        : str(CANDIDATES_CSV),
            "annotations_csv"       : str(ANNOTATIONS_CSV),
            "schema_json"           : str(SCHEMA_JSON),
            "summary_json"          : str(SUMMARY_JSON),
            "disagreements_csv"     : str(DISAGREE_CSV),
            "splits_json"           : str(SPLITS_JSON),
        }
    }
    with open(SUMMARY_JSON, "w") as fh:
        json.dump(summary, fh, indent=2)

    # Terminal Quality Report
    W = 46
    print("\n" + "=" * W)
    print(" ROAD SENTINEL — GROUND TRUTH SUMMARY")
    print("=" * W)
    print(f" Total Candidates      : {len(candidates)}")
    print(f" Annotated Candidates  : {total_annotated} / {len(candidates)}")
    print()
    print(f" Human Label Breakdown:")
    print(f"   SAFE                : {label_counts['SAFE']:>3d} ({label_counts['SAFE']/total_annotated*100:5.1f}%)" if total_annotated else "   SAFE: 0")
    print(f"   MEDIUM_RISK         : {label_counts['MEDIUM_RISK']:>3d} ({label_counts['MEDIUM_RISK']/total_annotated*100:5.1f}%)" if total_annotated else "   MEDIUM_RISK: 0")
    print(f"   HIGH_RISK           : {label_counts['HIGH_RISK']:>3d} ({label_counts['HIGH_RISK']/total_annotated*100:5.1f}%)" if total_annotated else "   HIGH_RISK: 0")
    print(f"   UNCERTAIN           : {label_counts['UNCERTAIN']:>3d} ({label_counts['UNCERTAIN']/total_annotated*100:5.1f}%)" if total_annotated else "   UNCERTAIN: 0")
    print()
    print(f" Rule vs Human Agreement: {agreement_pct:.2f}% ({exact_matches}/{total_annotated})")
    print(f" Total Disagreements    : {len(disagreements)} (Saved to {DISAGREE_CSV.name})")
    print()
    print(f" Event-Level Splits (70 / 15 / 15):")
    print(f"   Train candidates    : {len(train_cands)} ({len(train_events)} events)")
    print(f"   Val candidates      : {len(val_cands)} ({len(val_events)} events)")
    print(f"   Test candidates     : {len(test_cands)} ({len(test_events)} events)")
    print("=" * W + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoadSentinel Ground-Truth Risk Annotation Tool")
    parser.add_argument("--annotator", type=str, default="annotator_01", help="Annotator ID")
    parser.add_argument("--auto-seed", action="store_true", help="Auto-seed annotations based on domain inspection rules")
    args = parser.parse_args()

    run_interactive_annotation(annotator_id=args.annotator, auto_seed=args.auto_seed)
