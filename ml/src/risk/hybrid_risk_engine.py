"""
hybrid_risk_engine.py — RoadSentinel AI, Phase 11
Safety-First Hybrid Risk Assessment Engine.

Combines physical rule-based heuristic scores with ML model predictions.

Hybrid Decision Logic:
  1. RULE_HIGH_OVERRIDE: If rule_based_risk == "HIGH", final_risk = "HIGH"
     (Safety guardrail protecting against 0% ML HIGH recall).
  2. ML: If ml_confidence >= threshold (default 0.70), final_risk = ml_prediction.
  3. RULE_FALLBACK: Otherwise, fall back to rule_based_risk.

Logs explicit decision sources and exports disagreements to hybrid_disagreements.csv.
"""

import os
import sys
import csv
import json
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR     = Path(__file__).resolve().parent.parent.parent
HYBRID_DIR   = BASE_DIR / "outputs" / "hybrid"
DISAGREE_CSV = HYBRID_DIR / "hybrid_disagreements.csv"
REPORT_JSON  = HYBRID_DIR / "hybrid_report.json"

sys.path.insert(0, str(BASE_DIR / "src"))
from ml.predict_risk import RiskPredictor

DEFAULT_CONFIDENCE_THRESHOLD = 0.70


class HybridRiskEngine:
    def __init__(self, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
        self.predictor = RiskPredictor()
        self.confidence_threshold = confidence_threshold
        HYBRID_DIR.mkdir(parents=True, exist_ok=True)

    def set_confidence_threshold(self, threshold):
        self.confidence_threshold = threshold

    def evaluate_hybrid_interaction(self, feature_dict, rule_output):
        """
        Evaluates a single vehicle interaction using both Rule Engine and ML Model.

        Args:
            feature_dict (dict): Physical kinematics and class features.
            rule_output (dict): Rule engine output containing 'risk_level' (LOW, MEDIUM, HIGH) and 'risk_score'.

        Returns:
            dict: Complete hybrid assessment.
        """
        rule_risk = rule_output.get("risk_level", "LOW")

        # 1. Obtain ML prediction
        ml_res = self.predictor.predict_single(feature_dict)
        ml_pred = ml_res["prediction"]
        ml_conf = ml_res["confidence"]
        ml_probs = ml_res["probabilities"]

        # 2. Apply Safety-First Hybrid Logic
        if rule_risk == "HIGH":
            final_risk = "HIGH"
            source = "RULE_HIGH_OVERRIDE"
        elif ml_conf >= self.confidence_threshold:
            final_risk = ml_pred
            source = "ML"
        else:
            final_risk = rule_risk
            source = "RULE_FALLBACK"

        # Disagreement check (mapped for comparison)
        r_mapped = "HIGH_RISK" if rule_risk == "HIGH" else "MEDIUM_RISK" if rule_risk == "MEDIUM" else "SAFE"
        is_disagreement = (r_mapped != ml_pred)

        return {
            "v1_id"          : feature_dict.get("v1_id", 0),
            "v2_id"          : feature_dict.get("v2_id", 0),
            "frame_id"       : feature_dict.get("frame_id", 0),
            "rule_based_risk": rule_risk,
            "ml_prediction"  : ml_pred,
            "ml_confidence"  : ml_conf,
            "ml_probabilities": ml_probs,
            "final_risk"     : final_risk,
            "decision_source": source,
            "is_disagreement": is_disagreement,
        }

    def process_feature_dataframe(self, df_feats, rule_rows):
        """Processes a batch dataframe of interaction features and rule outputs."""
        results = []
        disagreements = []

        rule_by_pair_frame = {}
        for r in rule_rows:
            key = (int(r["v1_id"]), int(r["v2_id"]), int(r["frame_id"]))
            rule_by_pair_frame[key] = r

        for idx, row in df_feats.iterrows():
            v1, v2, fid = int(row["v1_id"]), int(row["v2_id"]), int(row["frame_id"])
            r_out = rule_by_pair_frame.get((v1, v2, fid), {"risk_level": "LOW", "risk_score": 0.0})

            feat_dict = dict(row)
            res = self.evaluate_hybrid_interaction(feat_dict, r_out)

            res["event_id"] = r_out.get("event_id", 0)
            res["TTC"] = row.get("ttc", "")
            res["distance"] = row.get("distance", "")
            res["closing_speed"] = row.get("closing_speed", "")
            res["risk_score"] = r_out.get("risk_score", 0.0)

            results.append(res)
            if res["is_disagreement"]:
                disagreements.append(res)

        # Write hybrid_disagreements.csv
        dis_cols = ["event_id", "v1_id", "v2_id", "frame_id", "rule_based_risk",
                    "ml_prediction", "ml_confidence", "final_risk", "decision_source",
                    "TTC", "distance", "closing_speed", "risk_score"]
        with open(DISAGREE_CSV, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=dis_cols)
            writer.writeheader()
            for d in disagreements:
                writer.writerow({c: d.get(c, "") for c in dis_cols})

        return results, disagreements


if __name__ == "__main__":
    engine = HybridRiskEngine(confidence_threshold=0.70)
    sample_feat = {
        "v1_id": 80, "v2_id": 83, "frame_id": 112,
        "distance": 133.4, "ttc": 1.85, "closing_speed": 72.1,
        "relative_speed": 72.4, "v1_speed": 35.0, "v2_speed": 22.0,
        "v1_acceleration": -2.0, "v2_acceleration": 1.0,
        "heading_difference": 8.7, "trajectory_crossing": 1.0,
        "approaching": 1.0, "v1_class": "car", "v2_class": "car"
    }
    # Test 1: Rule HIGH -> Override expected
    rule_out_high = {"risk_level": "HIGH", "risk_score": 0.70}
    res_high = engine.evaluate_hybrid_interaction(sample_feat, rule_out_high)
    print("\nTest 1 (Rule HIGH Override):")
    print(json.dumps(res_high, indent=2))

    # Test 2: Rule LOW, ML HIGH -> ML expected if conf >= 0.70
    rule_out_low = {"risk_level": "LOW", "risk_score": 0.02}
    res_low = engine.evaluate_hybrid_interaction(sample_feat, rule_out_low)
    print("\nTest 2 (ML Decision):")
    print(json.dumps(res_low, indent=2))
