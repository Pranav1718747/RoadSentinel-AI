"""
predict_risk.py — RoadSentinel AI, Phase 11
ML Risk Inference Engine.

Loads saved trained pipeline from ml/models/risk_model.pkl and performs
validated prediction on incoming traffic interaction features.

Returns:
  - prediction: "SAFE", "MEDIUM_RISK", or "HIGH_RISK"
  - probabilities: {"SAFE": p0, "MEDIUM_RISK": p1, "HIGH_RISK": p2}
  - confidence: max probability
"""

import os
import sys
import json
import joblib
import math
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
MODEL_PATH  = BASE_DIR / "models" / "risk_model.pkl"
CONFIG_PATH = BASE_DIR / "models" / "model_config.json"

REQUIRED_FEATURES = [
    "distance", "ttc", "closing_speed", "relative_speed",
    "v1_speed", "v2_speed", "v1_acceleration", "v2_acceleration",
    "heading_difference", "trajectory_crossing", "approaching",
    "v1_class", "v2_class"
]

LABEL_NAMES = ["SAFE", "MEDIUM_RISK", "HIGH_RISK"]


class RiskPredictor:
    def __init__(self, model_path=MODEL_PATH, config_path=CONFIG_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}. Run train_risk_model.py first.")

        self.pipeline = joblib.load(model_path)
        with open(config_path) as fh:
            self.config = json.load(fh)

        self.model_type = self.config.get("model_type", "ML Model")
        print(f"[RiskPredictor] Loaded {self.model_type} pipeline from {model_path.name}")

    def validate_features(self, df_input):
        """Verifies that all required feature columns exist in the input dataframe."""
        missing = [f for f in REQUIRED_FEATURES if f not in df_input.columns and f.upper() not in df_input.columns]
        if missing:
            raise ValueError(f"Feature Consistency Failure: Missing required input features {missing}")

    def preprocess_input(self, df_input):
        """Standardizes input dataframe columns and creates missing TTC indicator."""
        df_x = df_input.copy()

        # Handle column capitalization
        if "TTC" in df_x.columns and "ttc" not in df_x.columns:
            df_x.rename(columns={"TTC": "ttc"}, inplace=True)

        self.validate_features(df_x)

        # Create missing TTC indicator flag
        df_x["is_ttc_missing"] = df_x["ttc"].apply(
            lambda v: 1.0 if (pd.isna(v) or str(v).strip() == "" or v != v) else 0.0
        )

        for col in ["distance", "ttc", "closing_speed", "relative_speed",
                    "v1_speed", "v2_speed", "v1_acceleration", "v2_acceleration",
                    "heading_difference", "trajectory_crossing", "approaching"]:
            if col in df_x.columns:
                df_x[col] = pd.to_numeric(df_x[col], errors="coerce").astype(float)

        return df_x

    def predict_single(self, feature_dict):
        """Predict risk for a single interaction feature dictionary."""
        df_single = pd.DataFrame([feature_dict])
        results = self.predict_batch(df_single)
        return results[0]

    def predict_batch(self, df_input):
        """Predict risk for a batch of interaction feature records."""
        df_x = self.preprocess_input(df_input)

        preds = self.pipeline.predict(df_x)
        probs = self.pipeline.predict_proba(df_x)

        results = []
        for i in range(len(df_x)):
            pred_idx = int(preds[i])
            pred_label = LABEL_NAMES[pred_idx]
            p_vec = probs[i]

            prob_dict = {
                "SAFE"       : round(float(p_vec[0]), 4),
                "MEDIUM_RISK": round(float(p_vec[1]), 4) if len(p_vec) > 1 else 0.0,
                "HIGH_RISK"  : round(float(p_vec[2]), 4) if len(p_vec) > 2 else 0.0,
            }
            conf = round(float(np.max(p_vec)), 4)

            results.append({
                "prediction": pred_label,
                "predicted_class_id": pred_idx,
                "probabilities": prob_dict,
                "confidence": conf,
            })

        return results


if __name__ == "__main__":
    predictor = RiskPredictor()
    sample_feat = {
        "distance": 120.5,
        "ttc": 1.8,
        "closing_speed": 45.2,
        "relative_speed": 48.0,
        "v1_speed": 35.0,
        "v2_speed": 22.0,
        "v1_acceleration": -2.1,
        "v2_acceleration": 1.0,
        "heading_difference": 85.0,
        "trajectory_crossing": 1.0,
        "approaching": 1.0,
        "v1_class": "car",
        "v2_class": "car",
    }
    res = predictor.predict_single(sample_feat)
    print("\nSample Inference Result:")
    print(json.dumps(res, indent=2))
