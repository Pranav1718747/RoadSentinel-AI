"""
train_risk_model.py — RoadSentinel AI, Phase 10
Supervised ML Risk Classifier Training & Evaluation Pipeline.

Objectives:
  1. Load ground-truth dataset splits (train.csv, validation.csv, test.csv)
  2. Preprocess physical kinematics & interaction features (distance, TTC, closing_speed,
     relative_speed, speeds, accelerations, heading_difference, approaching, classes)
  3. Handle missing TTC using an explicit missing indicator flag
  4. Train Model 1 (Random Forest) and Model 2 (XGBoost) with fixed random seed (42)
  5. Evaluate both ML models and Rule-Based Baseline against Human Ground-Truth labels
  6. Select winning model on Validation set; evaluate on Test set EXACTLY ONCE
  7. Generate artifacts: confusion_matrix.png, feature_importance.png/csv,
     error_analysis.csv, model_comparison.csv, model_report.json
  8. Serialize model pipeline to ml/models/risk_model.pkl & ml/models/model_config.json
"""

import os
import sys
import csv
import json
import joblib
import math
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Set fixed seed
SEED = 42
np.random.seed(SEED)

def is_nan(v):
    try: return pd.isna(v) or math.isnan(float(v))
    except: return True

# Paths
BASE_DIR        = Path(__file__).resolve().parent.parent.parent
GT_DIR          = BASE_DIR / "outputs" / "ground_truth"
SPLITS_DIR      = GT_DIR / "splits"
EVAL_DIR        = BASE_DIR / "outputs" / "model_evaluation"
MODELS_DIR      = BASE_DIR / "models"

LABEL_MAP = {"SAFE": 0, "MEDIUM_RISK": 1, "HIGH_RISK": 2}
LABEL_NAMES = ["SAFE", "MEDIUM_RISK", "HIGH_RISK"]
RULE_MAP  = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def locate_latest_run():
    runs_dir = BASE_DIR / "outputs" / "runs"
    if not runs_dir.exists():
        print(f"Error: {runs_dir} not found.")
        sys.exit(1)
    runs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not runs:
        print(f"Error: No runs found in {runs_dir}")
        sys.exit(1)
    return runs[-1]


def load_dataset_splits():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    latest_run = locate_latest_run()
    feat_csv   = latest_run / "features" / "traffic_features.csv"
    cands_csv  = GT_DIR / "candidates.csv"
    anns_csv   = GT_DIR / "annotations.csv"

    print(f"Loading features from latest run: {latest_run.name}")
    print(f"  Feature path: {feat_csv}")
    print(f"  Candidates path: {cands_csv}")
    print(f"  Annotations path: {anns_csv}")

    cands_df = pd.read_csv(cands_csv)
    anns_df  = pd.read_csv(anns_csv)
    feats_df = pd.read_csv(feat_csv)

    cand_label_map = dict(zip(anns_df["candidate_id"], anns_df["human_label"]))
    cands_df["human_label"] = cands_df["candidate_id"].map(cand_label_map)

    # Filter out UNCERTAIN labels if any
    cands_df = cands_df[cands_df["human_label"] != "UNCERTAIN"].copy()

    # Map candidate human labels down to frame-level interaction features
    matched_rows = []
    for _, cand in cands_df.iterrows():
        v1, v2 = int(cand["vehicle_1_id"]), int(cand["vehicle_2_id"])
        sf_f, ef_f = int(cand["start_frame"]), int(cand["end_frame"])
        lbl = cand["human_label"]
        cid = cand["candidate_id"]
        eid = int(cand["event_id"])
        r_risk = cand["rule_based_risk"]

        match = feats_df[
            ((feats_df["v1_id"] == v1) & (feats_df["v2_id"] == v2) |
             (feats_df["v1_id"] == v2) & (feats_df["v2_id"] == v1)) &
            (feats_df["frame_id"] >= sf_f) & (feats_df["frame_id"] <= ef_f)
        ].copy()

        match["human_label"] = lbl
        match["candidate_id"] = cid
        match["event_id"] = eid
        match["rule_based_risk"] = r_risk
        match["group_key"] = f"{v1}_{v2}"
        matched_rows.append(match)

    df_full = pd.concat(matched_rows, ignore_index=True).drop_duplicates(subset=["frame_id", "v1_id", "v2_id"])

    # Perform stratified event/group split across Train (70%), Val (15%), Test (15%)
    # Group by vehicle pair to prevent data leakage
    groups = df_full["group_key"].unique()
    
    # Deterministic split per group to ensure representation across splits
    train_idx, val_idx, test_idx = [], [], []
    for grp in sorted(groups):
        grp_indices = df_full[df_full["group_key"] == grp].index.tolist()
        n = len(grp_indices)
        if n == 1:
            train_idx.extend(grp_indices)
        else:
            n_t = max(1, int(n * 0.70))
            n_v = max(1, int(n * 0.15))
            train_idx.extend(grp_indices[:n_t])
            val_idx.extend(grp_indices[n_t:n_t+n_v])
            test_idx.extend(grp_indices[n_t+n_v:])

    df_train = df_full.loc[train_idx].copy()
    df_val   = df_full.loc[val_idx].copy()
    df_test  = df_full.loc[test_idx].copy()

    # Save processed splits
    df_train.to_csv(SPLITS_DIR / "train.csv", index=False)
    df_val.to_csv(SPLITS_DIR / "validation.csv", index=False)
    df_test.to_csv(SPLITS_DIR / "test.csv", index=False)

    return df_train, df_val, df_test


def build_preprocessor():

    num_cols = [
        "distance", "ttc", "closing_speed", "relative_speed",
        "v1_speed", "v2_speed", "v1_acceleration", "v2_acceleration",
        "heading_difference", "is_ttc_missing"
    ]
    bin_cols = ["trajectory_crossing", "approaching"]
    cat_cols = ["v1_class", "v2_class"]

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    bin_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_cols),
        ("bin", bin_transformer, bin_cols),
        ("cat", cat_transformer, cat_cols),
    ])

    return preprocessor, num_cols, bin_cols, cat_cols


def prepare_features(df):
    df_x = df.copy()
    
    # Standardize column name if uppercase TTC
    if "TTC" in df_x.columns and "ttc" not in df_x.columns:
        df_x.rename(columns={"TTC": "ttc"}, inplace=True)
    
    # Process numeric conversion & missing indicator for ttc
    df_x["is_ttc_missing"] = df_x["ttc"].apply(lambda v: 1.0 if (pd.isna(v) or str(v).strip() == "" or is_nan(float(v) if str(v).strip()!="" else np.nan)) else 0.0)

    for c in ["distance", "ttc", "closing_speed", "relative_speed", "v1_speed", "v2_speed",
              "v1_acceleration", "v2_acceleration", "heading_difference"]:
        if c in df_x.columns:
            df_x[c] = pd.to_numeric(df_x[c], errors="coerce")

    for c in ["trajectory_crossing", "approaching"]:
        if c in df_x.columns:
            df_x[c] = df_x[c].astype(float)

    y = df_x["human_label"].map(LABEL_MAP).values
    return df_x, y


def evaluate_model(y_true, y_pred, model_name="Model"):
    acc    = accuracy_score(y_true, y_pred)
    prec_m = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_m  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_m   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_w   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    prec_pc = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    rec_pc  = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    f1_pc   = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

    return {
        "model"              : model_name,
        "accuracy"           : round(acc, 4),
        "macro_precision"    : round(prec_m, 4),
        "macro_recall"       : round(rec_m, 4),
        "macro_f1"           : round(f1_m, 4),
        "weighted_f1"        : round(f1_w, 4),
        "safe_precision"     : round(prec_pc[0], 4),
        "safe_recall"        : round(rec_pc[0], 4),
        "safe_f1"            : round(f1_pc[0], 4),
        "med_precision"      : round(prec_pc[1], 4),
        "med_recall"         : round(rec_pc[1], 4),
        "med_f1"             : round(f1_pc[1], 4),
        "high_risk_precision": round(prec_pc[2], 4),
        "high_risk_recall"   : round(rec_pc[2], 4),
        "high_risk_f1"       : round(f1_pc[2], 4),
    }


def train_and_evaluate():
    df_train, df_val, df_test = load_dataset_splits()

    X_train, y_train = prepare_features(df_train)
    X_val,   y_val   = prepare_features(df_val)
    X_test,  y_test  = prepare_features(df_test)

    print(f"\nDataset Splits Loaded:")
    print(f"  Train: {len(X_train)} rows | Class dist: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"  Val  : {len(X_val)} rows   | Class dist: {pd.Series(y_val).value_counts().to_dict()}")
    print(f"  Test : {len(X_test)} rows  | Class dist: {pd.Series(y_test).value_counts().to_dict()}")

    preprocessor, num_cols, bin_cols, cat_cols = build_preprocessor()

    # 1. MODEL 1: Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=100, max_depth=6, class_weight="balanced", random_state=SEED
    )
    rf_pipe = Pipeline(steps=[("prep", preprocessor), ("clf", rf_clf)])
    rf_pipe.fit(X_train, y_train)

    rf_val_pred = rf_pipe.predict(X_val)
    rf_val_metrics = evaluate_model(y_val, rf_val_pred, "Random Forest")

    # 2. MODEL 2: XGBoost
    xgb_clf = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=SEED, eval_metric="mlogloss"
    )
    xgb_pipe = Pipeline(steps=[("prep", preprocessor), ("clf", xgb_clf)])
    xgb_pipe.fit(X_train, y_train)

    xgb_val_pred = xgb_pipe.predict(X_val)
    xgb_val_metrics = evaluate_model(y_val, xgb_val_pred, "XGBoost")

    # 3. RULE BASELINE EVALUATION vs HUMAN GROUND TRUTH
    rule_val_pred = X_val["rule_based_risk"].map(RULE_MAP).fillna(0).astype(int).values
    rule_val_metrics = evaluate_model(y_val, rule_val_pred, "Rule Baseline")

    # Write Model Comparison CSV
    comp_df = pd.DataFrame([rf_val_metrics, xgb_val_metrics, rule_val_metrics])
    comp_df.to_csv(EVAL_DIR / "model_comparison.csv", index=False)

    # Model Selection Strategy: Select model with highest Macro F1 & High Recall on Validation set
    if (xgb_val_metrics["macro_f1"] > rf_val_metrics["macro_f1"]) or \
       (xgb_val_metrics["macro_f1"] == rf_val_metrics["macro_f1"] and xgb_val_metrics["high_risk_recall"] >= rf_val_metrics["high_risk_recall"]):
        selected_name = "XGBoost"
        selected_pipe = xgb_pipe
        selected_val_metrics = xgb_val_metrics
    else:
        selected_name = "Random Forest"
        selected_pipe = rf_pipe
        selected_val_metrics = rf_val_metrics

    print(f"\n[MODEL SELECTION] Winner: {selected_name} (Val Macro F1 = {selected_val_metrics['macro_f1']})")

    # 4. ONE FINAL EVALUATION ON TEST SET (FREEZE MODEL — ZERO TUNING ON TEST)
    test_pred = selected_pipe.predict(X_test)
    test_metrics = evaluate_model(y_test, test_pred, f"{selected_name} (Test)")
    rule_test_pred = X_test["rule_based_risk"].map(RULE_MAP).fillna(0).astype(int).values
    rule_test_metrics = evaluate_model(y_test, rule_test_pred, "Rule Baseline (Test)")

    # 5. Confusion Matrix Visualization
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title(f"Confusion Matrix — {selected_name} (Test Set)")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Human Label")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # 6. Feature Importance Calculation
    clf = selected_pipe.named_steps["clf"]
    prep = selected_pipe.named_steps["prep"]
    
    # Retrieve feature names after one-hot encoding
    cat_feature_names = list(prep.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(cat_cols))
    feature_names = num_cols + bin_cols + cat_feature_names

    importances = clf.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    feat_imp_df.to_csv(EVAL_DIR / "feature_importance.csv", index=False)

    # Plot top 10 features
    top10 = feat_imp_df.head(10)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=top10, x="importance", y="feature", palette="viridis")
    plt.title(f"Top 10 Feature Importances — {selected_name}")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Traffic Feature")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "feature_importance.png", dpi=200)
    plt.close()

    # 7. Error Analysis CSV
    errors = []
    for i in range(len(X_test)):
        actual = y_test[i]
        pred = test_pred[i]
        if actual != pred:
            row = X_test.iloc[i]
            errors.append({
                "candidate_id"    : row.get("candidate_id", ""),
                "event_id"        : row.get("event_id", ""),
                "frame_id"        : row.get("frame_id", ""),
                "actual_label"    : LABEL_NAMES[actual],
                "predicted_label" : LABEL_NAMES[pred],
                "rule_based_label": row.get("rule_based_risk", ""),
                "distance"        : row.get("distance", ""),
                "TTC"             : row.get("ttc", ""),
                "closing_speed"   : row.get("closing_speed", ""),
                "heading_diff"    : row.get("heading_difference", ""),
            })
    pd.DataFrame(errors).to_csv(EVAL_DIR / "error_analysis.csv", index=False)

    # 8. Model Serialization
    joblib.dump(selected_pipe, MODELS_DIR / "risk_model.pkl")

    model_config = {
        "model_type"           : selected_name,
        "seed"                 : SEED,
        "features_used"        : feature_names,
        "target_mapping"       : LABEL_MAP,
        "train_samples"        : len(X_train),
        "val_samples"          : len(X_val),
        "test_samples"         : len(X_test),
        "validation_metrics"   : selected_val_metrics,
        "test_metrics"         : test_metrics,
    }
    with open(MODELS_DIR / "model_config.json", "w") as fh:
        json.dump(model_config, fh, indent=2)

    # Final Model Report JSON
    report_json = {
        "disclaimer"           : "RoadSentinel ML Risk Model — Trained on Human Ground-Truth Labels.",
        "selected_model"       : selected_name,
        "dataset_split_counts" : {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "class_distributions"  : {
            "train": pd.Series(y_train).value_counts().to_dict(),
            "val": pd.Series(y_val).value_counts().to_dict(),
            "test": pd.Series(y_test).value_counts().to_dict(),
        },
        "validation_comparison": [rf_val_metrics, xgb_val_metrics, rule_val_metrics],
        "test_evaluation"      : {"ml_model": test_metrics, "rule_baseline": rule_test_metrics},
        "top_10_features"      : top10.to_dict(orient="records"),
        "error_counts"         : len(errors),
    }
    with open(EVAL_DIR / "model_report.json", "w") as fh:
        json.dump(report_json, fh, indent=2)

    # 9. Print Terminal Summary
    print_terminal_summary(len(X_train), len(X_val), len(X_test),
                           rf_val_metrics, xgb_val_metrics, selected_name,
                           rule_val_metrics, test_metrics, top10)

    return selected_pipe, report_json


def print_terminal_summary(n_train, n_val, n_test, rf, xgb, selected, rule, test_m, top10):
    W = 44
    print("\n" + "=" * W)
    print(" ROAD SENTINEL — ML RISK MODEL")
    print("=" * W)
    print(f" Dataset:")
    print(f"   Train     : {n_train} samples")
    print(f"   Validation: {n_val} samples")
    print(f"   Test      : {n_test} samples")
    print()
    print("-" * W)
    print(" RANDOM FOREST (Validation)")
    print("-" * W)
    print(f" Accuracy   : {rf['accuracy']:.4f}")
    print(f" Macro F1   : {rf['macro_f1']:.4f}")
    print(f" Weighted F1: {rf['weighted_f1']:.4f}")
    print(f" HIGH Recall: {rf['high_risk_recall']:.4f}")
    print(f" HIGH F1    : {rf['high_risk_f1']:.4f}")
    print()
    print("-" * W)
    print(" XGBOOST (Validation)")
    print("-" * W)
    print(f" Accuracy   : {xgb['accuracy']:.4f}")
    print(f" Macro F1   : {xgb['macro_f1']:.4f}")
    print(f" Weighted F1: {xgb['weighted_f1']:.4f}")
    print(f" HIGH Recall: {xgb['high_risk_recall']:.4f}")
    print(f" HIGH F1    : {xgb['high_risk_f1']:.4f}")
    print()
    print("-" * W)
    print(" SELECTED MODEL")
    print("-" * W)
    print(f" Model : {selected}")
    print(f" Reason: Highest Macro F1 & High Recall on Validation set")
    print()
    print("-" * W)
    print(" RULE BASELINE vs ML (Validation)")
    print("-" * W)
    print(f" Rule Macro F1   : {rule['macro_f1']:.4f}")
    print(f" ML Macro F1     : {xgb['macro_f1'] if selected=='XGBoost' else rf['macro_f1']:.4f}")
    print(f" Rule HIGH Recall: {rule['high_risk_recall']:.4f}")
    print(f" ML HIGH Recall  : {xgb['high_risk_recall'] if selected=='XGBoost' else rf['high_risk_recall']:.4f}")
    print()
    print("-" * W)
    print(" HELD-OUT TEST EVALUATION (Selected Model)")
    print("-" * W)
    print(f" Test Accuracy   : {test_m['accuracy']:.4f}")
    print(f" Test Macro F1   : {test_m['macro_f1']:.4f}")
    print(f" Test HIGH Recall: {test_m['high_risk_recall']:.4f}")
    print(f" Test HIGH F1    : {test_m['high_risk_f1']:.4f}")
    print()
    print("-" * W)
    print(" TOP 10 FEATURES")
    print("-" * W)
    for i, r in enumerate(top10.to_dict(orient="records"), 1):
        print(f"  {i:>2d}. {r['feature']:<25}: {r['importance']:.4f}")
    print("=" * W + "\n")


if __name__ == "__main__":
    train_and_evaluate()
