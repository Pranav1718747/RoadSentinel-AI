"""
risk_engine.py — RoadSentinel AI, Phase 7
Rule-based risk scorer for vehicle-pair interactions.

SCIENTIFIC LIMITATION: This is a BASELINE RULE-BASED system.
TTC and distance are image-plane quantities (pixels / px/s).
Thresholds are initial engineering assumptions, not trained parameters.
Do NOT claim this predicts real-world collisions with proven accuracy.
"""

import math
import risk_config as cfg

NaN = float("nan")

def is_nan(v):
    try: return v != v
    except: return True

def safe_float(v):
    if v is None or str(v).strip() == "": return NaN
    try: return float(v)
    except: return NaN

def safe_bool(v):
    return str(v).strip().lower() == "true"


# ──────────────────────────────────────────────────────────────
# Sub-score functions — each returns a value in [0, 1]
# and a list of human-readable reason strings.
# ──────────────────────────────────────────────────────────────

def score_ttc(ttc, approaching):
    """
    TTC score [0, 1].
    Only meaningful when approaching=True AND ttc is valid.
    If not approaching: 0.0 (no TTC risk for separating vehicles).
    """
    reasons = []
    if not approaching or is_nan(ttc) or ttc <= 0:
        return 0.0, reasons

    if ttc < cfg.TTC_HIGH:
        reasons.append(f"Very low TTC ({ttc:.2f}s < {cfg.TTC_HIGH}s threshold)")
        return 1.0, reasons
    elif ttc < cfg.TTC_MEDIUM:
        score = 1.0 - (ttc - cfg.TTC_HIGH) / (cfg.TTC_MEDIUM - cfg.TTC_HIGH)
        reasons.append(f"Elevated TTC risk ({ttc:.2f}s)")
        return round(score, 4), reasons
    return 0.0, reasons


def score_distance(distance):
    """
    Distance score [0, 1].
    Distance is image-plane pixels. Not metres.
    """
    reasons = []
    if is_nan(distance) or distance < 0:
        return 0.0, reasons

    if distance < cfg.DIST_HIGH:
        reasons.append(f"Very small image-plane separation ({distance:.1f}px < {cfg.DIST_HIGH}px)")
        return 1.0, reasons
    elif distance < cfg.DIST_MEDIUM:
        score = 1.0 - (distance - cfg.DIST_HIGH) / (cfg.DIST_MEDIUM - cfg.DIST_HIGH)
        reasons.append(f"Close image-plane proximity ({distance:.1f}px)")
        return round(score, 4), reasons
    return 0.0, reasons


def score_closing_speed(closing_speed, approaching):
    """
    Closing speed score [0, 1].
    Only applied when approaching=True and closing_speed > 0.
    Separating vehicles (closing_speed <= 0) never receive this score.
    """
    reasons = []
    if not approaching or is_nan(closing_speed) or closing_speed <= 0:
        return 0.0, reasons

    if closing_speed > cfg.CS_HIGH:
        reasons.append(f"Very high closing speed ({closing_speed:.1f}px/s > {cfg.CS_HIGH}px/s)")
        return 1.0, reasons
    elif closing_speed > cfg.CS_MEDIUM:
        score = (closing_speed - cfg.CS_MEDIUM) / (cfg.CS_HIGH - cfg.CS_MEDIUM)
        reasons.append(f"Elevated closing speed ({closing_speed:.1f}px/s)")
        return round(score, 4), reasons
    return 0.0, reasons


def score_trajectory(trajectory_crossing, approaching, distance):
    """
    Trajectory interaction score [0, 1].
    Crossing alone does NOT cause HIGH risk — must combine with motion.
    Score is modulated by whether vehicles are approaching AND close.
    """
    reasons = []
    if not trajectory_crossing:
        return 0.0, reasons

    # Crossing + approaching + close → meaningful risk signal
    if approaching and not is_nan(distance) and distance < cfg.DIST_MEDIUM:
        reasons.append("Trajectories intersect and vehicles are approaching")
        # Scale: 0.5 if far-ish, 1.0 if very close
        scale = 1.0 - min(1.0, max(0.0, (distance - cfg.DIST_HIGH) / (cfg.DIST_MEDIUM - cfg.DIST_HIGH)))
        return round(0.4 + 0.6 * scale, 4), reasons
    elif trajectory_crossing:
        # Crossing but not meeting these conditions — mild signal only
        reasons.append("Trajectories intersect (motion context inconclusive)")
        return 0.2, reasons
    return 0.0, reasons


def score_heading(heading_diff, approaching):
    """
    Heading difference score [0, 1].
    Large heading difference + approaching = potential crossing-path conflict.
    Heading difference alone (not approaching) gets a reduced signal.
    """
    reasons = []
    if is_nan(heading_diff):
        return 0.0, reasons

    if heading_diff >= cfg.HEADING_HIGH:
        if approaching:
            reasons.append(f"Near head-on / crossing paths ({heading_diff:.1f}° divergence)")
            return 1.0, reasons
        else:
            reasons.append(f"Large heading divergence ({heading_diff:.1f}°) — not approaching")
            return 0.3, reasons
    elif heading_diff >= cfg.HEADING_MEDIUM:
        if approaching:
            reasons.append(f"Significant heading difference ({heading_diff:.1f}°)")
            return 0.5, reasons
        else:
            return 0.1, reasons
    return 0.0, reasons


def distance_gate_factor(distance):
    """
    Distance-gating attenuation factor f(dist) in [0, 1].
    f(dist) = max(0.0, 1.0 - dist / DISTANCE_GATING_THRESHOLD)
    - dist >= 600 px -> f(dist) = 0.0 (closing-speed and heading contributions become zero)
    - dist = 300 px  -> f(dist) = 0.5
    - dist = 0 px    -> f(dist) = 1.0
    Attenuates closing-speed and heading risk sub-scores so long-distance pairs
    (>= 600px) cannot trigger false HIGH alerts solely from velocity projections.
    INITIAL ENGINEERING THRESHOLD — NOT a calibrated real-world distance.
    """
    if is_nan(distance) or distance < 0:
        return 0.0
    thresh = getattr(cfg, "DISTANCE_GATING_THRESHOLD", 600.0)
    return max(0.0, 1.0 - (distance / thresh))


# ──────────────────────────────────────────────────────────────
# Main scoring function
# ──────────────────────────────────────────────────────────────

def evaluate_interaction(row):
    """
    Evaluate a single vehicle-pair interaction row from traffic_features.csv.
    Returns a dict with all sub-scores, combined risk_score, risk_level,
    and risk_reasons list.

    Scoring is transparent: each sub-score is stored individually so the
    system can be debugged, inspected, and calibrated.
    """
    ttc           = safe_float(row.get("ttc"))
    distance      = safe_float(row.get("distance"))
    closing_speed = safe_float(row.get("closing_speed"))
    heading_diff  = safe_float(row.get("heading_difference"))
    approaching   = safe_bool(row.get("approaching"))
    traj_crossing = safe_bool(row.get("trajectory_crossing"))

    # ── Sanity guards ────────────────────────────────────────
    # Never use TTC as a risk signal when not approaching
    if not approaching:
        ttc = NaN

    # ── Sub-scores ───────────────────────────────────────────
    ttc_s,  ttc_r  = score_ttc(ttc, approaching)
    dist_s, dist_r = score_distance(distance)
    cs_s,   cs_r   = score_closing_speed(closing_speed, approaching)
    traj_s, traj_r = score_trajectory(traj_crossing, approaching, distance)
    hdg_s,  hdg_r  = score_heading(heading_diff, approaching)

    # ── Distance Gating Attenuation (Fix 1) ───────────────────
    gate = distance_gate_factor(distance)
    cs_s  = round(cs_s * gate, 4)
    hdg_s = round(hdg_s * gate, 4)

    # ── Weighted total ────────────────────────────────────────
    risk_score = round(
        cfg.WEIGHT_TTC        * ttc_s  +
        cfg.WEIGHT_DISTANCE   * dist_s +
        cfg.WEIGHT_CLOSING    * cs_s   +
        cfg.WEIGHT_TRAJECTORY * traj_s +
        cfg.WEIGHT_HEADING    * hdg_s,
        4
    )

    # ── Risk level ────────────────────────────────────────────
    if risk_score >= cfg.SCORE_HIGH:
        risk_level = "HIGH"
    elif risk_score >= cfg.SCORE_MEDIUM:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # ── Compile reasons ───────────────────────────────────────
    reasons = ttc_r + dist_r + cs_r + traj_r + hdg_r
    # Approaching is informational context, not a scored reason,
    # but worth noting for MEDIUM+ events.
    if approaching and risk_level in ("MEDIUM", "HIGH") and not any("approaching" in r.lower() for r in reasons):
        reasons.append("Vehicles are approaching each other")

    return {
        "risk_score"            : risk_score,
        "risk_level"            : risk_level,
        "ttc_score"             : round(ttc_s,  4),
        "distance_score"        : round(dist_s, 4),
        "closing_speed_score"   : round(cs_s,   4),
        "trajectory_score"      : round(traj_s, 4),
        "heading_score"         : round(hdg_s,  4),
        "risk_reasons"          : "; ".join(reasons) if reasons else "No significant risk indicators",
    }
