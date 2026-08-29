"""
risk_config.py — RoadSentinel AI, Phase 7
All configurable thresholds for the rule-based risk engine.

IMPORTANT SCIENTIFIC LIMITATION:
These thresholds are INITIAL ENGINEERING ASSUMPTIONS based on the
image-plane feature distribution of the traffic_features.csv dataset
(500-frame aerial traffic video, 10 VALID_MOVING tracks).

TTC and distance are measured in image-space units (seconds and pixels)
because no camera calibration / homography to real-world metres has been
performed. Thresholds must later be validated against manually annotated
ground-truth traffic-conflict data before making safety claims.

This is a BASELINE RULE-BASED system. It is NOT a trained ML model.
"""

# ──────────────────────────────────────────────────────────────
# TTC thresholds (seconds)
# Derived from actual TTC distribution in traffic_features.csv:
#   p5=3.1s  p10=3.7s  p25=4.5s  median=5.7s  p90=9.2s
# Logic:
#   < TTC_HIGH   → high TTC risk  (below ~p10 of observed approaching events)
#   < TTC_MEDIUM → medium TTC risk (below ~p25)
#   >= TTC_MEDIUM → no TTC risk contribution
# ──────────────────────────────────────────────────────────────
TTC_HIGH   = 3.5   # seconds — aggressive proximity; below ~p10
TTC_MEDIUM = 5.0   # seconds — cautionary zone;    below ~p25

# ──────────────────────────────────────────────────────────────
# Distance thresholds (image-plane pixels)
# Derived from actual distance distribution:
#   p5=171px  p10=268px  p25=457px  median=787px
# Logic:
#   < DIST_HIGH   → high proximity risk
#   < DIST_MEDIUM → medium proximity risk
# ──────────────────────────────────────────────────────────────
DIST_HIGH   = 200   # px — well below p10; extremely close in-frame
DIST_MEDIUM = 450   # px — approximately p25

# ──────────────────────────────────────────────────────────────
# Closing speed thresholds (px/s)
# Derived from actual positive closing speed distribution:
#   p50=161px/s  p75=228px/s  p90=295px/s  p95=340px/s
# Logic: Only applied when approaching=True
#   > CS_HIGH   → fast closure (above p90)
#   > CS_MEDIUM → elevated closure (above p75)
# ──────────────────────────────────────────────────────────────
CS_HIGH   = 300   # px/s — above ~p90 of observed closing speeds
CS_MEDIUM = 220   # px/s — above ~p75

# ──────────────────────────────────────────────────────────────
# Heading difference thresholds (degrees)
# Large heading difference → vehicles are crossing paths, not parallel
#   >= HEADING_HIGH   → nearly head-on or crossing (>120°)
#   >= HEADING_MEDIUM → significant angle (>60°)
# ──────────────────────────────────────────────────────────────
HEADING_HIGH   = 120  # degrees
HEADING_MEDIUM = 60   # degrees

# ──────────────────────────────────────────────────────────────
# Score weights (must sum to 1.0)
# TTC is the strongest physical indicator; distance second.
# ──────────────────────────────────────────────────────────────
WEIGHT_TTC        = 0.35
WEIGHT_DISTANCE   = 0.25
WEIGHT_CLOSING    = 0.20
WEIGHT_TRAJECTORY = 0.10
WEIGHT_HEADING    = 0.10

# ──────────────────────────────────────────────────────────────
# Risk level thresholds (total weighted score in [0, 1])
# ──────────────────────────────────────────────────────────────
SCORE_HIGH   = 0.55
SCORE_MEDIUM = 0.25

# ──────────────────────────────────────────────────────────────
# Distance gating threshold (image-plane pixels)
# INITIAL ENGINEERING THRESHOLD — NOT a calibrated real-world distance.
# At image-plane distances >= DISTANCE_GATING_THRESHOLD, closing speed
# and heading risk sub-scores are attenuated to 0 to prevent long-distance
# pairs on opposite sides of the frame from triggering false HIGH alerts.
# ──────────────────────────────────────────────────────────────
DISTANCE_GATING_THRESHOLD = 600.0   # px

# ──────────────────────────────────────────────────────────────
# Event grouping — consecutive HIGH/MEDIUM frames from the same
# vehicle pair are grouped into a single risk event if the gap
# between them is no larger than GROUP_GAP_FRAMES.
# ──────────────────────────────────────────────────────────────
GROUP_GAP_FRAMES = 15   # updated to prevent tracker flickers from splitting events

# ──────────────────────────────────────────────────────────────
# Minimum frames for an event to be reported
# (filters one-frame spikes from the event list)
# ──────────────────────────────────────────────────────────────
MIN_EVENT_FRAMES = 2

# ──────────────────────────────────────────────────────────────
# TTC validity — only use TTC as a risk signal if approaching=True
# This is enforced in the engine and is not configurable.
# ──────────────────────────────────────────────────────────────
