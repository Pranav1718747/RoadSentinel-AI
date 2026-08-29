"""
event_grouper.py — RoadSentinel AI, Phase 7
Groups consecutive HIGH/MEDIUM risk frames from the same vehicle pair
into single risk events to avoid alert flooding.
"""

import risk_config as cfg
from collections import defaultdict


def group_events(scored_rows):
    """
    Input : list of scored interaction dicts (one per frame-pair).
    Output: list of event dicts, one per grouped incident.

    Grouping rules:
    - Only HIGH and MEDIUM rows are grouped into events.
    - Consecutive rows for the same (v1_id, v2_id) pair within
      GROUP_GAP_FRAMES frames are merged into one event.
    - Events shorter than MIN_EVENT_FRAMES are discarded.
    """
    NaN = float("nan")

    def sf(v):
        if v is None or str(v).strip() == "": return NaN
        try: return float(v)
        except: return NaN

    def is_nan(v):
        try: return v != v
        except: return True

    # Sort by pair then by frame
    pairs = defaultdict(list)
    for row in scored_rows:
        if row["risk_level"] in ("HIGH", "MEDIUM"):
            key = (row["v1_id"], row["v2_id"])
            pairs[key].append(row)

    raw_events = []
    event_id = 0

    for (v1, v2), rows in pairs.items():
        rows.sort(key=lambda r: int(r["frame_id"]))

        # Sliding merge: if gap to previous row <= GROUP_GAP_FRAMES, extend event
        current_group = [rows[0]]
        for r in rows[1:]:
            gap = int(r["frame_id"]) - int(current_group[-1]["frame_id"])
            if gap <= cfg.GROUP_GAP_FRAMES:
                current_group.append(r)
            else:
                # Finalise current group
                if len(current_group) >= cfg.MIN_EVENT_FRAMES:
                    raw_events.append(_build_event(event_id, current_group))
                    event_id += 1
                current_group = [r]
        # Finalise last group
        if len(current_group) >= cfg.MIN_EVENT_FRAMES:
            raw_events.append(_build_event(event_id, current_group))
            event_id += 1

    # Sort events by max risk score descending
    raw_events.sort(key=lambda e: -e["max_risk_score"])
    return raw_events


def _build_event(event_id, rows):
    """Build a single event summary dict from a list of consecutive rows."""
    NaN = float("nan")
    def sf(v):
        if v is None or str(v).strip() == "": return NaN
        try: return float(v)
        except: return NaN
    def is_nan(v):
        try: return v != v
        except: return True

    frames      = [int(r["frame_id"])   for r in rows]
    timestamps  = [sf(r["timestamp"])   for r in rows]
    scores      = [sf(r["risk_score"])  for r in rows]
    ttcs        = [sf(r["ttc"])         for r in rows if not is_nan(sf(r["ttc"]))]
    dists       = [sf(r["distance"])    for r in rows if not is_nan(sf(r["distance"]))]
    css         = [sf(r["closing_speed"]) for r in rows if not is_nan(sf(r["closing_speed"]))]

    # Predominant risk level (most severe wins)
    levels = [r["risk_level"] for r in rows]
    peak_level = "HIGH" if "HIGH" in levels else "MEDIUM"

    # Collect unique reasons across all frames (top 3 most frequent)
    reason_freq = defaultdict(int)
    for r in rows:
        for reason in r["risk_reasons"].split(";"):
            rs = reason.strip()
            if rs and rs != "No significant risk indicators":
                reason_freq[rs] += 1
    top_reasons = [r for r, _ in sorted(reason_freq.items(), key=lambda x: -x[1])[:3]]

    v1 = rows[0]["v1_id"]
    v2 = rows[0]["v2_id"]

    return {
        "event_id"          : event_id,
        "v1_id"             : v1,
        "v2_id"             : v2,
        "v1_class"          : rows[0]["v1_class"],
        "v2_class"          : rows[0]["v2_class"],
        "start_frame"       : min(frames),
        "end_frame"         : max(frames),
        "event_duration_frames": max(frames) - min(frames) + 1,
        "start_time"        : round(min(t for t in timestamps if not is_nan(t)), 3),
        "end_time"          : round(max(t for t in timestamps if not is_nan(t)), 3),
        "duration_seconds"  : round(max(t for t in timestamps if not is_nan(t)) -
                                    min(t for t in timestamps if not is_nan(t)), 3),
        "peak_risk_level"   : peak_level,
        "max_risk_score"    : round(max(s for s in scores if not is_nan(s)), 4),
        "min_ttc"           : round(min(ttcs), 4) if ttcs else None,
        "min_distance"      : round(min(dists), 2) if dists else None,
        "max_closing_speed" : round(max(css), 2)  if css else None,
        "n_frames"          : len(rows),
        "top_reasons"       : "; ".join(top_reasons),
    }
