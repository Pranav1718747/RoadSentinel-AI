import math
from typing import Dict, Any, List

def parse_float(val, default=0.0):
    if val is None or val == "" or str(val).lower() == "nan":
        return default
    try:
        return float(val)
    except Exception:
        return default

def parse_nullable_float(val):
    if val is None or val == "" or str(val).lower() == "nan":
        return None
    try:
        return float(val)
    except Exception:
        return None

class EvidenceAggregator:
    def __init__(self, risk_events: List[Dict[str, Any]], interactions: List[Dict[str, Any]] = None, img_width: float = 3840.0, img_height: float = 2160.0):
        self.risk_events = risk_events or []
        self.interactions = interactions or []
        self.img_width = img_width or 3840.0
        self.img_height = img_height or 2160.0

    def aggregate(self) -> Dict[str, Any]:
        return self.build_evidence_summary()

    def build_evidence_summary(self) -> Dict[str, Any]:
        total_events = len(self.risk_events)
        high_events = [e for e in self.risk_events if str(e.get("risk_level")).upper() == "HIGH"]
        med_events = [e for e in self.risk_events if str(e.get("risk_level")).upper() == "MEDIUM"]
        low_events = [e for e in self.risk_events if str(e.get("risk_level")).upper() == "LOW"]

        # 1. Compute internal 2x2 spatial zones (Zone A, Zone B, Zone C, Zone D)
        zones_data = self._compute_risk_zones()

        zone_a_score = zones_data["Zone A"]["zone_risk_score"]
        zone_b_score = zones_data["Zone B"]["zone_risk_score"]
        zone_c_score = zones_data["Zone C"]["zone_risk_score"]
        zone_d_score = zones_data["Zone D"]["zone_risk_score"]

        # 2. OVERALL TRAFFIC AREA RISK SCORE = ARITHMETIC MEAN OF 4 ZONE SCORES
        overall_risk_score = (zone_a_score + zone_b_score + zone_c_score + zone_d_score) / 4.0
        overall_risk_score = round(overall_risk_score, 2)

        # 3. OVERALL RISK LEVEL THRESHOLDS
        if overall_risk_score >= 80.0:
            overall_risk_level = "CRITICAL"
        elif overall_risk_score >= 60.0:
            overall_risk_level = "HIGH"
        elif overall_risk_score >= 30.0:
            overall_risk_level = "MEDIUM"
        else:
            overall_risk_level = "LOW"

        # Quantitative metrics
        ttc_vals = [parse_nullable_float(e.get("ttc")) for e in self.risk_events]
        ttc_vals = [v for v in ttc_vals if v is not None and v > 0]
        min_ttc = min(ttc_vals) if ttc_vals else None

        dist_vals = [parse_nullable_float(e.get("minimum_distance", e.get("distance"))) for e in self.risk_events]
        dist_vals = [v for v in dist_vals if v is not None and v > 0]
        min_dist = min(dist_vals) if dist_vals else None

        spd_vals = [parse_nullable_float(e.get("closing_speed")) for e in self.risk_events]
        spd_vals = [v for v in spd_vals if v is not None]
        max_closing_spd = max(spd_vals) if spd_vals else 0.0

        scores = [parse_float(e.get("risk_score")) for e in self.risk_events]
        max_risk_score = max(scores) if scores else 0.0
        avg_risk_score = sum(scores) / len(scores) if scores else 0.0

        # Dominant risk patterns
        dominant_patterns = []
        if max_closing_spd >= 30.0:
            dominant_patterns.append("Rapid closing-speed interactions observed across the corridor")
        if min_ttc and min_ttc <= 1.5:
            dominant_patterns.append("Short time-to-collision conflicts (TTC ≤ 1.5s)")
        if len(high_events) >= 2:
            dominant_patterns.append("Repeated close vehicle interactions")
        if len(dominant_patterns) == 0 and total_events > 0:
            dominant_patterns.append("Low density vehicle interaction corridor")
        elif len(dominant_patterns) == 0:
            dominant_patterns.append("Normal traffic flow & clear safety margins")

        return {
            "overall_risk_score": overall_risk_score,
            "overall_risk_level": overall_risk_level,
            "zone_a_score": zone_a_score,
            "zone_b_score": zone_b_score,
            "zone_c_score": zone_c_score,
            "zone_d_score": zone_d_score,
            "total_events": total_events,
            "high_events_count": len(high_events),
            "med_events_count": len(med_events),
            "low_events_count": len(low_events),
            "avg_risk_score": round(avg_risk_score, 3),
            "max_risk_score": max_risk_score,
            "minimum_ttc": min_ttc,
            "minimum_distance": min_dist,
            "maximum_closing_speed": max_closing_spd,
            "dominant_risk_patterns": dominant_patterns,
            "zones_data": zones_data
        }

    def _compute_risk_zones(self) -> Dict[str, Dict[str, Any]]:
        zones = {
            "Zone A": {"x_min": 0.0, "y_min": 0.0, "x_max": 0.5, "y_max": 0.5, "events": []},
            "Zone B": {"x_min": 0.5, "y_min": 0.0, "x_max": 1.0, "y_max": 0.5, "events": []},
            "Zone C": {"x_min": 0.0, "y_min": 0.5, "x_max": 0.5, "y_max": 1.0, "events": []},
            "Zone D": {"x_min": 0.5, "y_min": 0.5, "x_max": 1.0, "y_max": 1.0, "events": []},
        }

        for idx, ev in enumerate(self.risk_events):
            cx = parse_nullable_float(ev.get("center_x", ev.get("x")))
            cy = parse_nullable_float(ev.get("center_y", ev.get("y")))

            if cx is None or cy is None:
                zones["Zone C"]["events"].append(ev)
                continue

            norm_x = min(1.0, max(0.0, cx / (self.img_width or 3840.0)))
            norm_y = min(1.0, max(0.0, cy / (self.img_height or 2160.0)))

            if norm_x <= 0.5 and norm_y <= 0.5:
                zones["Zone A"]["events"].append(ev)
            elif norm_x > 0.5 and norm_y <= 0.5:
                zones["Zone B"]["events"].append(ev)
            elif norm_x <= 0.5 and norm_y > 0.5:
                zones["Zone C"]["events"].append(ev)
            else:
                zones["Zone D"]["events"].append(ev)

        result = {}
        for z_id, z_data in zones.items():
            z_evs = z_data["events"]
            h_cnt = len([e for e in z_evs if str(e.get("risk_level")).upper() == "HIGH"])
            m_cnt = len([e for e in z_evs if str(e.get("risk_level")).upper() == "MEDIUM"])
            l_cnt = len([e for e in z_evs if str(e.get("risk_level")).upper() == "LOW"])

            ttcs = [parse_nullable_float(e.get("ttc")) for e in z_evs if parse_nullable_float(e.get("ttc")) is not None]
            min_z_ttc = min(ttcs) if ttcs else None

            dists = [parse_nullable_float(e.get("minimum_distance", e.get("distance"))) for e in z_evs if parse_nullable_float(e.get("minimum_distance", e.get("distance"))) is not None]
            min_z_dist = min(dists) if dists else None

            spds = [parse_nullable_float(e.get("closing_speed")) for e in z_evs if parse_nullable_float(e.get("closing_speed")) is not None]
            max_z_spd = max(spds) if spds else 0.0

            z_scores = [parse_float(e.get("risk_score")) for e in z_evs]
            max_z_risk = max(z_scores) if z_scores else 0.0
            avg_z_risk = sum(z_scores) / len(z_scores) if z_scores else 0.0

            raw_z_score = (h_cnt * 20.0) + (m_cnt * 4.0) + (l_cnt * 0.5)
            if min_z_ttc and min_z_ttc < 1.5:
                raw_z_score *= 1.2
            zone_risk_score = min(100.0, round(raw_z_score, 1))

            if zone_risk_score >= 80.0:
                zone_risk_level = "CRITICAL"
            elif zone_risk_score >= 60.0:
                zone_risk_level = "HIGH"
            elif zone_risk_score >= 30.0:
                zone_risk_level = "MEDIUM"
            else:
                zone_risk_level = "LOW"

            result[z_id] = {
                "zone_id": z_id,
                "zone_risk_score": zone_risk_score,
                "zone_risk_level": zone_risk_level,
                "total_events": len(z_evs),
                "high_events": h_cnt,
                "medium_events": m_cnt,
                "low_events": l_cnt,
                "average_event_risk": round(avg_z_risk, 3),
                "maximum_event_risk": max_z_risk,
                "minimum_ttc": min_z_ttc,
                "minimum_distance": min_z_dist,
                "maximum_closing_speed": max_z_spd,
                "x_min": z_data["x_min"],
                "y_min": z_data["y_min"],
                "x_max": z_data["x_max"],
                "y_max": z_data["y_max"],
            }

        return result
