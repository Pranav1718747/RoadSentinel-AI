from typing import Dict, Any, List
from recommendation_config import CATEGORIES, PRIORITIES, CONFIDENCES, TEMPLATES, THRESHOLDS
from recommendation_evidence import EvidenceAggregator

class RecommendationRuleEngine:
    def __init__(self, evidence_summary: Dict[str, Any]):
        self.ev = evidence_summary

    def generate_recommendations(self) -> List[Dict[str, Any]]:
        recs = []

        total_events = self.ev.get("total_events", 0)
        high_cnt = self.ev.get("high_events_count", 0)
        med_cnt = self.ev.get("med_events_count", 0)
        max_closing_spd = self.ev.get("maximum_closing_speed", 0.0)
        min_ttc = self.ev.get("minimum_ttc")
        min_dist = self.ev.get("minimum_distance")
        v_classes = [str(c).lower() for c in self.ev.get("vehicle_classes", [])]
        reason_map = self.ev.get("reason_event_map", {})
        spatial_clusters = self.ev.get("spatial_clusters", [])

        # Case 1: Zero risk events
        if total_events == 0:
            recs.append({
                "category": "FURTHER_INVESTIGATION",
                "priority": "LOW",
                "confidence": "HIGH",
                "title": "Insufficient Evidence for Infrastructure Change",
                "summary": "No risk events or hazardous vehicle conflicts were detected in the analyzed video sample.",
                "evidence": ["0 hazardous vehicle interactions recorded across video frames."],
                "recommended_action": "Continue routine traffic monitoring.",
                "limitations": "Short sample duration may not capture rare peak-hour traffic conflicts.",
                "source": "RULE_BASED",
                "risk_event_ids": []
            })
            return recs

        # Rule 1: Elevated Closing Speed & High Conflicts -> SPEED_MANAGEMENT
        if (high_cnt >= 1 or med_cnt >= 2) and max_closing_spd >= THRESHOLDS["HIGH_CLOSING_SPEED_PX_S"]:
            ev_ids = self.ev.get("high_event_ids", []) + self.ev.get("med_event_ids", [])
            priority = "CRITICAL" if high_cnt >= 3 else "HIGH"
            tmpl = TEMPLATES["SPEED_MANAGEMENT"]
            recs.append({
                "category": "SPEED_MANAGEMENT",
                "priority": priority,
                "confidence": "HIGH",
                "title": tmpl["title"],
                "summary": f"Observed {high_cnt} HIGH-risk and {med_cnt} MEDIUM-risk conflict events with elevated closing speeds (max {max_closing_spd:.1f} px/s).",
                "evidence": [
                    f"Recorded {high_cnt} HIGH-risk and {med_cnt} MEDIUM-risk conflict events.",
                    f"Maximum observed closing speed: {max_closing_spd:.1f} px/s.",
                    f"Minimum Time-to-Collision (TTC): {f'{min_ttc:.2f}s' if min_ttc else 'N/A (>30s)'}."
                ],
                "recommended_action": tmpl["action"],
                "limitations": tmpl["limitations"],
                "source": "RULE_BASED",
                "risk_event_ids": list(set(ev_ids[:10]))
            })

        # Rule 2: Low Time-to-Collision (< 1.5s) -> SPEED_MANAGEMENT / TRAFFIC_SIGNAL_REVIEW
        if min_ttc is not None and min_ttc <= THRESHOLDS["CRITICAL_TTC_SEC"]:
            ev_ids = self.ev.get("high_event_ids", []) or self.ev.get("med_event_ids", [])
            # Check if any signalized reason is present
            is_signalized = any("signal" in r.lower() or "intersection" in r.lower() for r in reason_map.keys())
            category = "TRAFFIC_SIGNAL_REVIEW" if is_signalized else "SPEED_MANAGEMENT"
            tmpl = TEMPLATES[category]
            recs.append({
                "category": category,
                "priority": "HIGH",
                "confidence": "HIGH",
                "title": tmpl["title"],
                "summary": f"Repeated low Time-to-Collision (TTC = {min_ttc:.2f}s) interactions detected.",
                "evidence": [
                    f"Critical minimum TTC observed: {min_ttc:.2f}s (Threshold: {THRESHOLDS['CRITICAL_TTC_SEC']}s).",
                    f"Associated conflict events: {len(ev_ids)}."
                ],
                "recommended_action": tmpl["action"],
                "limitations": tmpl["limitations"],
                "source": "RULE_BASED",
                "risk_event_ids": list(set(ev_ids[:10]))
            })

        # Rule 3: Pedestrian-Vehicle Conflicts -> PEDESTRIAN_SAFETY
        has_pedestrian = any("pedestrian" in c or "person" in c for c in v_classes) or any("pedestrian" in r.lower() for r in reason_map.keys())
        if has_pedestrian and (high_cnt >= 1 or med_cnt >= 1):
            ev_ids = self.ev.get("high_event_ids", []) + self.ev.get("med_event_ids", [])
            tmpl = TEMPLATES["PEDESTRIAN_SAFETY"]
            recs.append({
                "category": "PEDESTRIAN_SAFETY",
                "priority": "HIGH" if high_cnt >= 1 else "MEDIUM",
                "confidence": "HIGH",
                "title": tmpl["title"],
                "summary": f"Detected vulnerable road user / pedestrian conflict interactions.",
                "evidence": [
                    f"Pedestrian / vulnerable user class identified in conflict events.",
                    f"Conflict events count: {len(ev_ids)}."
                ],
                "recommended_action": tmpl["action"],
                "limitations": tmpl["limitations"],
                "source": "RULE_BASED",
                "risk_event_ids": list(set(ev_ids[:10]))
            })

        # Rule 4: Spatial Coordinate Clustering -> INTERSECTION_DESIGN
        if len(spatial_clusters) >= 1:
            top_cluster = max(spatial_clusters, key=lambda c: c["count"])
            tmpl = TEMPLATES["INTERSECTION_DESIGN"]
            recs.append({
                "category": "INTERSECTION_DESIGN",
                "priority": "HIGH" if top_cluster["count"] >= 3 else "MEDIUM",
                "confidence": "HIGH",
                "title": tmpl["title"],
                "summary": f"Concentrated conflict cluster detected around image coordinates ({top_cluster['center_x']}, {top_cluster['center_y']}).",
                "evidence": [
                    f"{top_cluster['count']} conflict events clustered at image coordinates ({top_cluster['center_x']}, {top_cluster['center_y']}).",
                    f"Cluster event IDs: {top_cluster['cluster_event_ids']}."
                ],
                "recommended_action": tmpl["action"],
                "limitations": tmpl["limitations"],
                "source": "RULE_BASED",
                "risk_event_ids": top_cluster["cluster_event_ids"]
            })

        # Rule 5: Fallback for isolated or low/medium risk events -> FURTHER_INVESTIGATION
        if len(recs) == 0:
            tmpl = TEMPLATES["FURTHER_INVESTIGATION"]
            recs.append({
                "category": "FURTHER_INVESTIGATION",
                "priority": "MEDIUM" if (high_cnt >= 1 or med_cnt >= 1) else "LOW",
                "confidence": "MEDIUM",
                "title": tmpl["title"],
                "summary": f"Recorded {total_events} interactions ({high_cnt} HIGH, {med_cnt} MEDIUM, {self.ev.get('low_events_count', 0)} LOW).",
                "evidence": [
                    f"Total recorded interactions: {total_events}.",
                    f"Minimum observed distance: {f'{min_dist:.1f} px' if min_dist else 'N/A'}."
                ],
                "recommended_action": tmpl["action"],
                "limitations": tmpl["limitations"],
                "source": "RULE_BASED",
                "risk_event_ids": self.ev.get("high_event_ids", []) or self.ev.get("med_event_ids", [])
            })

        # Deduplicate recommendations by category
        deduped = {}
        for r in recs:
            cat = r["category"]
            if cat not in deduped:
                deduped[cat] = r
            else:
                # Prefer higher priority
                p1_idx = PRIORITIES.index(r["priority"]) if r["priority"] in PRIORITIES else 99
                p2_idx = PRIORITIES.index(deduped[cat]["priority"]) if deduped[cat]["priority"] in PRIORITIES else 99
                if p1_idx < p2_idx:
                    deduped[cat] = r

        final_recs = list(deduped.values())

        # Sort recommendations by priority (CRITICAL > HIGH > MEDIUM > LOW)
        final_recs.sort(key=lambda r: PRIORITIES.index(r["priority"]) if r["priority"] in PRIORITIES else 99)
        return final_recs
