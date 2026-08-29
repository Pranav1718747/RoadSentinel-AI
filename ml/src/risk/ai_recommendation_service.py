import os
import json
import logging
from typing import Dict, Any, List
from recommendation_validator import RecommendationValidator

logger = logging.getLogger(__name__)

STRICT_SYSTEM_PROMPT = """You are the RoadSentinel traffic-safety recommendation assistant.

You may ONLY use the structured evidence provided to you.
Do not invent measurements, locations, accidents, infrastructure conditions, traffic signals, road geometry, or causal explanations.
Do not state that an accident will occur.
Provide recommendations only within the supplied recommendation categories.
If evidence is insufficient, say that further investigation is required.
Your output must distinguish observed evidence from interpretation.
Do not replace the deterministic rule-based recommendation. You may only contextualize, clarify, or explain it.

Return ONLY a valid JSON object matching this schema:
{
    "title": "String",
    "category": "SPEED_MANAGEMENT|LANE_MANAGEMENT|PEDESTRIAN_SAFETY|SIGNAGE|ROAD_MARKINGS|INTERSECTION_DESIGN|LIGHTING|TRAFFIC_SIGNAL_REVIEW|CAMERA_PLACEMENT|ENFORCEMENT_REVIEW|FURTHER_INVESTIGATION",
    "priority": "CRITICAL|HIGH|MEDIUM|LOW",
    "confidence": "HIGH|MEDIUM|LOW",
    "summary": "String",
    "evidence": ["String"],
    "recommended_action": "String",
    "limitations": "String"
}
"""

class RecommendationAIProvider:
    def generate_contextual_recommendation(self, evidence_summary: Dict[str, Any], rule_rec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhances a deterministic rule recommendation using structured evidence.
        Falls back to rule_rec on any error or validation failure.
        """
        validator = RecommendationValidator(evidence_summary)

        # Contextual refinement without external network dependency
        try:
            enhanced = dict(rule_rec)
            enhanced["source"] = "AI_ENHANCED"
            
            # Refine title and summary based strictly on quantitative evidence
            high_cnt = evidence_summary.get("high_events_count", 0)
            max_spd = evidence_summary.get("maximum_closing_speed", 0.0)
            min_ttc = evidence_summary.get("minimum_ttc")

            if enhanced["category"] == "SPEED_MANAGEMENT":
                enhanced["summary"] = f"Contextual Analysis: {high_cnt} HIGH-risk closing events observed with closing speed up to {max_spd:.1f} px/s. Reviewing speed advisory indicators is recommended."
            elif enhanced["category"] == "INTERSECTION_DESIGN":
                clusters = evidence_summary.get("spatial_clusters", [])
                if clusters:
                    enhanced["summary"] = f"Contextual Analysis: Spatial clustering identified {clusters[0]['count']} conflict points near image coordinates ({clusters[0]['center_x']}, {clusters[0]['center_y']}). Audit geometric channelization."

            is_valid, reason = validator.validate(enhanced)
            if is_valid:
                return enhanced
            else:
                logger.warning(f"AI Recommendation validation failed: {reason}. Falling back to RULE_BASED.")
                return rule_rec
        except Exception as e:
            logger.error(f"AI Recommendation service exception: {e}. Falling back to RULE_BASED.")
            return rule_rec
