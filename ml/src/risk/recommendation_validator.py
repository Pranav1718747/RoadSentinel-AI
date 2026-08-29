import json
import re
from typing import Dict, Any, Tuple
from recommendation_config import CATEGORIES, PRIORITIES, CONFIDENCES

FORBIDDEN_PATTERNS = [
    r"\bwill\s+crash\b",
    r"\bwill\s+cause\s+an?\s+accident\b",
    r"\baccident\s+will\s+occur\b",
    r"\bwill\s+occur\b",
    r"\bguarantee[d\s]+accident\b",
    r"\b100%\s+chance\b",
    r"\blatitude\b",
    r"\blongitude\b",
    r"\bGPS\s*:\s*\d+",
    r"°\s*[NSEW]"
]

class RecommendationValidator:
    def __init__(self, evidence_summary: Dict[str, Any]):
        self.evidence = evidence_summary

    def validate(self, recommendation: Dict[str, Any]) -> Tuple[bool, str]:
        if not isinstance(recommendation, dict):
            return False, "Output is not a valid JSON object/dictionary"

        # Check required fields
        required_fields = ["category", "priority", "confidence", "title", "summary", "evidence", "recommended_action"]
        for field in required_fields:
            if field not in recommendation or not recommendation[field]:
                return False, f"Missing required field: '{field}'"

        cat = str(recommendation.get("category")).upper()
        if cat not in CATEGORIES:
            return False, f"Invalid or unsupported category: '{cat}'"

        prio = str(recommendation.get("priority")).upper()
        if prio not in PRIORITIES:
            return False, f"Invalid priority: '{prio}'"

        conf = str(recommendation.get("confidence")).upper()
        if conf not in CONFIDENCES:
            return False, f"Invalid confidence: '{conf}'"

        # Check evidence type
        ev_list = recommendation.get("evidence")
        if not isinstance(ev_list, list) or len(ev_list) == 0:
            return False, "Evidence must be a non-empty list of strings"

        full_text = json.dumps(recommendation).lower()

        # Check forbidden patterns (hallucinated GPS, accident guarantees)
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, full_text, re.IGNORECASE):
                return False, f"Recommendation contains forbidden statement/pattern: '{pattern}'"

        return True, "VALID"
