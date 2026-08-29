# RoadSentinel AI — Recommendation Engine Configuration

# Controlled Recommendation Categories
CATEGORIES = [
    "SPEED_MANAGEMENT",
    "LANE_MANAGEMENT",
    "PEDESTRIAN_SAFETY",
    "SIGNAGE",
    "ROAD_MARKINGS",
    "INTERSECTION_DESIGN",
    "LIGHTING",
    "TRAFFIC_SIGNAL_REVIEW",
    "CAMERA_PLACEMENT",
    "ENFORCEMENT_REVIEW",
    "FURTHER_INVESTIGATION"
]

# Priority Levels
PRIORITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

# Confidence Levels
CONFIDENCES = ["HIGH", "MEDIUM", "LOW"]

# Sources
SOURCES = ["RULE_BASED", "AI_ENHANCED"]

# Rule Thresholds & Configuration Parameters
THRESHOLDS = {
    "HIGH_CLOSING_SPEED_PX_S": 30.0,
    "CRITICAL_TTC_SEC": 1.5,
    "WARNING_TTC_SEC": 3.0,
    "MIN_PROXIMITY_DIST_PX": 150.0,
    "MIN_EVENT_COUNT_FOR_PATTERN": 2,
    "SPATIAL_CLUSTER_RADIUS_PX": 200.0,
}

# Recommendation Templates
TEMPLATES = {
    "SPEED_MANAGEMENT": {
        "title": "Speed Management & Calibrations Review",
        "action": "Review speed-management measures, advisory speed limits, or traffic-calming features in the affected area.",
        "limitations": "Image-plane closing speed is calculated without metric ground survey. Physical site validation is required."
    },
    "LANE_MANAGEMENT": {
        "title": "Lane Management & Merging Review",
        "action": "Review lane channelization, merging zone pavement markings, and lane discipline enforcement.",
        "limitations": "Lane boundaries are inferred from trajectory clusters."
    },
    "PEDESTRIAN_SAFETY": {
        "title": "Pedestrian Safety Enhancement",
        "action": "Evaluate pedestrian crosswalk visibility, refuge islands, and pedestrian signal timing.",
        "limitations": "Pedestrian trajectory classification is based on motion footprint and bounding box dimensions."
    },
    "INTERSECTION_DESIGN": {
        "title": "Intersection Geometric & Conflict Point Audit",
        "action": "Conduct a formal road safety audit focusing on high-density conflict coordinates identified in aerial tracking.",
        "limitations": "Camera lens distortion and height angle may distort spatial cluster boundaries."
    },
    "TRAFFIC_SIGNAL_REVIEW": {
        "title": "Traffic Signal Phase & Cycle Review",
        "action": "Audit traffic signal phase timing and clearance intervals at signalized interaction points.",
        "limitations": "Signal state is inferred from vehicle deceleration profiles; signal head state was not directly recorded."
    },
    "FURTHER_INVESTIGATION": {
        "title": "Detailed Site Traffic Safety Audit Required",
        "action": "Conduct targeted site inspection and collect additional video footage to evaluate transient traffic hazards.",
        "limitations": "Observed risk events exhibit moderate pattern strength; sample duration may be limited."
    }
}
