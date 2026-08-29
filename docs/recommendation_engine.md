# RoadSentinel AI — Road Safety Recommendation Engine Specification

## Overview
The RoadSentinel Recommendation Engine is a downstream decision-support system that consumes validated risk outputs from the frozen ML/CV pipeline (YOLO26 + ByteTrack + Physics + XGBoost) and produces deterministic, evidence-backed road safety recommendations.

```
       [ VALIDATED RISK EVENTS & TRAJECTORIES ]
                          │
                          ▼
                [ EVIDENCE AGGREGATOR ]
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
      [ RULE ENGINE ]         [ AI CONTEXTUALIZER ]
             │                         │
             └────────────┬────────────┘
                          ▼
              [ RECOMMENDATION VALIDATOR ]
                          │
                          ▼
            [ DJANGO REST API & DATABASE ]
                          │
                          ▼
                [ REACT DASHBOARD UI ]
```

---

## Controlled Vocabulary Categories
- `SPEED_MANAGEMENT`: Advisory speeds, speed-calming features, corridor speed reviews.
- `LANE_MANAGEMENT`: Lane markings, channelization, merging zone enforcement.
- `PEDESTRIAN_SAFETY`: Crosswalk visibility, refuge islands, pedestrian timing.
- `SIGNAGE`: Warning signs, advisory speed signs.
- `ROAD_MARKINGS`: Pavement markings, stop bars, lane lines.
- `INTERSECTION_DESIGN`: Geometric layout, conflict point audits.
- `LIGHTING`: Illumination reviews for night/shadow zones.
- `TRAFFIC_SIGNAL_REVIEW`: Signal phase timing, clearance intervals.
- `CAMERA_PLACEMENT`: Surveillance coverage adjustments.
- `ENFORCEMENT_REVIEW`: Targeted traffic enforcement.
- `FURTHER_INVESTIGATION`: Site audit required due to insufficient pattern sample.

---

## Deterministic Rule Rules & Thresholds
- **Rule 1 (Speed Management)**: Triggered if `high_events >= 1` or `med_events >= 2` and `max_closing_speed >= 30.0 px/s`. Output Priority: `HIGH` or `CRITICAL`.
- **Rule 2 (Signal Review / Speed)**: Triggered if `minimum_ttc <= 1.5s`. Output Priority: `HIGH`.
- **Rule 3 (Pedestrian Safety)**: Triggered if pedestrian / person class is involved in conflict events. Output Priority: `HIGH` or `MEDIUM`.
- **Rule 4 (Intersection Design)**: Triggered if >=2 conflict events cluster within 200px image space radius. Output Priority: `HIGH` or `MEDIUM`.
- **Rule 5 (Further Investigation)**: Triggered if events exist but pattern sample is inconclusive. Output Priority: `LOW` or `MEDIUM`.

---

## AI Role & Validation Layer
- **Prompt Guardrails**: Strictly prohibits inventing GPS coordinates, accidents, or unmeasured metrics.
- **Validation Check**: `RecommendationValidator` verifies category, priority, confidence, JSON schema, and checks for forbidden strings (`"will crash"`, `"accident will occur"`, `"100% chance"`).
- **Fallback**: If AI contextualization fails or is invalid, the engine silently falls back to `RULE_BASED`.

---

## Database Model & API Schema

### Django Model (`Recommendation`)
- `video`: ForeignKey(`Video`)
- `risk_event`: ForeignKey(`RiskEvent`, null=True)
- `category`: CharField (`SPEED_MANAGEMENT`, etc.)
- `priority`: CharField (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`)
- `confidence`: CharField (`HIGH`, `MEDIUM`, `LOW`)
- `title`: CharField
- `summary`: TextField
- `evidence`: JSONField
- `recommended_action`: TextField
- `limitations`: TextField
- `source`: CharField (`RULE_BASED`, `AI_ENHANCED`)

### REST Endpoint
`GET /api/videos/{video_id}/recommendations/`
Filters: `priority`, `category`, `source`

---

## System Status Matrix
- **RECOMMENDATION ENGINE**: **PASS**
- **RULE ENGINE**: **PASS**
- **AI CONTEXTUALIZATION**: **PASS**
- **VALIDATION**: **PASS**
- **DATABASE**: **PASS**
- **API**: **PASS**
- **FRONTEND**: **PASS**
- **END-TO-END**: **PASS**
