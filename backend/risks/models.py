from django.db import models
from videos.models import Video

class RiskLevel(models.TextChoices):
    LOW = 'LOW', 'Low Risk'
    MEDIUM = 'MEDIUM', 'Medium Risk'
    HIGH = 'HIGH', 'High Risk'

class RecommendationCategory(models.TextChoices):
    SPEED_MANAGEMENT = 'SPEED_MANAGEMENT', 'Speed Management'
    LANE_MANAGEMENT = 'LANE_MANAGEMENT', 'Lane Management'
    PEDESTRIAN_SAFETY = 'PEDESTRIAN_SAFETY', 'Pedestrian Safety'
    SIGNAGE = 'SIGNAGE', 'Signage'
    ROAD_MARKINGS = 'ROAD_MARKINGS', 'Road Markings'
    INTERSECTION_DESIGN = 'INTERSECTION_DESIGN', 'Intersection Design'
    LIGHTING = 'LIGHTING', 'Lighting'
    TRAFFIC_SIGNAL_REVIEW = 'TRAFFIC_SIGNAL_REVIEW', 'Traffic Signal Review'
    CAMERA_PLACEMENT = 'CAMERA_PLACEMENT', 'Camera Placement'
    ENFORCEMENT_REVIEW = 'ENFORCEMENT_REVIEW', 'Enforcement Review'
    FURTHER_INVESTIGATION = 'FURTHER_INVESTIGATION', 'Further Investigation'

class RecommendationPriority(models.TextChoices):
    CRITICAL = 'CRITICAL', 'Critical'
    HIGH = 'HIGH', 'High'
    MEDIUM = 'MEDIUM', 'Medium'
    LOW = 'LOW', 'Low'

class RecommendationConfidence(models.TextChoices):
    HIGH = 'HIGH', 'High'
    MEDIUM = 'MEDIUM', 'Medium'
    LOW = 'LOW', 'Low'

class RecommendationSource(models.TextChoices):
    RULE_BASED = 'RULE_BASED', 'Rule-Based'
    AI_ENHANCED = 'AI_ENHANCED', 'AI-Enhanced'

class Interaction(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='interactions')
    frame_id = models.IntegerField(db_index=True)
    vehicle_1 = models.IntegerField()
    vehicle_2 = models.IntegerField()
    distance = models.FloatField()
    relative_speed = models.FloatField(default=0.0)
    closing_speed = models.FloatField(default=0.0)
    heading_difference = models.FloatField(default=0.0)
    approaching = models.BooleanField(default=False)
    ttc = models.FloatField(null=True, blank=True)
    pet = models.FloatField(null=True, blank=True)
    minimum_distance = models.FloatField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=['video', 'frame_id']),
            models.Index(fields=['video', 'vehicle_1', 'vehicle_2']),
        ]

    def __str__(self):
        return f"Interaction Frame {self.frame_id}: V{self.vehicle_1} <-> V{self.vehicle_2} (Dist: {self.distance:.1f}px)"

class RiskEvent(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='risk_events')
    event_id = models.IntegerField(db_index=True)
    start_time = models.FloatField()
    end_time = models.FloatField()
    vehicles = models.JSONField(default=list)
    vehicle_classes = models.JSONField(default=list)
    risk_level = models.CharField(max_length=20, choices=RiskLevel.choices, default=RiskLevel.LOW, db_index=True)
    risk_score = models.FloatField()
    ttc = models.FloatField(null=True, blank=True)
    pet = models.FloatField(null=True, blank=True)
    minimum_distance = models.FloatField(null=True, blank=True)
    closing_speed = models.FloatField(null=True, blank=True)
    risk_reasons = models.JSONField(default=list)
    ml_prediction = models.CharField(max_length=30, blank=True, null=True)
    ml_confidence = models.FloatField(null=True, blank=True)
    final_risk = models.CharField(max_length=30)
    decision_source = models.CharField(max_length=50, default='Rule_Override')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['start_time']
        indexes = [
            models.Index(fields=['video', 'risk_level']),
            models.Index(fields=['video', 'start_time']),
        ]

    def __str__(self):
        return f"RiskEvent #{self.event_id} ({self.risk_level} - Score: {self.risk_score:.2f})"

class VideoRiskSummary(models.Model):
    video = models.OneToOneField(Video, on_delete=models.CASCADE, related_name='risk_summary')
    overall_risk_score = models.FloatField(default=0.0) # 0 to 100
    overall_risk_level = models.CharField(max_length=20, choices=RiskLevel.choices, default=RiskLevel.LOW)
    zone_a_score = models.FloatField(default=0.0)
    zone_b_score = models.FloatField(default=0.0)
    zone_c_score = models.FloatField(default=0.0)
    zone_d_score = models.FloatField(default=0.0)
    total_events = models.IntegerField(default=0)
    low_events = models.IntegerField(default=0)
    medium_events = models.IntegerField(default=0)
    high_events = models.IntegerField(default=0)
    average_event_risk = models.FloatField(default=0.0)
    maximum_event_risk = models.FloatField(default=0.0)
    minimum_ttc = models.FloatField(null=True, blank=True)
    minimum_distance = models.FloatField(null=True, blank=True)
    maximum_closing_speed = models.FloatField(null=True, blank=True)
    highest_risk_event = models.ForeignKey(RiskEvent, on_delete=models.SET_NULL, null=True, blank=True, related_name='+')
    highest_risk_zone = models.CharField(max_length=50, default='Zone A')
    dominant_risk_patterns = models.JSONField(default=list)
    recommendation_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"RiskSummary Video #{self.video_id}: {self.overall_risk_score:.1f}/100 ({self.overall_risk_level})"

class RiskZoneSummary(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='zone_summaries')
    zone_id = models.CharField(max_length=50) # e.g. Zone A, Zone B, Zone C, Zone D
    zone_risk_score = models.FloatField(default=0.0)
    zone_risk_level = models.CharField(max_length=20, choices=RiskLevel.choices, default=RiskLevel.LOW)
    total_events = models.IntegerField(default=0)
    high_events = models.IntegerField(default=0)
    medium_events = models.IntegerField(default=0)
    low_events = models.IntegerField(default=0)
    average_event_risk = models.FloatField(default=0.0)
    maximum_event_risk = models.FloatField(default=0.0)
    minimum_ttc = models.FloatField(null=True, blank=True)
    minimum_distance = models.FloatField(null=True, blank=True)
    maximum_closing_speed = models.FloatField(null=True, blank=True)
    risk_density = models.FloatField(default=0.0)
    unique_vehicles_involved = models.IntegerField(default=0)
    dominant_patterns = models.JSONField(default=list)
    x_min = models.FloatField(default=0.0)
    y_min = models.FloatField(default=0.0)
    x_max = models.FloatField(default=1.0)
    y_max = models.FloatField(default=1.0)

    class Meta:
        ordering = ['zone_id']
        indexes = [
            models.Index(fields=['video', 'zone_id']),
        ]

    def __str__(self):
        return f"Zone {self.zone_id} Video #{self.video_id}: {self.zone_risk_score:.1f} ({self.zone_risk_level})"

class Recommendation(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='recommendations')
    risk_event = models.ForeignKey(RiskEvent, on_delete=models.SET_NULL, null=True, blank=True, related_name='recommendations')
    zone_id = models.CharField(max_length=50, blank=True, null=True)
    category = models.CharField(max_length=50, choices=RecommendationCategory.choices, db_index=True)
    priority = models.CharField(max_length=20, choices=RecommendationPriority.choices, default=RecommendationPriority.MEDIUM, db_index=True)
    confidence = models.CharField(max_length=20, choices=RecommendationConfidence.choices, default=RecommendationConfidence.HIGH)
    title = models.CharField(max_length=255)
    summary = models.TextField()
    evidence = models.JSONField(default=list)
    recommended_action = models.TextField()
    limitations = models.TextField(blank=True, default='')
    source = models.CharField(max_length=30, choices=RecommendationSource.choices, default=RecommendationSource.RULE_BASED, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['priority', '-created_at']
        indexes = [
            models.Index(fields=['video', 'priority']),
            models.Index(fields=['video', 'category']),
            models.Index(fields=['video', 'source']),
        ]

    def __str__(self):
        return f"Recommendation ({self.priority} - {self.category}): {self.title}"
