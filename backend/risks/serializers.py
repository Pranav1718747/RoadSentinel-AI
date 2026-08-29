from rest_framework import serializers
from risks.models import Interaction, RiskEvent, Recommendation, VideoRiskSummary, RiskZoneSummary

class InteractionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Interaction
        fields = [
            'id', 'video', 'frame_id', 'vehicle_1', 'vehicle_2', 'distance', 
            'relative_speed', 'closing_speed', 'heading_difference', 'approaching', 
            'ttc', 'pet', 'minimum_distance'
        ]

class RiskEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = RiskEvent
        fields = [
            'id', 'video', 'event_id', 'start_time', 'end_time', 'vehicles', 
            'vehicle_classes', 'risk_level', 'risk_score', 'ttc', 'pet', 
            'minimum_distance', 'closing_speed', 'risk_reasons', 'ml_prediction', 
            'ml_confidence', 'final_risk', 'decision_source', 'created_at'
        ]

class VideoRiskSummarySerializer(serializers.ModelSerializer):
    video_id = serializers.IntegerField(source='video.id', read_only=True)
    overall_risk_score = serializers.SerializerMethodField()
    risk_distribution = serializers.SerializerMethodField()
    critical_metrics = serializers.SerializerMethodField()

    class Meta:
        model = VideoRiskSummary
        fields = [
            'id', 'video_id', 'overall_risk_score', 'overall_risk_level',
            'risk_distribution', 'critical_metrics', 'dominant_risk_patterns',
            'recommendation_count', 'created_at'
        ]

    def get_overall_risk_score(self, obj):
        return round(obj.overall_risk_score)

    def get_risk_distribution(self, obj):
        return {
            "high": obj.high_events,
            "medium": obj.medium_events,
            "low": obj.low_events,
            "total": obj.total_events
        }

    def get_critical_metrics(self, obj):
        return {
            "minimum_ttc": obj.minimum_ttc,
            "minimum_distance": obj.minimum_distance,
            "maximum_closing_speed": obj.maximum_closing_speed
        }

class RiskZoneSummarySerializer(serializers.ModelSerializer):
    class Meta:
        model = RiskZoneSummary
        fields = [
            'id', 'video', 'zone_id', 'zone_risk_score', 'zone_risk_level',
            'total_events', 'high_events', 'medium_events', 'low_events',
            'average_event_risk', 'maximum_event_risk', 'minimum_ttc',
            'minimum_distance', 'maximum_closing_speed', 'risk_density',
            'unique_vehicles_involved', 'dominant_patterns',
            'x_min', 'y_min', 'x_max', 'y_max'
        ]

class RecommendationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recommendation
        fields = [
            'id', 'video', 'risk_event', 'zone_id', 'category', 'priority', 'confidence', 
            'title', 'summary', 'evidence', 'recommended_action', 'limitations', 
            'source', 'created_at', 'updated_at'
        ]
