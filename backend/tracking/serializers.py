from rest_framework import serializers
from tracking.models import Track

class TrackSerializer(serializers.ModelSerializer):
    class Meta:
        model = Track
        fields = ['id', 'video', 'track_id', 'class_id', 'class_name', 'start_frame', 'end_frame', 'duration', 'movement_state', 'total_displacement', 'avg_speed']
