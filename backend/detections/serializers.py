from rest_framework import serializers
from detections.models import Detection

class DetectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Detection
        fields = ['id', 'video', 'frame_id', 'timestamp', 'class_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2', 'center_x', 'center_y']
