from rest_framework import serializers
from trajectories.models import Trajectory

class TrajectorySerializer(serializers.ModelSerializer):
    track_id = serializers.IntegerField(source='track.track_id', read_only=True)

    class Meta:
        model = Trajectory
        fields = ['id', 'video', 'track_id', 'frame_id', 'timestamp', 'x', 'y', 'speed', 'acceleration', 'heading']
