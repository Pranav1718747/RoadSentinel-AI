from rest_framework import serializers
from videos.models import Video

class VideoSerializer(serializers.ModelSerializer):
    processing_job_id = serializers.SerializerMethodField()

    class Meta:
        model = Video
        fields = [
            'id', 'filename', 'file', 'duration', 'fps', 'resolution', 
            'total_frames', 'status', 'created_at', 'updated_at', 'processing_job_id'
        ]
        read_only_fields = ['id', 'duration', 'fps', 'resolution', 'total_frames', 'status', 'created_at', 'updated_at']

    def get_processing_job_id(self, obj):
        latest_job = obj.jobs.first()
        return latest_job.id if latest_job else None
