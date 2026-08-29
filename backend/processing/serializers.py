from rest_framework import serializers
from processing.models import ProcessingJob, ModelRun

class ModelRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelRun
        fields = ['detector_model', 'detector_checkpoint', 'risk_model', 'risk_model_version', 'pipeline_version', 'created_at']

class ProcessingJobSerializer(serializers.ModelSerializer):
    model_run = ModelRunSerializer(read_only=True)
    video_filename = serializers.CharField(source='video.filename', read_only=True)

    class Meta:
        model = ProcessingJob
        fields = [
            'id', 'video', 'video_filename', 'status', 'progress', 
            'current_stage', 'stage_progress', 'overall_progress',
            'current_frame', 'total_frames', 'video_position_seconds', 'video_duration_seconds',
            'elapsed_seconds', 'processing_fps', 'eta_seconds',
            'processed_detections', 'processed_tracks', 'processed_trajectories', 'processed_interactions',
            'last_progress_update', 'started_at', 'completed_at', 'error_message', 'model_run'
        ]
        read_only_fields = ['id', 'video', 'started_at', 'completed_at']
