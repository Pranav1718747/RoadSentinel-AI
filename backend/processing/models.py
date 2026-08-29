from django.db import models
from videos.models import Video

class JobStatus(models.TextChoices):
    PENDING = 'PENDING', 'Pending'
    QUEUED = 'QUEUED', 'Queued'
    PROCESSING = 'PROCESSING', 'Processing'
    COMPLETED = 'COMPLETED', 'Completed'
    FAILED = 'FAILED', 'Failed'
    CANCELLED = 'CANCELLED', 'Cancelled'

class JobStage(models.TextChoices):
    UPLOADING = 'UPLOADING', 'Uploading'
    VALIDATING = 'VALIDATING', 'Validating'
    DETECTING = 'DETECTING', 'Detecting'
    TRACKING = 'TRACKING', 'Tracking'
    TRAJECTORY_ANALYSIS = 'TRAJECTORY_ANALYSIS', 'Trajectory Analysis'
    INTERACTION_ANALYSIS = 'INTERACTION_ANALYSIS', 'Interaction Analysis'
    RISK_ANALYSIS = 'RISK_ANALYSIS', 'Risk Analysis'
    ML_PREDICTION = 'ML_PREDICTION', 'ML Prediction'
    FINALIZING = 'FINALIZING', 'Finalizing'
    COMPLETED = 'COMPLETED', 'Completed'

class ProcessingJob(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='jobs')
    status = models.CharField(max_length=30, choices=JobStatus.choices, default=JobStatus.PENDING)
    progress = models.FloatField(default=0.0) # 0.0 to 100.0 (overall_progress)
    current_stage = models.CharField(max_length=50, choices=JobStage.choices, default=JobStage.UPLOADING)
    
    # Real-time frame & speed progress metadata
    stage_progress = models.FloatField(default=0.0)
    overall_progress = models.FloatField(default=0.0)
    current_frame = models.IntegerField(default=0)
    total_frames = models.IntegerField(default=0)
    video_position_seconds = models.FloatField(default=0.0)
    video_duration_seconds = models.FloatField(default=0.0)
    elapsed_seconds = models.FloatField(default=0.0)
    processing_fps = models.FloatField(default=0.0)
    eta_seconds = models.FloatField(null=True, blank=True)
    
    processed_detections = models.IntegerField(default=0)
    processed_tracks = models.IntegerField(default=0)
    processed_trajectories = models.IntegerField(default=0)
    processed_interactions = models.IntegerField(default=0)
    
    last_progress_update = models.DateTimeField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-started_at']

    def __str__(self):
        return f"Job #{self.id} - Video {self.video_id} ({self.status})"

class ModelRun(models.Model):
    job = models.OneToOneField(ProcessingJob, on_delete=models.CASCADE, related_name='model_run')
    detector_model = models.CharField(max_length=50, default='YOLO26')
    detector_checkpoint = models.CharField(max_length=255, default='yolo26s.pt')
    risk_model = models.CharField(max_length=50, default='XGBoost')
    risk_model_version = models.CharField(max_length=50, default='1.0.0')
    pipeline_version = models.CharField(max_length=50, default='2.0.0')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"ModelRun for Job #{self.job_id} ({self.detector_model} + {self.risk_model})"
