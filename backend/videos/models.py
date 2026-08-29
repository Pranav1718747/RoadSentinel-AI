from django.db import models
from django.conf import settings

class VideoStatus(models.TextChoices):
    UPLOADED = 'UPLOADED', 'Uploaded'
    PROCESSING = 'PROCESSING', 'Processing'
    COMPLETED = 'COMPLETED', 'Completed'
    FAILED = 'FAILED', 'Failed'
    CANCELLED = 'CANCELLED', 'Cancelled'

class Video(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='videos')
    file = models.FileField(upload_to='uploads/')
    filename = models.CharField(max_length=255)
    duration = models.FloatField(default=0.0)
    fps = models.FloatField(default=0.0)
    resolution = models.CharField(max_length=50, default='3840x2160')
    total_frames = models.IntegerField(default=0)
    status = models.CharField(max_length=30, choices=VideoStatus.choices, default=VideoStatus.UPLOADED)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.filename} ({self.status})"
