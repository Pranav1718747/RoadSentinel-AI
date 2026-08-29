from django.db import models
from videos.models import Video

class ReportType(models.TextChoices):
    PDF = 'PDF', 'PDF Report'
    CSV = 'CSV', 'CSV Export'

class ReportStatus(models.TextChoices):
    GENERATING = 'GENERATING', 'Generating'
    COMPLETED = 'COMPLETED', 'Completed'
    FAILED = 'FAILED', 'Failed'

class Report(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='reports')
    report_type = models.CharField(max_length=20, choices=ReportType.choices, default=ReportType.PDF)
    file = models.FileField(upload_to='reports/', blank=True, null=True)
    status = models.CharField(max_length=20, choices=ReportStatus.choices, default=ReportStatus.GENERATING)
    summary = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Report #{self.id} ({self.report_type} - Video {self.video_id})"
