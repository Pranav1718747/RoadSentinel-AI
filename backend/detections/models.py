from django.db import models
from videos.models import Video

class Detection(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='detections')
    frame_id = models.IntegerField(db_index=True)
    timestamp = models.FloatField()
    class_id = models.IntegerField()
    class_name = models.CharField(max_length=50)
    confidence = models.FloatField()
    x1 = models.FloatField()
    y1 = models.FloatField()
    x2 = models.FloatField()
    y2 = models.FloatField()
    center_x = models.FloatField()
    center_y = models.FloatField()

    class Meta:
        indexes = [
            models.Index(fields=['video', 'frame_id']),
            models.Index(fields=['video', 'class_name']),
        ]

    def __str__(self):
        return f"Detection Frame {self.frame_id} - {self.class_name} ({self.confidence:.2f})"
