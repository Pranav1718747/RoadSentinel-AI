from django.db import models
from videos.models import Video
from tracking.models import Track

class Trajectory(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='trajectories')
    track = models.ForeignKey(Track, on_delete=models.CASCADE, related_name='points')
    frame_id = models.IntegerField(db_index=True)
    timestamp = models.FloatField()
    x = models.FloatField()
    y = models.FloatField()
    speed = models.FloatField(default=0.0)
    acceleration = models.FloatField(default=0.0)
    heading = models.FloatField(default=0.0)

    class Meta:
        ordering = ['frame_id']
        indexes = [
            models.Index(fields=['video', 'track', 'frame_id']),
        ]

    def __str__(self):
        return f"Trajectory Track #{self.track.track_id} Frame {self.frame_id} ({self.x:.1f}, {self.y:.1f})"
