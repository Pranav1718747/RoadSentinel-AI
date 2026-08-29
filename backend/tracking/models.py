from django.db import models
from videos.models import Video

class MovementState(models.TextChoices):
    MOVING = 'MOVING', 'Moving'
    STATIONARY = 'STATIONARY', 'Stationary'
    SHORT = 'SHORT', 'Short Track'
    NOISY = 'NOISY', 'Noisy Track'
    QUESTIONABLE = 'QUESTIONABLE', 'Questionable Track'

class Track(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='tracks')
    track_id = models.IntegerField(db_index=True)
    class_id = models.IntegerField()
    class_name = models.CharField(max_length=50)
    start_frame = models.IntegerField()
    end_frame = models.IntegerField()
    duration = models.FloatField(default=0.0)
    movement_state = models.CharField(max_length=30, choices=MovementState.choices, default=MovementState.MOVING)
    total_displacement = models.FloatField(default=0.0)
    avg_speed = models.FloatField(default=0.0)

    class Meta:
        indexes = [
            models.Index(fields=['video', 'track_id']),
            models.Index(fields=['video', 'movement_state']),
        ]

    def __str__(self):
        return f"Track #{self.track_id} ({self.class_name} - {self.movement_state})"
