from django.urls import path
from trajectories.views import VideoTrajectoriesView, TrackTrajectoryView

urlpatterns = [
    path('videos/<int:video_id>/trajectories/', VideoTrajectoriesView.as_view(), name='video-trajectories'),
    path('tracks/<int:track_id>/trajectory/', TrackTrajectoryView.as_view(), name='track-trajectory'),
]
