from django.urls import path
from tracking.views import VideoTracksView, TrackDetailView

urlpatterns = [
    path('videos/<int:video_id>/tracks/', VideoTracksView.as_view(), name='video-tracks'),
    path('tracks/<int:track_id>/', TrackDetailView.as_view(), name='track-detail'),
]
