from django.urls import path
from detections.views import VideoDetectionsView

urlpatterns = [
    path('videos/<int:video_id>/detections/', VideoDetectionsView.as_view(), name='video-detections'),
]
