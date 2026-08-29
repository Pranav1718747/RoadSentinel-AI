from django.urls import path
from risks.views import (
    VideoInteractionsView, VideoRiskEventsView, VideoRecommendationsView,
    VideoRiskSummaryView, VideoRiskZonesView
)

urlpatterns = [
    path('videos/<int:video_id>/interactions/', VideoInteractionsView.as_view(), name='video-interactions'),
    path('videos/<int:video_id>/risk-events/', VideoRiskEventsView.as_view(), name='video-risk-events'),
    path('videos/<int:video_id>/risk-summary/', VideoRiskSummaryView.as_view(), name='video-risk-summary'),
    path('videos/<int:video_id>/risk-zones/', VideoRiskZonesView.as_view(), name='video-risk-zones'),
    path('videos/<int:video_id>/recommendations/', VideoRecommendationsView.as_view(), name='video-recommendations'),
]
