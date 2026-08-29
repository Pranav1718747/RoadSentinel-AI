from django.urls import path
from dashboard.views import DashboardSummaryView, DashboardRiskDistributionView, DashboardTimelineView, DashboardHeatmapView

urlpatterns = [
    path('summary/', DashboardSummaryView.as_view(), name='dashboard-summary'),
    path('risk-distribution/', DashboardRiskDistributionView.as_view(), name='dashboard-risk-distribution'),
    path('timeline/', DashboardTimelineView.as_view(), name='dashboard-timeline'),
    path('heatmap/', DashboardHeatmapView.as_view(), name='dashboard-heatmap'),
]
