from django.urls import path
from reports.views import GenerateVideoReportView, ReportListView, ReportDetailView, DownloadReportView

urlpatterns = [
    path('videos/<int:video_id>/reports/', GenerateVideoReportView.as_view(), name='generate-video-report'),
    path('reports/', ReportListView.as_view(), name='report-list'),
    path('reports/<int:pk>/', ReportDetailView.as_view(), name='report-detail'),
    path('reports/<int:report_id>/download/', DownloadReportView.as_view(), name='report-download'),
]
