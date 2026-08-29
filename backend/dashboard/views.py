from rest_framework import views, permissions
from rest_framework.response import Response
from django.db.models import Avg, Count
from videos.models import Video, VideoStatus
from processing.models import ProcessingJob, JobStatus
from risks.models import RiskEvent, Interaction, Recommendation

class DashboardSummaryView(views.APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        user = request.user
        videos_qs = Video.objects.all() if user.is_admin() else Video.objects.filter(user=user)
        jobs_qs = ProcessingJob.objects.all() if user.is_admin() else ProcessingJob.objects.filter(video__user=user)
        risks_qs = RiskEvent.objects.all() if user.is_admin() else RiskEvent.objects.filter(video__user=user)
        interactions_qs = Interaction.objects.all() if user.is_admin() else Interaction.objects.filter(video__user=user)
        recs_qs = Recommendation.objects.all() if user.is_admin() else Recommendation.objects.filter(video__user=user)

        high_cnt = risks_qs.filter(risk_level='HIGH').count()
        med_cnt = risks_qs.filter(risk_level='MEDIUM').count()
        low_cnt = risks_qs.filter(risk_level='LOW').count()

        avg_ttc = interactions_qs.filter(ttc__isnull=False).aggregate(Avg('ttc'))['ttc__avg'] or 0.0
        avg_min_dist = interactions_qs.filter(minimum_distance__isnull=False).aggregate(Avg('minimum_distance'))['minimum_distance__avg'] or 0.0
        avg_risk_score = risks_qs.aggregate(Avg('risk_score'))['risk_score__avg'] or 0.0

        top_rec = recs_qs.first()
        highest_prio = top_rec.priority if top_rec else 'NONE'

        return Response({
            "total_videos": videos_qs.count(),
            "completed_jobs": jobs_qs.filter(status=JobStatus.COMPLETED).count(),
            "processing_jobs": jobs_qs.filter(status=JobStatus.PROCESSING).count(),
            "total_risk_events": risks_qs.count(),
            "risk_counts": {
                "HIGH": high_cnt,
                "MEDIUM": med_cnt,
                "LOW": low_cnt
            },
            "averages": {
                "avg_ttc": round(avg_ttc, 2),
                "avg_minimum_distance": round(avg_min_dist, 2),
                "avg_risk_score": round(avg_risk_score, 4)
            },
            "recommendation_summary": {
                "recommendation_count": recs_qs.count(),
                "highest_priority_recommendation": highest_prio
            }
        })

class DashboardRiskDistributionView(views.APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        user = request.user
        risks_qs = RiskEvent.objects.all() if user.is_admin() else RiskEvent.objects.filter(video__user=user)

        dist = list(risks_qs.values('risk_level').annotate(count=Count('id')))
        return Response({"distribution": dist})

class DashboardTimelineView(views.APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        user = request.user
        risks_qs = RiskEvent.objects.all() if user.is_admin() else RiskEvent.objects.filter(video__user=user)

        events = list(risks_qs.values('id', 'event_id', 'start_time', 'end_time', 'risk_level', 'risk_score', 'vehicles')[:100])
        return Response({"timeline_events": events})

class DashboardHeatmapView(views.APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        user = request.user
        interactions_qs = Interaction.objects.all() if user.is_admin() else Interaction.objects.filter(video__user=user)

        # Image/pixel coordinate heatmap points (No fake GPS)
        points = list(interactions_qs.filter(ttc__isnull=False).values('frame_id', 'distance', 'closing_speed', 'ttc')[:500])
        return Response({
            "coordinate_system": "IMAGE_PIXEL",
            "heatmap_points": points
        })
