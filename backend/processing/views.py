from django.utils import timezone
from datetime import timedelta
from rest_framework import status, viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from processing.models import ProcessingJob, JobStatus
from processing.serializers import ProcessingJobSerializer

class ProcessingJobViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = ProcessingJobSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        queryset = ProcessingJob.objects.all() if user.is_admin() else ProcessingJob.objects.filter(video__user=user)

        # Stale Job Detection: Check for stuck PROCESSING jobs with no progress update in >60s
        now = timezone.now()
        stale_cutoff = now - timedelta(seconds=60)

        stale_jobs = queryset.filter(
            status=JobStatus.PROCESSING,
            last_progress_update__lt=stale_cutoff
        )
        for s_job in stale_jobs:
            s_job.status = JobStatus.FAILED
            s_job.error_message = "Worker process stopped responding (Heartbeat Timeout > 60s)."
            s_job.save()

        return queryset

    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        job = self.get_object()
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return Response({
                "error": "INVALID_STATE",
                "message": f"Cannot cancel job in '{job.status}' state.",
                "details": {}
            }, status=status.HTTP_400_BAD_REQUEST)

        job.status = JobStatus.CANCELLED
        job.save()
        if hasattr(job, 'video') and job.video:
            job.video.status = 'CANCELLED'
            job.video.save()

        return Response({
            "job_id": job.id,
            "status": job.status,
            "message": "Processing job has been cancelled."
        })
