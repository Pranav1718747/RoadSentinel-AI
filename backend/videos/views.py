import os
import cv2
from rest_framework import status, viewsets, permissions
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
from videos.models import Video, VideoStatus
from videos.serializers import VideoSerializer
from processing.models import ProcessingJob, JobStatus, JobStage
from processing.tasks import process_video_pipeline

class VideoViewSet(viewsets.ModelViewSet):
    serializer_class = VideoSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def get_queryset(self):
        user = self.request.user
        if user.is_admin():
            return Video.objects.all()
        return Video.objects.filter(user=user)

    def create(self, request, *args, **kwargs):
        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({
                "error": "NO_FILE_PROVIDED",
                "message": "Please upload a valid video file.",
                "details": {}
            }, status=status.HTTP_400_BAD_REQUEST)

        # Validate video file extension
        ext = os.path.splitext(file_obj.name)[1].lower()
        if ext not in ['.mp4', '.avi', '.mov', '.mkv']:
            return Response({
                "error": "INVALID_FILE_TYPE",
                "message": "Only MP4, AVI, MOV, and MKV video formats are supported.",
                "details": {"allowed_extensions": ['.mp4', '.avi', '.mov', '.mkv']}
            }, status=status.HTTP_400_BAD_REQUEST)

        video = Video.objects.create(
            user=request.user,
            file=file_obj,
            filename=file_obj.name,
            status=VideoStatus.UPLOADED
        )

        # Create Processing Job
        job = ProcessingJob.objects.create(
            video=video,
            status=JobStatus.QUEUED,
            current_stage=JobStage.UPLOADING,
            progress=0.0
        )

        # Queue async processing via Celery (or run task)
        if getattr(settings, 'CELERY_TASK_ALWAYS_EAGER', False):
            process_video_pipeline(job.id)
        else:
            process_video_pipeline.delay(job.id)

        return Response({
            "video_id": video.id,
            "job_id": job.id,
            "filename": video.filename,
            "status": job.status,
            "message": "Video uploaded successfully and processing queued."
        }, status=status.HTTP_201_CREATED)
