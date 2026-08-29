from rest_framework import generics, permissions, status
from rest_framework.response import Response
from detections.models import Detection
from detections.serializers import DetectionSerializer
from videos.models import Video

class VideoDetectionsView(generics.ListAPIView):
    serializer_class = DetectionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        video_id = self.kwargs.get('video_id')
        queryset = Detection.objects.filter(video_id=video_id)

        # Filters
        frame = self.request.query_params.get('frame')
        class_name = self.request.query_params.get('class')
        min_conf = self.request.query_params.get('confidence')

        if frame is not None:
            queryset = queryset.filter(frame_id=frame)
        if class_name is not None:
            queryset = queryset.filter(class_name__iexact=class_name)
        if min_conf is not None:
            try:
                queryset = queryset.filter(confidence__gte=float(min_conf))
            except ValueError:
                pass

        return queryset
