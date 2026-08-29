from rest_framework import generics, permissions, status
from rest_framework.response import Response
from tracking.models import Track
from tracking.serializers import TrackSerializer

class VideoTracksView(generics.ListAPIView):
    serializer_class = TrackSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        video_id = self.kwargs.get('video_id')
        queryset = Track.objects.filter(video_id=video_id)

        movement = self.request.query_params.get('movement')
        class_name = self.request.query_params.get('class')

        if movement is not None:
            queryset = queryset.filter(movement_state__iexact=movement)
        if class_name is not None:
            queryset = queryset.filter(class_name__iexact=class_name)

        return queryset

class TrackDetailView(generics.RetrieveAPIView):
    serializer_class = TrackSerializer
    permission_classes = [permissions.IsAuthenticated]
    lookup_field = 'track_id'

    def get_queryset(self):
        return Track.objects.all()
