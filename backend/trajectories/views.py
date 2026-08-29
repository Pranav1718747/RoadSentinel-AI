from rest_framework import generics, permissions
from trajectories.models import Trajectory
from trajectories.serializers import TrajectorySerializer

class VideoTrajectoriesView(generics.ListAPIView):
    serializer_class = TrajectorySerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        video_id = self.kwargs.get('video_id')
        queryset = Trajectory.objects.filter(video_id=video_id)
        
        track_id = self.request.query_params.get('track_id')
        if track_id is not None:
            queryset = queryset.filter(track__track_id=track_id)

        return queryset

class TrackTrajectoryView(generics.ListAPIView):
    serializer_class = TrajectorySerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        track_id = self.kwargs.get('track_id')
        return Trajectory.objects.filter(track__track_id=track_id).order_by('frame_id')
