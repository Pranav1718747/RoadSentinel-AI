from django.urls import path, include
from rest_framework.routers import DefaultRouter
from videos.views import VideoViewSet

router = DefaultRouter()
router.register(r'', VideoViewSet, basename='video')

urlpatterns = [
    path('', include(router.urls)),
]
