from django.urls import path, include
from rest_framework.routers import DefaultRouter
from processing.views import ProcessingJobViewSet

router = DefaultRouter()
router.register(r'', ProcessingJobViewSet, basename='job')

urlpatterns = [
    path('', include(router.urls)),
]
