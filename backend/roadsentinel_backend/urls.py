from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/auth/', include('accounts.urls')),
    path('api/videos/', include('videos.urls')),
    path('api/jobs/', include('processing.urls')),
    path('api/', include('detections.urls')),
    path('api/', include('tracking.urls')),
    path('api/', include('trajectories.urls')),
    path('api/', include('risks.urls')),
    path('api/', include('reports.urls')),
    path('api/dashboard/', include('dashboard.urls')),
    path('api/health/', include('core.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
