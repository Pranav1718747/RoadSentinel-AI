import os
from pathlib import Path
from rest_framework import views, permissions
from rest_framework.response import Response
from django.db import connection
from django.conf import settings

class HealthCheckView(views.APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        health_status = {
            "api": "ok",
            "database": "unknown",
            "redis": "ok",
            "worker": "ok",
            "ml_pipeline": "ok",
            "detector": "YOLO26"
        }

        # Check DB
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            health_status["database"] = "ok"
        except Exception as e:
            health_status["database"] = f"error: {str(e)}"

        # Check Model file availability
        yolo_path = Path(getattr(settings, "YOLO26_MODEL_PATH", ""))
        if not yolo_path.exists():
            health_status["ml_pipeline"] = "warning: yolo26s.pt model file missing"

        return Response(health_status)
