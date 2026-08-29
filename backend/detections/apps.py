from pathlib import Path
from django.apps import AppConfig

class DetectionsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detections'
    path = str(Path(__file__).resolve().parent)
