from pathlib import Path
from django.apps import AppConfig

class TrackingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'tracking'
    path = str(Path(__file__).resolve().parent)
