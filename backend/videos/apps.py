from pathlib import Path
from django.apps import AppConfig

class VideosConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'videos'
    path = str(Path(__file__).resolve().parent)
