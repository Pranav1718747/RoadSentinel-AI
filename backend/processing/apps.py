from pathlib import Path
from django.apps import AppConfig

class ProcessingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'processing'
    path = str(Path(__file__).resolve().parent)
