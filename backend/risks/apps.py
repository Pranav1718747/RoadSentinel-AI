from pathlib import Path
from django.apps import AppConfig

class RisksConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'risks'
    path = str(Path(__file__).resolve().parent)
