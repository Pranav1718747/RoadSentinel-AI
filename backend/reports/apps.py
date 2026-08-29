from pathlib import Path
from django.apps import AppConfig

class ReportsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'reports'
    path = str(Path(__file__).resolve().parent)
