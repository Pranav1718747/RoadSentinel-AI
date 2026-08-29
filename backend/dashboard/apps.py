from pathlib import Path
from django.apps import AppConfig

class DashboardConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'dashboard'
    path = str(Path(__file__).resolve().parent)
