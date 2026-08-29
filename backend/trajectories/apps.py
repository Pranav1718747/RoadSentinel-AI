from pathlib import Path
from django.apps import AppConfig

class TrajectoriesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'trajectories'
    path = str(Path(__file__).resolve().parent)
