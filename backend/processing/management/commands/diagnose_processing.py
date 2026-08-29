import sys
from pathlib import Path
from django.core.management.base import BaseCommand
from django.db import connection
from django.conf import settings
import redis

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

class Command(BaseCommand):
    help = 'Runs comprehensive system diagnostics across Django, DB, Redis, Celery, YOLO26, XGBoost, and storage.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("=== ROAD SENTINEL SYSTEM DIAGNOSTICS ==="))

        results = {}

        # 1. Django
        try:
            results["Django Framework"] = ("PASS", f"Django settings '{settings.SETTINGS_MODULE}' loaded")
        except Exception as e:
            results["Django Framework"] = ("FAIL", str(e))

        # 2. Database
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            results["Database Connectivity"] = ("PASS", f"Connected to {settings.DATABASES['default']['ENGINE']}")
        except Exception as e:
            results["Database Connectivity"] = ("FAIL", str(e))

        # 3. Redis
        try:
            r = redis.Redis.from_url(getattr(settings, "CELERY_BROKER_URL", "redis://localhost:6379/0"))
            if r.ping():
                results["Redis Broker"] = ("PASS", "Redis PONG received")
            else:
                results["Redis Broker"] = ("FAIL", "Redis PING returned False")
        except Exception as e:
            results["Redis Broker"] = ("FAIL", str(e))

        # 4. Celery Task Registration
        try:
            from roadsentinel_backend.celery import app
            registered = [t for t in app.tasks.keys() if not t.startswith("celery.")]
            if "processing.tasks.process_video_pipeline" in app.tasks:
                results["Celery Task Discovery"] = ("PASS", f"Registered tasks: {registered}")
            else:
                results["Celery Task Discovery"] = ("FAIL", f"process_video_pipeline missing in {registered}")
        except Exception as e:
            results["Celery Task Discovery"] = ("FAIL", str(e))

        # 5. YOLO26 Checkpoint
        try:
            yolo_path = BASE_DIR / "yolo26s.pt"
            if yolo_path.exists():
                results["YOLO26 Checkpoint"] = ("PASS", f"Found {yolo_path} ({yolo_path.stat().st_size / (1024*1024):.1f} MB)")
            else:
                results["YOLO26 Checkpoint"] = ("FAIL", f"Missing at {yolo_path}")
        except Exception as e:
            results["YOLO26 Checkpoint"] = ("FAIL", str(e))

        # 6. XGBoost Model
        try:
            xgb_path = BASE_DIR / "ml" / "models" / "risk_model.pkl"
            if xgb_path.exists():
                results["XGBoost Model"] = ("PASS", f"Found {xgb_path}")
            else:
                results["XGBoost Model"] = ("FAIL", f"Missing at {xgb_path}")
        except Exception as e:
            results["XGBoost Model"] = ("FAIL", str(e))

        # 7. Video Storage
        try:
            media_dir = BASE_DIR / "storage" / "media" / "uploads"
            if media_dir.exists():
                results["Video Storage Dir"] = ("PASS", f"Found {media_dir}")
            else:
                results["Video Storage Dir"] = ("FAIL", f"Missing at {media_dir}")
        except Exception as e:
            results["Video Storage Dir"] = ("FAIL", str(e))

        # Print final report
        all_pass = True
        for component, (status, detail) in results.items():
            if status == "PASS":
                self.stdout.write(self.style.SUCCESS(f"[PASS] {component}: {detail}"))
            else:
                all_pass = False
                self.stdout.write(self.style.ERROR(f"[FAIL] {component}: {detail}"))

        self.stdout.write("=" * 40)
        if all_pass:
            self.stdout.write(self.style.SUCCESS("OVERALL SYSTEM STATUS: PASS"))
        else:
            self.stdout.write(self.style.ERROR("OVERALL SYSTEM STATUS: FAIL"))
