from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from accounts.models import UserRole

User = get_user_model()

class Command(BaseCommand):
    help = 'Seeds initial demo users for all three roles: TRAFFIC_OPERATOR, ADMINISTRATOR, GOVERNMENT_AUTHORITY'

    def handle(self, *args, **options):
        users_data = [
            {
                "username": "operator",
                "email": "operator@roadsentinel.ai",
                "password": "operator123",
                "role": UserRole.TRAFFIC_OPERATOR,
                "first_name": "Traffic",
                "last_name": "Operator",
                "organization": "City Traffic Control"
            },
            {
                "username": "admin",
                "email": "admin@roadsentinel.ai",
                "password": "admin123",
                "role": UserRole.ADMINISTRATOR,
                "first_name": "System",
                "last_name": "Admin",
                "is_staff": True,
                "is_superuser": True,
                "organization": "RoadSentinel Admin HQ"
            },
            {
                "username": "authority",
                "email": "authority@roadsentinel.ai",
                "password": "authority123",
                "role": UserRole.GOVERNMENT_AUTHORITY,
                "first_name": "Department",
                "last_name": "Authority",
                "organization": "Department of Transportation"
            }
        ]

        for u_info in users_data:
            user, created = User.objects.get_or_create(
                username=u_info["username"],
                defaults={
                    "email": u_info["email"],
                    "role": u_info["role"],
                    "first_name": u_info["first_name"],
                    "last_name": u_info["last_name"],
                    "organization": u_info["organization"],
                    "is_staff": u_info.get("is_staff", False),
                    "is_superuser": u_info.get("is_superuser", False)
                }
            )
            user.set_password(u_info["password"])
            user.save()
            status = "Created" if created else "Updated"
            self.stdout.write(self.style.SUCCESS(f"{status} user: {user.username} ({user.role})"))
