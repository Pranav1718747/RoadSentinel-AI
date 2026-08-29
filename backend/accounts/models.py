from django.db import models
from django.contrib.auth.models import AbstractUser

class UserRole(models.TextChoices):
    TRAFFIC_OPERATOR = 'TRAFFIC_OPERATOR', 'Traffic Operator'
    ADMINISTRATOR = 'ADMINISTRATOR', 'Administrator'
    GOVERNMENT_AUTHORITY = 'GOVERNMENT_AUTHORITY', 'Government Authority'

class User(AbstractUser):
    role = models.CharField(
        max_length=30,
        choices=UserRole.choices,
        default=UserRole.TRAFFIC_OPERATOR
    )
    phone_number = models.CharField(max_length=20, blank=True, null=True)
    organization = models.CharField(max_length=100, blank=True, null=True)

    def is_operator(self):
        return self.role in [UserRole.TRAFFIC_OPERATOR, UserRole.ADMINISTRATOR]

    def is_admin(self):
        return self.role == UserRole.ADMINISTRATOR

    def is_authority(self):
        return self.role == UserRole.GOVERNMENT_AUTHORITY

    def __str__(self):
        return f"{self.username} ({self.role})"
