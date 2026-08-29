from rest_framework import permissions
from accounts.models import UserRole

class IsTrafficOperator(permissions.BasePermission):
    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and 
                    request.user.role in [UserRole.TRAFFIC_OPERATOR, UserRole.ADMINISTRATOR])

class IsAdministrator(permissions.BasePermission):
    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and 
                    request.user.role == UserRole.ADMINISTRATOR)

class IsGovernmentAuthority(permissions.BasePermission):
    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and 
                    request.user.role in [UserRole.GOVERNMENT_AUTHORITY, UserRole.ADMINISTRATOR])
