from rest_framework import serializers
from reports.models import Report

class ReportSerializer(serializers.ModelSerializer):
    video_filename = serializers.CharField(source='video.filename', read_only=True)

    class Meta:
        model = Report
        fields = ['id', 'video', 'video_filename', 'report_type', 'file', 'status', 'summary', 'created_at']
        read_only_fields = ['id', 'file', 'status', 'summary', 'created_at']
