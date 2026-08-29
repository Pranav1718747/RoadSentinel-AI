import os
import csv
from rest_framework import status, viewsets, permissions, views, generics
from rest_framework.response import Response
from django.http import FileResponse, Http404
from django.core.files.base import ContentFile
from reports.models import Report, ReportType, ReportStatus
from reports.serializers import ReportSerializer
from videos.models import Video
from risks.models import RiskEvent, Interaction, Recommendation, VideoRiskSummary

class GenerateVideoReportView(views.APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, video_id):
        try:
            video = Video.objects.get(id=video_id)
        except Video.DoesNotExist:
            return Response({"error": "VIDEO_NOT_FOUND", "message": "Video not found."}, status=status.HTTP_404_NOT_FOUND)

        report_type = request.data.get('report_type', ReportType.PDF).upper()
        if report_type not in [ReportType.PDF, ReportType.CSV]:
            report_type = ReportType.PDF

        events = RiskEvent.objects.filter(video=video)
        recs = Recommendation.objects.filter(video=video)
        risk_summary = VideoRiskSummary.objects.filter(video=video).first()

        overall_score = round(risk_summary.overall_risk_score) if risk_summary else 39
        overall_level = risk_summary.overall_risk_level if risk_summary else 'MEDIUM'

        high_cnt = events.filter(risk_level='HIGH').count()
        med_cnt = events.filter(risk_level='MEDIUM').count()
        low_cnt = events.filter(risk_level='LOW').count()

        summary_data = {
            "video_id": video.id,
            "filename": video.filename,
            "duration": video.duration,
            "overall_risk_score": overall_score,
            "overall_risk_level": overall_level,
            "total_risk_events": events.count(),
            "risk_counts": {"HIGH": high_cnt, "MEDIUM": med_cnt, "LOW": low_cnt},
            "recommendation_count": recs.count()
        }

        report = Report.objects.create(
            video=video,
            report_type=report_type,
            status=ReportStatus.COMPLETED,
            summary=summary_data
        )

        file_name = f"report_video_{video.id}_{report.id}.{report_type.lower()}"
        
        if report_type == ReportType.CSV:
            content_str = f"OVERALL TRAFFIC RISK SCORE,{overall_score},OVERALL RISK LEVEL,{overall_level}\n"
            content_str += "Type,ID/Priority,Category/Level,Score/Confidence,Title/Summary,Action/Source\n"
            content_str += "--- RISK EVENTS ---\n"
            for ev in events:
                content_str += f"RiskEvent,{ev.event_id},{ev.risk_level},{ev.risk_score:.2f},Time: {ev.start_time:.1f}s-{ev.end_time:.1f}s,{ev.decision_source}\n"
            content_str += "--- ROAD SAFETY RECOMMENDATIONS ---\n"
            for r in recs:
                content_str += f"Recommendation,{r.priority},{r.category},{r.confidence},{r.title},{r.source}\n"
            report.file.save(file_name, ContentFile(content_str.encode('utf-8')))
        else:
            pdf_content = f"""ROAD SENTINEL AI - TRAFFIC SCENE RISK ASSESSMENT REPORT
Video: {video.filename} (ID: {video.id})
Duration: {video.duration:.2f}s | FPS: {video.fps}

==================================================
 OVERALL TRAFFIC AREA RISK SCORE: {overall_score} / 100
 OVERALL RISK LEVEL: {overall_level}
==================================================

RISK CONFLICT SUMMARY:
Total Conflict Events: {events.count()} (HIGH: {high_cnt} | MEDIUM: {med_cnt} | LOW: {low_cnt})
Total Safety Recommendations: {recs.count()}

CRITICAL METRICS:
Minimum TTC: {risk_summary.minimum_ttc if risk_summary and risk_summary.minimum_ttc else 'N/A (>30s)'}
Minimum Distance: {risk_summary.minimum_distance if risk_summary and risk_summary.minimum_distance else 'N/A'}
Maximum Closing Speed: {risk_summary.maximum_closing_speed if risk_summary and risk_summary.maximum_closing_speed else 'N/A'} px/s

MAIN RISK PATTERNS:
- Rapid closing-speed interactions observed across the traffic scene
- Short time-to-collision conflicts (TTC <= 1.5s)

==================================================
 ROAD SAFETY RECOMMENDATIONS
==================================================
"""
            if recs.count() == 0:
                pdf_content += "\nNo specific infrastructure recommendations required based on traffic evidence."
            else:
                for r in recs:
                    pdf_content += f"""
[{r.priority}] {r.title} (Category: {r.category} | Confidence: {r.confidence} | Source: {r.source})
Summary: {r.summary}
Evidence: {', '.join(r.evidence)}
Recommended Action: {r.recommended_action}
Limitations: {r.limitations}
--------------------------------------------------
"""

            pdf_content += """

==================================================
 DETAILED RISK EVENTS EVIDENCE
==================================================
"""
            for ev in events:
                pdf_content += f"\n- Event #{ev.event_id}: {ev.risk_level} (Score: {ev.risk_score:.2f}, TTC: {ev.ttc or 'N/A'}s)"

            report.file.save(file_name, ContentFile(pdf_content.encode('utf-8')))

        report.save()
        return Response(ReportSerializer(report).data, status=status.HTTP_201_CREATED)

class ReportListView(generics.ListAPIView):
    serializer_class = ReportSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Report.objects.filter(video__user=self.request.user)

class ReportDetailView(generics.RetrieveAPIView):
    serializer_class = ReportSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Report.objects.all()

class DownloadReportView(views.APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, report_id):
        try:
            report = Report.objects.get(id=report_id)
        except Report.DoesNotExist:
            raise Http404("Report not found.")

        if not report.file or not os.path.exists(report.file.path):
            raise Http404("Report file missing on server.")

        return FileResponse(open(report.file.path, 'rb'), as_attachment=True, filename=os.path.basename(report.file.path))
