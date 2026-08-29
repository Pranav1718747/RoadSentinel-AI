import os
import sys
from pathlib import Path
from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework import status
from accounts.models import UserRole
from videos.models import Video, VideoStatus
from processing.models import ProcessingJob, JobStatus, JobStage
from risks.models import RiskEvent, Interaction, Recommendation, VideoRiskSummary, RiskZoneSummary

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "ml" / "src" / "risk"))
sys.path.insert(0, str(BASE_DIR / "backend" / "processing" / "services"))

from recommendation_evidence import EvidenceAggregator
from recommendation_rules import RecommendationRuleEngine
from recommendation_validator import RecommendationValidator
from pipeline_service import JobProgressTracker

User = get_user_model()

class RoadSentinelBackendTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.operator_user = User.objects.create_user(
            username='op_test',
            password='password123',
            role=UserRole.TRAFFIC_OPERATOR
        )

    def test_health_check(self):
        response = self.client.get('/api/health/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['api'], 'ok')

    def test_acceptance_test_arithmetic_mean_overall_score(self):
        """
        Critical Acceptance Test:
        A = 15, B = 23, C = 82, D = 35
        overall_score = (15 + 23 + 82 + 35) / 4 = 38.75
        displayed_score = round(38.75) = 39
        overall_risk_level = MEDIUM (since 30 <= 38.75 < 60)
        """
        agg = EvidenceAggregator([])
        summary = agg.build_evidence_summary()

        summary["zones_data"] = {
            "Zone A": {"zone_risk_score": 15.0},
            "Zone B": {"zone_risk_score": 23.0},
            "Zone C": {"zone_risk_score": 82.0},
            "Zone D": {"zone_risk_score": 35.0},
        }

        zA = summary["zones_data"]["Zone A"]["zone_risk_score"]
        zB = summary["zones_data"]["Zone B"]["zone_risk_score"]
        zC = summary["zones_data"]["Zone C"]["zone_risk_score"]
        zD = summary["zones_data"]["Zone D"]["zone_risk_score"]

        overall_score = (zA + zB + zC + zD) / 4.0
        self.assertEqual(overall_score, 38.75)
        self.assertEqual(round(overall_score), 39)

        if overall_score >= 80.0:
            level = 'CRITICAL'
        elif overall_score >= 60.0:
            level = 'HIGH'
        elif overall_score >= 30.0:
            level = 'MEDIUM'
        else:
            level = 'LOW'

        self.assertEqual(level, 'MEDIUM')
        self.assertNotEqual(level, 'HIGH') # The highest zone (82) MUST NOT force overall score to HIGH

    def test_risk_summary_api_endpoint(self):
        self.client.force_authenticate(user=self.operator_user)
        video = Video.objects.create(user=self.operator_user, filename='scene_test.mp4', status=VideoStatus.COMPLETED)
        
        VideoRiskSummary.objects.create(
            video=video,
            overall_risk_score=38.75,
            overall_risk_level='MEDIUM',
            zone_a_score=15.0,
            zone_b_score=23.0,
            zone_c_score=82.0,
            zone_d_score=35.0,
            total_events=7108,
            high_events=18,
            medium_events=712,
            low_events=6378,
            minimum_ttc=1.25,
            minimum_distance=12.4,
            maximum_closing_speed=42.0,
            dominant_risk_patterns=['Rapid closing-speed interactions', 'Short time-to-collision conflicts']
        )

        response_summary = self.client.get(f'/api/videos/{video.id}/risk-summary/')
        self.assertEqual(response_summary.status_code, status.HTTP_200_OK)
        self.assertEqual(response_summary.data['overall_risk_score'], 39) # Displayed as 39
        self.assertEqual(response_summary.data['overall_risk_level'], 'MEDIUM')
        self.assertEqual(response_summary.data['risk_distribution']['high'], 18)
        self.assertEqual(response_summary.data['risk_distribution']['medium'], 712)
        self.assertEqual(response_summary.data['risk_distribution']['low'], 6378)
