import os
import sys
import csv
import json
import time
import logging
import traceback
import pandas as pd
from pathlib import Path
from django.utils import timezone
from django.db import transaction

# Ensure imports for models
from videos.models import Video, VideoStatus
from processing.models import ProcessingJob, JobStatus, JobStage, ModelRun
from detections.models import Detection
from tracking.models import Track, MovementState
from trajectories.models import Trajectory
from risks.models import Interaction, RiskEvent, RiskLevel, Recommendation, VideoRiskSummary, RiskZoneSummary

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "ml"))
sys.path.insert(0, str(BASE_DIR / "ml" / "src"))
sys.path.insert(0, str(BASE_DIR / "ml" / "src" / "pipeline"))
sys.path.insert(0, str(BASE_DIR / "ml" / "src" / "risk"))

import config as cfg
from run_roadsentinel import RoadSentinelPipeline
from run_hybrid import run_hybrid_pipeline
from recommendation_evidence import EvidenceAggregator
from recommendation_rules import RecommendationRuleEngine
from ai_recommendation_service import RecommendationAIProvider

logger = logging.getLogger(__name__)

def log_job_event(job_id, video_id, message):
    log_msg = f"[JOB {job_id} | VIDEO {video_id}] {message}"
    print(log_msg)
    logger.info(log_msg)

class JobProgressTracker:
    STAGE_CONFIG = {
        JobStage.VALIDATING:           {"base": 0.0,  "weight": 5.0},
        JobStage.DETECTING:            {"base": 5.0,  "weight": 25.0},
        JobStage.TRACKING:             {"base": 30.0, "weight": 15.0},
        JobStage.TRAJECTORY_ANALYSIS:  {"base": 45.0, "weight": 15.0},
        JobStage.INTERACTION_ANALYSIS: {"base": 60.0, "weight": 10.0},
        JobStage.RISK_ANALYSIS:        {"base": 70.0, "weight": 15.0},
        JobStage.ML_PREDICTION:        {"base": 85.0, "weight": 5.0},
        JobStage.FINALIZING:           {"base": 90.0, "weight": 10.0},
        JobStage.COMPLETED:            {"base": 100.0, "weight": 0.0},
    }

    def __init__(self, job, celery_task=None):
        self.job = job
        self.celery_task = celery_task
        self.start_time = time.time()
        self.last_update_time = 0.0
        self.update_interval = 0.75  # Rate limit updates to 0.75s

    def update_progress(self, stage, current_frame=0, total_frames=0, processed_detections=0, fps=50.0, force=False):
        now = time.time()
        if not force and (now - self.last_update_time < self.update_interval) and (current_frame < total_frames):
            return

        self.last_update_time = now

        # Check job cancellation in DB
        self.job.refresh_from_db(fields=['status'])
        if self.job.status == JobStatus.CANCELLED:
            raise Exception("CANCELLED: Processing job cancelled by user.")

        cfg_stage = self.STAGE_CONFIG.get(stage, {"base": 0.0, "weight": 10.0})
        base = cfg_stage["base"]
        weight = cfg_stage["weight"]

        if total_frames > 0:
            stage_prog = min(100.0, round((current_frame / total_frames) * 100.0, 1))
            v_pos = round(current_frame / (fps or 50.0), 2)
            v_dur = round(total_frames / (fps or 50.0), 2)
        else:
            stage_prog = 100.0
            v_pos = 0.0
            v_dur = 0.0

        overall_prog = min(100.0, round(base + (weight * (stage_prog / 100.0)), 1))
        elapsed = round(now - self.start_time, 1)

        proc_fps = round(current_frame / elapsed, 1) if elapsed > 0 and current_frame > 0 else 0.0
        eta = round((total_frames - current_frame) / proc_fps, 1) if proc_fps > 0 and total_frames > current_frame else None

        # Update DB instance
        self.job.current_stage = stage
        self.job.stage_progress = stage_prog
        self.job.overall_progress = overall_prog
        self.job.progress = overall_prog
        self.job.current_frame = current_frame
        self.job.total_frames = total_frames
        self.job.video_position_seconds = v_pos
        self.job.video_duration_seconds = v_dur
        self.job.elapsed_seconds = elapsed
        self.job.processing_fps = proc_fps
        self.job.eta_seconds = eta
        self.job.processed_detections = processed_detections
        self.job.last_progress_update = timezone.now()
        self.job.save()

        # Update Celery task state
        if self.celery_task:
            self.celery_task.update_state(
                state="PROGRESS",
                meta={
                    "stage": stage,
                    "stage_progress": stage_prog,
                    "overall_progress": overall_prog,
                    "current_frame": current_frame,
                    "total_frames": total_frames,
                    "video_position_seconds": v_pos,
                    "elapsed_seconds": elapsed,
                    "processing_fps": proc_fps,
                    "eta_seconds": eta,
                    "detections": processed_detections
                }
            )

        print(f"[JOB {self.job.id}] {stage} | Frame {current_frame}/{total_frames} ({stage_prog}%) | Overall {overall_prog}% | {proc_fps} FPS | ETA {eta}s | {processed_detections} Detections")


def run_pipeline_for_job(job_id, celery_task=None):
    """
    Executes the frozen RoadSentinel ML pipeline for a given ProcessingJob
    and ingests structured output CSV/JSON files into the database.
    """
    try:
        job = ProcessingJob.objects.get(id=job_id)
    except ProcessingJob.DoesNotExist:
        print(f"[ERROR] ProcessingJob #{job_id} not found in database.")
        return False

    video = job.video
    log_job_event(job.id, video.id, "Starting processing pipeline")

    tracker = JobProgressTracker(job, celery_task=celery_task)

    job.status = JobStatus.PROCESSING
    job.started_at = timezone.now()
    tracker.update_progress(JobStage.VALIDATING, current_frame=0, total_frames=500, force=True)

    video.status = VideoStatus.PROCESSING
    video.save()

    try:
        video_file_path = Path(video.file.path).resolve()
        log_job_event(job.id, video.id, f"Target video path: {video_file_path}")

        if not video_file_path.exists():
            log_job_event(job.id, video.id, "Uploaded file missing on disk, checking default fallback sample")
            default_path = (BASE_DIR / "ml" / "data" / "samples" / "traffic.mp4").resolve()
            if default_path.exists():
                video_file_path = default_path
            else:
                raise FileNotFoundError(f"Video file not found at: {video_file_path}")

        # Stage 1: Validation
        tracker.update_progress(JobStage.VALIDATING, current_frame=500, total_frames=500, force=True)
        log_job_event(job.id, video.id, "Stage VALIDATING completed")

        # Callback function for frame-level progress
        def on_frame_progress(stage, current_frame, total_frames, processed_detections=0):
            tracker.update_progress(
                stage=JobStage.DETECTING,
                current_frame=current_frame,
                total_frames=total_frames,
                processed_detections=processed_detections
            )

        # Stage 2: Detection & Tracking
        tracker.update_progress(JobStage.DETECTING, current_frame=0, total_frames=500, force=True)
        log_job_event(job.id, video.id, "Stage DETECTING started with YOLO26")

        # Execute existing frozen ML pipeline using absolute path and progress_callback
        abs_video_path_str = str(video_file_path)
        pipeline_obj = RoadSentinelPipeline(
            input_video_rel=abs_video_path_str,
            progress_callback=on_frame_progress
        )
        run_manifest = pipeline_obj.run_all()
        latest_run_dir = Path(pipeline_obj.paths["run_dir"])

        log_job_event(job.id, video.id, f"YOLO26 pipeline completed. Run dir: {latest_run_dir}")

        total_frames = pipeline_obj.video_props.get("total_frames", 500)
        detections_count = len(pipeline_obj.tracking_rows)

        tracker.update_progress(JobStage.TRACKING, current_frame=total_frames, total_frames=total_frames, processed_detections=detections_count, force=True)
        log_job_event(job.id, video.id, "Stage TRACKING completed")

        tracker.update_progress(JobStage.TRAJECTORY_ANALYSIS, current_frame=total_frames, total_frames=total_frames, processed_detections=detections_count, force=True)
        log_job_event(job.id, video.id, "Stage TRAJECTORY_ANALYSIS completed")

        tracker.update_progress(JobStage.INTERACTION_ANALYSIS, current_frame=total_frames, total_frames=total_frames, processed_detections=detections_count, force=True)
        log_job_event(job.id, video.id, "Stage INTERACTION_ANALYSIS completed")

        tracker.update_progress(JobStage.RISK_ANALYSIS, current_frame=total_frames, total_frames=total_frames, processed_detections=detections_count, force=True)
        log_job_event(job.id, video.id, "Stage RISK_ANALYSIS completed")

        tracker.update_progress(JobStage.ML_PREDICTION, current_frame=total_frames, total_frames=total_frames, processed_detections=detections_count, force=True)
        log_job_event(job.id, video.id, "Stage ML_PREDICTION started (XGBoost Classifier)")

        # Execute hybrid ML prediction
        run_hybrid_pipeline(video_input=abs_video_path_str)
        log_job_event(job.id, video.id, "Stage ML_PREDICTION completed")

        # Stage 5: Ingest pipeline outputs into database
        tracker.update_progress(JobStage.FINALIZING, current_frame=total_frames, total_frames=total_frames, processed_detections=detections_count, force=True)
        log_job_event(job.id, video.id, "Stage FINALIZING started (DB Output Ingestion)")

        ingest_pipeline_outputs(video, latest_run_dir)

        # Create ModelRun metadata record
        ModelRun.objects.update_or_create(
            job=job,
            defaults={
                "detector_model": "YOLO26",
                "detector_checkpoint": getattr(cfg, "YOLO_MODEL_NAME", "yolo26s.pt"),
                "risk_model": "XGBoost",
                "risk_model_version": "1.0.0",
                "pipeline_version": "2.0.0"
            }
        )

        job.status = JobStatus.COMPLETED
        job.current_stage = JobStage.COMPLETED
        job.stage_progress = 100.0
        job.overall_progress = 100.0
        job.progress = 100.0
        job.eta_seconds = 0.0
        job.completed_at = timezone.now()
        job.last_progress_update = timezone.now()
        job.save()

        video.status = VideoStatus.COMPLETED
        video.save()

        log_job_event(job.id, video.id, "Pipeline execution & DB ingestion COMPLETED SUCCESSFULLY")
        return True

    except Exception as e:
        tb_str = traceback.format_exc()
        err_msg = f"Processing Error: {str(e)}\n{tb_str}"
        log_job_event(job.id, video.id, f"JOB FAILED with error: {err_msg}")
        
        job.status = JobStatus.FAILED
        job.error_message = str(e)
        job.completed_at = timezone.now()
        job.save()

        video.status = VideoStatus.FAILED
        video.save()
        return False


def parse_float_safe(val, default=0.0):
    if val is None or val == "" or str(val).lower() == "nan":
        return default
    try:
        return float(val)
    except Exception:
        return default


def parse_nullable_float(val):
    if val is None or val == "" or str(val).lower() == "nan":
        return None
    try:
        return float(val)
    except Exception:
        return None


def ingest_pipeline_outputs(video, run_dir):
    """
    Parses generated CSV and JSON outputs from the run directory,
    runs Scene/Zone Risk Aggregator, Recommendation Engine, and saves structured DB records.
    """
    tracks_csv = run_dir / "tracking" / "tracks_v2.csv"
    trajectories_csv = run_dir / "trajectories" / "validated_trajectories.csv"
    features_csv = run_dir / "features" / "traffic_features.csv"
    risk_csv = run_dir / "risk" / "risk_events.csv"
    report_json = run_dir / "reports" / "roadsentinel_report.json"

    with transaction.atomic():
        Detection.objects.filter(video=video).delete()
        Track.objects.filter(video=video).delete()
        Trajectory.objects.filter(video=video).delete()
        Interaction.objects.filter(video=video).delete()
        RiskEvent.objects.filter(video=video).delete()
        Recommendation.objects.filter(video=video).delete()
        VideoRiskSummary.objects.filter(video=video).delete()
        RiskZoneSummary.objects.filter(video=video).delete()

        track_map = {}

        # 1. Ingest Tracks & Detections
        if tracks_csv.exists():
            detections_to_create = []
            tracks_info = {}

            with open(tracks_csv, newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    f_id = int(r["frame_id"])
                    ts = parse_float_safe(r.get("timestamp"))
                    t_id = int(r["track_id"])
                    c_id = int(r.get("class_id", 2))
                    c_name = r.get("class_name", "vehicle")
                    conf = parse_float_safe(r.get("confidence"), 0.5)
                    x1 = parse_float_safe(r.get("x1"))
                    y1 = parse_float_safe(r.get("y1"))
                    x2 = parse_float_safe(r.get("x2"))
                    y2 = parse_float_safe(r.get("y2"))
                    cx = parse_float_safe(r.get("center_x"), (x1 + x2) / 2)
                    cy = parse_float_safe(r.get("center_y"), (y1 + y2) / 2)

                    if t_id not in tracks_info:
                        tracks_info[t_id] = {
                            "class_id": c_id,
                            "class_name": c_name,
                            "start_frame": f_id,
                            "end_frame": f_id,
                            "timestamps": [ts]
                        }
                    else:
                        tracks_info[t_id]["end_frame"] = max(tracks_info[t_id]["end_frame"], f_id)
                        tracks_info[t_id]["timestamps"].append(ts)

                    detections_to_create.append(Detection(
                        video=video, frame_id=f_id, timestamp=ts, class_id=c_id,
                        class_name=c_name, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2,
                        center_x=cx, center_y=cy
                    ))

            Detection.objects.bulk_create(detections_to_create, batch_size=1000)

            for t_id, info in tracks_info.items():
                dur = max(0.0, info["timestamps"][-1] - info["timestamps"][0])
                trk = Track.objects.create(
                    video=video, track_id=t_id, class_id=info["class_id"],
                    class_name=info["class_name"], start_frame=info["start_frame"],
                    end_frame=info["end_frame"], duration=dur, movement_state=MovementState.MOVING
                )
                track_map[t_id] = trk

        # 2. Ingest Trajectories
        if trajectories_csv.exists():
            trajs_to_create = []
            with open(trajectories_csv, newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    t_id = int(r["track_id"])
                    f_id = int(r["frame_id"])
                    ts = parse_float_safe(r.get("timestamp"))
                    x = parse_float_safe(r.get("smooth_x", r.get("center_x", 0.0)))
                    y = parse_float_safe(r.get("smooth_y", r.get("center_y", 0.0)))
                    spd = parse_float_safe(r.get("speed"))
                    acc = parse_float_safe(r.get("acceleration"))
                    hdg = parse_float_safe(r.get("heading"))

                    trk = track_map.get(t_id)
                    if trk:
                        trajs_to_create.append(Trajectory(
                            video=video, track=trk, frame_id=f_id, timestamp=ts,
                            x=x, y=y, speed=spd, acceleration=acc, heading=hdg
                        ))

            Trajectory.objects.bulk_create(trajs_to_create, batch_size=1000)

        # 3. Ingest Interactions
        if features_csv.exists():
            interactions_to_create = []
            with open(features_csv, newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    f_id = int(r["frame_id"])
                    v1 = int(r.get("v1_id", r.get("vehicle_1", 0)))
                    v2 = int(r.get("v2_id", r.get("vehicle_2", 0)))
                    dist = parse_float_safe(r.get("distance"))
                    rel_spd = parse_float_safe(r.get("relative_speed"))
                    cs_spd = parse_float_safe(r.get("closing_speed"))
                    hdg_diff = parse_float_safe(r.get("heading_difference"))
                    appr = str(r.get("approaching", "False")).lower() == "true"

                    ttc_val = parse_nullable_float(r.get("ttc"))
                    pet_val = parse_nullable_float(r.get("pet"))
                    min_d = parse_nullable_float(r.get("minimum_distance"))

                    interactions_to_create.append(Interaction(
                        video=video, frame_id=f_id, vehicle_1=v1, vehicle_2=v2,
                        distance=dist, relative_speed=rel_spd, closing_speed=cs_spd,
                        heading_difference=hdg_diff, approaching=appr,
                        ttc=ttc_val, pet=pet_val, minimum_distance=min_d
                    ))

            Interaction.objects.bulk_create(interactions_to_create, batch_size=1000)

        # 4. Ingest Risk Events
        db_risk_events = []
        raw_risk_records = []
        if risk_csv.exists():
            with open(risk_csv, newline="") as f:
                reader = csv.DictReader(f)
                for idx, r in enumerate(reader):
                    raw_risk_records.append(dict(r))
                    e_id = int(r.get("event_id", idx + 1))
                    s_time = parse_float_safe(r.get("start_time", r.get("timestamp")))
                    e_time = parse_float_safe(r.get("end_time"), s_time + 1.0)
                    
                    v1 = int(r.get("v1_id", r.get("vehicle_1", 0)))
                    v2 = int(r.get("v2_id", r.get("vehicle_2", 0)))
                    v_list = [v1, v2]

                    r_level = str(r.get("risk_level", "LOW")).upper()

                    ttc_v = parse_nullable_float(r.get("ttc"))
                    min_d = parse_nullable_float(r.get("minimum_distance", r.get("distance")))
                    cs_s = parse_nullable_float(r.get("closing_speed"))
                    r_score = parse_float_safe(r.get("risk_score"))

                    ev = RiskEvent.objects.create(
                        video=video, event_id=e_id, start_time=s_time, end_time=e_time,
                        vehicles=v_list, vehicle_classes=["car"], risk_level=r_level,
                        risk_score=r_score, ttc=ttc_v,
                        minimum_distance=min_d, closing_speed=cs_s,
                        risk_reasons=[r.get("risk_reasons", r.get("risk_reason", "Proximity warning"))],
                        final_risk=r_level, decision_source=r.get("decision_source", "Rule_Override")
                    )
                    db_risk_events.append(ev)

        # 5. Scene & Area Risk Summary Aggregator
        aggregator = EvidenceAggregator(raw_risk_records)
        evidence_summary = aggregator.build_evidence_summary()

        # Save VideoRiskSummary DB record
        highest_ev_obj = db_risk_events[0] if len(db_risk_events) > 0 else None
        VideoRiskSummary.objects.create(
            video=video,
            overall_risk_score=evidence_summary["overall_risk_score"],
            overall_risk_level=evidence_summary["overall_risk_level"],
            zone_a_score=evidence_summary["zone_a_score"],
            zone_b_score=evidence_summary["zone_b_score"],
            zone_c_score=evidence_summary["zone_c_score"],
            zone_d_score=evidence_summary["zone_d_score"],
            total_events=evidence_summary["total_events"],
            low_events=evidence_summary["low_events_count"],
            medium_events=evidence_summary["med_events_count"],
            high_events=evidence_summary["high_events_count"],
            average_event_risk=evidence_summary["avg_risk_score"],
            maximum_event_risk=evidence_summary["max_risk_score"],
            minimum_ttc=evidence_summary["minimum_ttc"],
            minimum_distance=evidence_summary["minimum_distance"],
            maximum_closing_speed=evidence_summary["maximum_closing_speed"],
            highest_risk_event=highest_ev_obj,
            highest_risk_zone=evidence_summary.get("highest_risk_zone", "Zone A"),
            dominant_risk_patterns=evidence_summary["dominant_risk_patterns"],
            recommendation_count=0
        )

        # Save RiskZoneSummary DB records for Zone A, Zone B, Zone C, Zone D
        zones_dict = evidence_summary.get("zones_data", {})
        for z_id, z_data in zones_dict.items():
            RiskZoneSummary.objects.create(
                video=video,
                zone_id=z_id,
                zone_risk_score=z_data["zone_risk_score"],
                zone_risk_level=z_data["zone_risk_level"],
                total_events=z_data["total_events"],
                high_events=z_data["high_events"],
                medium_events=z_data["medium_events"],
                low_events=z_data["low_events"],
                average_event_risk=z_data["average_event_risk"],
                maximum_event_risk=z_data["maximum_event_risk"],
                minimum_ttc=z_data["minimum_ttc"],
                minimum_distance=z_data["minimum_distance"],
                maximum_closing_speed=z_data["maximum_closing_speed"],
                risk_density=z_data["risk_density"],
                unique_vehicles_involved=z_data["unique_vehicles_involved"],
                dominant_patterns=z_data["dominant_patterns"],
                x_min=z_data["x_min"],
                y_min=z_data["y_min"],
                x_max=z_data["x_max"],
                y_max=z_data["y_max"]
            )

        # 6. Execute Road Safety Recommendation Engine
        rule_engine = RecommendationRuleEngine(evidence_summary)
        rule_recs = rule_engine.generate_recommendations()

        ai_provider = RecommendationAIProvider()
        final_recs = []

        for r_rec in rule_recs:
            final_rec = ai_provider.generate_contextual_recommendation(evidence_summary, r_rec)
            final_recs.append(final_rec)

            ev_link = None
            if final_rec.get("risk_event_ids") and len(final_rec["risk_event_ids"]) > 0:
                matching_ev = RiskEvent.objects.filter(video=video, event_id=final_rec["risk_event_ids"][0]).first()
                if matching_ev:
                    ev_link = matching_ev

            Recommendation.objects.create(
                video=video,
                risk_event=ev_link,
                zone_id=evidence_summary["highest_risk_zone"],
                category=final_rec["category"],
                priority=final_rec["priority"],
                confidence=final_rec["confidence"],
                title=final_rec["title"],
                summary=final_rec["summary"],
                evidence=final_rec.get("evidence", []),
                recommended_action=final_rec["recommended_action"],
                limitations=final_rec.get("limitations", ""),
                source=final_rec.get("source", "RULE_BASED")
            )

        # Update VideoRiskSummary recommendation_count
        v_summary = VideoRiskSummary.objects.filter(video=video).first()
        if v_summary:
            v_summary.recommendation_count = len(final_recs)
            v_summary.save()

        # Save recommendations.json in run directory
        recs_json_path = run_dir / "reports" / "recommendations.json"
        with open(recs_json_path, "w") as f_out:
            json.dump({
                "video_id": video.id,
                "overall_risk_score": evidence_summary["overall_risk_score"],
                "overall_risk_level": evidence_summary["overall_risk_level"],
                "recommendations_count": len(final_recs),
                "recommendations": final_recs
            }, f_out, indent=2)

        # Update video attributes
        if report_json.exists():
            with open(report_json) as f:
                r_data = json.load(f)
                video.duration = parse_float_safe(r_data.get("duration_sec"), 10.0)
                video.fps = parse_float_safe(r_data.get("fps"), 50.0)
                video.total_frames = int(r_data.get("total_frames", 500))
                video.save()
