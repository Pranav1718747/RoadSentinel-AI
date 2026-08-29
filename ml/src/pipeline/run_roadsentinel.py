"""
run_roadsentinel.py — RoadSentinel AI, Phase 8
Master Pipeline Orchestrator for the Complete End-to-End Baseline System.

Executes the complete RoadSentinel traffic risk pipeline:
  RAW VIDEO
      ↓
  1. VIDEO VALIDATION
      ↓
  2. YOLO DETECTION & BYTE TRACKING
      ↓
  3. TRAJECTORY EXTRACTION & STATIONARY FILTERING
      ↓
  4. TRAJECTORY QUALITY VALIDATION & IOU DUP FILTERING
      ↓
  5. FEATURE ENGINEERING (Causal Trailing Smoothing)
      ↓
  6. RULE-BASED RISK ENGINE (Distance-Gated Sub-Scoring)
      ↓
  7. EVENT GROUPING
      ↓
  8. FINAL RISK VISUALIZATION (roadsentinel_final.mp4)
      ↓
  9. REPORTS & PIPELINE MANIFEST
      ↓
 10. FINAL DATA QUALITY & CONSISTENCY CHECKS

Usage:
  python3 src/pipeline/run_roadsentinel.py --input data/samples/traffic.mp4
"""

import os
import sys
import csv
import json
import time
import math
import argparse
import datetime
import itertools
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add current and parent paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "src" / "pipeline"))
sys.path.insert(0, str(BASE_DIR / "src" / "risk"))

import config as cfg
from ultralytics import YOLO
from risk_engine import evaluate_interaction, distance_gate_factor
from event_grouper import group_events

NaN = float("nan")

def is_nan(v):
    try: return v != v
    except: return True

def sf(v):
    if v is None or str(v).strip() == "": return NaN
    try: return float(v)
    except: return NaN


# ──────────────────────────────────────────────────────────────
# Master Pipeline Class
# ──────────────────────────────────────────────────────────────
class RoadSentinelPipeline:
    def __init__(self, input_video_rel, run_id=None, progress_callback=None):
        self.start_time = time.time()
        self.run_id = run_id or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.progress_callback = progress_callback
        
        target_path = Path(input_video_rel)
        if target_path.is_absolute():
            self.input_video_path = target_path
        elif (BASE_DIR / input_video_rel).exists():
            self.input_video_path = (BASE_DIR / input_video_rel).resolve()
        elif (BASE_DIR.parent / input_video_rel).exists():
            self.input_video_path = (BASE_DIR.parent / input_video_rel).resolve()
        else:
            self.input_video_path = target_path.resolve()

        self.paths = cfg.get_run_paths(self.run_id, BASE_DIR)
        
        self.stage_timings = {}
        self.stage_statuses = {}
        self.manifest_stages = []

        # State data across stages
        self.video_props = {}
        self.raw_bboxes = {}         # (tid, fid) -> (x1, y1, x2, y2)
        self.tracking_rows = []
        self.traj_by_track = {}
        self.all_metrics = {}
        self.track_labels = {}
        self.valid_moving_tids = []
        self.enriched_kinematics = {}
        self.feature_rows = []
        self.scored_rows = []
        self.risk_events = []
        self.quality_checks = {}

    def log_stage_start(self, stage_name):
        print(f"\n[RUNNING] {stage_name} ...")
        return time.time()

    def log_stage_end(self, stage_name, t0, status="SUCCESS", output_files=None):
        elapsed = round(time.time() - t0, 3)
        self.stage_timings[stage_name] = elapsed
        self.stage_statuses[stage_name] = status
        print(f"[{'OK' if status == 'SUCCESS' else 'FAILED'}] {stage_name} ({elapsed}s)")
        
        self.manifest_stages.append({
            "stage_name"     : stage_name,
            "status"         : status,
            "duration_sec"   : elapsed,
            "output_files"   : [str(f) for f in (output_files or [])],
        })

    # ──────────────────────────────────────────────────────────
    # Stage 0: Validate Input Video
    # ──────────────────────────────────────────────────────────
    def stage_validate_video(self):
        t0 = self.log_stage_start("Input Video Validation")
        if not self.input_video_path.exists():
            print(f"  [ERROR] Input video not found at {self.input_video_path}")
            self.log_stage_end("Input Video Validation", t0, "FAILED")
            raise FileNotFoundError(f"Input video not found at {self.input_video_path}")

        cap = cv2.VideoCapture(str(self.input_video_path))
        if not cap.isOpened():
            print(f"  [ERROR] Could not open input video {self.input_video_path}")
            self.log_stage_end("Input Video Validation", t0, "FAILED")
            raise ValueError(f"Could not open input video {self.input_video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size_b = self.input_video_path.stat().st_size
        cap.release()

        self.video_props = {
            "path"        : str(self.input_video_path),
            "size_bytes"  : size_b,
            "width"       : width,
            "height"      : height,
            "fps"         : fps,
            "total_frames": total,
            "duration_sec": round(total / fps, 2) if fps > 0 else 0.0,
        }
        print(f"  Video: {width}x{height} @ {fps:.1f} fps, {total} frames ({self.video_props['duration_sec']}s)")
        self.log_stage_end("Input Video Validation", t0, "SUCCESS")

    # ──────────────────────────────────────────────────────────
    # Stage 1: Detection & Tracking
    # ──────────────────────────────────────────────────────────
    def stage_tracking(self):
        t0 = self.log_stage_start("YOLOv8 Detection & ByteTrack Tracking")
        model = YOLO(cfg.YOLO_MODEL_NAME)
        
        cap = cv2.VideoCapture(str(self.input_video_path))
        fps    = self.video_props["fps"]
        width  = self.video_props["width"]
        height = self.video_props["height"]
        total  = self.video_props["total_frames"]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid = cv2.VideoWriter(str(self.paths["tracked_video"]), fourcc, fps, (width, height))

        csv_fh = open(self.paths["tracks_csv"], mode="w", newline="")
        csv_wtr = csv.writer(csv_fh)
        csv_wtr.writerow([
            "frame_id", "timestamp", "track_id", "class_id", "class_name", 
            "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"
        ])

        frame_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1
                ts = frame_id / fps

                results = model.track(
                    source=frame,
                    persist=True,
                    tracker=cfg.TRACKER_CONFIG,
                    verbose=False,
                    imgsz=cfg.DETECTION_IMGSZ,
                    conf=cfg.DETECTION_CONF,
                    classes=cfg.VEHICLE_CLASSES,
                )
                res = results[0]

                if res.boxes is not None and len(res.boxes) > 0:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    cls_ids = res.boxes.cls.cpu().numpy()
                    track_ids = res.boxes.id.cpu().numpy() if res.boxes.id is not None else None

                    for idx, box in enumerate(boxes):
                        if track_ids is None or idx >= len(track_ids) or track_ids[idx] is None:
                            continue
                        tid = int(track_ids[idx])
                        cid = int(cls_ids[idx])
                        cname = model.names[cid] if hasattr(model, "names") else str(cid)
                        conf = float(confs[idx])
                        x1, y1, x2, y2 = [float(v) for v in box]
                        cx = round((x1 + x2) / 2.0, 2)
                        cy = round((y1 + y2) / 2.0, 2)

                        row = [frame_id, f"{ts:.3f}", tid, cid, cname, f"{conf:.3f}",
                               f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}", cx, cy]
                        csv_wtr.writerow(row)
                        self.raw_bboxes[(tid, frame_id)] = (x1, y1, x2, y2)
                        self.tracking_rows.append(row)

                        # Annotate frame
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID:{tid} {cname}", (int(x1), int(y1)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                out_vid.write(frame)
                if self.progress_callback:
                    self.progress_callback(
                        stage="DETECTING",
                        current_frame=frame_id,
                        total_frames=total,
                        processed_detections=len(self.tracking_rows)
                    )
        finally:
            cap.release()
            out_vid.release()
            csv_fh.close()

        print(f"  Processed {frame_id} frames. Detections: {len(self.tracking_rows)}")
        self.log_stage_end("YOLOv8 Detection & ByteTrack Tracking", t0, "SUCCESS",
                           [self.paths["tracks_csv"], self.paths["tracked_video"]])

    # ──────────────────────────────────────────────────────────
    # Stage 2: Trajectory Extraction
    # ──────────────────────────────────────────────────────────
    def stage_trajectory_extraction(self):
        t0 = self.log_stage_start("Trajectory Extraction & Causal Smoothing")
        raw = defaultdict(list)
        with open(self.paths["tracks_csv"], newline="") as fh:
            for row in csv.DictReader(fh):
                tid = int(row["track_id"])
                raw[tid].append({
                    "track_id"   : tid,
                    "class_name" : row["class_name"].strip(),
                    "frame_id"   : int(row["frame_id"]),
                    "timestamp"  : float(row["timestamp"]),
                    "center_x"   : float(row["center_x"]),
                    "center_y"   : float(row["center_y"]),
                    "confidence" : float(row["confidence"]),
                })

        for tid in raw:
            raw[tid].sort(key=lambda r: r["frame_id"])
            xs = [r["center_x"] for r in raw[tid]]
            ys = [r["center_y"] for r in raw[tid]]
            # Apply CAUSAL TRAILING SMOOTHING (Fix 2)
            n = len(xs)
            sx = []
            sy = []
            for i in range(n):
                lo = max(0, i - cfg.CAUSAL_SMOOTH_WINDOW + 1)
                hi = i + 1
                sx.append(round(sum(xs[lo:hi]) / (hi - lo), 2))
                sy.append(round(sum(ys[lo:hi]) / (hi - lo), 2))
            for i, r in enumerate(raw[tid]):
                r["smooth_x"] = sx[i]
                r["smooth_y"] = sy[i]

        self.traj_by_track = dict(raw)

        with open(self.paths["traj_csv"], "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["track_id", "class_name", "frame_id", "timestamp",
                             "center_x", "center_y", "smooth_x", "smooth_y", "confidence"])
            for tid, obs in self.traj_by_track.items():
                for o in obs:
                    writer.writerow([tid, o["class_name"], o["frame_id"], f"{o['timestamp']:.3f}",
                                     o["center_x"], o["center_y"], o["smooth_x"], o["smooth_y"],
                                     f"{o['confidence']:.3f}"])

        print(f"  Extracted {len(self.traj_by_track)} raw trajectories with causal smoothing.")
        self.log_stage_end("Trajectory Extraction & Causal Smoothing", t0, "SUCCESS", [self.paths["traj_csv"]])

    # ──────────────────────────────────────────────────────────
    # Stage 3: Trajectory Quality Validation & IoU Filtering
    # ──────────────────────────────────────────────────────────
    def stage_trajectory_validation(self):
        t0 = self.log_stage_start("Trajectory Quality Validation & IoU Duplicate Filtering")
        
        # 1. Metrics per track
        for tid, obs in self.traj_by_track.items():
            n = len(obs)
            fids = [o["frame_id"] for o in obs]
            disps = [math.sqrt((obs[i]["center_x"]-obs[i-1]["center_x"])**2 +
                               (obs[i]["center_y"]-obs[i-1]["center_y"])**2)
                     for i in range(1, n)]
            total_disp = sum(disps)
            avg_disp   = total_disp / len(disps) if disps else 0.0
            gaps       = [fids[i+1] - fids[i] - 1 for i in range(len(fids) - 1)]
            above      = sum(1 for d in disps if d >= cfg.MOVE_STEP)
            ratio      = above / len(disps) if disps else 0.0

            # Std dev of displacements
            m = avg_disp
            std = math.sqrt(sum((d - m)**2 for d in disps) / len(disps)) if len(disps) >= 2 else 0.0
            has_jump = any(d > cfg.MAX_JUMP_PX for d in disps)

            self.all_metrics[tid] = {
                "track_id": tid, "class_name": obs[0]["class_name"], "n_obs": n,
                "start_frame": fids[0], "end_frame": fids[-1],
                "total_displacement": round(total_disp, 3), "avg_displacement": round(avg_disp, 4),
                "movement_ratio": round(ratio, 4), "n_gaps": sum(1 for g in gaps if g > 0),
                "max_gap_size": max(gaps) if gaps else 0, "displacement_std": round(std, 4),
                "has_jump": has_jump
            }

        # 2. IoU Candidate Detection (Fix 3)
        frame_boxes = defaultdict(dict)
        for (tid, fid), box in self.raw_bboxes.items():
            if tid in self.traj_by_track:
                frame_boxes[fid][tid] = box

        pair_high_iou = defaultdict(int)
        pair_overlap = defaultdict(int)
        for fid, tdict in frame_boxes.items():
            tids = sorted(tdict.keys())
            for i in range(len(tids)):
                for j in range(i+1, len(tids)):
                    t1, t2 = tids[i], tids[j]
                    pair = (t1, t2)
                    pair_overlap[pair] += 1
                    # IoU
                    b1, b2 = tdict[t1], tdict[t2]
                    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou > cfg.IOU_DUPLICATE_THRESH:
                        pair_high_iou[pair] += 1

        filtered_duplicates = set()
        questionable_tracks = set()
        for (t1, t2), high_cnt in pair_high_iou.items():
            tot = pair_overlap[(t1, t2)]
            if high_cnt >= cfg.IOU_DUPLICATE_FRAMES or (high_cnt / tot >= 0.30 if tot > 0 else False):
                len1 = self.all_metrics[t1]["n_obs"]
                len2 = self.all_metrics[t2]["n_obs"]
                shorter = t2 if len2 <= len1 else t1
                filtered_duplicates.add(shorter)
            else:
                questionable_tracks.add(t1)
                questionable_tracks.add(t2)

        # 3. Classify tracks
        for tid, m in self.all_metrics.items():
            if tid in filtered_duplicates or tid in questionable_tracks:
                lbl = "QUESTIONABLE"
            elif m["n_obs"] < cfg.MIN_OBS:
                lbl = "SHORT"
            elif m["has_jump"]:
                lbl = "QUESTIONABLE"
            elif m["n_gaps"] > cfg.MAX_GAPS_ALLOWED or m["max_gap_size"] > cfg.MAX_GAP_SIZE:
                lbl = "FRAGMENTED"
            elif (m["avg_displacement"] < cfg.MIN_AVG_DISPLACEMENT or
                  m["movement_ratio"] < cfg.MIN_MOVEMENT_RATIO or
                  m["total_displacement"] < cfg.MIN_TOTAL_DISP):
                lbl = "STATIONARY"
            elif m["displacement_std"] > cfg.MAX_JITTER_STD:
                lbl = "NOISY"
            else:
                lbl = "VALID_MOVING"
            self.track_labels[tid] = lbl

        self.valid_moving_tids = sorted([tid for tid, lbl in self.track_labels.items() if lbl == "VALID_MOVING"])

        # Write validated CSV
        with open(self.paths["val_traj_csv"], "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["track_id", "class_name", "frame_id", "timestamp",
                             "center_x", "center_y", "smooth_x", "smooth_y", "confidence", "quality_label"])
            for tid in self.valid_moving_tids:
                for o in self.traj_by_track[tid]:
                    writer.writerow([tid, o["class_name"], o["frame_id"], f"{o['timestamp']:.3f}",
                                     o["center_x"], o["center_y"], o["smooth_x"], o["smooth_y"],
                                     f"{o['confidence']:.3f}", "VALID_MOVING"])

        # Write quality report
        report = {
            "total_tracks": len(self.track_labels),
            "valid_moving_count": len(self.valid_moving_tids),
            "iou_duplicate_candidates": len(pair_high_iou),
            "iou_filtered_duplicates": sorted(filtered_duplicates),
            "valid_moving_ids": self.valid_moving_tids,
        }
        with open(self.paths["traj_report"], "w") as fh:
            json.dump(report, fh, indent=2)

        print(f"  Valid moving tracks: {len(self.valid_moving_tids)} / {len(self.track_labels)}")
        print(f"  IoU Candidate Pairs: {len(pair_high_iou)}, Filtered Duplicates: {len(filtered_duplicates)}")
        self.log_stage_end("Trajectory Quality Validation & IoU Duplicate Filtering", t0, "SUCCESS",
                           [self.paths["val_traj_csv"], self.paths["traj_report"]])

    # ──────────────────────────────────────────────────────────
    # Stage 4: Feature Engineering
    # ──────────────────────────────────────────────────────────
    def stage_feature_engineering(self):
        t0 = self.log_stage_start("Traffic Feature Engineering (Causal Trailing)")
        
        # Load validated trajectories
        valid_tracks = {tid: self.traj_by_track[tid] for tid in self.valid_moving_tids}
        
        # Kinematics with causal smoothing
        enriched = {}
        for tid, obs in valid_tracks.items():
            n = len(obs)
            disps, dts, dirs, speeds_raw = [NaN], [NaN], [NaN], [NaN]
            for i in range(1, n):
                prev, curr = obs[i-1], obs[i]
                dx = curr["smooth_x"] - prev["smooth_x"]
                dy = curr["smooth_y"] - prev["smooth_y"]
                dt = curr["timestamp"] - prev["timestamp"]
                disp = math.sqrt(dx*dx + dy*dy)
                if dt <= 0:
                    disps.append(NaN); dts.append(NaN); dirs.append(NaN); speeds_raw.append(NaN)
                else:
                    disps.append(disp); dts.append(dt)
                    dirs.append(math.degrees(math.atan2(dy, dx)) % 360.0)
                    speeds_raw.append(disp / dt)

            # Causal trailing rolling mean over speeds_raw (Fix 2)
            speeds_smooth = []
            for i in range(n):
                lo = max(0, i - cfg.CAUSAL_SMOOTH_WINDOW + 1)
                hi = i + 1
                seg = [v for v in speeds_raw[lo:hi] if not is_nan(v)]
                speeds_smooth.append(sum(seg)/len(seg) if seg else NaN)

            # Accel on smoothed speed
            accels = [NaN] * n
            for i in range(1, n):
                dt = dts[i]
                if not is_nan(dt) and dt > 1e-6:
                    accels[i] = (speeds_smooth[i] - speeds_smooth[i-1]) / dt

            rows = []
            for i, o in enumerate(obs):
                rows.append({
                    "track_id": tid, "class_name": o["class_name"],
                    "frame_id": o["frame_id"], "timestamp": o["timestamp"],
                    "smooth_x": o["smooth_x"], "smooth_y": o["smooth_y"],
                    "speed": speeds_raw[i], "smoothed_speed": speeds_smooth[i],
                    "acceleration": accels[i], "direction": dirs[i],
                })
            enriched[tid] = rows

        self.enriched_kinematics = enriched

        # Build interaction table frame by frame
        frame_lookup = defaultdict(dict)
        for tid, rows in enriched.items():
            for r in rows:
                frame_lookup[r["frame_id"]][tid] = r

        # Path crossings
        crossings = {}
        for tid1, tid2 in itertools.combinations(self.valid_moving_tids, 2):
            p1 = [(r["smooth_x"], r["smooth_y"]) for r in enriched[tid1]]
            p2 = [(r["smooth_x"], r["smooth_y"]) for r in enriched[tid2]]
            crossed = False
            for i in range(len(p1)-1):
                for j in range(len(p2)-1):
                    # Segment intersection test
                    o1, a1, b1 = p2[j], p2[j+1], p1[i]
                    o2, a2, b2 = p2[j], p2[j+1], p1[i+1]
                    d1 = (a1[0]-o1[0])*(b1[1]-o1[1]) - (a1[1]-o1[1])*(b1[0]-o1[0])
                    d2 = (a1[0]-o1[0])*(b2[1]-o1[1]) - (a1[1]-o1[1])*(b2[0]-o1[0])
                    o3, a3, b3 = p1[i], p1[i+1], p2[j]
                    o4, a4, b4 = p1[i], p1[i+1], p2[j+1]
                    d3 = (a3[0]-o3[0])*(b3[1]-o3[1]) - (a3[1]-o3[1])*(b3[0]-o3[0])
                    d4 = (a3[0]-o3[0])*(b4[1]-o3[1]) - (a3[1]-o3[1])*(b4[0]-o3[0])
                    if ((d1>0<d2) or (d1<0>d2)) and ((d3>0<d4) or (d3<0>d4)):
                        crossed = True; break
                if crossed: break
            crossings[(min(tid1,tid2), max(tid1,tid2))] = crossed

        feature_rows = []
        for fid in sorted(frame_lookup.keys()):
            present = frame_lookup[fid]
            tids = sorted(present.keys())
            if len(tids) < 2: continue

            for tid1, tid2 in itertools.combinations(tids, 2):
                r1, r2 = present[tid1], present[tid2]
                dx = r2["smooth_x"] - r1["smooth_x"]
                dy = r2["smooth_y"] - r1["smooth_y"]
                dist = math.sqrt(dx*dx + dy*dy)

                # Velocity
                spd1, d1 = r1["smoothed_speed"], r1["direction"]
                spd2, d2 = r2["smoothed_speed"], r2["direction"]
                vx1 = spd1*math.cos(math.radians(d1)) if not (is_nan(spd1) or is_nan(d1)) else NaN
                vy1 = spd1*math.sin(math.radians(d1)) if not (is_nan(spd1) or is_nan(d1)) else NaN
                vx2 = spd2*math.cos(math.radians(d2)) if not (is_nan(spd2) or is_nan(d2)) else NaN
                vy2 = spd2*math.sin(math.radians(d2)) if not (is_nan(spd2) or is_nan(d2)) else NaN

                rel_vx = vx1 - vx2 if not (is_nan(vx1) or is_nan(vx2)) else NaN
                rel_vy = vy1 - vy2 if not (is_nan(vy1) or is_nan(vy2)) else NaN
                rel_spd = math.sqrt(rel_vx**2 + rel_vy**2) if not (is_nan(rel_vx) or is_nan(rel_vy)) else NaN

                closing_spd = NaN
                if dist > 1e-6 and not is_nan(rel_vx) and not is_nan(rel_vy):
                    closing_spd = rel_vx * (dx / dist) + rel_vy * (dy / dist)

                approaching = (closing_spd > 0) if not is_nan(closing_spd) else False
                ttc = (dist / closing_spd) if (approaching and closing_spd > 1e-6 and dist / closing_spd <= cfg.TTC_CAP) else NaN

                hdg_diff = NaN
                if not (is_nan(d1) or is_nan(d2)):
                    hdg_diff = abs((d1 - d2 + 180.0) % 360.0 - 180.0)

                crossed = crossings.get((min(tid1,tid2), max(tid1,tid2)), False)

                def fmt(v): return "" if is_nan(v) else v

                feature_rows.append({
                    "timestamp": r1["timestamp"], "frame_id": fid,
                    "v1_id": tid1, "v1_class": r1["class_name"], "v1_x": r1["smooth_x"], "v1_y": r1["smooth_y"],
                    "v1_speed": fmt(r1["smoothed_speed"]), "v1_acceleration": fmt(r1["acceleration"]), "v1_direction": fmt(r1["direction"]),
                    "v2_id": tid2, "v2_class": r2["class_name"], "v2_x": r2["smooth_x"], "v2_y": r2["smooth_y"],
                    "v2_speed": fmt(r2["smoothed_speed"]), "v2_acceleration": fmt(r2["acceleration"]), "v2_direction": fmt(r2["direction"]),
                    "distance": dist, "rel_vx": fmt(rel_vx), "rel_vy": fmt(rel_vy), "relative_speed": fmt(rel_spd),
                    "closing_speed": fmt(closing_spd), "ttc": fmt(ttc), "approaching": approaching,
                    "heading_difference": fmt(hdg_diff), "trajectory_crossing": crossed,
                    "interaction_type": "CROSSING" if crossed else "APPROACHING" if approaching else "PARALLEL"
                })

        self.feature_rows = feature_rows

        # Write feature CSV
        cols = ["timestamp", "frame_id", "v1_id", "v1_class", "v1_x", "v1_y", "v1_speed", "v1_acceleration", "v1_direction",
                "v2_id", "v2_class", "v2_x", "v2_y", "v2_speed", "v2_acceleration", "v2_direction",
                "distance", "rel_vx", "rel_vy", "relative_speed", "closing_speed", "ttc",
                "approaching", "heading_difference", "trajectory_crossing", "interaction_type"]
        with open(self.paths["feat_csv"], "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for r in feature_rows:
                writer.writerow({c: r.get(c, "") for c in cols})

        # Feature Summary JSON
        ttcs = [sf(r["ttc"]) for r in feature_rows if not is_nan(sf(r["ttc"]))]
        dists = [sf(r["distance"]) for r in feature_rows if not is_nan(sf(r["distance"]))]
        spds = [sf(r["v1_speed"]) for r in feature_rows if not is_nan(sf(r["v1_speed"]))]
        accs = [sf(r["v1_acceleration"]) for r in feature_rows if not is_nan(sf(r["v1_acceleration"]))]

        summary = {
            "total_feature_rows": len(feature_rows),
            "valid_ttc_rows": len(ttcs),
            "distance_stats": {"min": round(min(dists), 2), "mean": round(sum(dists)/len(dists), 2)},
            "speed_stats": {"min": round(min(spds), 2), "max": round(max(spds), 2), "mean": round(sum(spds)/len(spds), 2)},
            "accel_stats": {"min": round(min(accs), 2), "max": round(max(accs), 2), "mean": round(sum(accs)/len(accs), 2)},
            "ttc_stats": {"min": round(min(ttcs), 2) if ttcs else None, "mean": round(sum(ttcs)/len(ttcs), 2) if ttcs else None},
        }
        with open(self.paths["feat_summary"], "w") as fh:
            json.dump(summary, fh, indent=2)

        print(f"  Generated {len(feature_rows)} feature rows. Valid TTC rows: {len(ttcs)}")
        self.log_stage_end("Traffic Feature Engineering (Causal Trailing)", t0, "SUCCESS",
                           [self.paths["feat_csv"], self.paths["feat_summary"]])

    # ──────────────────────────────────────────────────────────
    # Stage 5 & 6: Risk Engine & Event Grouping
    # ──────────────────────────────────────────────────────────
    def stage_risk_engine(self):
        t0 = self.log_stage_start("Rule-Based Risk Engine & Event Grouping (Distance-Gated)")
        
        # Score each row
        scored = []
        for r in self.feature_rows:
            res = evaluate_interaction(r)
            sc = dict(r)
            sc.update(res)
            scored.append(sc)
        self.scored_rows = scored

        # Write risk events CSV
        risk_cols = [
            "timestamp", "frame_id", "v1_id", "v1_class", "v2_id", "v2_class",
            "distance", "relative_speed", "closing_speed", "ttc", "trajectory_crossing", "approaching", "heading_difference",
            "risk_score", "risk_level", "ttc_score", "distance_score", "closing_speed_score", "trajectory_score", "heading_score", "risk_reasons"
        ]
        with open(self.paths["risk_csv"], "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=risk_cols)
            writer.writeheader()
            for r in scored:
                writer.writerow({c: r.get(c, "") for c in risk_cols})

        # Group events (GROUP_GAP_FRAMES = 15)
        events = group_events(scored)
        self.risk_events = events

        # Write events JSON
        with open(self.paths["risk_json"], "w") as fh:
            json.dump(events, fh, indent=2)

        # Risk Summary JSON
        level_counts = defaultdict(int)
        for r in scored:
            level_counts[r["risk_level"]] += 1

        high_evts = [e for e in events if e["peak_risk_level"] == "HIGH"]
        med_evts = [e for e in events if e["peak_risk_level"] == "MEDIUM"]

        summary = {
            "disclaimer": "RoadSentinel Rule-Based Traffic Risk Baseline — NOT a trained ML model.",
            "totals": {
                "total_interactions": len(scored),
                "low_risk": level_counts["LOW"],
                "medium_risk": level_counts["MEDIUM"],
                "high_risk": level_counts["HIGH"],
                "unique_risk_events": len(events),
                "high_risk_events": len(high_evts),
                "medium_risk_events": len(med_evts),
            },
            "top_high_events": high_evts,
        }
        with open(self.paths["risk_summary"], "w") as fh:
            json.dump(summary, fh, indent=2)

        print(f"  Scored {len(scored)} rows. LOW: {level_counts['LOW']}, MEDIUM: {level_counts['MEDIUM']}, HIGH: {level_counts['HIGH']}")
        print(f"  Grouped {len(events)} unique risk events (HIGH: {len(high_evts)}, MEDIUM: {len(med_evts)})")
        self.log_stage_end("Rule-Based Risk Engine & Event Grouping (Distance-Gated)", t0, "SUCCESS",
                           [self.paths["risk_csv"], self.paths["risk_json"], self.paths["risk_summary"]])

    # ──────────────────────────────────────────────────────────
    # Stage 7: Final Risk Visualization (roadsentinel_final.mp4)
    # ──────────────────────────────────────────────────────────
    def stage_visualization(self):
        t0 = self.log_stage_start("Final Risk Visualization Rendering (roadsentinel_final.mp4)")
        
        cap = cv2.VideoCapture(str(self.input_video_path))
        fps    = self.video_props["fps"]
        width  = self.video_props["width"]
        height = self.video_props["height"]
        total  = self.video_props["total_frames"]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        wtr    = cv2.VideoWriter(str(self.paths["final_video"]), fourcc, fps, (width, height))

        # Build frame lookup
        frame_scored = defaultdict(list)
        for r in self.scored_rows:
            frame_scored[int(r["frame_id"])].append(r)

        # Smooth position history lookup: tid -> {fid: (sx, sy)}
        sp_lookup = defaultdict(dict)
        for tid in self.valid_moving_tids:
            for r in self.traj_by_track[tid]:
                sp_lookup[tid][r["frame_id"]] = (r["smooth_x"], r["smooth_y"])

        colors = {"LOW": (0, 200, 0), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 220)}
        trail_hist = {tid: [] for tid in self.valid_moving_tids}

        frame_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_id += 1
                scored_here = frame_scored.get(frame_id, [])

                # Vehicle risk level in this frame
                v_risk = {}
                for r in scored_here:
                    for vid in [r["v1_id"], r["v2_id"]]:
                        cur = v_risk.get(vid, "LOW")
                        if r["risk_level"] == "HIGH": v_risk[vid] = "HIGH"
                        elif r["risk_level"] == "MEDIUM" and cur != "HIGH": v_risk[vid] = "MEDIUM"
                        else: v_risk.setdefault(vid, "LOW")

                # Draw trails
                for tid in self.valid_moving_tids:
                    sp = sp_lookup[tid].get(frame_id)
                    if sp: trail_hist[tid].append((int(sp[0]), int(sp[1])))
                    trail = trail_hist[tid][-30:]
                    color = colors[v_risk.get(tid, "LOW")]
                    for i in range(1, len(trail)):
                        t = max(1, int((i/len(trail))*3))
                        cv2.line(frame, trail[i-1], trail[i], color, t)

                # Draw Bboxes
                for tid in self.valid_moving_tids:
                    bbox = self.raw_bboxes.get((tid, frame_id))
                    if not bbox: continue
                    x1, y1, x2, y2 = bbox
                    lvl = v_risk.get(tid, "LOW")
                    color = colors[lvl]
                    lw = 3 if lvl == "HIGH" else 2
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, lw)
                    cv2.putText(frame, f"ID{tid} [{lvl}]", (int(x1), max(int(y1)-5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                # Draw pair connecting lines & card overlays
                active_high_event = None
                for r in scored_here:
                    if r["risk_level"] in ("MEDIUM", "HIGH"):
                        sp1 = sp_lookup[r["v1_id"]].get(frame_id)
                        sp2 = sp_lookup[r["v2_id"]].get(frame_id)
                        if sp1 and sp2:
                            pt1 = (int(sp1[0]), int(sp1[1]))
                            pt2 = (int(sp2[0]), int(sp2[1]))
                            color = colors[r["risk_level"]]
                            cv2.line(frame, pt1, pt2, color, 2)
                        if r["risk_level"] == "HIGH" and active_high_event is None:
                            active_high_event = r

                # Render active HIGH-risk overlay banner (if any)
                if active_high_event:
                    r = active_high_event
                    # Top-left card
                    card_w, card_h = 460, 110
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (15, 15), (15 + card_w, 15 + card_h), (0, 0, 120), -1)
                    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
                    cv2.rectangle(frame, (15, 15), (15 + card_w, 15 + card_h), (0, 0, 255), 2)
                    
                    ttc_v = sf(r.get("ttc"))
                    dist_v = sf(r.get("distance"))
                    cs_v = sf(r.get("closing_speed"))
                    ttc_str = f"{ttc_v:.2f}s" if not is_nan(ttc_v) else "n/a"
                    dist_str = f"{dist_v:.1f}px" if not is_nan(dist_v) else "n/a"
                    cs_str = f"{cs_v:.1f}px/s" if not is_nan(cs_v) else "n/a"

                    cv2.putText(frame, f"HIGH RISK EVENT — Score: {r['risk_score']:.2f}", (25, 38),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.putText(frame, f"Vehicle {r['v1_id']} ({r['v1_class']}) <-> Vehicle {r['v2_id']} ({r['v2_class']})", (25, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
                    cv2.putText(frame, f"TTC: {ttc_str}  |  Dist: {dist_str}  |  CS: {cs_str}", (25, 78),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    cv2.putText(frame, f"Reason: {r['risk_reasons'][:55]}", (25, 98),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 255), 1)

                wtr.write(frame)
        finally:
            cap.release()
            wtr.release()

        print(f"  Rendered final video: {self.paths['final_video']}")
        self.log_stage_end("Final Risk Visualization Rendering (roadsentinel_final.mp4)", t0, "SUCCESS",
                           [self.paths["final_video"]])

    # ──────────────────────────────────────────────────────────
    # Stage 8: Reports & Manifest
    # ──────────────────────────────────────────────────────────
    def stage_reports(self):
        t0 = self.log_stage_start("Reports & Pipeline Manifest Generation")
        
        # High events
        high_evts = [e for e in self.risk_events if e["peak_risk_level"] == "HIGH"]
        med_evts  = [e for e in self.risk_events if e["peak_risk_level"] == "MEDIUM"]
        low_count = sum(1 for r in self.scored_rows if r["risk_level"] == "LOW")
        med_count = sum(1 for r in self.scored_rows if r["risk_level"] == "MEDIUM")
        high_count = sum(1 for r in self.scored_rows if r["risk_level"] == "HIGH")

        ttcs = [sf(r["ttc"]) for r in self.feature_rows if not is_nan(sf(r["ttc"]))]
        dists = [sf(r["distance"]) for r in self.feature_rows if not is_nan(sf(r["distance"]))]
        spds = [sf(r["v1_speed"]) for r in self.feature_rows if not is_nan(sf(r["v1_speed"]))]

        # Detailed Report JSON
        report_data = {
            "disclaimer": "RoadSentinel Rule-Based Traffic Risk Baseline — NOT a trained ML model.",
            "video_summary": self.video_props,
            "object_summary": {
                "total_detections": len(self.tracking_rows),
                "total_unique_tracks": len(self.traj_by_track),
                "valid_moving_tracks": len(self.valid_moving_tids),
                "stationary_tracks": sum(1 for lbl in self.track_labels.values() if lbl == "STATIONARY"),
            },
            "trajectory_summary": {
                "valid_moving_trajectories": len(self.valid_moving_tids),
                "questionable_trajectories": sum(1 for lbl in self.track_labels.values() if lbl == "QUESTIONABLE"),
                "short_trajectories": sum(1 for lbl in self.track_labels.values() if lbl == "SHORT"),
                "fragmented_trajectories": sum(1 for lbl in self.track_labels.values() if lbl == "FRAGMENTED"),
            },
            "feature_summary": {
                "total_feature_rows": len(self.feature_rows),
                "valid_ttc_rows": len(ttcs),
                "avg_speed_px_s": round(sum(spds)/len(spds), 2) if spds else None,
                "max_speed_px_s": round(max(spds), 2) if spds else None,
                "min_distance_px": round(min(dists), 2) if dists else None,
                "min_ttc_seconds": round(min(ttcs), 2) if ttcs else None,
            },
            "risk_summary": {
                "total_interactions": len(self.scored_rows),
                "low_interactions": low_count,
                "medium_interactions": med_count,
                "high_interactions": high_count,
                "total_risk_events": len(self.risk_events),
                "high_risk_events": len(high_evts),
                "medium_risk_events": len(med_evts),
            },
            "top_high_events": high_evts,
        }
        with open(self.paths["report_json"], "w") as fh:
            json.dump(report_data, fh, indent=2)

        # Human-Readable Report TXT
        W = 47
        txt_lines = [
            "=" * W,
            "ROADSENTINEL AI — TRAFFIC SAFETY REPORT",
            "=" * W,
            f"Input Video : {self.input_video_path.name}",
            f"Run ID      : {self.run_id}",
            f"Resolution  : {self.video_props['width']}x{self.video_props['height']}",
            f"FPS         : {self.video_props['fps']:.1f}",
            f"Total Frames: {self.video_props['total_frames']}",
            f"Duration    : {self.video_props['duration_sec']} seconds",
            "",
            "-" * W,
            "OBJECT ANALYSIS",
            "-" * W,
            f"Total Detections   : {len(self.tracking_rows):,}",
            f"Unique Tracks      : {len(self.traj_by_track)}",
            f"Moving Vehicles    : {len(self.valid_moving_tids)}",
            "",
            "-" * W,
            "TRAJECTORY ANALYSIS",
            "-" * W,
            f"Valid Trajectories : {len(self.valid_moving_tids)}",
            f"Questionable Tracks: {sum(1 for lbl in self.track_labels.values() if lbl == 'QUESTIONABLE')} (IoU duplicate reclassifications)",
            f"Stationary Tracks  : {sum(1 for lbl in self.track_labels.values() if lbl == 'STATIONARY')}",
            "",
            "-" * W,
            "RISK ANALYSIS",
            "-" * W,
            f"LOW Interactions   : {low_count:>5}  ({low_count/len(self.scored_rows)*100:5.1f}%)",
            f"MEDIUM Interactions: {med_count:>5}  ({med_count/len(self.scored_rows)*100:5.1f}%)",
            f"HIGH Interactions  : {high_count:>5}  ({high_count/len(self.scored_rows)*100:5.1f}%)",
            f"Total Risk Events  : {len(self.risk_events)} (HIGH: {len(high_evts)}, MEDIUM: {len(med_evts)})",
            "",
            "-" * W,
            "TOP HIGH-RISK EVENTS",
            "-" * W,
        ]

        if high_evts:
            for e in high_evts:
                txt_lines.extend([
                    f"Event #{e['event_id']}",
                    f"  Vehicles     : Vehicle {e['v1_id']} ({e['v1_class']}) <-> Vehicle {e['v2_id']} ({e['v2_class']})",
                    f"  Frames       : {e['start_frame']} – {e['end_frame']} ({e['event_duration_frames']} frames / {e['duration_seconds']}s)",
                    f"  Peak Score   : {e['max_risk_score']:.4f}",
                    f"  Minimum TTC  : {e['min_ttc']:.2f}s" if e["min_ttc"] is not None else "  Minimum TTC  : n/a",
                    f"  Minimum Dist : {e['min_distance']:.1f}px" if e["min_distance"] is not None else "  Minimum Dist : n/a",
                    f"  Max Closing  : {e['max_closing_speed']:.1f}px/s" if e["max_closing_speed"] is not None else "  Max Closing  : n/a",
                    f"  Primary Cause: {e['top_reasons']}",
                    ""
                ])
        else:
            txt_lines.append("  No HIGH-risk events detected in this video run.\n")

        txt_lines.extend([
            "=" * W,
            "SCIENTIFIC LIMITATION DISCLAIMER:",
            "This report is generated by the RoadSentinel Rule-Based",
            "Traffic Risk Baseline. It is NOT a trained ML model.",
            "Speed, distance, and TTC are image-plane quantities.",
            "=" * W,
        ])

        with open(self.paths["report_txt"], "w") as fh:
            fh.write("\n".join(txt_lines))

        # Pipeline Manifest JSON
        end_time_sec = time.time()
        total_pipeline_time = round(end_time_sec - self.start_time, 3)

        manifest = {
            "run_id": self.run_id,
            "disclaimer": "RoadSentinel Rule-Based Traffic Risk Baseline — NOT a trained ML model.",
            "input_video": str(self.input_video_path),
            "input_video_size_bytes": self.video_props["size_bytes"],
            "video_width": self.video_props["width"],
            "video_height": self.video_props["height"],
            "fps": self.video_props["fps"],
            "total_frames": self.video_props["total_frames"],
            "processing_start_time": datetime.datetime.fromtimestamp(self.start_time).isoformat(),
            "processing_end_time": datetime.datetime.fromtimestamp(end_time_sec).isoformat(),
            "total_processing_duration_sec": total_pipeline_time,
            "processing_fps": round(self.video_props["total_frames"] / total_pipeline_time, 2) if total_pipeline_time > 0 else 0.0,
            "stages": self.manifest_stages,
        }

        with open(self.paths["manifest_json"], "w") as fh:
            json.dump(manifest, fh, indent=2)

        print(f"  Generated report JSON, TXT, and pipeline manifest.")
        self.log_stage_end("Reports & Pipeline Manifest Generation", t0, "SUCCESS",
                           [self.paths["report_json"], self.paths["report_txt"], self.paths["manifest_json"]])

    # ──────────────────────────────────────────────────────────
    # Stage 9: Final Quality & Consistency Checks
    # ──────────────────────────────────────────────────────────
    def stage_quality_checks(self):
        t0 = self.log_stage_start("Final Quality & Data Consistency Checks")
        
        checks = {}
        # 1. Output files exist & non-empty
        for key in ["tracks_csv", "tracked_video", "val_traj_csv", "feat_csv",
                    "risk_csv", "final_video", "manifest_json", "report_json", "report_txt"]:
            p = self.paths[key]
            exists = p.exists()
            size = p.stat().st_size if exists else 0
            checks[f"file_{key}_valid"] = exists and size > 0

        # 2. Can video be opened?
        cap = cv2.VideoCapture(str(self.paths["final_video"]))
        checks["final_video_openable"] = cap.isOpened()
        if cap.isOpened(): cap.release()

        # 3. Data consistency: ID propagation
        # All vehicle IDs in risk events exist in valid_moving_tids
        valid_set = set(self.valid_moving_tids)
        risk_ids = set()
        for r in self.scored_rows:
            risk_ids.add(int(r["v1_id"]))
            risk_ids.add(int(r["v2_id"]))
        checks["risk_vehicle_ids_valid"] = risk_ids.issubset(valid_set)

        # 4. No NaN/inf in critical risk columns
        nan_count = 0
        for r in self.scored_rows:
            for c in ["distance", "risk_score"]:
                if is_nan(sf(r.get(c))): nan_count += 1
        checks["no_nan_in_critical_risk_cols"] = (nan_count == 0)

        # 5. Distance gating check: 0 HIGH rows at distance >= 600px
        high_gt_600 = sum(1 for r in self.scored_rows if sf(r.get("distance")) >= 600.0 and r["risk_level"] == "HIGH")
        checks["distance_gating_high_count_zero"] = (high_gt_600 == 0)

        self.quality_checks = checks
        all_passed = all(checks.values())
        
        print("  Quality Checks Status:")
        for k, v in checks.items():
            print(f"    {'[OK]' if v else '[FAIL]'} {k}")

        self.log_stage_end("Final Quality & Data Consistency Checks", t0,
                           "SUCCESS" if all_passed else "FAILED")

    # ──────────────────────────────────────────────────────────
    # Run Complete Pipeline
    # ──────────────────────────────────────────────────────────
    def run_all(self):
        print("\n" + "=" * 60)
        print(" ROAD SENTINEL AI — COMPLETE END-TO-END BASELINE PIPELINE")
        print("=" * 60)
        print(f" Run ID: {self.run_id}")
        print(f" Output Directory: {self.paths['run_dir']}")

        self.stage_validate_video()
        self.stage_tracking()
        self.stage_trajectory_extraction()
        self.stage_trajectory_validation()
        self.stage_feature_engineering()
        self.stage_risk_engine()
        self.stage_visualization()
        self.stage_reports()
        self.stage_quality_checks()

        total_time = round(time.time() - self.start_time, 2)
        fps_perf = round(self.video_props["total_frames"] / total_time, 2) if total_time > 0 else 0.0

        # Print Final Terminal Summary
        self.print_final_summary(total_time, fps_perf)

    def print_final_summary(self, total_time, fps_perf):
        low_count = sum(1 for r in self.scored_rows if r["risk_level"] == "LOW")
        med_count = sum(1 for r in self.scored_rows if r["risk_level"] == "MEDIUM")
        high_count = sum(1 for r in self.scored_rows if r["risk_level"] == "HIGH")
        high_evts = [e for e in self.risk_events if e["peak_risk_level"] == "HIGH"]
        ttcs = [sf(r["ttc"]) for r in self.feature_rows if not is_nan(sf(r["ttc"]))]

        W = 50
        print("\n" + "=" * W)
        print(" ROAD SENTINEL — END-TO-END PIPELINE SUMMARY")
        print("=" * W)
        print(f" Input Video : {self.input_video_path.name}")
        print(f" Resolution  : {self.video_props['width']}x{self.video_props['height']}")
        print(f" FPS         : {self.video_props['fps']:.1f}")
        print(f" Total Frames: {self.video_props['total_frames']}")
        print()

        for stage, status in self.stage_statuses.items():
            print(f" [{'OK' if status == 'SUCCESS' else 'FAILED'}] {stage:<40}")
        print()

        print("-" * W)
        print(" RESULTS")
        print("-" * W)
        print(f" Total Detections    : {len(self.tracking_rows):,}")
        print(f" Unique Tracks       : {len(self.traj_by_track)}")
        print(f" Valid Moving Tracks : {len(self.valid_moving_tids)}")
        print(f" Feature Rows        : {len(self.feature_rows):,}")
        print()
        print(f" LOW Interactions    : {low_count:>5}  ({low_count/len(self.scored_rows)*100:5.1f}%)")
        print(f" MEDIUM Interactions : {med_count:>5}  ({med_count/len(self.scored_rows)*100:5.1f}%)")
        print(f" HIGH Interactions   : {high_count:>5}  ({high_count/len(self.scored_rows)*100:5.1f}%)")
        print(f" Total Risk Events   : {len(self.risk_events)} (HIGH: {len(high_evts)})")
        print()
        print(f" Highest Risk Score  : {max(sf(r['risk_score']) for r in self.scored_rows):.4f}")
        print(f" Lowest Valid TTC    : {min(ttcs):.3f}s" if ttcs else " Lowest Valid TTC    : n/a")
        print()

        print("-" * W)
        print(" PERFORMANCE")
        print("-" * W)
        for stage, t_sec in self.stage_timings.items():
            print(f"   {stage:<40}: {t_sec:>6.2f}s")
        print(f" Total Processing Time: {total_time:.2f}s")
        print(f" Processing Speed     : {fps_perf:.2f} FPS")
        print()

        print("-" * W)
        print(" OUTPUT MANIFEST")
        print("-" * W)
        print(f" Final Video: {self.paths['final_video']}")
        print(f" Report JSON: {self.paths['report_json']}")
        print(f" Report TXT : {self.paths['report_txt']}")
        print(f" Manifest   : {self.paths['manifest_json']}")
        print("=" * W)
        print("\n SCIENTIFIC LIMITATION DISCLAIMER:")
        print(" This pipeline represents the RoadSentinel Rule-Based")
        print(" Traffic Risk Baseline. It is NOT a trained ML model.")
        print(" Speed, distance, and TTC are image-plane quantities.")
        print("=" * W + "\n")


# ──────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RoadSentinel End-to-End Master Pipeline")
    parser.add_argument("--input", type=str, default=cfg.DEFAULT_INPUT_VIDEO,
                        help="Path to input video relative to project root")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Optional unique run identifier")
    args = parser.parse_args()

    pipeline = RoadSentinelPipeline(args.input, args.run_id)
    pipeline.run_all()


if __name__ == "__main__":
    main()
