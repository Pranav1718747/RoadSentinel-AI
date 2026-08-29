# RoadSentinel AI - Phase 1

This module contains the Phase 1 YOLOv8 object detection baseline for the RoadSentinel AI platform.

## What it does
The `detect.py` script loads a pretrained YOLOv8 model (`yolov8n.pt`) and runs it frame-by-frame on an input video. It detects road users, draws bounding boxes (along with class name and confidence) on each frame, and preserves the original video's FPS and resolution. 

## How to run it

From the `ml` directory, run:
```bash
python src/detection/detect.py
```

### Inputs
By default, the script looks for:
`data/samples/traffic.mp4`

### Outputs
The script produces two outputs:
1. **Annotated Video**: `outputs/detections/detected_traffic.mp4`
2. **Detections CSV**: `outputs/detections/detections.csv`

### CSV Schema
The CSV file contains the following columns for each detection:
- `frame_id`: Frame index (1-based or 0-based index)
- `timestamp`: Timestamp of the frame in seconds
- `class_id`: The YOLOv8 class ID for the detected object
- `class_name`: The human-readable name of the class (e.g., person, car)
- `confidence`: The detection confidence score
- `x1`: Top-left x-coordinate of the bounding box
- `y1`: Top-left y-coordinate of the bounding box
- `x2`: Bottom-right x-coordinate of the bounding box
- `y2`: Bottom-right y-coordinate of the bounding box
- `center_x`: Center x-coordinate of the bounding box
- `center_y`: Center y-coordinate of the bounding box

## Phase 1B: ByteTrack Multi-Object Tracking

This phase introduces multi-object tracking by combining the YOLOv8 detections with ByteTrack. 

### Why Tracking is Required
While detection identifies road users frame-by-frame, tracking assigns a persistent identity (`track_id`) to each user across multiple frames. This continuity is essential for the later phases of the RoadSentinel pipeline, which require trajectory analysis, speed estimation, and risk analysis.

### What ByteTrack Does
ByteTrack associates detection bounding boxes across video frames. It leverages YOLOv8's built-in tracking capabilities to seamlessly map road users from one frame to the next without sacrificing performance.

### How to run the tracker
From the `ml` directory, run:
```bash
python src/tracking/track.py
```

### Outputs
The tracker script processes the same input video (`data/samples/traffic.mp4`) and produces:
1. **Annotated Tracking Video**: `outputs/tracking/tracked_traffic.mp4` (includes `track_id` overlaid on boxes)
2. **Tracks CSV**: `outputs/tracking/tracks.csv`

### Tracks CSV Schema
The CSV file contains the following columns for each tracked object:
- `frame_id`: Frame index (1-based index)
- `timestamp`: Timestamp of the frame in seconds
- `track_id`: The persistent unique identifier assigned by ByteTrack
- `class_id`: The YOLOv8 class ID
- `class_name`: The human-readable name of the class
- `confidence`: The detection confidence score
- `x1`: Top-left x-coordinate of the bounding box
- `y1`: Top-left y-coordinate of the bounding box
- `x2`: Bottom-right x-coordinate of the bounding box
- `y2`: Bottom-right y-coordinate of the bounding box
- `center_x`: Center x-coordinate of the bounding box
- `center_y`: Center y-coordinate of the bounding box

