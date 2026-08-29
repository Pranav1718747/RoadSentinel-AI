"""
download_dataset.py — RoadSentimentalAI, Phase 3
Downloads the Kaggle dataset 'daudshah/vehicle-detection-dataset' using kagglehub
and programmatically inspects its directory structure, files, annotations, and format.
"""

import os
import sys
import json
import glob
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("Installing kagglehub package...")
    os.system("pip install kagglehub")
    import kagglehub


def download_and_inspect():
    print("Downloading Kaggle dataset 'daudshah/vehicle-detection-dataset' via kagglehub...")
    path_str = kagglehub.dataset_download("daudshah/vehicle-detection-dataset")
    print("Path to dataset files:", path_str)

    ds_path = Path(path_str)
    if not ds_path.exists():
        print(f"Error: Path {ds_path} does not exist.")
        sys.exit(1)

    # Inspect directory structure
    all_files = list(ds_path.rglob("*"))
    all_dirs  = [f for f in all_files if f.is_dir()]
    all_files = [f for f in all_files if f.is_file()]

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    label_exts = {".txt", ".xml", ".json", ".csv"}

    images = [f for f in all_files if f.suffix.lower() in image_exts]
    labels = [f for f in all_files if f.suffix.lower() in label_exts]

    print(f"\nDataset Exploration:")
    print(f"  Root Path       : {ds_path}")
    print(f"  Total Subdirs   : {len(all_dirs)}")
    print(f"  Total Files     : {len(all_files)}")
    print(f"  Image Files     : {len(images)}")
    print(f"  Label/Annotation Files: {len(labels)}")

    # Sample subdirectories
    print("\nSubdirectories:")
    for d in all_dirs[:15]:
        print(f"  {d.relative_to(ds_path)}")

    # Inspect annotation format
    anno_format = "UNKNOWN"
    sample_label = labels[0] if labels else None

    if sample_label:
        ext = sample_label.suffix.lower()
        if ext == ".txt":
            # Check if YOLO format (class_id x_center y_center width height)
            with open(sample_label) as f:
                content = f.read().strip()
            lines = content.splitlines()
            if lines and len(lines[0].split()) == 5:
                anno_format = "YOLO_TXT (class x_center y_center width height)"
            else:
                anno_format = f"TXT (lines: {len(lines)})"
        elif ext == ".xml":
            anno_format = "PASCAL_VOC_XML"
        elif ext == ".json":
            anno_format = "COCO_JSON"
        elif ext == ".csv":
            anno_format = "CSV_ANNOTATIONS"

    print(f"\nProgrammatically Detected Annotation Format: {anno_format}")

    # Read class names if data.yaml or classes.txt exist
    yaml_files = [f for f in all_files if f.name.endswith(".yaml") or f.name.endswith(".yml")]
    classes_files = [f for f in all_files if f.name == "classes.txt"]

    print("\nConfiguration Files:")
    print(f"  YAML files: {[str(f.relative_to(ds_path)) for f in yaml_files]}")
    print(f"  classes.txt: {[str(f.relative_to(ds_path)) for f in classes_files]}")

    report = {
        "dataset_root": str(ds_path),
        "total_files": len(all_files),
        "total_images": len(images),
        "total_labels": len(labels),
        "detected_format": anno_format,
        "sample_label_file": str(sample_label.relative_to(ds_path)) if sample_label else "",
        "yaml_files": [str(f.relative_to(ds_path)) for f in yaml_files],
        "classes_files": [str(f.relative_to(ds_path)) for f in classes_files],
    }

    return report, ds_path

if __name__ == "__main__":
    download_and_inspect()
