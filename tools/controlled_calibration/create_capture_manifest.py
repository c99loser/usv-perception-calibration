#!/usr/bin/env python3
"""Write provenance for a restricted controlled-calibration recording."""
import argparse
import json
from pathlib import Path

LIDARS = ("front", "back", "left", "left2", "right", "right2", "forward")
CAMERAS = ("front", "back", "left", "left2", "right", "right2")

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--layout-index", type=int, required=True)
parser.add_argument("--world", default="perception_user")
args = parser.parse_args()
topics = [f"/wamv/sensors/lidars/{name}_lidar_sensor/points" for name in LIDARS]
topics += [f"/wamv/sensors/cameras/{name}_camera_sensor/image_raw" for name in CAMERAS]
manifest = {"schema": "controlled-calibration-capture/v1", "world": args.world,
            "layout_index": args.layout_index, "target_manifest_schema": "controlled-calibration-target-manifest/v1",
            "target_manifest": "targets/targets.json", "bag": "bag", "input_topics": topics,
            "truth_usage": "offline_evaluation_only", "static_scene": True,
            "note": "No /world, /tf, odometry, camera_info or auxiliary-camera topic is recorded."}
(args.run_dir / "capture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
