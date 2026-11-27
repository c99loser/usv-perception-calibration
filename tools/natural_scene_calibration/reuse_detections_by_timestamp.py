#!/usr/bin/env python3
"""Reuse frozen image detections when only the point-cloud pipeline changed."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--detections-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    target = json.loads(args.observations.read_text(encoding="utf-8"))
    source = json.loads(args.detections_from.read_text(encoding="utf-8"))
    if target.get("purpose") != "natural_scene_sensor_only_observation_extraction":
        raise SystemExit("target is not a sensor-only observation extraction")
    by_stamp = {(frame["image_stamp_ns"], frame["lidar_stamp_ns"]): frame for frame in source["frames"]}
    for frame in target["frames"]:
        key = (frame["image_stamp_ns"], frame["lidar_stamp_ns"])
        if key not in by_stamp or "detections" not in by_stamp[key]:
            raise SystemExit(f"missing frozen detections for timestamps {key}")
        frame["detections"] = copy.deepcopy(by_stamp[key]["detections"])
    target["detector"] = copy.deepcopy(source.get("detector", {}))
    target["detector_reuse"] = {
        "source": str(args.detections_from),
        "join_key": ["image_stamp_ns", "lidar_stamp_ns"],
        "same_timestamp_pairs": True,
    }
    target["purpose"] = "natural_scene_sensor_only_yolo_lidar_observations"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(target, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(target["frames"]), "same_timestamp_pairs": True}))


if __name__ == "__main__":
    main()
