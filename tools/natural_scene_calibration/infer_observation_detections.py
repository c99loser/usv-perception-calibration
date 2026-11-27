#!/usr/bin/env python3
"""Attach frozen raw-fisheye detector results to sensor-only observation JSON."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.50)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("confidence must be within [0, 1]")
    payload = json.loads(args.observations.read_text(encoding="utf-8"))
    if payload.get("purpose") != "natural_scene_sensor_only_observation_extraction":
        raise SystemExit("input is not a sensor-only observation extraction")
    from ultralytics import YOLO
    model = YOLO(str(args.model)); counts = Counter()
    for frame in payload["frames"]:
        result = model(frame["image"], imgsz=960, conf=args.confidence, iou=0.7, verbose=False)[0]
        detections = []
        for box, conf, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            label = str(model.names[int(cls)])
            if label == "boat":
                continue
            counts[label] += 1
            detections.append({"label": label, "confidence": round(float(conf), 5),
                               "xyxy": [round(float(value), 3) for value in box]})
        frame["detections"] = detections
    payload["detector"] = {"model": str(args.model), "confidence": args.confidence,
                           "excluded_dynamic_class": "boat", "detections_by_class": dict(counts)}
    payload["purpose"] = "natural_scene_sensor_only_yolo_lidar_observations"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(payload["frames"]), "detections": dict(counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
