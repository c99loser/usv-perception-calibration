#!/usr/bin/env python3
"""One-shot six-camera evaluation on a bag held out from all training choices."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float


def iou(a: Box, b: Box) -> float:
    left, top, right, bottom = max(a.x1, b.x1), max(a.y1, b.y1), min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union else 0.0


def center_error(a: Box, b: Box) -> float:
    return (((a.x1 + a.x2 - b.x1 - b.x2) / 2) ** 2 + ((a.y1 + a.y2 - b.y1 - b.y2) / 2) ** 2) ** 0.5


def read_labels(path: Path, width: int, height: int) -> list[Box]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cls, cx, cy, box_w, box_h = map(float, line.split())
        result.append(Box(int(cls), (cx - box_w / 2) * width, (cy - box_h / 2) * height,
                          (cx + box_w / 2) * width, (cy + box_h / 2) * height))
    return result


def metric(predictions: list[Box], targets: list[Box], min_iou: float) -> dict[str, float | int | None]:
    remaining = list(targets)
    matches: list[tuple[Box, Box]] = []
    for prediction in predictions:
        options = [(iou(prediction, target), index, target) for index, target in enumerate(remaining)
                   if target.cls == prediction.cls]
        if options:
            score, index, target = max(options, key=lambda item: item[0])
            if score >= min_iou:
                matches.append((prediction, target)); remaining.pop(index)
    errors = [center_error(prediction, target) for prediction, target in matches]
    return {"predictions": len(predictions), "targets": len(targets), "matches": len(matches),
            "precision": len(matches) / len(predictions) if predictions else 0.0,
            "recall": len(matches) / len(targets) if targets else 0.0,
            "median_center_error_px": statistics.median(errors) if errors else None}


def label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = parts.index("images")
    except ValueError as exc:
        raise SystemExit(f"cannot derive labels path from {image}") from exc
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="three-bag config directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.50, help="fixed before holdout evaluation")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--min-matches", type=int, default=20)
    parser.add_argument("--max-center-error", type=float, default=12.0)
    parser.add_argument("--min-iou", type=float, default=0.20)
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("confidence must be within [0, 1]")
    audit = json.loads((args.config / "holdout_audit.json").read_text(encoding="utf-8"))
    if not audit.get("formal_bag_holdout") or audit.get("timestamp_overlap") != 0:
        raise SystemExit("config does not prove a formal disjoint bag holdout")
    with (args.config / "holdout_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("empty holdout manifest")
    from ultralytics import YOLO
    import cv2
    model = YOLO(str(args.model))
    per_camera: dict[str, dict[str, list[Box]]] = defaultdict(lambda: {"predictions": [], "targets": []})
    for row in rows:
        image_path = Path(row["source_image"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"cannot read {image_path}")
        height, width = image.shape[:2]
        label_path = label_for_image(image_path)
        if not label_path.is_file():
            raise SystemExit(f"missing label {label_path}")
        prediction = model(str(image_path), imgsz=args.imgsz, conf=args.confidence, iou=0.7, verbose=False)[0]
        camera = row["camera"]
        per_camera[camera]["predictions"].extend(Box(int(cls), *box)
                                                   for box, cls in zip(prediction.boxes.xyxy.cpu().tolist(), prediction.boxes.cls.cpu().tolist()))
        per_camera[camera]["targets"].extend(read_labels(label_path, width, height))
    gates = {"min_precision": args.min_precision, "min_matches": args.min_matches,
             "max_center_error_px": args.max_center_error, "min_iou": args.min_iou}
    report = {"purpose": "offline_yolo_formal_three_bag_holdout", "model": str(args.model),
              "holdout_config": str(args.config), "holdout_bag": audit["holdout_bag"],
              "fixed_confidence": args.confidence, "confidence_selection": "predeclared; no holdout tuning",
              "gates": gates, "cameras": {}}
    for camera, values in sorted(per_camera.items()):
        summary = metric(values["predictions"], values["targets"], args.min_iou)
        accepted = (summary["precision"] >= args.min_precision and summary["matches"] >= args.min_matches and
                    summary["median_center_error_px"] is not None and summary["median_center_error_px"] <= args.max_center_error)
        report["cameras"][camera] = {"metrics": summary, "accepted": accepted}
    report["accepted_cameras"] = [camera for camera, value in report["cameras"].items() if value["accepted"]]
    report["formal_acceptance"] = len(report["accepted_cameras"]) == 6
    report["decision"] = ("proceed_to_lidar_camera_multiframe_optimization" if report["formal_acceptance"]
                          else "switch_to_simulation_plus_real_small_sample_finetuning")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"accepted_cameras": report["accepted_cameras"], "decision": report["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
