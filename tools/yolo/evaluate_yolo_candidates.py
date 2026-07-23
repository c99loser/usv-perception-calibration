#!/usr/bin/env python3
"""Evaluate YOLO image candidates with calibrated per-camera confidence gates.

This is offline-only.  It reads labelled images and model predictions, tunes a
threshold on ``val``, and evaluates the held-out ``test`` split.  A dataset
audit can explicitly forbid treating a single-bag temporal result as formal.
"""
from __future__ import annotations

import argparse
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
    score: float = 1.0


def iou(a: Box, b: Box) -> float:
    left, top, right, bottom = max(a.x1, b.x1), max(a.y1, b.y1), min(a.x2, b.x2), min(a.y2, b.y2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def center_error(a: Box, b: Box) -> float:
    return (((a.x1 + a.x2 - b.x1 - b.x2) / 2) ** 2 + ((a.y1 + a.y2 - b.y1 - b.y2) / 2) ** 2) ** 0.5


def match(predictions: list[Box], targets: list[Box], min_iou: float) -> list[tuple[Box, Box]]:
    remaining = list(targets)
    result = []
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        choices = [(iou(prediction, target), index, target) for index, target in enumerate(remaining) if target.cls == prediction.cls]
        if not choices:
            continue
        score, index, target = max(choices, key=lambda item: item[0])
        if score >= min_iou:
            result.append((prediction, target))
            remaining.pop(index)
    return result


def metric(predictions: list[Box], targets: list[Box], min_iou: float) -> dict:
    matches = match(predictions, targets, min_iou)
    errors = [center_error(prediction, target) for prediction, target in matches]
    return {"predictions": len(predictions), "targets": len(targets), "matches": len(matches),
            "precision": len(matches) / len(predictions) if predictions else 0.0,
            "recall": len(matches) / len(targets) if targets else 0.0,
            "median_center_error_px": statistics.median(errors) if errors else None}


def labels(path: Path, width: int, height: int) -> list[Box]:
    result = []
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        cls, cx, cy, box_w, box_h = map(float, line.split())
        result.append(Box(int(cls), (cx - box_w / 2) * width, (cy - box_h / 2) * height,
                          (cx + box_w / 2) * width, (cy + box_h / 2) * height))
    return result


def collect(model, dataset: Path, split: str) -> dict[str, dict[str, list[Box]]]:
    import cv2
    records: dict[str, dict[str, list[Box]]] = defaultdict(lambda: {"predictions": [], "targets": []})
    images = sorted((dataset / "images" / split).glob("*.jpg"))
    for image_path in images:
        camera = image_path.name.split("__", 1)[0]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"cannot read {image_path}")
        height, width = image.shape[:2]
        result = model(str(image_path), imgsz=640, conf=0.001, iou=0.7, verbose=False)[0]
        records[camera]["predictions"].extend(
            Box(int(cls), *box, float(conf))
            for box, conf, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())
        )
        records[camera]["targets"].extend(labels(dataset / "labels" / split / f"{image_path.stem}.txt", width, height))
    return records


def select_threshold(values: dict[str, list[Box]], min_precision: float, min_matches: int, max_error: float, min_iou: float) -> float | None:
    scores = sorted({0.0, *(item.score for item in values["predictions"])})
    viable = []
    for threshold in scores:
        report = metric([item for item in values["predictions"] if item.score >= threshold], values["targets"], min_iou)
        if report["precision"] >= min_precision and report["matches"] >= min_matches and report["median_center_error_px"] is not None and report["median_center_error_px"] <= max_error:
            viable.append((report["matches"], threshold))
    return max(viable)[1] if viable else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--min-matches", type=int, default=20)
    parser.add_argument("--max-center-error", type=float, default=12.0)
    parser.add_argument("--min-iou", type=float, default=0.20)
    args = parser.parse_args()
    from ultralytics import YOLO
    audit = json.loads((args.dataset / "split_audit.json").read_text(encoding="utf-8"))
    model = YOLO(str(args.model))
    validation, test = collect(model, args.dataset, "val"), collect(model, args.dataset, "test")
    report = {"purpose": "offline_yolo_candidate_validation", "model": str(args.model), "dataset": str(args.dataset),
              "split_audit": audit, "gates": {"min_precision": args.min_precision, "min_matches": args.min_matches,
              "max_center_error_px": args.max_center_error, "min_iou": args.min_iou}, "cameras": {}}
    for camera in sorted(set(validation) | set(test)):
        threshold = select_threshold(validation[camera], args.min_precision, args.min_matches, args.max_center_error, args.min_iou)
        heldout = metric([item for item in test[camera]["predictions"] if threshold is not None and item.score >= threshold], test[camera]["targets"], args.min_iou) if threshold is not None else None
        accepted = bool(heldout and heldout["precision"] >= args.min_precision and heldout["matches"] >= args.min_matches and
                        heldout["median_center_error_px"] is not None and heldout["median_center_error_px"] <= args.max_center_error)
        report["cameras"][camera] = {"calibrated_min_confidence": threshold, "heldout": heldout, "accepted": accepted}
    report["accepted_cameras"] = [camera for camera, value in report["cameras"].items() if value["accepted"]]
    report["formal_acceptance"] = bool(audit.get("formal_bag_holdout") and len(report["accepted_cameras"]) == 6)
    report["decision"] = "proceed_to_lidar_camera_multiframe_optimization" if report["formal_acceptance"] else "switch_to_simulation_plus_real_small_sample_finetuning"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"accepted_cameras": report["accepted_cameras"], "decision": report["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
