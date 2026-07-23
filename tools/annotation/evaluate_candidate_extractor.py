#!/usr/bin/env python3
"""Offline-only COCO evaluation for a natural-scene image candidate extractor.

It has no ROS subscriptions and no access to truth/TF.  It compares candidate
boxes with *human* COCO annotations, learns an area-proxy threshold on selected
development bags, and reports held-out precision/localisation gates.  A failed
gate explicitly recommends the model-fine-tuning route rather than permitting
an unverified candidate into calibration.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "camera_fisheye"))
from camera_fisheye_pipeline import CAMERA_NAMES, detect_candidates  # noqa: E402


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float
    score: float = 1.0


def iou(a: Box, b: Box) -> float:
    left, top = max(a.x, b.x), max(a.y, b.y)
    right, bottom = min(a.x + a.width, b.x + b.width), min(a.y + a.height, b.y + b.height)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union else 0.0


def center_error(a: Box, b: Box) -> float:
    return ((a.x + a.width / 2 - b.x - b.width / 2) ** 2 +
            (a.y + a.height / 2 - b.y - b.height / 2) ** 2) ** 0.5


def match(predictions: Iterable[Box], truth: Iterable[Box], *, min_iou: float = 0.20) -> list[tuple[Box, Box]]:
    remaining = list(truth)
    matches: list[tuple[Box, Box]] = []
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        options = [(iou(prediction, target), index, target) for index, target in enumerate(remaining)]
        if not options:
            continue
        overlap, index, target = max(options, key=lambda item: item[0])
        if overlap >= min_iou:
            matches.append((prediction, target))
            remaining.pop(index)
    return matches


def metrics(predictions: list[Box], truth: list[Box], *, min_iou: float = 0.20) -> dict:
    matches = match(predictions, truth, min_iou=min_iou)
    errors = [center_error(prediction, target) for prediction, target in matches]
    return {
        "predictions": len(predictions), "annotations": len(truth), "matches": len(matches),
        "precision": len(matches) / len(predictions) if predictions else 0.0,
        "recall": len(matches) / len(truth) if truth else 0.0,
        "median_center_error_px": statistics.median(errors) if errors else None,
    }


def thresholds(scores: Iterable[float]) -> list[float]:
    values = sorted(set(float(score) for score in scores))
    return [0.0] + values


def choose_threshold(predictions: list[Box], truth: list[Box], min_precision: float,
                     min_matches: int, min_iou: float) -> float | None:
    """Choose the greatest coverage threshold that meets development precision."""
    viable: list[tuple[int, float]] = []
    for threshold in thresholds(item.score for item in predictions):
        report = metrics([item for item in predictions if item.score >= threshold], truth, min_iou=min_iou)
        if report["precision"] >= min_precision and report["matches"] >= min_matches:
            viable.append((report["matches"], threshold))
    return max(viable)[1] if viable else None


def load_labels(path: Path) -> tuple[list[dict], dict[int, list[Box]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = list(payload.get("images", []))
    labels: dict[int, list[Box]] = {int(item["id"]): [] for item in images}
    for annotation in payload.get("annotations", []):
        if annotation.get("iscrowd", 0):
            continue
        x, y, width, height = annotation["bbox"]
        labels.setdefault(int(annotation["image_id"]), []).append(Box(float(x), float(y), float(width), float(height)))
    return images, labels


def extract_predictions(dataset_root: Path, images: list[dict]) -> dict[int, list[Box]]:
    import cv2
    result: dict[int, list[Box]] = {}
    for item in images:
        image = cv2.imread(str(dataset_root / item["file_name"]))
        if image is None:
            raise RuntimeError(f"Missing image: {item['file_name']}")
        _debug, detections = detect_candidates(image)
        result[int(item["id"])] = [Box(item.x, item.y, item.width, item.height, item.score) for item in detections]
    return result


def aggregate(ids: Iterable[int], predictions: dict[int, list[Box]], labels: dict[int, list[Box]], threshold: float,
              min_iou: float) -> dict:
    candidate = [item for image_id in ids for item in predictions[image_id] if item.score >= threshold]
    truth = [item for image_id in ids for item in labels[image_id]]
    return metrics(candidate, truth, min_iou=min_iou)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-bag", required=True, help="source_bag value reserved for final validation")
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--max-median-center-error-px", type=float, default=12.0)
    parser.add_argument("--min-matches-per-camera", type=int, default=20)
    parser.add_argument("--min-iou", type=float, default=0.20)
    args = parser.parse_args()
    images, labels = load_labels(args.coco)
    if not any(labels.values()):
        raise SystemExit("COCO contains no annotations; annotate images before evaluation.")
    predictions = extract_predictions(args.dataset_root, images)
    report = {"purpose": "offline_candidate_extractor_validation_only", "heldout_bag": args.heldout,
              "gates": {"min_precision": args.min_precision, "max_median_center_error_px": args.max_median_center_error_px,
                        "min_matches_per_camera": args.min_matches_per_camera, "min_iou": args.min_iou}, "cameras": {}}
    for camera in CAMERA_NAMES:
        camera_images = [item for item in images if item["camera"] == camera]
        development = [int(item["id"]) for item in camera_images if item["source_bag"] != args.heldout_bag]
        heldout = [int(item["id"]) for item in camera_images if item["source_bag"] == args.heldout_bag]
        development_predictions = [item for image_id in development for item in predictions[image_id]]
        development_truth = [item for image_id in development for item in labels[image_id]]
        threshold = choose_threshold(development_predictions, development_truth, args.min_precision,
                                     args.min_matches_per_camera, args.min_iou)
        heldout_metrics = aggregate(heldout, predictions, labels, threshold, args.min_iou) if threshold is not None else None
        accepted = bool(heldout_metrics and heldout_metrics["precision"] >= args.min_precision and
                        heldout_metrics["matches"] >= args.min_matches_per_camera and
                        heldout_metrics["median_center_error_px"] is not None and
                        heldout_metrics["median_center_error_px"] <= args.max_median_center_error_px)
        report["cameras"][camera] = {"calibrated_min_score": threshold, "heldout": heldout_metrics, "accepted": accepted}
    accepted_cameras = [camera for camera, value in report["cameras"].items() if value["accepted"]]
    report["accepted_cameras"] = accepted_cameras
    report["decision"] = ("candidate_extractor_may_proceed_to_temporal_and_lidar_association_validation"
                          if len(accepted_cameras) == len(CAMERA_NAMES)
                          else "switch_to_simulation_plus_manual_annotation_model_finetuning")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"accepted_cameras": accepted_cameras, "decision": report["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
