#!/usr/bin/env python3
"""Prior-bounded target-free LiDAR-to-fisheye camera candidate optimisation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import differential_evolution, linear_sum_assignment


def rotation(rpy_deg: list[float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(rpy_deg)
    rx = np.array(((1, 0, 0), (0, math.cos(roll), -math.sin(roll)), (0, math.sin(roll), math.cos(roll))))
    ry = np.array(((math.cos(pitch), 0, math.sin(pitch)), (0, 1, 0), (-math.sin(pitch), 0, math.cos(pitch))))
    rz = np.array(((math.cos(yaw), -math.sin(yaw), 0), (math.sin(yaw), math.cos(yaw), 0), (0, 0, 1)))
    return rz @ ry @ rx


SENSOR_TO_OPTICAL = np.array(((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)))


def nominal_transform(camera_rpy: list[float], lidar_rpy: list[float], lidar_minus_camera_m: list[float]) -> tuple[np.ndarray, np.ndarray]:
    camera_to_base = rotation(camera_rpy)
    lidar_to_base = rotation(lidar_rpy)
    sensor_rotation = camera_to_base.T @ lidar_to_base
    sensor_translation = camera_to_base.T @ np.asarray(lidar_minus_camera_m, dtype=float)
    return SENSOR_TO_OPTICAL @ sensor_rotation, SENSOR_TO_OPTICAL @ sensor_translation


def project(points: np.ndarray, rmat: np.ndarray, translation: np.ndarray, K: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = (rmat @ points.T).T + translation
    valid = camera[:, 2] > 0.1
    pixels = np.full((len(points), 2), np.nan)
    if np.any(valid):
        projected, _ = cv2.fisheye.projectPoints(camera[valid].reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), K, D)
        pixels[valid] = projected.reshape(-1, 2)
    return pixels, valid


def box_cost(pixel: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    if not len(boxes):
        return np.empty((len(pixel), 0))
    matrix = np.empty((len(pixel), len(boxes)))
    for i, (u, v) in enumerate(pixel):
        for j, (x1, y1, x2, y2) in enumerate(boxes):
            dx, dy = max(x1 - u, 0.0, u - x2), max(y1 - v, 0.0, v - y2)
            diagonal = max(1.0, math.hypot(x2 - x1, y2 - y1))
            matrix[i, j] = math.hypot(dx, dy) / diagonal
    return matrix


def prepare_frames(payload: dict) -> list[tuple[np.ndarray, list[list[float]], np.ndarray]]:
    result = []
    for frame in payload["frames"]:
        eligible = [
            item
            for item in frame["clusters"]
            if item.get("calibration_quality", {}).get("eligible") is True
            and "calibration_center_m" in item
        ]
        points = np.asarray(
            [item["calibration_center_m"] for item in eligible], dtype=float
        ).reshape((-1, 3))
        weights = np.asarray(
            [
                float(item["calibration_quality"].get("inverse_variance_weight", 1.0))
                for item in eligible
            ],
            dtype=float,
        )
        boxes = [item["xyxy"] for item in frame.get("detections", []) if item["label"] in {"buoy", "pillar", "block"}]
        if len(points) and boxes:
            result.append((points, boxes, weights))
    return result


def residual_score(params: np.ndarray, frames, nominal_r: np.ndarray, nominal_t: np.ndarray, K, D, image_shape: tuple[int, int]) -> tuple[float, int, list[float]]:
    delta_r, _ = cv2.Rodrigues(np.deg2rad(params[:3]))
    rmat, translation = delta_r @ nominal_r, nominal_t + params[3:]
    height, width = image_shape
    objective, accepted, costs = 0.0, 0, []
    for points, boxes, weights in frames:
        pixels, valid = project(points, rmat, translation, K, D)
        on_image = valid & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        if not np.any(on_image):
            objective += 0.35 * min(len(points), len(boxes)); continue
        matrix = box_cost(pixels[on_image], boxes)
        rows, cols = linear_sum_assignment(matrix)
        matched = matrix[rows, cols]
        matched_weights = weights[on_image][rows]
        scale = np.median(matched_weights) if len(matched_weights) else 1.0
        normalized_weights = np.clip(matched_weights / max(scale, 1e-9), 0.25, 4.0)
        good = matched < 0.20
        accepted += int(np.sum(good)); costs.extend(matched[good].tolist())
        objective += float(np.sum(np.minimum(matched, 0.35) * normalized_weights)) + 0.05 * abs(len(points) - len(boxes))
    return objective, accepted, costs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-rpy-deg", nargs=3, type=float, required=True)
    parser.add_argument("--lidar-rpy-deg", nargs=3, type=float, required=True)
    parser.add_argument("--lidar-minus-camera-m", nargs=3, type=float, required=True)
    parser.add_argument("--max-rotation-deg", type=float, default=20.0)
    parser.add_argument("--max-translation-m", type=float, default=1.5)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    payload = json.loads(args.observations.read_text(encoding="utf-8"))
    if payload.get("purpose") != "natural_scene_sensor_only_yolo_lidar_observations":
        raise SystemExit("input must contain frozen detector observations")
    intrinsics = json.loads(args.intrinsics.read_text(encoding="utf-8"))["cameras"][args.camera]
    K, D = np.asarray(intrinsics["K"], dtype=float), np.asarray(intrinsics["D"], dtype=float).reshape(4, 1)
    frames = prepare_frames(payload)
    if len(frames) < 20:
        raise SystemExit("fewer than 20 frames contain both clusters and detections")
    nominal_r, nominal_t = nominal_transform(args.camera_rpy_deg, args.lidar_rpy_deg, args.lidar_minus_camera_m)
    bounds = [(-args.max_rotation_deg, args.max_rotation_deg)] * 3 + [(-args.max_translation_m, args.max_translation_m)] * 3
    shape = (int(payload["camera_info"]["height"]), int(payload["camera_info"]["width"]))
    optimum = differential_evolution(lambda x: residual_score(x, frames, nominal_r, nominal_t, K, D, shape)[0], bounds,
                                     seed=17, maxiter=80, popsize=10, polish=True, workers=1, updating="immediate")
    score, matches, costs = residual_score(optimum.x, frames, nominal_r, nominal_t, K, D, shape)
    result = {"purpose": "natural_scene_prior_bounded_candidate_only", "prohibited_inputs": ["/tf", "/tf_static", "odometry", "/world/*"],
              "observations": str(args.observations), "intrinsics": str(args.intrinsics), "camera": args.camera,
              "prior": {"camera_rpy_deg": args.camera_rpy_deg, "lidar_rpy_deg": args.lidar_rpy_deg,
                        "lidar_minus_camera_m": args.lidar_minus_camera_m, "bounds": {"rotation_deg": args.max_rotation_deg, "translation_m": args.max_translation_m}},
              "corrections": {"rotation_deg": optimum.x[:3].round(5).tolist(), "translation_m": optimum.x[3:].round(5).tolist()},
              "objective": round(float(score), 6), "matched_cluster_box_pairs": matches,
              "median_normalized_box_distance": round(float(np.median(costs)), 6) if costs else None,
              "frames_with_candidates": len(frames),
              "state": "candidate", "accepted": False,
              "next_required": "independent-bag repeatability and offline-only truth evaluation; do not publish or fuse"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
