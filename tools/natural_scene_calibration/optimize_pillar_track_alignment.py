#!/usr/bin/env python3
"""Prior-bounded LiDAR/camera candidate alignment using pillar trajectories.

Every association is constrained to be consistent over a 3-D LiDAR candidate
track and a 2-D frozen-detector pillar track.  This tool accepts no TF,
odometry, world state, or ground truth and always emits ``candidate`` only.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import differential_evolution, linear_sum_assignment


SENSOR_TO_OPTICAL = np.array(((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)))


def rotation(rpy_deg: list[float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(rpy_deg)
    rx = np.array(((1, 0, 0), (0, math.cos(roll), -math.sin(roll)), (0, math.sin(roll), math.cos(roll))))
    ry = np.array(((math.cos(pitch), 0, math.sin(pitch)), (0, 1, 0), (-math.sin(pitch), 0, math.cos(pitch))))
    rz = np.array(((math.cos(yaw), -math.sin(yaw), 0), (math.sin(yaw), math.cos(yaw), 0), (0, 0, 1)))
    return rz @ ry @ rx


def nominal_transform(camera_rpy: list[float], lidar_rpy: list[float], lidar_minus_camera_m: list[float]) -> tuple[np.ndarray, np.ndarray]:
    camera_to_base, lidar_to_base = rotation(camera_rpy), rotation(lidar_rpy)
    return SENSOR_TO_OPTICAL @ camera_to_base.T @ lidar_to_base, SENSOR_TO_OPTICAL @ camera_to_base.T @ np.asarray(lidar_minus_camera_m, float)


def project(point: np.ndarray, rmat: np.ndarray, translation: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray | None:
    optical = rmat @ point + translation
    if optical[2] <= 0.1:
        return None
    return cv2.fisheye.projectPoints(optical.reshape(1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), K, D)[0].reshape(2)


def iou(a: list[float], b: list[float]) -> float:
    left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, right-left) * max(0.0, bottom-top)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union else 0.0


def centre_cost(pixel: np.ndarray, box: list[float]) -> float:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return float(math.hypot(pixel[0] - cx, pixel[1] - cy) / max(1.0, math.hypot(box[2] - box[0], box[3] - box[1])))


@dataclass
class Track:
    track_id: int
    entries: list[tuple[int, object]] = field(default_factory=list)


@dataclass(frozen=True)
class LidarObservation:
    point: np.ndarray
    weight: float


def lidar_tracks(payload: dict, *, min_length: int) -> list[Track]:
    tracks: dict[int, Track] = {}
    last: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    next_id = 1
    for frame_index, frame in enumerate(payload["frames"]):
        assigned: set[int] = set()
        for local in frame.get("pillar_candidate_indices", []):
            cluster = frame["clusters"][local]
            quality = cluster.get("calibration_quality", {})
            if quality.get("eligible") is not True or "calibration_center_m" not in cluster:
                continue
            point = np.asarray(cluster["calibration_center_m"], float)
            extent = np.asarray(cluster["extent_m"], float)
            temporal = cluster.get("temporal_track", {})
            embedded_id = temporal.get("track_id")
            if embedded_id is not None:
                track_id = int(embedded_id)
                if track_id in assigned:
                    raise ValueError(f"duplicate temporal track_id {track_id} in frame")
                tracks.setdefault(track_id, Track(track_id))
                weight = float(quality.get("inverse_variance_weight", 1.0))
                tracks[track_id].entries.append(
                    (frame_index, LidarObservation(point, weight))
                )
                assigned.add(track_id)
                continue
            candidates = [(float(np.linalg.norm(point - old[0]) + .25*np.linalg.norm(extent-old[1])), tid)
                          for tid, old in last.items() if tid not in assigned and np.linalg.norm(point-old[0]) <= 1.5 and np.linalg.norm(extent-old[1]) <= 1.0]
            track_id = min(candidates)[1] if candidates else next_id
            if track_id == next_id:
                tracks[track_id] = Track(track_id); next_id += 1
            weight = float(quality.get("inverse_variance_weight", 1.0))
            tracks[track_id].entries.append(
                (frame_index, LidarObservation(point, weight))
            )
            last[track_id] = (point, extent); assigned.add(track_id)
    return [track for track in tracks.values() if len(track.entries) >= min_length]


def image_tracks(payload: dict, *, min_length: int) -> list[Track]:
    tracks: dict[int, Track] = {}; last: dict[int, list[float]] = {}; next_id = 1
    for frame_index, frame in enumerate(payload["frames"]):
        assigned: set[int] = set()
        boxes = [item["xyxy"] for item in frame.get("detections", []) if item["label"] == "pillar"]
        for box in boxes:
            candidates = [(iou(box, previous), tid) for tid, previous in last.items() if tid not in assigned]
            score, track_id = max(candidates, default=(0.0, next_id))
            if score < 0.05:
                track_id = next_id
            if track_id == next_id:
                tracks[track_id] = Track(track_id); next_id += 1
            tracks[track_id].entries.append((frame_index, box)); last[track_id] = box; assigned.add(track_id)
    return [track for track in tracks.values() if len(track.entries) >= min_length]


def pair_cost(lidar: Track, image: Track, rmat, translation, K, D, shape: tuple[int, int]) -> tuple[float, int]:
    lmap, imap = dict(lidar.entries), dict(image.entries); common = sorted(set(lmap) & set(imap)); costs = []; weights = []
    height, width = shape
    for index in common:
        observation = lmap[index]
        pixel = project(observation.point, rmat, translation, K, D)
        if pixel is None or not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
            costs.append(2.0)
        else:
            costs.append(min(2.0, centre_cost(pixel, imap[index])))
        weights.append(max(float(observation.weight), 1e-9))
    if not costs:
        return 2.0, len(common)
    order = np.argsort(costs)
    ordered_costs = np.asarray(costs, dtype=float)[order]
    ordered_weights = np.asarray(weights, dtype=float)[order]
    middle = 0.5 * float(np.sum(ordered_weights))
    selected = int(np.searchsorted(np.cumsum(ordered_weights), middle, side="left"))
    return float(ordered_costs[min(selected, len(ordered_costs) - 1)]), len(common)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--camera", required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-rpy-deg", nargs=3, type=float, required=True); parser.add_argument("--lidar-rpy-deg", nargs=3, type=float, required=True)
    parser.add_argument("--lidar-minus-camera-m", nargs=3, type=float, required=True)
    parser.add_argument("--min-track-length", type=int, default=6); parser.add_argument("--max-rotation-deg", type=float, default=10.0); parser.add_argument("--max-translation-m", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists(): raise SystemExit(f"refusing to overwrite {args.output}")
    payload = json.loads(args.observations.read_text(encoding="utf-8"))
    if payload.get("purpose") != "natural_scene_sensor_only_yolo_lidar_pillar_candidates": raise SystemExit("input lacks frozen pillar candidate stage")
    ltracks, itracks = lidar_tracks(payload, min_length=args.min_track_length), image_tracks(payload, min_length=args.min_track_length)
    if len(ltracks) < 3 or len(itracks) < 3: raise SystemExit(f"insufficient persistent tracks: lidar={len(ltracks)}, image={len(itracks)}")
    intrinsics = json.loads(args.intrinsics.read_text(encoding="utf-8"))["cameras"][args.camera]
    K, D = np.asarray(intrinsics["K"], float), np.asarray(intrinsics["D"], float).reshape(4, 1)
    nominal_r, nominal_t = nominal_transform(args.camera_rpy_deg, args.lidar_rpy_deg, args.lidar_minus_camera_m)
    shape = int(payload["camera_info"]["height"]), int(payload["camera_info"]["width"])
    def evaluate(params: np.ndarray):
        delta, _ = cv2.Rodrigues(np.deg2rad(params[:3])); rmat, translation = delta @ nominal_r, nominal_t + params[3:]
        costs = np.full((len(ltracks), len(itracks)), 2.0); coverage = np.zeros_like(costs, dtype=int)
        for row, lidar in enumerate(ltracks):
            for column, image in enumerate(itracks): costs[row, column], coverage[row, column] = pair_cost(lidar, image, rmat, translation, K, D, shape)
        rows, columns = linear_sum_assignment(costs); matched = costs[rows, columns]
        return float(np.sum(np.minimum(matched, 1.5))), rows, columns, matched, coverage[rows, columns]
    bounds = [(-args.max_rotation_deg, args.max_rotation_deg)]*3 + [(-args.max_translation_m, args.max_translation_m)]*3
    optimum = differential_evolution(lambda x: evaluate(x)[0], bounds, seed=23, maxiter=70, popsize=10, polish=True, workers=1, updating="immediate")
    _, rows, columns, costs, coverage = evaluate(optimum.x)
    result = {"purpose": "natural_scene_track_constrained_candidate_only", "prohibited_inputs": ["/tf", "/tf_static", "odometry", "/world/*"],
              "observations": str(args.observations), "camera": args.camera, "tracks": {"lidar": len(ltracks), "image": len(itracks), "min_length": args.min_track_length},
              "corrections": {"rotation_deg": optimum.x[:3].round(5).tolist(), "translation_m": optimum.x[3:].round(5).tolist()},
              "matches": [{"lidar_track": ltracks[row].track_id, "image_track": itracks[column].track_id, "median_normalized_center_error": round(float(cost), 6), "overlap_frames": int(count)} for row, column, cost, count in zip(rows, columns, costs, coverage)],
              "state": "candidate", "accepted": False, "next_required": "independent-bag correction agreement and offline-only truth evaluation"}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result))


if __name__ == "__main__": main()
