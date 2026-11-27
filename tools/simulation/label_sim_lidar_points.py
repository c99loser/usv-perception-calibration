#!/usr/bin/env python3
"""Project synthetic LiDAR points into a training-only semantic fisheye map.

The supplied transform is simulator installation geometry and is used solely
to make synthetic supervision.  This script refuses non-training artifacts;
it must never be used on an MCAP or in competition runtime.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


SENSOR_TO_OPTICAL = np.array(((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)))
VALID_LABELS = {0, 1, 2, 3, 4}


def rotation(rpy_deg: list[float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(rpy_deg)
    rx = np.array(((1, 0, 0), (0, math.cos(roll), -math.sin(roll)), (0, math.sin(roll), math.cos(roll))))
    ry = np.array(((math.cos(pitch), 0, math.sin(pitch)), (0, 1, 0), (-math.sin(pitch), 0, math.cos(pitch))))
    rz = np.array(((math.cos(yaw), -math.sin(yaw), 0), (math.sin(yaw), math.cos(yaw), 0), (0, 0, 1)))
    return rz @ ry @ rx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-rpy-deg", nargs=3, type=float, required=True)
    parser.add_argument("--lidar-rpy-deg", nargs=3, type=float, required=True)
    parser.add_argument("--lidar-minus-camera-m", nargs=3, type=float, required=True)
    parser.add_argument("--projection", choices=("opencv_fisheye", "gazebo_custom_tan"), required=True)
    parser.add_argument("--horizontal-fov-rad", type=float, default=3.054)
    parser.add_argument("--custom-c2", type=float, default=4.0)
    parser.add_argument("--semantic-neighborhood-px", type=float, default=0.0,
                        help="Training-only semantic mask proximity; 0 requires exact pixel overlap.")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    metadata = json.loads((args.pair / "pair.json").read_text(encoding="utf-8"))
    if metadata.get("purpose") != "simulation_training_only" or metadata.get("runtime_allowed") is not False:
        raise SystemExit("input is not explicitly marked simulation_training_only")
    xyz = np.load(args.pair / "points.npz")["xyz"].astype(float)
    labels_image = cv2.imread(str(args.pair / "labels_map.png"), cv2.IMREAD_COLOR)
    if labels_image is None or labels_image.ndim != 3 or not np.array_equal(labels_image[..., 0], labels_image[..., 1]) or not np.array_equal(labels_image[..., 1], labels_image[..., 2]):
        raise SystemExit("semantic labels_map must be grayscale RGB")
    semantic = labels_image[..., 0]
    unknown = set(int(value) for value in np.unique(semantic)) - VALID_LABELS
    if unknown:
        raise SystemExit(f"unknown semantic ids: {sorted(unknown)}")
    intrinsics = json.loads(args.intrinsics.read_text(encoding="utf-8"))["cameras"][args.camera]
    K = np.asarray(intrinsics["K"], dtype=float)
    D = np.asarray(intrinsics["D"], dtype=float).reshape(4, 1)
    r = SENSOR_TO_OPTICAL @ rotation(args.camera_rpy_deg).T @ rotation(args.lidar_rpy_deg)
    t = SENSOR_TO_OPTICAL @ rotation(args.camera_rpy_deg).T @ np.asarray(args.lidar_minus_camera_m, dtype=float)
    finite = np.isfinite(xyz).all(axis=1)
    optical = np.full_like(xyz, np.nan)
    optical[finite] = (r @ xyz[finite].T).T + t
    forward = finite & (optical[:, 2] > 0.1)
    pixels = np.full((len(xyz), 2), np.nan)
    height, width = semantic.shape
    if args.projection == "opencv_fisheye":
        pixels[forward] = cv2.fisheye.projectPoints(optical[forward].reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), K, D)[0].reshape(-1, 2)
    else:
        if args.horizontal_fov_rad <= 0 or args.custom_c2 <= 0:
            raise SystemExit("custom wide-angle parameters must be positive")
        direction = optical[forward]
        radial = np.linalg.norm(direction[:, :2], axis=1)
        theta = np.arctan2(radial, direction[:, 2])
        scale = (width / 2.0) / math.tan(args.horizontal_fov_rad / (2.0 * args.custom_c2))
        radius = scale * np.tan(theta / args.custom_c2)
        unit = np.divide(direction[:, :2], radial[:, None], out=np.zeros_like(direction[:, :2]), where=radial[:, None] > 1e-9)
        pixels[forward] = unit * radius[:, None] + np.array((width / 2.0, height / 2.0))
    on_image = forward & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    point_labels = np.zeros(len(xyz), dtype=np.uint8)
    point_labels[on_image] = semantic[pixels[on_image, 1].astype(int), pixels[on_image, 0].astype(int)]
    exact_semantic = point_labels.copy()
    if args.semantic_neighborhood_px < 0:
        raise SystemExit("semantic-neighborhood-px must be non-negative")
    ambiguous = np.zeros(len(xyz), dtype=bool)
    if args.semantic_neighborhood_px:
        # Sparse LiDAR beams do not always land on a few-pixel semantic
        # region.  Use only the nearest *unambiguous* class within a measured
        # tolerance; background and ties remain negative labels.
        distance_by_class = np.stack([
            cv2.distanceTransform((semantic != class_id).astype(np.uint8), cv2.DIST_L2, 3)
            for class_id in range(1, 5)
        ])
        rows, columns = pixels[on_image, 1].astype(int), pixels[on_image, 0].astype(int)
        distances = distance_by_class[:, rows, columns]
        best = np.argmin(distances, axis=0)
        sorted_distance = np.sort(distances, axis=0)
        close = sorted_distance[0] <= args.semantic_neighborhood_px
        unique = (sorted_distance[1] - sorted_distance[0]) > 1.0
        selected = np.flatnonzero(on_image)
        point_labels[selected[close & unique]] = (best[close & unique] + 1).astype(np.uint8)
        ambiguous[selected[close & ~unique]] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, xyz=xyz.astype(np.float32), semantic_id=point_labels, uv=pixels.astype(np.float32))
    rgb = cv2.imread(str(args.pair / "rgb.png"), cv2.IMREAD_COLOR)
    if rgb is None:
        raise SystemExit("missing rgb.png")
    overlay = rgb.copy()
    palette = {1: (0, 255, 255), 2: (0, 165, 255), 3: (255, 0, 255), 4: (0, 0, 255)}
    for index in np.flatnonzero(on_image & (point_labels > 0))[::max(1, int(np.sum(on_image & (point_labels > 0)) // 1500))]:
        cv2.circle(overlay, tuple(np.round(pixels[index]).astype(int)), 1, palette[int(point_labels[index])], -1)
    overlay_path = args.output.with_suffix(".overlay.jpg")
    cv2.imwrite(str(overlay_path), overlay)
    report = {"purpose": "simulation_training_only", "runtime_allowed": False, "pair": str(args.pair),
              "points": int(len(xyz)), "forward_points": int(np.sum(forward)), "on_image_points": int(np.sum(on_image)),
              "semantic_points": {str(key): value for key, value in sorted(Counter(point_labels.tolist()).items())},
              "exact_semantic_points": {str(key): value for key, value in sorted(Counter(exact_semantic.tolist()).items())},
              "ambiguous_neighborhood_points": int(np.sum(ambiguous)), "semantic_neighborhood_px": args.semantic_neighborhood_px,
              "installation_geometry": {"camera_rpy_deg": args.camera_rpy_deg, "lidar_rpy_deg": args.lidar_rpy_deg,
                                        "lidar_minus_camera_m": args.lidar_minus_camera_m},
              "projection": {"kind": args.projection, "horizontal_fov_rad": args.horizontal_fov_rad, "custom_c2": args.custom_c2},
              "overlay": str(overlay_path)}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
