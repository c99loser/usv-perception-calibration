#!/usr/bin/env python3
"""Label synthetic LiDAR points from exact obstacle collision geometry.

This script is simulation-training-only.  It uses the sampled Gazebo model
poses and declared sphere/cylinder/box collisions.  Camera masks and any real
sensor extrinsic are deliberately not inputs.
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


CLASS_IDS = {"buoy": 1, "pillar": 2, "block": 3}
BUOY_RADIUS_M = 0.25  # Fuel mb_round_buoy_{orange,black} collision sphere.


def quat_matrix(q: list[float]) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=float)
    norm = np.linalg.norm((x, y, z, w))
    if norm <= 0.0:
        raise ValueError("zero quaternion")
    x, y, z, w = np.asarray((x, y, z, w)) / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def surface_distance_sphere(local: np.ndarray, radius: float) -> np.ndarray:
    return np.abs(np.linalg.norm(local, axis=1) - radius)


def surface_distance_box(local: np.ndarray, size_xyz: np.ndarray) -> np.ndarray:
    q = np.abs(local) - size_xyz / 2.0
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return np.abs(outside + inside)


def surface_distance_cylinder(local: np.ndarray, radius: float, length: float) -> np.ndarray:
    q = np.column_stack((np.linalg.norm(local[:, :2], axis=1) - radius, np.abs(local[:, 2]) - length / 2.0))
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return np.abs(outside + inside)


def declared_geometries(world: Path) -> list[dict[str, Any]]:
    root = ET.parse(world).getroot()
    result: list[dict[str, Any]] = []
    for model in root.iter("model"):
        name = model.get("name") or ""
        if name.startswith("pillar_"):
            cylinder = model.find("./link/collision/geometry/cylinder")
            if cylinder is None:
                raise ValueError(f"pillar without cylinder collision: {name}")
            result.append({"name": name, "class_name": "pillar", "shape": "cylinder",
                           "radius": float(cylinder.findtext("radius")), "length": float(cylinder.findtext("length"))})
        elif name.startswith("block_"):
            box = model.find("./link/collision/geometry/box")
            if box is None:
                raise ValueError(f"block without box collision: {name}")
            result.append({"name": name, "class_name": "block", "shape": "box",
                           "size_xyz": [float(value) for value in (box.findtext("size") or "").split()]})
    for include in root.iter("include"):
        name = include.findtext("name") or ""
        uri = include.findtext("uri") or ""
        if name.startswith("buoy_") and "mb_round_buoy_" in uri:
            result.append({"name": name, "class_name": "buoy", "shape": "sphere", "radius": BUOY_RADIUS_M,
                           "model_uri": uri})
    if not any(item["class_name"] == "pillar" for item in result):
        raise ValueError("world has no pillar geometry")
    return result


def sensor_to_world(xyz: np.ndarray, poses: list[dict[str, Any]], camera: str) -> np.ndarray:
    by_name = {item["name"]: item for item in poses}
    wamv = by_name.get("wamv")
    sensors = [item for item in poses if f"{camera}_lidar_airy_dome_visual" in item["name"]]
    if wamv is None or len(sensors) != 1:
        raise ValueError(f"cannot identify WAM-V and one {camera} LiDAR pose")
    sensor = sensors[0]
    finite = np.isfinite(xyz).all(axis=1)
    output = np.full_like(xyz, np.nan, dtype=np.float64)
    root_rotation = quat_matrix(wamv["orientation_xyzw"]); root_translation = np.asarray(wamv["position"], dtype=float)
    sensor_rotation = quat_matrix(sensor["orientation_xyzw"]); sensor_translation = np.asarray(sensor["position"], dtype=float)
    output[finite] = (root_rotation @ ((sensor_rotation @ xyz[finite].T).T + sensor_translation).T).T + root_translation
    return output


def label_points(
    xyz: np.ndarray,
    world_xyz: np.ndarray,
    poses: list[dict[str, Any]],
    geometries: list[dict[str, Any]],
    tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_name = {item["name"]: item for item in poses}
    finite_indices = np.flatnonzero(np.isfinite(world_xyz).all(axis=1))
    label = np.zeros(len(xyz), dtype=np.uint8)
    object_index = np.full(len(xyz), -1, dtype=np.int32)
    nearest = np.full(len(xyz), np.inf, dtype=np.float32)
    for geometry_index, geometry in enumerate(geometries):
        pose = by_name.get(geometry["name"])
        if pose is None:
            raise ValueError(f"sampled pose missing for {geometry['name']}")
        rotation = quat_matrix(pose["orientation_xyzw"])
        translation = np.asarray(pose["position"], dtype=float)
        local = (rotation.T @ (world_xyz[finite_indices] - translation).T).T
        if geometry["shape"] == "sphere":
            distance = surface_distance_sphere(local, float(geometry["radius"]))
        elif geometry["shape"] == "box":
            distance = surface_distance_box(local, np.asarray(geometry["size_xyz"], dtype=float))
        elif geometry["shape"] == "cylinder":
            distance = surface_distance_cylinder(local, float(geometry["radius"]), float(geometry["length"]))
        else:
            raise ValueError(f"unsupported shape: {geometry['shape']}")
        better = distance < nearest[finite_indices]
        selected = finite_indices[better]
        nearest[selected] = distance[better]
        object_index[selected] = geometry_index
    within = np.flatnonzero(nearest <= tolerance_m)
    for point_index in within:
        label[point_index] = CLASS_IDS[geometries[int(object_index[point_index])]["class_name"]]
    return label, object_index, nearest


def build(pair: Path, world: Path, camera: str, output: Path, tolerance_m: float) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    metadata = json.loads((pair / "pair.json").read_text(encoding="utf-8"))
    if metadata.get("purpose") != "simulation_training_only" or metadata.get("runtime_allowed") is not False:
        raise ValueError("pair is not simulation_training_only")
    poses = json.loads((pair / "poses.json").read_text(encoding="utf-8"))["poses"]
    xyz = np.load(pair / "points.npz", allow_pickle=False)["xyz"].astype(np.float64)
    geometries = declared_geometries(world)
    world_xyz = sensor_to_world(xyz, poses, camera)
    labels, object_indices, distances = label_points(xyz, world_xyz, poses, geometries, tolerance_m)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, xyz=xyz.astype(np.float32), semantic_id=labels, world_xyz=world_xyz.astype(np.float32),
                        collision_surface_distance_m=distances, collision_object_index=object_indices)
    counts = Counter(int(value) for value in labels)
    report = {
        "purpose": "simulation_training_only", "runtime_allowed": False,
        "label_source": "sampled_model_pose_plus_declared_collision_geometry",
        "camera_mask_projection_used": False, "world": str(world), "camera": camera,
        "points": int(len(xyz)), "finite_points": int(np.sum(np.isfinite(xyz).all(axis=1))),
        "surface_tolerance_m": tolerance_m,
        "class_points": {name: counts[class_id] for name, class_id in CLASS_IDS.items()},
        "geometry_models": dict(Counter(item["class_name"] for item in geometries)),
        "buoy_collision_contract": {"models": ["mb_round_buoy_orange", "mb_round_buoy_black"], "shape": "sphere", "radius_m": BUOY_RADIUS_M},
    }
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--surface-tolerance-m", type=float, default=0.12)
    args = parser.parse_args()
    if args.surface_tolerance_m <= 0.0:
        raise SystemExit("surface tolerance must be positive")
    report = build(args.pair.resolve(), args.world.resolve(), args.camera, args.output.resolve(), args.surface_tolerance_m)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
