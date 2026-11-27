#!/usr/bin/env python3
"""Create exact pillar / block labels for a simulation-only LiDAR cloud.

Unlike image-mask projection, this uses the training world's declared collision
geometry and the sampled simulator pose snapshot.  It is deliberately limited
to synthetic data and refuses any artifact not marked ``simulation_training_only``.
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def quat_matrix(q: list[float]) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=float)
    norm = np.linalg.norm((x, y, z, w))
    if norm == 0:
        raise ValueError("zero quaternion")
    x, y, z, w = np.asarray((x, y, z, w)) / norm
    return np.array(((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
                     (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
                     (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y))))


def pillars(world: Path) -> list[tuple[np.ndarray, float, float]]:
    result = []
    for model in ET.parse(world).getroot().iter("model"):
        if not (model.get("name") or "").startswith("pillar_"):
            continue
        pose = [float(value) for value in (model.findtext("pose") or "").split()]
        cylinder = model.find("./link/collision/geometry/cylinder")
        if len(pose) != 6 or cylinder is None:
            raise ValueError(f"invalid pillar model {model.get('name')}")
        radius, length = float(cylinder.findtext("radius")), float(cylinder.findtext("length"))
        result.append((np.asarray(pose[:3], dtype=float), radius, length))
    if not result:
        raise ValueError("world has no pillar collision models")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--surface-tolerance-m", type=float, default=0.12)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.surface_tolerance_m <= 0:
        raise SystemExit("surface-tolerance-m must be positive")
    metadata = json.loads((args.pair / "pair.json").read_text(encoding="utf-8"))
    if metadata.get("purpose") != "simulation_training_only" or metadata.get("runtime_allowed") is not False:
        raise SystemExit("input is not explicitly marked simulation_training_only")
    poses = json.loads((args.pair / "poses.json").read_text(encoding="utf-8"))["poses"]
    by_name = {item["name"]: item for item in poses}
    wamv = by_name.get("wamv")
    sensor_candidates = [item for item in poses if f"{args.camera}_lidar_airy_dome_visual" in item["name"]]
    if wamv is None or len(sensor_candidates) != 1:
        raise SystemExit(f"could not identify WAM-V and one {args.camera} LiDAR sensor pose")
    sensor = sensor_candidates[0]
    xyz = np.load(args.pair / "points.npz")["xyz"].astype(float)
    finite = np.isfinite(xyz).all(axis=1)
    root_r, root_t = quat_matrix(wamv["orientation_xyzw"]), np.asarray(wamv["position"], dtype=float)
    lidar_r, lidar_t = quat_matrix(sensor["orientation_xyzw"]), np.asarray(sensor["position"], dtype=float)
    world_xyz = np.full_like(xyz, np.nan)
    world_xyz[finite] = ((root_r @ ((lidar_r @ xyz[finite].T).T + lidar_t).T).T + root_t)
    label = np.zeros(len(xyz), dtype=np.uint8)
    distance = np.full(len(xyz), np.inf)
    for center, radius, height in pillars(args.world):
        point = world_xyz[finite] - center
        radial = np.linalg.norm(point[:, :2], axis=1)
        dz = np.abs(point[:, 2])
        d = np.maximum(np.abs(radial - radius), np.maximum(dz - height / 2.0, 0.0))
        local = np.flatnonzero(finite)
        better = d < distance[local]
        selected = local[better]
        distance[selected] = d[better]
        label[selected[d[better] <= args.surface_tolerance_m]] = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, xyz=xyz.astype(np.float32), semantic_id=label, world_xyz=world_xyz.astype(np.float32), surface_distance_m=distance.astype(np.float32))
    report = {"purpose": "simulation_training_only", "runtime_allowed": False, "label_source": "world_collision_geometry",
              "world": str(args.world), "camera": args.camera, "points": int(len(xyz)), "finite_points": int(np.sum(finite)),
              "pillar_points": int(np.sum(label == 2)), "pillar_models": len(pillars(args.world)),
              "surface_tolerance_m": args.surface_tolerance_m, "sensor_pose_name": sensor["name"]}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
