#!/usr/bin/env python3
"""Build a high-precision cluster dataset from simulation-only point labels.

The clusterer deliberately matches the real-MCAP candidate extractor.  A
cluster is positive only when at least two exact training labels support one
static class.  Callers must declare the label artifact: image-projection
proximity labels are intentionally not accepted as LiDAR supervision.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from competition_user.pointcloud_v2 import PIPELINE_VERSION as V2_VERSION, PointCloudV2Config, cluster_features, process_pointcloud_v2
from competition_user.pointcloud_v3 import PIPELINE_VERSION as V3_VERSION, PointCloudV3Config, process_pointcloud_v3
from competition_user.pointcloud_v4 import PIPELINE_VERSION as V4_VERSION, PointCloudV4Config, process_pointcloud_v4


STATIC_CLASSES = {1: "buoy", 2: "pillar", 3: "block"}
CAMERAS = ("front", "back", "left", "left2", "right", "right2")


def plane_mask(points: np.ndarray, *, tolerance_m: float = 0.12, trials: int = 128) -> np.ndarray:
    if len(points) < 100:
        return np.zeros(len(points), dtype=bool)
    rng, best = np.random.default_rng(17), np.zeros(len(points), dtype=bool)
    for _ in range(trials):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0]); length = np.linalg.norm(normal)
        if length <= 1e-8:
            continue
        normal /= length
        candidate = np.abs((points - sample[0]) @ normal) <= tolerance_m
        if np.sum(candidate) > np.sum(best):
            best = candidate
    return best if np.sum(best) >= max(100, int(0.20 * len(points))) else np.zeros(len(points), dtype=bool)


def components_v2(xyz: np.ndarray) -> list[np.ndarray]:
    processed = process_pointcloud_v2(xyz, PointCloudV2Config())
    return [processed.source_indices[cluster.indices] for cluster in processed.clusters]


def components_v3(xyz: np.ndarray, *, organized_width: int) -> list[np.ndarray]:
    if organized_width <= 0 or len(xyz) % organized_width:
        raise ValueError(f"cannot restore organized scan: points={len(xyz)}, width={organized_width}")
    organized = xyz.reshape((len(xyz) // organized_width, organized_width, 3))
    processed = process_pointcloud_v3(organized, PointCloudV3Config())
    return [processed.source_indices[cluster.indices] for cluster in processed.clusters]


def components_v4(xyz: np.ndarray, *, organized_width: int) -> list[np.ndarray]:
    if organized_width <= 0 or len(xyz) % organized_width:
        raise ValueError(f"cannot restore organized scan: points={len(xyz)}, width={organized_width}")
    organized = xyz.reshape((len(xyz) // organized_width, organized_width, 3))
    processed = process_pointcloud_v4(organized, PointCloudV4Config())
    return [processed.source_indices[cluster.indices] for cluster in processed.clusters]


def components(xyz: np.ndarray, *, voxel_m: float = 0.35) -> list[np.ndarray]:
    finite = np.isfinite(xyz).all(axis=1); ranges = np.linalg.norm(np.where(finite[:, None], xyz, 0.0), axis=1)
    indices = np.flatnonzero(finite & (ranges > 1.0) & (ranges < 30.0)); points = xyz[indices]
    keep = ~plane_mask(points); indices, points = indices[keep], points[keep]
    cells: dict[tuple[int, int, int], list[int]] = {}
    for local, key in enumerate(map(tuple, np.floor(points / voxel_m).astype(np.int32))): cells.setdefault(key, []).append(local)
    remaining, result = set(cells), []
    while remaining:
        seed = remaining.pop(); todo, group = [seed], [seed]
        while todo:
            x, y, z = todo.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        key = (x+dx, y+dy, z+dz)
                        if key in remaining: remaining.remove(key); todo.append(key); group.append(key)
        chosen = np.concatenate([np.asarray(cells[key]) for key in group])
        if len(chosen) >= 12 and np.ptp(points[chosen], axis=0).max() <= 8.0: result.append(indices[chosen])
    return result


def features_v2(cloud: np.ndarray) -> dict[str, float]:
    item = cluster_features(cloud)
    extent, ratios = item["extent_m"], item["pca_ratios"]
    return {"points": float(item["points"]), "range_m": float(item["range_m"]),
            "extent_x": float(extent[0]), "extent_y": float(extent[1]), "extent_z": float(extent[2]),
            "pca1": float(ratios[0]), "pca2": float(ratios[1]), "pca3": float(ratios[2]),
            "linearity": float(item["linearity"]), "planarity": float(item["planarity"]),
            "scattering": float(item["scattering"]), "elongation": float(item["elongation"]),
            "radial_size_m": float(item["radial_size_m"]), "tangential_size_m": float(item["tangential_size_m"]),
            "range_normalized_density": float(item["range_normalized_density"])}


def features(cloud: np.ndarray) -> dict[str, float]:
    low, high = cloud.min(axis=0), cloud.max(axis=0)
    values = np.linalg.eigvalsh(np.cov(cloud, rowvar=False))[::-1]
    return {"points": float(len(cloud)), "range_m": float(np.linalg.norm(cloud.mean(axis=0))),
            "extent_x": float(high[0]-low[0]), "extent_y": float(high[1]-low[1]), "extent_z": float(high[2]-low[2]),
            "pca1": float(values[0]), "pca2": float(values[1]), "pca3": float(values[2]),
            "linearity": float((values[0]-values[1])/max(values[0], 1e-9))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--holdout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-exact-support", type=int, default=2)
    parser.add_argument("--label-filename", default="labeled_points_exact.npz")
    parser.add_argument("--pointcloud-pipeline", choices=("v1", "v2", "v3", "v4"), default="v1")
    parser.add_argument("--organized-width", type=int, default=360,
                        help="required to restore topology of legacy flattened simulation artifacts")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.min_exact_support < 1:
        raise SystemExit("min-exact-support must be positive")
    roots = [path.resolve() for path in args.input_root]
    holdout = args.holdout_root.resolve()
    if holdout not in roots or len(roots) < 2:
        raise SystemExit("holdout-root must be one of at least two input roots")
    rows: list[dict[str, object]] = []
    for root in roots:
        for camera in CAMERAS:
            path = root / f"{camera}_0001" / args.label_filename
            metadata = root / f"{camera}_0001" / "pair.json"
            if not path.is_file() or not metadata.is_file():
                raise SystemExit(f"missing labelled simulation pair: {path}")
            pair = json.loads(metadata.read_text(encoding="utf-8"))
            if pair.get("purpose") != "simulation_training_only" or pair.get("runtime_allowed") is not False:
                raise SystemExit(f"not simulation_training_only: {metadata}")
            data = np.load(path); xyz, semantic = data["xyz"].astype(float), data["semantic_id"].astype(int)
            if args.pointcloud_pipeline == "v4":
                component_groups = components_v4(xyz, organized_width=args.organized_width)
                feature_fn = features_v2
            elif args.pointcloud_pipeline == "v3":
                component_groups = components_v3(xyz, organized_width=args.organized_width)
                feature_fn = features_v2
            elif args.pointcloud_pipeline == "v2":
                component_groups, feature_fn = components_v2(xyz), features_v2
            else:
                component_groups, feature_fn = components(xyz), features
            for local_id, indices in enumerate(component_groups):
                cloud = xyz[indices]; support = Counter(int(value) for value in semantic[indices] if int(value) in STATIC_CLASSES)
                label_id, label_support = (0, 0) if not support else max(support.items(), key=lambda item: item[1])
                # Exact pixels for another class make a cluster ambiguous.
                accepted = label_support >= args.min_exact_support and len(support) == 1
                row = {"source_kind": "simulation_training_only_exact_semantic", "source_root": str(root),
                       "split": "val" if root == holdout else "train", "camera": camera, "cluster_id": local_id,
                       "label_id": label_id if accepted else 0, "label": STATIC_CLASSES.get(label_id, "background"),
                       "exact_support": label_support, "accepted_positive": accepted, **feature_fn(cloud)}
                rows.append(row)
    args.output.mkdir(parents=True)
    fields = list(rows[0]) if rows else []
    with (args.output / "clusters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    positives = [row for row in rows if row["accepted_positive"]]
    report = {"purpose": "simulation_training_only", "runtime_allowed": False,
              "pointcloud_pipeline_version": {"v1": "pointcloud_v1_fixed_voxel_0.35",
                                               "v2": V2_VERSION, "v3": V3_VERSION,
                                               "v4": V4_VERSION}[args.pointcloud_pipeline],
              "organized_width": args.organized_width if args.pointcloud_pipeline in ("v3", "v4") else None,
              "label_rule": {"declared_label_artifact": args.label_filename, "min_exact_support": args.min_exact_support,
                             "static_classes": STATIC_CLASSES, "dynamic_boat_excluded": True},
              "roots": [str(root) for root in roots], "holdout_root": str(holdout), "clusters": len(rows),
              "positive_clusters": len(positives),
              "positive_by_split_class": {split: dict(Counter(row["label"] for row in positives if row["split"] == split)) for split in ("train", "val")}}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
