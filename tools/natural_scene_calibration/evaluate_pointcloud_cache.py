#!/usr/bin/env python3
"""Evaluate a point-cloud pipeline on an immutable organized-scan cache."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from competition_user.pointcloud_v2 import FixedEchoMaskBuilder, PointCloudV2Config
from competition_user.pointcloud_v3 import (
    PIPELINE_VERSION as V3_VERSION,
    PointCloudV3Config,
    process_pointcloud_v3,
)


def load_config(kind: str, path: Path | None):
    values = {} if path is None else json.loads(path.read_text(encoding="utf-8"))
    if kind == "v3":
        return PointCloudV3Config(**values)
    if kind == "v4":
        from competition_user.pointcloud_v4 import PointCloudV4Config

        return PointCloudV4Config(**values)
    raise ValueError(kind)


def process(kind: str, cloud: np.ndarray, config, echo_mask):
    if kind == "v3":
        return process_pointcloud_v3(cloud, config, echo_mask=echo_mask)
    from competition_user.pointcloud_v4 import process_pointcloud_v4

    return process_pointcloud_v4(cloud, config, echo_mask=echo_mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pipeline", choices=("v3", "v4"), required=True)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-cluster-points", action="store_true")
    parser.add_argument("--max-export-points", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    cache = np.load(args.cache)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    xyz = cache["xyz"].astype(float)
    lidar_stamps = cache["lidar_stamp_ns"].astype(np.int64).tolist()
    image_stamps = cache["image_stamp_ns"].astype(np.int64).tolist()
    expected = [
        (int(frame["image_stamp_ns"]), int(frame["lidar_stamp_ns"]))
        for frame in reference["frames"]
    ]
    if list(zip(image_stamps, lidar_stamps)) != expected:
        raise SystemExit("cache/reference timestamps differ")

    echo_builder = FixedEchoMaskBuilder(PointCloudV2Config())
    for cloud in xyz:
        echo_builder.observe(cloud)
    echo_mask = echo_builder.frozen_mask()
    config = load_config(args.pipeline, args.config_json)
    frames = []
    for index, cloud in enumerate(xyz):
        processed = process(args.pipeline, cloud, config, echo_mask)
        clusters = []
        for cluster in processed.clusters:
            item = dict(cluster.features)
            if args.include_cluster_points:
                points = processed.points[cluster.indices]
                stride = max(1, int(np.ceil(len(points) / args.max_export_points)))
                item["points_sensor_m"] = points[::stride].round(4).tolist()
            clusters.append(item)
        clusters.sort(key=lambda item: (-int(item["points"]), float(item["range_m"])))
        source = reference["frames"][index]
        frames.append(
            {
                "image_stamp_ns": image_stamps[index],
                "lidar_stamp_ns": lidar_stamps[index],
                "sync_delta_ms": source["sync_delta_ms"],
                "image": source["image"],
                "clusters": clusters,
                "pointcloud_audit": processed.audit,
            }
        )

    version = V3_VERSION if args.pipeline == "v3" else frames[0]["pointcloud_audit"]["pipeline_version"]
    result = {
        "purpose": "natural_scene_sensor_only_observation_extraction",
        "prohibited_inputs": ["/tf", "/tf_static", "odometry", "/world/*"],
        "bag": reference.get("bag"),
        "camera": reference["camera"],
        "lidar": reference.get("lidar", reference["camera"]),
        "camera_info": reference.get("camera_info"),
        "cache": str(args.cache),
        "cluster_points_exported_for_manual_annotation": bool(args.include_cluster_points),
        "pointcloud_pipeline_version": version,
        "self_echo_mask": {
            "source": "selected_sensor_only_cached_frames",
            "frames": echo_builder.frames,
            "cells": [[int(value) for value in cell] for cell in sorted(echo_mask)],
        },
        "frame_selection": reference.get("frame_selection"),
        "start_index": reference.get("start_index"),
        "frames": frames,
        "summary": {
            "selected_frames": len(frames),
            "clusters": sum(len(frame["clusters"]) for frame in frames),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
