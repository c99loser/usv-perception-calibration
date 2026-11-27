#!/usr/bin/env python3
"""ROS-Jazzy extraction of sensor-only image/point-cloud observations from a bag."""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from competition_user.pointcloud_v2 import FixedEchoMaskBuilder, PIPELINE_VERSION as V2_VERSION, PointCloudV2Config, process_pointcloud_v2
from competition_user.pointcloud_v3 import PIPELINE_VERSION as V3_VERSION, PointCloudV3Config, process_pointcloud_v3
from competition_user.pointcloud_v4 import PIPELINE_VERSION as V4_VERSION, PointCloudV4Config, process_pointcloud_v4


def camera_topic(camera: str, suffix: str) -> str:
    return f"/wamv/sensors/cameras/{camera}_camera_sensor/{suffix}"


def lidar_topic(camera: str) -> str:
    return f"/wamv/sensors/lidars/{camera}_lidar_sensor/points"


def nearest(values: list[int], stamp: int) -> int:
    index = bisect.bisect_left(values, stamp)
    return min(values[max(0, index - 1):index + 1], key=lambda item: abs(item - stamp))


def decode_xyz(message, *, organized: bool = False) -> np.ndarray:
    fields = {field.name: field for field in message.fields}
    if not all(name in fields and fields[name].datatype == 7 for name in ("x", "y", "z")):
        raise RuntimeError("PointCloud2 requires float32 x/y/z")
    rows = np.frombuffer(message.data, np.uint8).reshape(message.height, message.width, message.point_step)
    xyz = np.stack([
        rows[:, :, fields[name].offset:fields[name].offset + 4].copy().view("<f4").reshape(message.height, message.width)
        for name in ("x", "y", "z")
    ], axis=-1)
    return xyz if organized else xyz.reshape((-1, 3))


def remove_dominant_plane(points: np.ndarray, *, tolerance_m: float = 0.12, trials: int = 128) -> np.ndarray:
    """Deterministically suppress one dominant water/ground plane only.

    This is not SOR/ROR.  A plane is removed only when it explains at least
    20% of the retained cloud, which prevents a sparse object face from being
    treated as the scene plane.
    """
    if len(points) < 100:
        return points
    rng, best = np.random.default_rng(17), np.zeros(len(points), dtype=bool)
    for _ in range(trials):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length <= 1e-8:
            continue
        normal /= length
        inliers = np.abs((points - sample[0]) @ normal) <= tolerance_m
        if np.sum(inliers) > np.sum(best):
            best = inliers
    return points[~best] if np.sum(best) >= max(100, int(0.20 * len(points))) else points


def clusters(points: np.ndarray, voxel_m: float = 0.35, *, include_points: bool = False,
             max_export_points: int = 512,
             echo_mask: frozenset[tuple[int, int, int]] = frozenset()) -> list[dict[str, object]]:
    return clusters_v1_with_audit(points, voxel_m=voxel_m, include_points=include_points,
                                  max_export_points=max_export_points)[0]


def clusters_v1_with_audit(points: np.ndarray, *, voxel_m: float = 0.35, include_points: bool = False,
                           max_export_points: int = 512) -> tuple[list[dict[str, object]], dict[str, object]]:
    raw_count = len(points)
    points = points[np.isfinite(points).all(axis=1)]
    ranges = np.linalg.norm(points, axis=1); points = points[(ranges > 1.0) & (ranges < 30.0)]
    before_plane = len(points); points = remove_dominant_plane(points)
    cells: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(map(tuple, np.floor(points / voxel_m).astype(np.int32))):
        cells.setdefault(key, []).append(index)
    remaining, result = set(cells), []
    while remaining:
        seed = remaining.pop(); todo, component = [seed], [seed]
        while todo:
            x, y, z = todo.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        key = (x + dx, y + dy, z + dz)
                        if key in remaining:
                            remaining.remove(key); todo.append(key); component.append(key)
        selected = np.concatenate([np.asarray(cells[key]) for key in component])
        if len(selected) < 12:
            continue
        cloud = points[selected]; extent = np.ptp(cloud, axis=0)
        if extent.max() > 8.0:
            continue
        center = cloud.mean(axis=0); values, vectors = np.linalg.eigh(np.cov(cloud, rowvar=False))
        order = np.argsort(values)[::-1]; values, axis = values[order], vectors[:, order[0]]
        if axis[np.argmax(np.abs(axis))] < 0: axis = -axis
        item = {"points": int(len(cloud)), "center_m": center.round(4).tolist(), "extent_m": extent.round(4).tolist(),
                "range_m": round(float(np.linalg.norm(center)), 4), "pca_eigenvalues_m2": values.round(6).tolist(),
                "pca_principal_axis": axis.round(6).tolist(),
                "pca_linearity": round(float((values[0]-values[1])/max(values[0], 1e-9)), 6)}
        if include_points:
            stride = max(1, int(np.ceil(len(cloud) / max_export_points)))
            item["points_sensor_m"] = cloud[::stride].round(4).tolist()
        result.append(item)
    result.sort(key=lambda item: (-int(item["points"]), float(item["range_m"])))
    audit = {"pipeline_version": "pointcloud_v1_fixed_voxel_0.35", "raw_points": raw_count,
             "range_filtered_points": before_plane, "dominant_plane_removed_points": before_plane-len(points),
             "final_clusters": len(result)}
    return result, audit


def clusters_with_audit(points: np.ndarray, *, include_points: bool = False,
                        max_export_points: int = 512,
                        echo_mask: frozenset[tuple[int, int, int]] = frozenset()) -> tuple[list[dict[str, object]], dict[str, object]]:
    # ``voxel_m`` remains in the signature only for compatibility with older
    # callers.  V2 parameters come exclusively from the versioned config.
    processed = process_pointcloud_v2(points, PointCloudV2Config(), echo_mask=echo_mask)
    result = []
    for cluster in processed.clusters:
        item = dict(cluster.features)
        cloud = processed.points[cluster.indices]
        if include_points:
            stride = max(1, int(np.ceil(len(cloud) / max_export_points)))
            item["points_sensor_m"] = cloud[::stride].round(4).tolist()
        result.append(item)
    return sorted(result, key=lambda item: (-int(item["points"]), float(item["range_m"]))), processed.audit


def clusters_v3_with_audit(points: np.ndarray, *, include_points: bool = False,
                           max_export_points: int = 512,
                           echo_mask: frozenset[tuple[int, int, int]] = frozenset()) -> tuple[list[dict[str, object]], dict[str, object]]:
    processed = process_pointcloud_v3(points, PointCloudV3Config(), echo_mask=echo_mask)
    result = []
    for cluster in processed.clusters:
        item = dict(cluster.features)
        cloud = processed.points[cluster.indices]
        if include_points:
            stride = max(1, int(np.ceil(len(cloud) / max_export_points)))
            item["points_sensor_m"] = cloud[::stride].round(4).tolist()
        result.append(item)
    return sorted(result, key=lambda item: (-int(item["points"]), float(item["range_m"]))), processed.audit


def clusters_v4_with_audit(points: np.ndarray, *, include_points: bool = False,
                           max_export_points: int = 512,
                           echo_mask: frozenset[tuple[int, int, int]] = frozenset()) -> tuple[list[dict[str, object]], dict[str, object]]:
    processed = process_pointcloud_v4(points, PointCloudV4Config(), echo_mask=echo_mask)
    result = []
    for cluster in processed.clusters:
        item = dict(cluster.features)
        cloud = processed.points[cluster.indices]
        if include_points:
            stride = max(1, int(np.ceil(len(cloud) / max_export_points)))
            item["points_sensor_m"] = cloud[::stride].round(4).tolist()
        result.append(item)
    return sorted(result, key=lambda item: (-int(item["points"]), float(item["range_m"]))), processed.audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--max-sync-ms", type=float, default=50.0)
    parser.add_argument("--frame-selection", choices=("contiguous", "uniform"), default="contiguous",
                        help="contiguous is required for tracking; uniform is annotation-coverage only")
    parser.add_argument("--start-index", type=int, default=0,
                        help="eligible-pair index at which a contiguous tracking window begins")
    parser.add_argument("--include-cluster-points", action="store_true",
                        help="training/annotation only: include bounded direct LiDAR points per cluster")
    parser.add_argument("--pointcloud-pipeline", choices=("v1", "v2", "v3", "v4"), default="v1")
    args = parser.parse_args()
    image_name, lidar_name, info_name = camera_topic(args.camera, "image_raw"), lidar_topic(args.camera), camera_topic(args.camera, "camera_info")
    image_dir = args.output.parent / f"{args.output.stem}_images"
    if args.output.exists() or image_dir.exists():
        raise SystemExit("refusing to overwrite output JSON or image directory")
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    def open_reader():
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
        topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
        reader.set_filter(rosbag2_py.StorageFilter(topics=[image_name, lidar_name, info_name]))
        return reader, topic_types

    reader, _ = open_reader(); times = {image_name: [], lidar_name: []}
    while reader.has_next():
        topic, _, stamp = reader.read_next()
        if topic in times:
            times[topic].append(stamp)
    if not times[image_name] or not times[lidar_name]:
        raise SystemExit("required image or LiDAR stream is empty")
    eligible = [(stamp, nearest(times[lidar_name], stamp)) for stamp in times[image_name]]
    eligible = [(image, lidar) for image, lidar in eligible if abs(image - lidar) <= args.max_sync_ms * 1e6]
    if not eligible:
        raise SystemExit("no image/LiDAR pair satisfies the requested synchronisation bound")
    if args.start_index < 0:
        raise SystemExit("start-index must be non-negative")
    if args.frame_selection == "contiguous":
        eligible = eligible[args.start_index:args.start_index + args.max_frames]
    else:
        step = max(1, len(eligible) // args.max_frames)
        eligible = eligible[::step][:args.max_frames]
    if not eligible:
        raise SystemExit("requested frame selection is empty")
    image_stamps = [image for image, _ in eligible]
    paired = dict(eligible)
    wanted_lidar = set(paired.values()); reader, types = open_reader()
    image_type, lidar_type, info_type = get_message(types[image_name]), get_message(types[lidar_name]), get_message(types[info_name])
    images, clouds, info = {}, {}, None
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        if topic == image_name and stamp in paired:
            images[stamp] = deserialize_message(raw, image_type)
        elif topic == lidar_name and stamp in wanted_lidar:
            clouds[stamp] = deserialize_message(raw, lidar_type)
        elif topic == info_name and info is None:
            info = deserialize_message(raw, info_type)
    image_dir.mkdir(parents=True)
    echo_builder = FixedEchoMaskBuilder(PointCloudV2Config())
    for lidar_stamp in wanted_lidar:
        if lidar_stamp in clouds:
            echo_builder.observe(decode_xyz(clouds[lidar_stamp]))
    echo_mask = echo_builder.frozen_mask()
    frames = []
    for image_stamp in image_stamps:
        lidar_stamp = paired[image_stamp]; image, cloud = images.get(image_stamp), clouds.get(lidar_stamp)
        if image is None or cloud is None:
            continue
        raw = np.frombuffer(image.data, np.uint8).reshape(image.height, image.width, 3)
        bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR) if image.encoding.lower() == "rgb8" else raw
        image_path = image_dir / f"{image_stamp}.jpg"
        if not cv2.imwrite(str(image_path), bgr):
            raise RuntimeError(f"cannot write {image_path}")
        processor = {"v1": clusters_v1_with_audit, "v2": clusters_with_audit,
                     "v3": clusters_v3_with_audit, "v4": clusters_v4_with_audit}[args.pointcloud_pipeline]
        kwargs = {"include_points": args.include_cluster_points}
        if args.pointcloud_pipeline in ("v2", "v3", "v4"):
            kwargs["echo_mask"] = echo_mask
        decoded = decode_xyz(cloud, organized=args.pointcloud_pipeline in ("v3", "v4"))
        frame_clusters, pointcloud_audit = processor(decoded, **kwargs)
        frames.append({"image_stamp_ns": image_stamp, "lidar_stamp_ns": lidar_stamp,
                       "sync_delta_ms": round(abs(image_stamp - lidar_stamp) / 1e6, 3),
                       "image": str(image_path), "clusters": frame_clusters,
                       "pointcloud_audit": pointcloud_audit})
    result = {"purpose": "natural_scene_sensor_only_observation_extraction", "prohibited_inputs": ["/tf", "/tf_static", "odometry", "/world/*"],
              "bag": str(args.bag), "camera": args.camera, "lidar": args.camera,
              "camera_info": {"width": int(info.width), "height": int(info.height), "distortion_model": str(info.distortion_model),
                              "K": [float(value) for value in info.k], "D": [float(value) for value in info.d]},
              "cluster_points_exported_for_manual_annotation": bool(args.include_cluster_points),
              "pointcloud_pipeline_version": {"v1": "pointcloud_v1_fixed_voxel_0.35",
                                               "v2": V2_VERSION, "v3": V3_VERSION,
                                               "v4": V4_VERSION}[args.pointcloud_pipeline],
              "self_echo_mask": {"source": "selected_sensor_only_frames", "frames": echo_builder.frames,
                                   "cells": [[int(value) for value in cell] for cell in sorted(echo_mask)]},
              "frame_selection": args.frame_selection, "start_index": args.start_index,
              "frames": frames, "summary": {"selected_frames": len(frames), "clusters": sum(len(frame["clusters"]) for frame in frames)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
