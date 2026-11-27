#!/usr/bin/env python3
"""Capture one training-only RGB / semantic-map / LiDAR sample from Gazebo.

This collector is intentionally outside the competition runtime.  The semantic
map is a simulator label source used only to supervise a later LiDAR candidate
extractor; no semantic topic is bridged or read by runtime code.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import gz.transport13 as gzt
import numpy as np
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked
from gz.msgs10.pose_v_pb2 import Pose_V


def stamp_ns(message) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(getattr(stamp, "nsec", 0))


def image_rgb(message: Image) -> np.ndarray:
    raw = np.frombuffer(message.data, dtype=np.uint8)
    expected = int(message.width) * int(message.height) * 3
    if raw.size != expected:
        raise ValueError(f"expected {expected} RGB bytes, received {raw.size}")
    return raw.reshape((int(message.height), int(message.width), 3))


def cloud_xyz(message: PointCloudPacked) -> np.ndarray:
    fields = {field.name: field for field in message.field}
    expected = {name: int(fields[name].datatype) for name in ("x", "y", "z") if name in fields}
    # Gazebo's protobuf enum is not the ROS PointField enum.  Do not assume a
    # numeric value here: the diagnostic is deliberately explicit so the
    # mapping can be verified on the installed simulator version.
    if set(expected) != {"x", "y", "z"}:
        available = {field.name: int(field.datatype) for field in message.field}
        raise ValueError(f"PointCloudPacked lacks x/y/z fields: {available}")
    if len(set(expected.values())) != 1:
        raise ValueError(f"PointCloudPacked x/y/z datatypes differ: {expected}")
    # gz.msgs.PointCloudPacked.Field FLOAT32 is 6 (unlike ROS PointField's
    # FLOAT32=7); this mapping comes from the installed protobuf descriptor.
    if next(iter(expected.values())) != 6:
        raise ValueError(f"PointCloudPacked x/y/z must be Gazebo FLOAT32=6: {expected}")
    count = int(message.width) * int(message.height)
    if count <= 0 or int(message.point_step) <= 0:
        raise ValueError("invalid PointCloudPacked dimensions")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    expected = count * int(message.point_step)
    if raw.size != expected:
        raise ValueError(f"PointCloudPacked data size mismatch: {raw.size} != {expected}")
    rows = raw.reshape(count, int(message.point_step))
    return np.column_stack([rows[:, int(fields[name].offset):int(fields[name].offset) + 4].copy().view("<f4").ravel()
                            for name in ("x", "y", "z")])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rgb-topic", required=True)
    parser.add_argument("--label-topic", required=True)
    parser.add_argument("--lidar-topic", required=True)
    parser.add_argument("--pose-topic", default="")
    parser.add_argument("--max-sync-ms", type=float, default=50.0)
    parser.add_argument("--timeout-seconds", type=float, default=12.0)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite {args.output_root}")
    if args.max_sync_ms <= 0:
        raise SystemExit("max-sync-ms must be positive")

    received: dict[str, dict[int, object]] = {"rgb": {}, "labels": {}, "lidar": {}, "poses": {}}

    def receive(key: str):
        def callback(message) -> None:
            bucket = received[key]
            bucket[stamp_ns(message)] = message
            while len(bucket) > 30:
                del bucket[min(bucket)]
        return callback

    node = gzt.Node()
    if not node.subscribe(Image, args.rgb_topic, receive("rgb")):
        raise RuntimeError(f"could not subscribe to {args.rgb_topic}")
    if not node.subscribe(Image, args.label_topic, receive("labels")):
        raise RuntimeError(f"could not subscribe to {args.label_topic}")
    if not node.subscribe(PointCloudPacked, args.lidar_topic, receive("lidar")):
        raise RuntimeError(f"could not subscribe to {args.lidar_topic}")
    if args.pose_topic and not node.subscribe(Pose_V, args.pose_topic, receive("poses")):
        raise RuntimeError(f"could not subscribe to {args.pose_topic}")

    deadline = time.monotonic() + args.timeout_seconds
    selected: tuple[int, int] | None = None
    max_delta_ns = int(args.max_sync_ms * 1e6)
    while selected is None and time.monotonic() < deadline:
        common = set(received["rgb"]) & set(received["labels"])
        for image_stamp in sorted(common, reverse=True):
            if received["lidar"]:
                lidar_stamp = min(received["lidar"], key=lambda stamp: abs(stamp - image_stamp))
                if abs(lidar_stamp - image_stamp) <= max_delta_ns:
                    selected = image_stamp, lidar_stamp
                    break
        time.sleep(0.02)
    if selected is None:
        raise TimeoutError("no RGB/semantic pair with a synchronized LiDAR cloud")

    image_stamp, lidar_stamp = selected
    rgb, labels = received["rgb"][image_stamp], received["labels"][image_stamp]
    if (rgb.width, rgb.height) != (labels.width, labels.height):
        raise ValueError("RGB and semantic label dimensions differ")
    rgb_pixels, label_pixels = image_rgb(rgb), image_rgb(labels)
    lidar_message = received["lidar"][lidar_stamp]
    points = cloud_xyz(lidar_message)
    if not len(points):
        raise ValueError("captured LiDAR cloud is empty")

    args.output_root.mkdir(parents=True)
    cv2.imwrite(str(args.output_root / "rgb.png"), cv2.cvtColor(rgb_pixels, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.output_root / "labels_map.png"), cv2.cvtColor(label_pixels, cv2.COLOR_RGB2BGR))
    np.savez_compressed(args.output_root / "points.npz", xyz=points,
                        organized_shape=np.asarray([int(lidar_message.height), int(lidar_message.width)], dtype=np.int32))
    pose_snapshot = None
    if args.pose_topic:
        if not received["poses"]:
            raise RuntimeError("no Gazebo pose snapshot received")
        pose_stamp = min(received["poses"], key=lambda stamp: abs(stamp - image_stamp))
        pose_snapshot = []
        for pose in received["poses"][pose_stamp].pose:
            pose_snapshot.append({"name": pose.name,
                                  "position": [pose.position.x, pose.position.y, pose.position.z],
                                  "orientation_xyzw": [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]})
        (args.output_root / "poses.json").write_text(json.dumps({"timestamp_ns": pose_stamp, "poses": pose_snapshot}, indent=2), encoding="utf-8")
    metadata = {
        "purpose": "simulation_training_only", "runtime_allowed": False,
        "rgb_topic": args.rgb_topic, "label_topic": args.label_topic, "lidar_topic": args.lidar_topic,
        "image_timestamp_ns": image_stamp, "lidar_timestamp_ns": lidar_stamp,
        "sync_delta_ms": round(abs(image_stamp - lidar_stamp) / 1e6, 3),
        "width": int(rgb.width), "height": int(rgb.height), "points": int(len(points)),
        "lidar_width": int(lidar_message.width), "lidar_height": int(lidar_message.height),
        "lidar_point_step": int(lidar_message.point_step),
        "pose_topic": args.pose_topic or None, "pose_entity_count": len(pose_snapshot) if pose_snapshot is not None else None,
    }
    (args.output_root / "pair.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
