#!/usr/bin/env python3
"""Cache the exact organized PointCloud2 samples referenced by an evaluation JSON.

The cache is intentionally sensor-only.  It stores no TF, odometry, world
state, or semantic labels, and it preserves the original ring/column layout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def decode_xyz(message) -> np.ndarray:
    fields = {field.name: field for field in message.fields}
    if not all(name in fields and fields[name].datatype == 7 for name in ("x", "y", "z")):
        raise RuntimeError("PointCloud2 requires float32 x/y/z fields")
    rows = np.frombuffer(message.data, np.uint8).reshape(
        message.height, message.width, message.point_step
    )
    return np.stack(
        [
            rows[:, :, fields[name].offset : fields[name].offset + 4]
            .copy()
            .view("<f4")
            .reshape(message.height, message.width)
            for name in ("x", "y", "z")
        ],
        axis=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    frames = reference.get("frames", [])
    if not frames:
        raise SystemExit("reference contains no frames")
    camera = str(reference["camera"])
    topic = f"/wamv/sensors/lidars/{camera}_lidar_sensor/points"
    stamps = [int(frame["lidar_stamp_ns"]) for frame in frames]
    # Several camera frames may legitimately select the same nearest LiDAR
    # message. Cache the message once and reproduce the reference sequence;
    # rejecting this would make slower LiDAR streams impossible to evaluate.

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if topic not in topic_types:
        raise SystemExit(f"missing topic: {topic}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    message_type = get_message(topic_types[topic])
    wanted, clouds = set(stamps), {}
    while reader.has_next() and len(clouds) < len(wanted):
        _, raw, stamp = reader.read_next()
        if stamp in wanted:
            clouds[stamp] = decode_xyz(deserialize_message(raw, message_type))
    missing = [stamp for stamp in stamps if stamp not in clouds]
    if missing:
        raise SystemExit(f"missing {len(missing)} referenced LiDAR messages")
    shapes = {clouds[stamp].shape for stamp in stamps}
    if len(shapes) != 1:
        raise SystemExit(f"organized shapes differ: {sorted(shapes)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        xyz=np.stack([clouds[stamp] for stamp in stamps]).astype(np.float32),
        lidar_stamp_ns=np.asarray(stamps, dtype=np.int64),
        image_stamp_ns=np.asarray(
            [int(frame["image_stamp_ns"]) for frame in frames], dtype=np.int64
        ),
        camera=np.asarray(camera),
        topic=np.asarray(topic),
        purpose=np.asarray("sensor_only_organized_pointcloud_evaluation_cache"),
        prohibited_inputs=np.asarray(["/tf", "/tf_static", "odometry", "/world/*"]),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "frames": len(stamps),
                "unique_lidar_messages": len(wanted),
                "organized_shape": list(next(iter(shapes))),
                "camera": camera,
            }
        )
    )


if __name__ == "__main__":
    main()
