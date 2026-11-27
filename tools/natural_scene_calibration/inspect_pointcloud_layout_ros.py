#!/usr/bin/env python3
"""Inspect PointCloud2 layout without consuming TF, odometry, or world state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def organized_scan_summary(message) -> dict[str, object]:
    fields = {field.name: field for field in message.fields}
    rows = np.frombuffer(message.data, np.uint8).reshape(message.height, message.width, message.point_step)
    xyz = np.stack([
        rows[:, :, fields[name].offset:fields[name].offset + 4].copy().view("<f4").reshape(message.height, message.width)
        for name in ("x", "y", "z")
    ], axis=-1)
    finite = np.isfinite(xyz).all(axis=2)
    directions = np.zeros_like(xyz)
    np.divide(xyz, np.maximum(np.linalg.norm(xyz, axis=2, keepdims=True), 1e-12),
              out=directions, where=finite[:, :, None])
    horizontal = np.arccos(np.clip(np.sum(directions[:, :-1] * directions[:, 1:], axis=2), -1.0, 1.0))
    vertical = np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=2), -1.0, 1.0))
    valid_h = finite[:, :-1] & finite[:, 1:]
    valid_v = finite[:-1] & finite[1:]
    result: dict[str, object] = {
        "finite_points": int(np.sum(finite)),
        "finite_per_row": np.sum(finite, axis=1).astype(int).tolist(),
        "median_horizontal_step_deg": round(float(np.degrees(np.median(horizontal[valid_h]))), 6) if np.any(valid_h) else None,
        "median_vertical_step_deg": round(float(np.degrees(np.median(vertical[valid_v]))), 6) if np.any(valid_v) else None,
    }
    ring = fields.get("ring")
    if ring is not None and ring.datatype == 4:
        rings = rows[:, :, ring.offset:ring.offset + 2].copy().view("<u2").reshape(message.height, message.width)
        result["dominant_ring_per_row"] = [
            int(np.bincount(row[mask].astype(int)).argmax()) if np.any(mask) else None
            for row, mask in zip(rings, finite)
        ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    point_topics = sorted(name for name, kind in topic_types.items() if kind == "sensor_msgs/msg/PointCloud2")
    reader.set_filter(rosbag2_py.StorageFilter(topics=point_topics))
    unseen = set(point_topics)
    layouts: dict[str, object] = {}
    message_type = get_message("sensor_msgs/msg/PointCloud2")
    while unseen and reader.has_next():
        topic, raw, stamp = reader.read_next()
        if topic not in unseen:
            continue
        message = deserialize_message(raw, message_type)
        layouts[topic] = {
            "stamp_ns": int(stamp),
            "frame_id": str(message.header.frame_id),
            "width": int(message.width),
            "height": int(message.height),
            "point_step": int(message.point_step),
            "row_step": int(message.row_step),
            "is_bigendian": bool(message.is_bigendian),
            "is_dense": bool(message.is_dense),
            "fields": [
                {"name": str(field.name), "offset": int(field.offset), "datatype": int(field.datatype),
                 "count": int(field.count)}
                for field in message.fields
            ],
            "scan_summary": organized_scan_summary(message),
        }
        unseen.remove(topic)
    report = {
        "purpose": "sensor_only_pointcloud_layout_inspection",
        "prohibited_inputs": ["/tf", "/tf_static", "odometry", "/world/*"],
        "bag": str(args.bag),
        "topics": layouts,
        "missing_topics": sorted(unseen),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
