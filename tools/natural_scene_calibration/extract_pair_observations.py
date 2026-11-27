#!/usr/bin/env python3
"""Extract time-synchronised YOLO boxes and geometric LiDAR clusters from MCAP.

Only one camera topic, its matching LiDAR PointCloud2 topic and CameraInfo are
opened.  TF, odometry and all `/world/*` topics are intentionally excluded.
The output is an audit input, not an extrinsic result.
"""
from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rosbags.rosbag2.reader import McapReader
from rosbags.typesys import Stores, get_typestore


def camera_topic(camera: str, suffix: str) -> str:
    return f"/wamv/sensors/cameras/{camera}_camera_sensor/{suffix}"


def lidar_topic(camera: str) -> str:
    return f"/wamv/sensors/lidars/{camera}_lidar_sensor/points"


def nearest(values: list[int], stamp: int) -> int:
    index = bisect.bisect_left(values, stamp)
    choices = values[max(0, index - 1):index + 1]
    return min(choices, key=lambda value: abs(value - stamp))


def cluster_points(points: np.ndarray, voxel_m: float = 0.35) -> list[dict[str, object]]:
    valid = points[np.isfinite(points).all(axis=1)]
    ranges = np.linalg.norm(valid, axis=1)
    valid = valid[(ranges > 1.0) & (ranges < 30.0)]
    if not len(valid):
        return []
    keys = np.floor(valid / voxel_m).astype(np.int32)
    cells: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(map(tuple, keys)):
        cells.setdefault(key, []).append(index)
    unseen = set(cells)
    clusters: list[dict[str, object]] = []
    while unseen:
        seed = unseen.pop(); stack = [seed]; component = [seed]
        while stack:
            x, y, z = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        candidate = (x + dx, y + dy, z + dz)
                        if candidate in unseen:
                            unseen.remove(candidate); stack.append(candidate); component.append(candidate)
        indices = np.concatenate([np.asarray(cells[key], dtype=np.int32) for key in component])
        if len(indices) < 12:
            continue
        cloud = valid[indices]
        lo, hi = cloud.min(axis=0), cloud.max(axis=0)
        extent = hi - lo
        if np.max(extent) > 8.0:
            continue
        clusters.append({"points": int(len(cloud)), "center_m": cloud.mean(axis=0).round(4).tolist(),
                         "extent_m": extent.round(4).tolist(), "range_m": round(float(np.linalg.norm(cloud.mean(axis=0))), 4)})
    return sorted(clusters, key=lambda item: (-int(item["points"]), float(item["range_m"])))


def decode_xyz(message) -> np.ndarray:
    fields = {field.name: field for field in message.fields}
    if not all(name in fields and fields[name].datatype == 7 for name in ("x", "y", "z")):
        raise RuntimeError("PointCloud2 does not expose float32 x/y/z")
    count = message.width * message.height
    rows = np.frombuffer(message.data, np.uint8).reshape(count, message.point_step)
    return np.column_stack([rows[:, fields[name].offset:fields[name].offset + 4].copy().view("<f4").ravel()
                            for name in ("x", "y", "z")])


def image_array(message) -> np.ndarray:
    import cv2
    channels = 3
    raw = np.frombuffer(message.data, np.uint8).reshape(message.height, message.width, channels)
    if message.encoding.lower() == "rgb8":
        return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    if message.encoding.lower() == "bgr8":
        return raw
    raise RuntimeError(f"unsupported image encoding {message.encoding!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--max-sync-ms", type=float, default=50.0)
    parser.add_argument("--confidence", type=float, default=0.50)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    image_name, lidar_name, info_name = camera_topic(args.camera, "image_raw"), lidar_topic(args.camera), camera_topic(args.camera, "camera_info")
    store = get_typestore(Stores.ROS2_JAZZY)
    reader = McapReader(args.mcap); reader.open()
    connections = {connection.topic: connection for connection in reader.connections if connection.topic in {image_name, lidar_name, info_name}}
    if set(connections) != {image_name, lidar_name, info_name}:
        raise SystemExit("required direct image, CameraInfo or same-name LiDAR topic is missing")
    time_connections = [connections[image_name], connections[lidar_name]]
    timeline = {image_name: [], lidar_name: []}
    for connection, stamp, _ in reader.messages(time_connections):
        timeline[connection.topic].append(stamp)
    reader.close()
    if not timeline[image_name] or not timeline[lidar_name]:
        raise SystemExit("empty image or LiDAR stream")
    stride = max(1, len(timeline[image_name]) // args.max_frames)
    selected_images = timeline[image_name][::stride][:args.max_frames]
    selected_lidar = {nearest(timeline[lidar_name], stamp) for stamp in selected_images}
    paired = {stamp: nearest(timeline[lidar_name], stamp) for stamp in selected_images}
    bad = [stamp for stamp, other in paired.items() if abs(stamp - other) > args.max_sync_ms * 1e6]
    if bad:
        raise SystemExit(f"{len(bad)} image frames exceed max synchronisation error")
    reader = McapReader(args.mcap); reader.open(); wanted = [connections[image_name], connections[lidar_name], connections[info_name]]
    images, clouds, info = {}, {}, None
    for connection, stamp, raw in reader.messages(wanted):
        if connection.topic == image_name and stamp in paired:
            images[stamp] = store.deserialize_cdr(raw, connection.msgtype)
        elif connection.topic == lidar_name and stamp in selected_lidar:
            clouds[stamp] = store.deserialize_cdr(raw, connection.msgtype)
        elif connection.topic == info_name and info is None:
            info = store.deserialize_cdr(raw, connection.msgtype)
    reader.close()
    from ultralytics import YOLO
    model = YOLO(str(args.model))
    frames = []
    labels = Counter()
    for image_stamp in selected_images:
        lidar_stamp = paired[image_stamp]
        image, cloud = images.get(image_stamp), clouds.get(lidar_stamp)
        if image is None or cloud is None:
            continue
        result = model(image_array(image), imgsz=960, conf=args.confidence, iou=0.7, verbose=False)[0]
        detections = []
        for box, confidence, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            label = model.names[int(cls)]
            if label == "boat":
                continue
            labels[label] += 1
            detections.append({"label": label, "confidence": round(float(confidence), 5), "xyxy": [round(float(v), 3) for v in box]})
        frames.append({"image_stamp_ns": image_stamp, "lidar_stamp_ns": lidar_stamp,
                       "sync_delta_ms": round(abs(image_stamp - lidar_stamp) / 1e6, 3),
                       "detections": detections, "clusters": cluster_points(decode_xyz(cloud))})
    report = {"purpose": "natural_scene_sensor_only_observation_extraction", "prohibited_inputs": ["/tf", "/tf_static", "odometry", "/world/*"],
              "mcap": str(args.mcap), "camera": args.camera, "lidar": args.camera, "model": str(args.model),
              "confidence": args.confidence, "max_sync_ms": args.max_sync_ms,
              "camera_info": {"width": int(info.width), "height": int(info.height), "distortion_model": str(info.distortion_model),
                              "K": [float(value) for value in info.k], "D": [float(value) for value in info.d]} if info else None,
              "frames": frames, "summary": {"selected_frames": len(frames), "detections_by_class": dict(labels),
                                                "clusters": sum(len(frame["clusters"]) for frame in frames)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(frames), "detections": dict(labels), "clusters": report["summary"]["clusters"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
