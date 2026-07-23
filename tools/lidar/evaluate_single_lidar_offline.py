#!/usr/bin/env python3
"""Evaluate the fixed replay manifest using truth exclusively offline.

The transform/odometry and /world/obstacles streams are read here only.  This
script is deliberately outside ``competition_user`` ROS runtime nodes.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

from competition_user.lidar_baseline import SingleLidarPipeline
from competition_user.offline_evaluation import SingleLidarEvaluator, TruthObject
from preprocessing.filters import PCD_DTYPE, validate_pointcloud2_layout

ROOT = Path(__file__).resolve().parents[2]
TRUTH_TOPICS = {f"/world/obstacles/{name}": name[:-1] if name.endswith("s") else name
                for name in ("buoys", "pillars", "blocks", "boats")}


def matrix(translation, rotation) -> np.ndarray:
    x, y, z, w = rotation.x, rotation.y, rotation.z, rotation.w
    xx, yy, zz = x*x, y*y, z*z; xy, xz, yz = x*y, x*z, y*z; wx, wy, wz = w*x, w*y, w*z
    out = np.array([[1-2*(yy+zz), 2*(xy-wz), 2*(xz+wy), translation.x],
                    [2*(xy+wz), 1-2*(xx+zz), 2*(yz-wx), translation.y],
                    [2*(xz-wy), 2*(yz+wx), 1-2*(xx+yy), translation.z], [0, 0, 0, 1]], dtype=float)
    return out


def truth_bev(objects: list[TruthObject], extent: float, resolution: float) -> np.ndarray:
    size = int(round(2 * extent / resolution)); image = np.zeros((size, size), dtype=np.uint8)
    for item in objects:
        x, y = np.floor((item.center[:2] + extent) / resolution).astype(int)
        if 0 <= x < size and 0 <= y < size: image[y, x] = 255
    return image


def sensor_name(topic: str) -> str:
    return topic.split("/")[-2]


def visible(objects: list[TruthObject], topic: str) -> list[TruthObject]:
    maximum = 200.0 if "forward_lidar" in topic else 60.0
    result = []
    for item in objects:
        distance = np.linalg.norm(item.center[:2])
        angle = abs(np.degrees(np.arctan2(item.center[1], item.center[0])))
        if 0.5 <= distance <= maximum and ("forward_lidar" not in topic or angle <= 62.0): result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "output" / "lidar_audit" / "replay_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "single_lidar_offline_evaluation" / "report.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    selected = defaultdict(set)
    for sample in json.loads(args.manifest.read_text(encoding="utf-8"))["samples"][:args.limit]: selected[sample["topic"]].add(sample["timestamp_ns"])
    evaluators, pipelines = {}, {topic: SingleLidarPipeline() for topic in selected}
    for topic in selected: evaluators[topic] = SingleLidarEvaluator()
    for path in sorted((ROOT / "数据").rglob("*.mcap")):
        reader = AnyReader([path]); reader.open(); base_world = None; sensor_base = {}; truth_world = {}; pending = []
        try:
            for connection, stamp, data in reader.messages():
                if connection.topic == "/wamv/sensors/position/ground_truth_odometry":
                    odom = reader.deserialize(data, connection.msgtype)
                    base_world = matrix(odom.pose.pose.position, odom.pose.pose.orientation)
                elif connection.topic in ("/wamv/pose", "/tf"):
                    message = reader.deserialize(data, connection.msgtype)
                    for transform in message.transforms:
                        child = transform.child_frame_id.split("/")[-1]
                        if child.endswith("lidar_sensor"): sensor_base[child] = matrix(transform.transform.translation, transform.transform.rotation)
                elif connection.topic in TRUTH_TOPICS:
                    message = reader.deserialize(data, connection.msgtype)
                    truth_world[TRUTH_TOPICS[connection.topic]] = [np.array([pose.position.x, pose.position.y, pose.position.z, 1.0]) for pose in message.poses]
                if stamp not in selected.get(connection.topic, set()): continue
                msg = reader.deserialize(data, connection.msgtype)
                validate_pointcloud2_layout(msg.point_step, msg.row_step, msg.width, msg.height, msg.is_bigendian, msg.fields)
                pending.append((connection.topic, stamp, np.frombuffer(bytes(msg.data), dtype=PCD_DTYPE).copy()))
            # MCAP ordering can place the first LiDAR sample before the first
            # odometry/TF sample.  For these static replay scenes, evaluate it
            # after the state streams have been observed; runtime never sees it.
            for topic, stamp, raw in pending:
                if base_world is None or sensor_name(topic) not in sensor_base:
                    continue
                result = pipelines[topic].process(raw, topic, stamp)
                world_sensor = base_world @ sensor_base[sensor_name(topic)]
                sensor_world = np.linalg.inv(world_sensor)
                truth = [TruthObject((sensor_world @ point)[:3], label) for label, points in truth_world.items() for point in points]
                truth = visible(truth, topic)
                evaluators[topic].update(result.detections, result.tracks, truth, result.bev,
                                         truth_bev(truth, pipelines[topic].bev_extent_m, pipelines[topic].bev_resolution_m), result.timings_ms)
        finally:
            reader.close()
    report = {"offline_evaluation_only": True, "provenance": "world obstacles + ground-truth odometry + recorded TF; never runtime input",
              "by_lidar_topic": {topic: evaluator.summary() for topic, evaluator in evaluators.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__": main()
