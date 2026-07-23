#!/usr/bin/env python3
"""Run the deterministic replay manifest through the single-LiDAR baseline."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

from competition_user.lidar_baseline import SingleLidarPipeline
from preprocessing.filters import PCD_DTYPE, validate_pointcloud2_layout


ROOT = Path(__file__).resolve().parents[2]


def overview(raw: np.ndarray, result, extent_m: float = 30.0, pixels: int = 600) -> np.ndarray:
    """Top-down diagnostic: raw gray, filtered white, cluster colors and tracks."""
    image = np.zeros((pixels, pixels, 3), dtype=np.uint8)
    scale = pixels / (2 * extent_m)
    def project(points):
        xy = np.asarray(points)[:, :2]
        out = np.empty((len(xy), 2), dtype=int)
        out[:, 0] = np.floor((xy[:, 0] + extent_m) * scale).astype(int)
        out[:, 1] = np.floor((extent_m - xy[:, 1]) * scale).astype(int)
        return out
    raw_xyz = np.column_stack((raw["x"], raw["y"], raw["z"]))
    finite = np.isfinite(raw_xyz).all(axis=1)
    for x, y in project(raw_xyz[finite]):
        if 0 <= x < pixels and 0 <= y < pixels: image[y, x] = (50, 50, 50)
    for x, y in project(result.filtered_xyz):
        if 0 <= x < pixels and 0 <= y < pixels: image[y, x] = (210, 210, 210)
    colors = ((0, 120, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0))
    for index, cluster in enumerate(result.clusters):
        color = colors[index % len(colors)]
        points = project(cluster)
        for x, y in points:
            if 0 <= x < pixels and 0 <= y < pixels: image[y, x] = color
        low, high = project(np.vstack((cluster.min(axis=0), cluster.max(axis=0))))
        cv2.rectangle(image, tuple(np.minimum(low, high)), tuple(np.maximum(low, high)), color, 1)
    for track in result.tracks:
        x, y = project(np.asarray([track.center]))[0]
        cv2.putText(image, str(track.track_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.line(image, (pixels // 2, 0), (pixels // 2, pixels - 1), (80, 80, 80), 1)
    cv2.line(image, (0, pixels // 2), (pixels - 1, pixels // 2), (80, 80, 80), 1)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "output" / "lidar_audit" / "replay_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "single_lidar_baseline")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for quick smoke runs.")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected: dict[str, set[int]] = defaultdict(set)
    for sample in manifest["samples"][:args.limit]:
        selected[sample["topic"]].add(sample["timestamp_ns"])
    args.output.mkdir(parents=True, exist_ok=True)
    pipelines = {topic: SingleLidarPipeline() for topic in selected}
    frames, timing, labels = [], defaultdict(list), defaultdict(int)
    mcaps = sorted((ROOT / "数据").rglob("*.mcap"))
    for path in mcaps:
        reader = AnyReader([path]); reader.open()
        try:
            for connection, timestamp, data in reader.messages():
                if timestamp not in selected.get(connection.topic, set()):
                    continue
                msg = reader.deserialize(data, connection.msgtype)
                validate_pointcloud2_layout(msg.point_step, msg.row_step, msg.width, msg.height,
                                            msg.is_bigendian, msg.fields)
                raw = np.frombuffer(bytes(msg.data), dtype=PCD_DTYPE)
                result = pipelines[connection.topic].process(raw, connection.topic, timestamp)
                for key, value in result.timings_ms.items(): timing[key].append(value)
                for item in result.detections: labels[item.label] += 1
                name = f"{Path(path).stem}_{connection.topic.split('/')[-3]}_{timestamp}"
                cv2.imwrite(str(args.output / f"{name}_bev.png"), result.bev)
                cv2.imwrite(str(args.output / f"{name}_overview.png"), overview(raw, result))
                frames.append({"topic": connection.topic, "timestamp_ns": timestamp,
                               "raw_points": len(raw), "filtered_points": len(result.filtered_xyz),
                               "detections": len(result.detections), "tracks": len(result.tracks),
                               "timings_ms": result.timings_ms, "bev": f"{name}_bev.png",
                               "overview": f"{name}_overview.png"})
        finally:
            reader.close()
    report = {"scope": "single_lidar_only_no_extrinsics_or_fusion", "frames": frames,
              "detection_labels": dict(labels),
              "timing_ms": {key: {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))} for key, values in timing.items()}}
    (args.output / "replay_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"processed {len(frames)} replay frames into {args.output}")


if __name__ == "__main__":
    main()
