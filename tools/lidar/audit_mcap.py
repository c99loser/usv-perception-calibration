#!/usr/bin/env python3
"""Audit recorded MCAPs and write a deterministic single-LiDAR replay manifest.

Ground-truth topics are inventoried only.  They are never used by this tool to
drive a detector or alter replay selection.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

from preprocessing.filters import ALL_LIDAR_TOPICS, PCD_DTYPE, validate_pointcloud2_layout


ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH_PREFIX = "/world/obstacles/"


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def audit_mcap(path: Path, samples_per_lidar: int = 12) -> dict:
    reader = AnyReader([path])
    reader.open()
    try:
        connections = {connection.id: connection for connection in reader.connections}
        lidar_topics = {topic for topic in ALL_LIDAR_TOPICS if any(c.topic == topic for c in reader.connections)}
        records: dict[str, dict] = defaultdict(lambda: {"timestamps_ns": [], "point_counts": [], "invalid_points": []})
        other_topics = {connection.topic: connection.msgtype for connection in reader.connections}
        for connection, timestamp, data in reader.messages():
            if connection.topic not in lidar_topics:
                continue
            entry = records[connection.topic]
            entry["timestamps_ns"].append(int(timestamp))
            try:
                msg = reader.deserialize(data, connection.msgtype)
                validate_pointcloud2_layout(msg.point_step, msg.row_step, msg.width, msg.height,
                                            msg.is_bigendian, msg.fields)
                raw = np.frombuffer(bytes(msg.data), dtype=PCD_DTYPE)
                finite = np.isfinite(raw["x"]) & np.isfinite(raw["y"]) & np.isfinite(raw["z"])
                nonzero = np.hypot(np.hypot(raw["x"], raw["y"]), raw["z"]) > 0
                entry["point_counts"].append(int(len(raw)))
                entry["invalid_points"].append(int((~(finite & nonzero)).sum()))
            except Exception as exc:
                entry.setdefault("decode_errors", []).append(str(exc))
        lidar_report, replay = {}, []
        for topic in sorted(lidar_topics):
            entry = records[topic]
            stamps = entry["timestamps_ns"]
            intervals_ms = [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:]) if b > a]
            median_interval = _percentile(intervals_ms, 50)
            abnormal = sum(value > max(100.0, 2.5 * median_interval) for value in intervals_ms) if median_interval else 0
            count = len(stamps)
            chosen = sorted({round(index * (count - 1) / max(1, samples_per_lidar - 1)) for index in range(samples_per_lidar)})
            lidar_report[topic] = {
                "messages": count,
                "duration_s": (stamps[-1] - stamps[0]) / 1e9 if count > 1 else 0.0,
                "frame_rate_hz_median": 1000.0 / median_interval if median_interval else None,
                "timestamp_interval_ms": {"p50": median_interval, "p95": _percentile(intervals_ms, 95), "max": max(intervals_ms) if intervals_ms else None},
                "abnormal_gap_frames": abnormal,
                "point_count": {"p05": _percentile(entry["point_counts"], 5), "p50": _percentile(entry["point_counts"], 50), "p95": _percentile(entry["point_counts"], 95)},
                "invalid_point_ratio": float(sum(entry["invalid_points"]) / max(1, sum(entry["point_counts"]))),
                "decode_errors": entry.get("decode_errors", []),
            }
            replay.extend({"topic": topic, "timestamp_ns": stamps[index], "frame_index": index} for index in chosen)
        return {
            "source_mcap": str(path),
            "scope": "single_lidar_only_no_extrinsics_or_fusion",
            "lidar": lidar_report,
            "ground_truth_topics_offline_evaluation_only": sorted(topic for topic in other_topics if topic.startswith(GROUND_TRUTH_PREFIX)),
            "all_topics": other_topics,
            "replay_samples": replay,
        }
    finally:
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "数据")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "lidar_audit" / "audit.json")
    parser.add_argument("--samples-per-lidar", type=int, default=12)
    args = parser.parse_args()
    mcaps = sorted(args.data_root.rglob("*.mcap"))
    if not mcaps:
        raise SystemExit(f"No MCAP files below {args.data_root}")
    report = {"bags": [audit_mcap(path, args.samples_per_lidar) for path in mcaps]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = args.output.with_name("replay_manifest.json")
    manifest.write_text(json.dumps({"scope": "single_lidar_only", "samples": [sample for bag in report["bags"] for sample in bag["replay_samples"]]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output} and {manifest}")


if __name__ == "__main__":
    main()
