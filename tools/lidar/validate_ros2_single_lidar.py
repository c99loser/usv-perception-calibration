#!/usr/bin/env python3
"""Run one deterministic, real-MCAP integration check for the ROS 2 baseline.

This test deliberately starts one perception node and feeds it one original
PointCloud2 topic.  It does not publish TF, consume any other LiDAR topic, or
perform extrinsic calibration/fusion.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


SENSOR_NAMES = ("front", "back", "left", "left2", "right", "right2", "forward")


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path, help="MCAP bag directory")
    parser.add_argument("--sensor", choices=SENSOR_NAMES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--messages", type=int, default=8)
    parser.add_argument("--rate-hz", type=float, default=2.0)
    args = parser.parse_args()
    if args.messages < 1 or args.rate_hz <= 0:
        parser.error("--messages must be positive and --rate-hz must be > 0")

    topic = f"/wamv/sensors/lidars/{args.sensor}_lidar_sensor/points"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    node_log = args.output_dir / "node.log"
    status_log = args.output_dir / "status.yaml"
    replay_log = args.output_dir / "replay.log"
    report_path = args.output_dir / "report.json"
    node_cmd = ["ros2", "run", "competition_user", "single_lidar_perception", "--ros-args", "-p", f"input_topic:={topic}"]
    replay_cmd = [sys.executable, "tools/lidar/ros2_mcap_lidar_replay.py", str(args.bag), "--topic", topic,
                  "--max-messages", str(args.messages), "--rate-hz", str(args.rate_hz)]

    started = time.monotonic()
    with node_log.open("w", encoding="utf-8") as node_file, status_log.open("w", encoding="utf-8") as status_file:
        node = subprocess.Popen(node_cmd, stdout=node_file, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            # This is outside the node callback.  It gives DDS discovery a bounded setup window.
            time.sleep(3.0)
            status = subprocess.Popen(["ros2", "topic", "echo", "--once", "/perception/single_lidar/status"],
                                      stdout=status_file, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            with replay_log.open("w", encoding="utf-8") as replay_file:
                replay = subprocess.run(replay_cmd, stdout=replay_file, stderr=subprocess.STDOUT, text=True, timeout=90)
            try:
                status.wait(timeout=8)
            except subprocess.TimeoutExpired:
                stop_process(status)
        finally:
            stop_process(node)

    node_text = node_log.read_text(encoding="utf-8", errors="replace")
    status_text = status_log.read_text(encoding="utf-8", errors="replace")
    replay_text = replay_log.read_text(encoding="utf-8", errors="replace")
    processed = [int(value) for value in re.findall(r"processed=(\d+)", node_text)]
    # ros2 topic echo abbreviates long string payloads with "..." by default.
    # Treat delivery as the integration assertion and extract only the stable
    # prefix fields that remain visible in abbreviated output.
    status_received = "data:" in status_text and '"scope": "single_lidar_only"' in status_text
    status_payload = None
    if status_received:
        status_payload = {"scope": "single_lidar_only"}
        for key in ("frame_id", "detections", "tracks"):
            match = re.search(rf'"{key}":\s*("[^"]*"|\d+)', status_text)
            if match:
                value = match.group(1)
                status_payload[key] = int(value) if value.isdigit() else value.strip('"')
    sent_match = re.search(r"published (\d+) recorded PointCloud2", replay_text)
    sent = int(sent_match.group(1)) if sent_match else 0
    report = {
        "scope": "single_lidar_only_no_extrinsics_no_fusion",
        "sensor": args.sensor,
        "input_topic": topic,
        "bag": str(args.bag),
        "requested_messages": args.messages,
        "published_messages": sent,
        "processed_log_markers": processed,
        "status_received": status_received,
        "status": status_payload,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "passed": sent == args.messages and bool(processed) and status_received,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
