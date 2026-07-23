#!/usr/bin/env python3
"""Refuse calibration export unless controlled-data evidence passes every gate.

This intentionally does not estimate transforms.  Estimation may evolve; the
acceptance contract must not.  A candidate JSON is copied to the approved path
only after all six 2D--3D projections and the multi-frame LiDAR pose graph are
validated.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

CAMERAS = {"front", "back", "left", "left2", "right", "right2"}
LIDARS = CAMERAS | {"forward"}
MAX_PIXEL_RMSE = 3.0
MAX_PIXEL_P95 = 6.0
MIN_CORRESPONDENCES = 40
MIN_ICP_FITNESS = 0.15
MAX_ICP_RMSE_M = 0.15
MAX_ICP_TRANSLATION_DELTA_M = 0.75
MAX_ICP_ROTATION_DELTA_DEG = 12.0


def read(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def connected_with_loop(edges: list[dict], errors: list[str]) -> None:
    parent = {node: node for node in LIDARS}
    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    useful = 0
    for edge in edges:
        a, b = edge.get("source"), edge.get("target")
        if a not in LIDARS or b not in LIDARS or a == b:
            fail(errors, f"ICP edge has invalid endpoints: {a!r}->{b!r}")
            continue
        useful += 1
        parent[find(a)] = find(b)
        for key, limit, lower_is_better in (
            ("fitness", MIN_ICP_FITNESS, False),
            ("rmse_m", MAX_ICP_RMSE_M, True),
            ("translation_delta_m", MAX_ICP_TRANSLATION_DELTA_M, True),
            ("rotation_delta_deg", MAX_ICP_ROTATION_DELTA_DEG, True),
        ):
            value = edge.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                fail(errors, f"ICP {a}->{b} lacks finite {key}")
            elif (lower_is_better and value > limit) or (not lower_is_better and value < limit):
                op = "≤" if lower_is_better else "≥"
                fail(errors, f"ICP {a}->{b}: {key}={value} violates {op}{limit}")
        if edge.get("stable") is not True:
            fail(errors, f"ICP {a}->{b} is not marked stable across frames")
    if useful < len(LIDARS):
        fail(errors, "LiDAR pose graph lacks the seven accepted edges required for a closed loop")
    if any(find(node) != find("front") for node in LIDARS):
        fail(errors, "LiDAR pose graph is disconnected")


def validate(args) -> tuple[dict, list[str]]:
    manifest, intrinsics, icp, projections, candidate = map(read, (
        args.manifest, args.intrinsics, args.icp, args.projections, args.candidate))
    errors: list[str] = []
    topics = set(manifest.get("input_topics", []))
    if len(topics) != 13 or any("/world/" in t or t.startswith("/tf") or "odom" in t for t in topics):
        fail(errors, "manifest must contain exactly the 13 permitted production sensor topics and no truth/TF/odom")
    if manifest.get("truth_usage") != "offline_evaluation_only":
        fail(errors, "truth_usage must be exactly 'offline_evaluation_only'")
    if manifest.get("target_manifest_schema") != "controlled-calibration-target-manifest/v1":
        fail(errors, "missing controlled target manifest provenance")

    camera_data = intrinsics.get("cameras", {})
    if set(camera_data) != CAMERAS:
        fail(errors, "intrinsics must contain exactly six named cameras")
    for name, data in camera_data.items():
        if data.get("source") != "controlled_checkerboard_images":
            fail(errors, f"{name} intrinsics are not from controlled checkerboard images")
        if data.get("model") not in {"fisheye", "equidistant"} or not data.get("version"):
            fail(errors, f"{name} has no versioned fisheye/equidistant intrinsic model")

    connected_with_loop(icp.get("edges", []), errors)
    by_pair = {item.get("pair"): item for item in projections.get("pairs", [])}
    if set(by_pair) != CAMERAS:
        fail(errors, "projection evidence must contain exactly six camera/LiDAR pairs")
    for pair, item in by_pair.items():
        for key, limit in (("pixel_rmse", MAX_PIXEL_RMSE), ("pixel_p95", MAX_PIXEL_P95)):
            value = item.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value > limit:
                fail(errors, f"{pair} projection {key} must be finite and ≤{limit}px")
        if item.get("correspondence_count", 0) < MIN_CORRESPONDENCES or item.get("accepted") is not True:
            fail(errors, f"{pair} projection lacks {MIN_CORRESPONDENCES} accepted 2D–3D correspondences")

    if set(candidate.get("camera_extrinsics", {})) != CAMERAS or set(candidate.get("lidar_extrinsics", {})) != LIDARS:
        fail(errors, "candidate must provide all six camera and seven LiDAR extrinsics")
    report = {"schema": "controlled-calibration-gate-report/v1", "passed": not errors,
              "thresholds": {"pixel_rmse": MAX_PIXEL_RMSE, "pixel_p95": MAX_PIXEL_P95,
                             "icp_fitness": MIN_ICP_FITNESS, "icp_rmse_m": MAX_ICP_RMSE_M},
              "errors": errors}
    return report, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--icp", type=Path, required=True)
    parser.add_argument("--projections", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--approved-output", type=Path, required=True)
    args = parser.parse_args()
    report, errors = validate(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors:
        print("EXPORT REJECTED; see", args.report, file=sys.stderr)
        raise SystemExit(2)
    args.approved_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.candidate, args.approved_output)
    print("EXPORT APPROVED:", args.approved_output)


if __name__ == "__main__":
    main()
