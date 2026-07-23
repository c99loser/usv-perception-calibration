#!/usr/bin/env python3
"""Turn an audit report into an explicit calibration-export decision.

This adapter is intentionally fail-closed.  It prevents legacy calibration
callers that expose a bare ``success`` boolean from publishing an ICP result
unless every retained LiDAR edge and every camera validation has passed.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, help="Optional simulated-target projection report; when supplied it is a required gate.")
    parser.add_argument("--max-groundtruth-median-px", type=float, default=5.0)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    lidar = audit.get("lidar_pairs", [])
    camera = audit.get("camera_pairs", [])
    lidar_names = ("front", "back", "left", "left2", "right", "right2", "forward")
    camera_names = ("front", "back", "left", "left2", "right", "right2")
    lidar_fail = [f"{x.get('lidar_a')}–{x.get('lidar_b')}" for x in lidar if not x.get("accepted", False)]
    camera_fail = [f"{x.get('lidar')}–{x.get('camera')}" for x in camera if not x.get("accepted_for_extrinsic", False)]
    seen_lidar = {frozenset((x.get("lidar_a"), x.get("lidar_b"))) for x in lidar}
    missing_lidar = [f"{a}–{b}" for a,b in itertools.combinations(lidar_names,2) if frozenset((a,b)) not in seen_lidar]
    seen_camera = {(x.get("lidar"), x.get("camera")) for x in camera}
    missing_camera = [f"{n}–{n}" for n in camera_names if (n,n) not in seen_camera]
    gt_fail = []
    if args.groundtruth:
        gt = json.loads(args.groundtruth.read_text(encoding="utf-8"))
        by_camera = {x.get("camera"):x for x in gt.get("results", [])}
        for name in camera_names:
            metric = by_camera.get(name, {})
            median = metric.get("median_target_mask_distance_px")
            if median is None or median > args.max_groundtruth_median_px:
                gt_fail.append(name)
    accepted = bool(lidar) and bool(camera) and not lidar_fail and not camera_fail and not missing_lidar and not missing_camera and not gt_fail
    result = {
        "schema": "calibration-export-gate/v1",
        "source_audit": str(args.audit),
        "accepted_for_calibrated_export": accepted,
        "required_policy": "all retained LiDAR pairs and all six LiDAR-camera checks must pass",
        "rejected_lidar_pairs": lidar_fail,
        "rejected_camera_pairs": camera_fail,
        "missing_lidar_pairs": missing_lidar,
        "missing_camera_pairs": missing_camera,
        "groundtruth_projection_report": str(args.groundtruth) if args.groundtruth else None,
        "max_groundtruth_median_px": args.max_groundtruth_median_px if args.groundtruth else None,
        "failed_groundtruth_cameras": gt_fail,
        "accepted_lidar_pairwise_edges": [f"{x.get('lidar_a')}–{x.get('lidar_b')}" for x in lidar if x.get("accepted", False)],
        "accepted_lidar_camera_pairs": [f"{x.get('lidar')}–{x.get('camera')}" for x in camera if x.get("accepted_for_extrinsic", False)],
        "calibrated_extrinsics": {},
        "nominal_unverified_transforms": audit.get("nominal_unverified_transforms", {}),
    }
    if accepted:
        result["next_step"] = "Attach independently estimated extrinsics and preserve this audit as provenance."
    else:
        result["next_step"] = "Do not export ICP/legacy success values. Capture or label common 2D–3D targets, then rerun the audit."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
