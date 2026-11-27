#!/usr/bin/env python3
"""Attach a frozen simulation-validated pillar-candidate rule to MCAP clusters.

The rule may only emit a candidate flag.  It consumes no TF, odometry, world
state, or ground truth and cannot mark an external calibration as accepted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FEATURES = ("points", "range_m", "extent_x", "extent_y", "extent_z", "pca1", "pca2", "pca3", "linearity")


def feature(cluster: dict) -> list[float]:
    extent = cluster["extent_m"]
    values = cluster["pca_eigenvalues_m2"]
    return [float(cluster["points"]), float(cluster["range_m"]), *map(float, extent), *map(float, values), float(cluster["pca_linearity"])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--rule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    payload = json.loads(args.observations.read_text(encoding="utf-8"))
    if payload.get("purpose") != "natural_scene_sensor_only_yolo_lidar_observations":
        raise SystemExit("input must be frozen sensor-only YOLO/LiDAR observations")
    rule = json.loads(args.rule.read_text(encoding="utf-8"))
    if rule.get("accepted_for_real_mcap_candidate_stage") is not True:
        raise SystemExit("simulation rule is not admitted to the real-MCAP candidate stage")
    params = rule["rule"]
    if tuple(params["kind"] for _ in [0]) != ("scaled_1nn_positive_vs_negative",) or tuple(rule["features"]) != FEATURES:
        raise SystemExit("unsupported or mismatched frozen rule")
    center, scale = np.asarray(params["center"], float), np.asarray(params["scale"], float)
    positive, negative = np.asarray(params["positive_prototypes"], float), np.asarray(params["negative_prototypes"], float)
    threshold = float(params["positive_distance_threshold"])
    max_ratio = float(params.get("max_positive_negative_ratio", 1.0))
    total = accepted = 0
    for frame in payload["frames"]:
        candidates = []
        for index, cluster in enumerate(frame["clusters"]):
            values = (np.asarray(feature(cluster), float) - center) / scale
            dp = float(np.linalg.norm(positive - values, axis=1).min())
            dn = float(np.linalg.norm(negative - values, axis=1).min())
            is_candidate = dp <= threshold and dp < max_ratio * dn
            cluster["pillar_rule"] = {"candidate": bool(is_candidate), "positive_distance": round(dp, 6),
                                       "negative_distance": round(dn, 6), "threshold": round(threshold, 6),
                                       "max_positive_negative_ratio": max_ratio}
            total += 1
            if is_candidate:
                accepted += 1
                candidates.append(index)
        frame["pillar_candidate_indices"] = candidates
    payload["pillar_candidate_rule"] = {"source": str(args.rule), "candidate_only": True,
                                         "prohibited_inputs": ["/tf", "/tf_static", "odometry", "/world/*"],
                                         "clusters_examined": total, "pillar_candidates": accepted}
    payload["purpose"] = "natural_scene_sensor_only_yolo_lidar_pillar_candidates"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["pillar_candidate_rule"], ensure_ascii=False))


if __name__ == "__main__":
    main()
