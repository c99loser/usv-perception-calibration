#!/usr/bin/env python3
"""Benchmark a bounded V4 geometry grid on immutable sensor-only caches."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from competition_user.pointcloud_v2 import FixedEchoMaskBuilder, PointCloudV2Config
from competition_user.pointcloud_v4 import PointCloudV4Config, process_pointcloud_v4
from tools.natural_scene_calibration.compare_pointcloud_pipelines import metrics


def numbers(value: str, kind=float) -> list:
    return [kind(item) for item in value.split(",")]


def load_case(value: str) -> dict:
    fields = value.split("|", 2)
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("case must be NAME|CACHE.npz|BASELINE.json")
    name, cache_path, baseline_path = fields
    data = np.load(cache_path)
    clouds = data["xyz"].astype(float)
    builder = FixedEchoMaskBuilder(PointCloudV2Config())
    for cloud in clouds:
        builder.observe(cloud)
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    return {
        "name": name,
        "clouds": clouds,
        "echo_mask": builder.frozen_mask(),
        "baseline": metrics(baseline),
    }


def candidate_metrics(case: dict, config: PointCloudV4Config) -> dict:
    frames = []
    for cloud in case["clouds"]:
        processed = process_pointcloud_v4(cloud, config, echo_mask=case["echo_mask"])
        frames.append({"clusters": [dict(cluster.features) for cluster in processed.clusters]})
    return metrics({"frames": frames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--beam-spacing-factors", default="0.75,0.9,1.05,1.2")
    parser.add_argument("--beam-link-bases-m", default="0.03,0.08")
    parser.add_argument("--cross-row-min-links", default="2,3")
    parser.add_argument("--cross-row-min-support", default="0.34,0.5")
    parser.add_argument("--max-cluster-extent-m", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    cases = [load_case(item) for item in args.case]
    base = PointCloudV4Config(max_cluster_extent_m=args.max_cluster_extent_m)
    combinations = itertools.product(
        numbers(args.beam_spacing_factors),
        numbers(args.beam_link_bases_m),
        numbers(args.cross_row_min_links, int),
        numbers(args.cross_row_min_support),
    )
    results = []
    for spacing, link_base, min_links, support in combinations:
        config = replace(
            base,
            beam_spacing_factor=spacing,
            beam_link_base_m=link_base,
            cross_row_min_links=min_links,
            cross_row_min_support=support,
        )
        case_results = []
        for case in cases:
            candidate = candidate_metrics(case, config)
            baseline = case["baseline"]
            gate = {
                "near_not_increased": candidate["near_clusters_lt_2m"]
                <= baseline["near_clusters_lt_2m"],
                "large_reduced": candidate["large_clusters_gt_2m"]
                < baseline["large_clusters_gt_2m"],
                "tracks_not_reduced": candidate["persistent_tracks_ge_6"]
                >= baseline["persistent_tracks_ge_6"],
                "jitter_reduced": candidate["median_track_extent_std_m"] is not None
                and baseline["median_track_extent_std_m"] is not None
                and candidate["median_track_extent_std_m"]
                < baseline["median_track_extent_std_m"],
            }
            case_results.append(
                {
                    "name": case["name"],
                    "baseline": baseline,
                    "candidate": candidate,
                    "gate": gate,
                    "pass": all(gate.values()),
                }
            )
        results.append(
            {
                "config": config.to_dict(),
                "cases": case_results,
                "pass_all": all(item["pass"] for item in case_results),
                "large_cluster_total": sum(
                    item["candidate"]["large_clusters_gt_2m"] for item in case_results
                ),
                "persistent_track_total": sum(
                    item["candidate"]["persistent_tracks_ge_6"] for item in case_results
                ),
            }
        )
    results.sort(
        key=lambda item: (
            not item["pass_all"],
            item["large_cluster_total"],
            -item["persistent_track_total"],
        )
    )
    report = {
        "purpose": "bounded_sensor_only_v4_geometry_search",
        "prohibited_inputs": ["/tf", "/tf_static", "odometry", "/world/*"],
        "cases": [case["name"] for case in cases],
        "configurations": len(results),
        "passing_configurations": sum(item["pass_all"] for item in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("configurations", "passing_configurations")}))
    print(json.dumps(results[0]))


if __name__ == "__main__":
    main()
