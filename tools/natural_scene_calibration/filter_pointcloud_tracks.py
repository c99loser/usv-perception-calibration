#!/usr/bin/env python3
"""Apply the same sensor-only temporal cluster filter to an observation JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from competition_user.pointcloud_tracking import PointCloudTrackConfig, PointCloudTrackFilter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-confirmed-hits", type=int, default=3)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--reference-extent-hits", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    payload = json.loads(args.observations.read_text(encoding="utf-8"))
    config = PointCloudTrackConfig(
        min_confirmed_hits=args.min_confirmed_hits,
        ema_alpha=args.ema_alpha,
        reference_extent_hits=args.reference_extent_hits,
    )
    tracker = PointCloudTrackFilter(config)
    raw_clusters = 0
    confirmed_clusters = 0
    for frame_index, frame in enumerate(payload["frames"]):
        raw_clusters += len(frame["clusters"])
        frame["clusters"] = tracker.update(frame_index, frame["clusters"])
        confirmed_clusters += len(frame["clusters"])
        frame.pop("pillar_candidate_indices", None)
    payload.pop("pillar_candidate_rule", None)
    payload["pointcloud_temporal_filter"] = {
        "config": config.to_dict(),
        "raw_clusters": raw_clusters,
        "confirmed_cluster_observations": confirmed_clusters,
        "tracks_created": tracker.next_id - 1,
        "sensor_only": True,
        "raw_geometry_preserved": True,
    }
    payload["pointcloud_pipeline_version"] = (
        f"{payload.get('pointcloud_pipeline_version', 'pointcloud_v1_fixed_voxel_0.35')}"
        f"+{config.to_dict()['version']}"
    )
    payload["summary"] = {
        "selected_frames": len(payload["frames"]),
        "clusters": confirmed_clusters,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["pointcloud_temporal_filter"]))


if __name__ == "__main__":
    main()
