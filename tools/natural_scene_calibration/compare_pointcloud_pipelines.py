#!/usr/bin/env python3
"""Compare two point-cloud pipelines on exactly the same sensor timestamps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _embedded_tracks(frames: list[dict], *, min_length: int) -> list[list[dict]] | None:
    """Use producer track identities when available instead of tracking twice."""
    grouped: dict[int, list[dict]] = {}
    saw_embedded_id = False
    for frame in frames:
        seen_in_frame: set[int] = set()
        for cluster in frame["clusters"]:
            track = cluster.get("temporal_track")
            if not isinstance(track, dict) or "track_id" not in track:
                continue
            saw_embedded_id = True
            track_id = int(track["track_id"])
            if track_id in seen_in_frame:
                raise ValueError(f"duplicate temporal track_id {track_id} in one frame")
            seen_in_frame.add(track_id)
            grouped.setdefault(track_id, []).append(cluster)
    if not saw_embedded_id:
        return None
    return [items for items in grouped.values() if len(items) >= min_length]


def tracks(frames: list[dict], *, min_length: int = 6) -> list[list[dict]]:
    embedded = _embedded_tracks(frames, min_length=min_length)
    if embedded is not None:
        return embedded
    active: dict[int, dict] = {}
    result: dict[int, list[dict]] = {}
    next_id = 1
    for frame_index, frame in enumerate(frames):
        assigned = set()
        for cluster in frame["clusters"]:
            center = np.asarray(cluster["center_m"])
            extent = np.asarray(cluster["extent_m"])
            options = []
            for track_id, previous in active.items():
                if track_id in assigned or previous["frame"] != frame_index - 1:
                    continue
                dc = np.linalg.norm(center - previous["center"])
                de = np.linalg.norm(extent - previous["extent"])
                if dc <= 1.5 and de <= 1.0:
                    options.append((dc + 0.25 * de, track_id))
            track_id = min(options)[1] if options else next_id
            if track_id == next_id:
                result[track_id] = []
                next_id += 1
            result[track_id].append(cluster)
            active[track_id] = {"frame": frame_index, "center": center, "extent": extent}
            assigned.add(track_id)
    return [items for items in result.values() if len(items) >= min_length]


def metrics(payload: dict) -> dict[str, object]:
    all_clusters = [cluster for frame in payload["frames"] for cluster in frame["clusters"]]
    persistent = tracks(payload["frames"])
    variations = []
    raw_variations = []
    for track in persistent:
        extents = np.asarray([item["extent_m"] for item in track], dtype=float)
        variations.append(float(np.mean(np.std(extents, axis=0))))
        raw_extents = np.asarray(
            [item.get("raw_extent_m", item["extent_m"]) for item in track], dtype=float
        )
        raw_variations.append(float(np.mean(np.std(raw_extents, axis=0))))
    persistent_lengths = sorted((len(track) for track in persistent), reverse=True)
    large = sum(max(map(float, item["extent_m"])) > 2 for item in all_clusters)
    return {
        "frames": len(payload["frames"]),
        "clusters": len(all_clusters),
        "clusters_per_frame": len(all_clusters) / max(len(payload["frames"]), 1),
        "near_clusters_lt_2m": sum(float(item["range_m"]) < 2 for item in all_clusters),
        "large_clusters_gt_2m": large,
        "large_cluster_fraction": large / max(len(all_clusters), 1),
        "very_large_clusters_gt_4m": sum(
            max(map(float, item["extent_m"])) > 4 for item in all_clusters
        ),
        "persistent_tracks_ge_6": len(persistent),
        # A raw count rewards identity fragmentation (several six-frame pieces can
        # describe one object), so keep it as a diagnostic and gate on evidence.
        "persistent_track_lengths": persistent_lengths,
        "persistent_track_observations_ge_6": sum(persistent_lengths),
        "persistent_track_observation_fraction": (
            sum(persistent_lengths) / max(len(all_clusters), 1)
        ),
        "median_persistent_track_length": (
            float(np.median(persistent_lengths)) if persistent_lengths else None
        ),
        "long_tracks_ge_20": sum(length >= 20 for length in persistent_lengths),
        "median_track_extent_std_m": (
            float(np.median(variations)) if variations else None
        ),
        "median_track_raw_extent_std_m": (
            float(np.median(raw_variations)) if raw_variations else None
        ),
        "uses_embedded_track_ids": any(
            "temporal_track" in cluster for cluster in all_clusters
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--v1", type=Path, help="backward-compatible alias for --baseline")
    parser.add_argument("--v2", type=Path, help="backward-compatible alias for --candidate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_path = args.baseline or args.v1
    candidate_path = args.candidate or args.v2
    if baseline_path is None or candidate_path is None:
        raise SystemExit("both --baseline and --candidate are required")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    stamps1 = [(f["image_stamp_ns"], f["lidar_stamp_ns"]) for f in baseline["frames"]]
    stamps2 = [(f["image_stamp_ns"], f["lidar_stamp_ns"]) for f in candidate["frames"]]
    if stamps1 != stamps2:
        raise SystemExit("V1/candidate timestamps differ; comparison would be invalid")
    first, second = metrics(baseline), metrics(candidate)
    gate = {
        "near_clusters_not_increased": (
            second["near_clusters_lt_2m"] <= first["near_clusters_lt_2m"]
        ),
        "large_clusters_reduced": (
            second["large_clusters_gt_2m"] < first["large_clusters_gt_2m"]
        ),
        "persistent_track_evidence_increased": (
            second["persistent_track_observations_ge_6"]
            > first["persistent_track_observations_ge_6"]
        ),
        "persistent_track_coverage_increased": (
            second["persistent_track_observation_fraction"]
            > first["persistent_track_observation_fraction"]
        ),
        "extent_jitter_reduced": (
            second["median_track_extent_std_m"] is not None
            and first["median_track_extent_std_m"] is not None
            and second["median_track_extent_std_m"]
            < first["median_track_extent_std_m"]
        ),
    }
    report = {
        "purpose": "sensor_only_pointcloud_pipeline_comparison",
        "prohibited_inputs": ["/tf", "/tf_static", "odometry", "/world/*"],
        "same_timestamps": True,
        "baseline_pipeline": baseline.get("pointcloud_pipeline_version"),
        "candidate_pipeline": candidate.get("pointcloud_pipeline_version"),
        "baseline": first,
        "candidate": second,
        "automated_gate": gate,
        "automated_pass": all(gate.values()),
        "human_recognisability_review_required": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
