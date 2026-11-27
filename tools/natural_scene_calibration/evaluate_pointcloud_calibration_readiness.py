#!/usr/bin/env python3
"""Evaluate whether point-cloud candidates are useful for extrinsic calibration.

This evaluator deliberately separates geometry readiness from semantic and
end-to-end calibration acceptance.  Point-cloud geometry can be admitted for
candidate generation while automatic extrinsic publication remains blocked.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA = "pointcloud-calibration-readiness/v1"
PROHIBITED_INPUTS = {"/tf", "/tf_static", "odometry", "/world/*"}
DEFAULT_THRESHOLDS: dict[str, float] = {
    "min_frames": 48,
    "max_sync_p95_ms": 50.0,
    "max_near_fraction": 0.02,
    "max_large_fraction": 0.10,
    "max_very_large_fraction": 0.0,
    "min_persistent_tracks": 3,
    "min_persistent_observations": 40,
    "min_persistent_coverage": 0.80,
    "min_track_occupancy": 0.70,
    "max_reference_extent_jitter_m": 0.02,
    "max_center_standard_error_p95_m": 0.15,
    "max_center_acceleration_p95_m_per_frame2": 0.50,
    "min_calibration_eligible_fraction": 0.10,
    "min_bearing_span_deg": 20.0,
    "min_elevation_span_deg": 3.0,
    "min_range_span_m": 3.0,
    "min_se3_information_rank": 6,
    "max_se3_condition_number": 1000.0,
    "min_distinct_bags": 2,
    "min_ready_pairs": 3,
    "min_real_semantic_precision": 0.90,
    "min_real_semantic_recall": 0.70,
    "min_real_semantic_matches": 20,
}


def _percentile(values: Iterable[float], q: float) -> float | None:
    array = np.asarray(list(values), dtype=float)
    return float(np.percentile(array, q)) if len(array) else None


def _finite_vector(value: object, length: int = 3) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.shape != (length,) or not np.isfinite(array).all():
        return None
    return array


def _calibration_center(cluster: dict) -> np.ndarray | None:
    return _finite_vector(
        cluster.get("calibration_center_m", cluster.get("raw_center_m"))
    )


def _raw_center(cluster: dict) -> np.ndarray | None:
    return _finite_vector(cluster.get("raw_center_m", cluster.get("center_m")))


def _center_standard_error(cluster: dict) -> float | None:
    explicit = cluster.get("center_standard_error_m")
    if isinstance(explicit, (int, float)) and math.isfinite(explicit):
        return float(explicit)
    eigenvalues = _finite_vector(cluster.get("pca_eigenvalues_m2"))
    points = cluster.get("points")
    if eigenvalues is None or not isinstance(points, (int, float)) or points <= 0:
        return None
    return float(np.sqrt(np.sum(np.maximum(eigenvalues, 0.0)) / points))


def _track_groups(frames: list[dict]) -> tuple[dict[int, list[tuple[int, dict]]], list[str]]:
    groups: dict[int, list[tuple[int, dict]]] = {}
    errors: list[str] = []
    for frame_index, frame in enumerate(frames):
        seen: set[int] = set()
        for cluster in frame.get("clusters", []):
            temporal = cluster.get("temporal_track")
            if not isinstance(temporal, dict) or "track_id" not in temporal:
                errors.append(f"frame_{frame_index}:missing_temporal_track_id")
                continue
            track_id = int(temporal["track_id"])
            if track_id in seen:
                errors.append(f"frame_{frame_index}:duplicate_track_id_{track_id}")
                continue
            seen.add(track_id)
            groups.setdefault(track_id, []).append((frame_index, cluster))
    return groups, errors


def _circular_span_deg(angles: np.ndarray) -> float:
    if len(angles) < 2:
        return 0.0
    wrapped = np.sort(np.mod(angles, 2.0 * np.pi))
    gaps = np.diff(np.r_[wrapped, wrapped[0] + 2.0 * np.pi])
    return float(np.degrees(2.0 * np.pi - np.max(gaps)))


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def observability(points: np.ndarray) -> dict[str, object]:
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    points = points[np.isfinite(points).all(axis=1)]
    ranges = np.linalg.norm(points, axis=1) if len(points) else np.empty(0)
    points = points[ranges > 1e-6]
    ranges = ranges[ranges > 1e-6]
    if not len(points):
        return {
            "representative_points": 0,
            "bearing_span_deg": 0.0,
            "elevation_span_deg": 0.0,
            "range_span_m": 0.0,
            "se3_information_rank": 0,
            "se3_condition_number": None,
            "se3_information_eigenvalues": [],
        }
    bearings = np.arctan2(points[:, 1], points[:, 0])
    elevations = np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1))
    characteristic_range = float(np.median(ranges))
    rows = []
    identity = np.eye(3)
    for point, distance in zip(points, ranges):
        direction = point / distance
        tangent = identity - np.outer(direction, direction)
        rotation_jacobian = tangent @ (-_skew(direction))
        translation_jacobian = tangent * (characteristic_range / distance)
        rows.append(np.hstack((rotation_jacobian, translation_jacobian)))
    information = np.vstack(rows).T @ np.vstack(rows)
    eigenvalues = np.maximum(np.linalg.eigvalsh(information), 0.0)
    maximum = max(float(eigenvalues[-1]), 1e-12)
    rank = int(np.sum(eigenvalues > maximum * 1e-6))
    condition = None
    if rank == 6 and eigenvalues[0] > 0.0:
        condition = float(eigenvalues[-1] / eigenvalues[0])
    return {
        "representative_points": int(len(points)),
        "bearing_span_deg": _circular_span_deg(bearings),
        "elevation_span_deg": float(np.degrees(np.ptp(elevations))),
        "range_span_m": float(np.ptp(ranges)),
        "se3_information_rank": rank,
        "se3_condition_number": condition,
        "se3_information_eigenvalues": eigenvalues.round(8).tolist(),
    }


def window_metrics(payload: dict, thresholds: dict[str, float]) -> tuple[dict, np.ndarray]:
    frames = payload.get("frames", [])
    all_clusters = [cluster for frame in frames for cluster in frame.get("clusters", [])]
    groups, identity_errors = _track_groups(frames)
    persistent = {track_id: entries for track_id, entries in groups.items() if len(entries) >= 6}
    persistent_observations = sum(len(entries) for entries in persistent.values())
    sync = [abs(float(frame.get("sync_delta_ms", math.inf))) for frame in frames]
    calibration_contract_errors = []
    standard_errors = []
    eligible_observations = 0
    near = large = very_large = invalid_geometry = 0
    for cluster in all_clusters:
        center = _calibration_center(cluster)
        raw = _raw_center(cluster)
        extent = _finite_vector(cluster.get("extent_m"))
        if center is None or raw is None or extent is None:
            invalid_geometry += 1
            continue
        temporal = cluster.get("temporal_track", {})
        if (
            "calibration_center_m" not in cluster
            or temporal.get("calibration_center_source")
            != "causal_current_timestamp_alpha_beta"
        ):
            calibration_contract_errors.append("missing_causal_current_time_center")
        distance = float(np.linalg.norm(center))
        near += int(distance < 2.0)
        maximum_extent = float(np.max(extent))
        large += int(maximum_extent > 2.0)
        very_large += int(maximum_extent > 4.0)
        quality = cluster.get("calibration_quality", {})
        if quality.get("eligible") is True:
            eligible_observations += 1

    eligible_groups = {
        track_id: [
            entry
            for entry in entries
            if entry[1].get("calibration_quality", {}).get("eligible") is True
        ]
        for track_id, entries in groups.items()
    }
    eligible_persistent = {
        track_id: entries
        for track_id, entries in eligible_groups.items()
        if len(entries) >= 6
    }
    eligible_persistent_observations = sum(
        len(entries) for entries in eligible_persistent.values()
    )

    occupancies = []
    extent_jitter = []
    accelerations = []
    representatives = []
    for entries in persistent.values():
        occupancies.append(len(entries) / (entries[-1][0] - entries[0][0] + 1))
        extents = np.asarray([entry[1]["extent_m"] for entry in entries], dtype=float)
        extent_jitter.append(float(np.mean(np.std(extents, axis=0))))
    for entries in eligible_persistent.values():
        centers = []
        frame_indices = []
        for frame_index, cluster in entries:
            center = _calibration_center(cluster)
            if center is not None:
                centers.append(center)
                frame_indices.append(frame_index)
            error = _center_standard_error(cluster)
            if error is not None and math.isfinite(error):
                standard_errors.append(error)
        if centers:
            representatives.append(np.median(np.asarray(centers), axis=0))
        accelerations.extend(
            float(cluster["calibration_quality"]["center_acceleration_m_per_frame2"])
            for _, cluster in entries
            if isinstance(
                cluster.get("calibration_quality", {}).get(
                    "center_acceleration_m_per_frame2"
                ),
                (int, float),
            )
        )

    count = max(len(all_clusters), 1)
    metrics = {
        "frames": len(frames),
        "clusters": len(all_clusters),
        "sync_p95_ms": _percentile(sync, 95),
        "invalid_geometry_clusters": invalid_geometry,
        "track_identity_errors": sorted(set(identity_errors)),
        "calibration_center_contract_errors": len(calibration_contract_errors),
        "near_fraction": near / count,
        "large_fraction": large / count,
        "very_large_fraction": very_large / count,
        "persistent_tracks_ge_6": len(persistent),
        "persistent_observations_ge_6": persistent_observations,
        "persistent_observation_fraction": persistent_observations / count,
        "calibration_eligible_observations": eligible_observations,
        "calibration_eligible_fraction": eligible_observations / count,
        "calibration_eligible_tracks_ge_6": len(eligible_persistent),
        "calibration_eligible_persistent_observations": eligible_persistent_observations,
        "calibration_eligible_track_coverage": (
            eligible_persistent_observations / max(eligible_observations, 1)
        ),
        "median_track_occupancy": float(np.median(occupancies)) if occupancies else None,
        "median_reference_extent_jitter_m": (
            float(np.median(extent_jitter)) if extent_jitter else None
        ),
        "center_standard_error_p95_m": _percentile(standard_errors, 95),
        "calibration_center_acceleration_p95_m_per_frame2": _percentile(
            accelerations, 95
        ),
    }
    contract = set(payload.get("prohibited_inputs", [])) == PROHIBITED_INPUTS
    gates = {
        "minimum_frames": metrics["frames"] >= thresholds["min_frames"],
        "sensor_only_contract": contract,
        "synchronization": metrics["sync_p95_ms"] is not None
        and metrics["sync_p95_ms"] <= thresholds["max_sync_p95_ms"],
        "finite_geometry": invalid_geometry == 0,
        "production_track_identity": not identity_errors and bool(all_clusters),
        "unlagged_calibration_center": not calibration_contract_errors and bool(all_clusters),
        "near_field_safety": metrics["near_fraction"] <= thresholds["max_near_fraction"],
        "large_cluster_safety": metrics["large_fraction"] <= thresholds["max_large_fraction"],
        "very_large_cluster_safety": (
            metrics["very_large_fraction"] <= thresholds["max_very_large_fraction"]
        ),
        "persistent_track_count": (
            metrics["persistent_tracks_ge_6"] >= thresholds["min_persistent_tracks"]
        ),
        "persistent_evidence": (
            metrics["persistent_observations_ge_6"]
            >= thresholds["min_persistent_observations"]
        ),
        "persistent_coverage": (
            metrics["persistent_observation_fraction"]
            >= thresholds["min_persistent_coverage"]
        ),
        "track_occupancy": metrics["median_track_occupancy"] is not None
        and metrics["median_track_occupancy"] >= thresholds["min_track_occupancy"],
        "reference_extent_stability": (
            metrics["median_reference_extent_jitter_m"] is not None
            and metrics["median_reference_extent_jitter_m"]
            <= thresholds["max_reference_extent_jitter_m"]
        ),
        "calibration_candidate_fraction": (
            metrics["calibration_eligible_fraction"]
            >= thresholds["min_calibration_eligible_fraction"]
        ),
        "calibration_candidate_tracks": (
            metrics["calibration_eligible_tracks_ge_6"]
            >= thresholds["min_persistent_tracks"]
        ),
        "calibration_candidate_evidence": (
            metrics["calibration_eligible_persistent_observations"]
            >= thresholds["min_persistent_observations"]
        ),
        "calibration_candidate_coverage": (
            metrics["calibration_eligible_track_coverage"]
            >= thresholds["min_persistent_coverage"]
        ),
        "center_measurement_uncertainty": metrics["center_standard_error_p95_m"] is not None
        and metrics["center_standard_error_p95_m"]
        <= thresholds["max_center_standard_error_p95_m"],
        "center_temporal_smoothness": (
            metrics["calibration_center_acceleration_p95_m_per_frame2"] is not None
            and metrics["calibration_center_acceleration_p95_m_per_frame2"]
            <= thresholds["max_center_acceleration_p95_m_per_frame2"]
        ),
    }
    metrics["gates"] = gates
    metrics["passed"] = all(gates.values())
    return metrics, np.asarray(representatives, dtype=float).reshape((-1, 3))


def _semantic_gate(manifest: dict, base: Path, thresholds: dict[str, float]) -> dict:
    value = manifest.get("real_lidar_semantic_report")
    if not value:
        return {
            "evaluated": False,
            "passed": False,
            "reason": "missing_real_lidar_semantic_holdout_report",
        }
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    report = json.loads(path.read_text(encoding="utf-8"))
    bags = report.get("by_bag", {})
    per_bag = bool(bags) and all(
        float(item.get("precision", 0.0)) >= thresholds["min_real_semantic_precision"]
        for item in bags.values()
    )
    gates = {
        "real_holdout_only": report.get("domain") == "real_mcap_holdout",
        "precision": float(report.get("precision", 0.0))
        >= thresholds["min_real_semantic_precision"],
        "recall": float(report.get("recall", 0.0))
        >= thresholds["min_real_semantic_recall"],
        "matches": int(report.get("matches", 0))
        >= thresholds["min_real_semantic_matches"],
        "per_bag_precision": per_bag,
    }
    return {"evaluated": True, "path": str(path), "gates": gates, "passed": all(gates.values())}


def evaluate_manifest(manifest: dict, base: Path) -> dict:
    thresholds = {**DEFAULT_THRESHOLDS, **manifest.get("thresholds", {})}
    windows = []
    pair_points: dict[str, list[np.ndarray]] = {}
    pair_bags: dict[str, set[str]] = {}
    for spec in manifest.get("windows", []):
        path = Path(spec["path"])
        if not path.is_absolute():
            path = base / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics, representatives = window_metrics(payload, thresholds)
        role = spec.get("role", "development")
        pair = spec.get("pair", f"{payload.get('lidar')}--{payload.get('camera')}")
        required = bool(spec.get("required", role != "diagnostic"))
        windows.append({
            "path": str(path), "role": role, "pair": pair, "required": required,
            "bag": payload.get("bag"), "pipeline": payload.get("pointcloud_pipeline_version"),
            **metrics,
        })
        if required:
            pair_points.setdefault(pair, []).append(representatives)
            pair_bags.setdefault(pair, set()).add(str(payload.get("bag")))

    pair_reports = {}
    for pair, chunks in pair_points.items():
        points = np.vstack([chunk for chunk in chunks if len(chunk)]) if any(len(chunk) for chunk in chunks) else np.empty((0, 3))
        report = observability(points)
        condition = report["se3_condition_number"]
        gates = {
            "distinct_bags": len(pair_bags[pair]) >= thresholds["min_distinct_bags"],
            "bearing_diversity": report["bearing_span_deg"] >= thresholds["min_bearing_span_deg"],
            "elevation_diversity": report["elevation_span_deg"] >= thresholds["min_elevation_span_deg"],
            "range_diversity": report["range_span_m"] >= thresholds["min_range_span_m"],
            "se3_information_rank": report["se3_information_rank"] >= thresholds["min_se3_information_rank"],
            "se3_conditioning": condition is not None and condition <= thresholds["max_se3_condition_number"],
        }
        pair_reports[pair] = {**report, "distinct_bags": len(pair_bags[pair]), "gates": gates, "passed": all(gates.values())}

    required_windows = [item for item in windows if item["required"]]
    blind_windows = [item for item in required_windows if item["role"] == "blind"]
    development_windows = [item for item in required_windows if item["role"] == "development"]
    window_gate = bool(required_windows) and all(item["passed"] for item in required_windows)
    blind_gate = bool(blind_windows) and all(item["passed"] for item in blind_windows)
    pair_gate = bool(pair_reports) and all(item["passed"] for item in pair_reports.values())
    geometry_ready = window_gate and blind_gate and pair_gate
    semantic = _semantic_gate(manifest, base, thresholds)
    ready_pairs = sum(item["passed"] for item in pair_reports.values())
    automatic_ready = (
        geometry_ready
        and semantic["passed"]
        and ready_pairs >= thresholds["min_ready_pairs"]
        and bool(manifest.get("cross_bag_extrinsic_report"))
        and bool(manifest.get("closed_loop_report"))
    )
    blockers = []
    if not window_gate:
        blockers.append("one_or_more_required_windows_failed")
    if not blind_gate:
        blockers.append("no_passing_frozen_blind_window")
    if not pair_gate:
        blockers.append("one_or_more_pairs_lack_se3_observability")
    if not semantic["passed"]:
        blockers.append("real_lidar_semantic_holdout_not_passed")
    if ready_pairs < thresholds["min_ready_pairs"]:
        blockers.append("fewer_than_three_geometry_ready_sensor_pairs")
    if not manifest.get("cross_bag_extrinsic_report"):
        blockers.append("cross_bag_extrinsic_repeatability_not_evaluated")
    if not manifest.get("closed_loop_report"):
        blockers.append("three_pair_closed_loop_not_evaluated")
    return {
        "schema": SCHEMA,
        "purpose": "pointcloud_support_for_natural_scene_extrinsic_calibration",
        "thresholds": thresholds,
        "windows": windows,
        "pairs": pair_reports,
        "summary": {
            "required_windows": len(required_windows),
            "development_windows": len(development_windows),
            "blind_windows": len(blind_windows),
            "ready_pairs": ready_pairs,
            "geometry_candidate_ready": geometry_ready,
            "automatic_extrinsic_ready": automatic_ready,
            "blockers": blockers,
        },
        "real_lidar_semantic": semantic,
        "decision": (
            "geometry_admitted_for_calibration_candidate_generation"
            if geometry_ready
            else "pointcloud_geometry_not_admitted"
        ),
        "publication_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = evaluate_manifest(manifest, args.manifest.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
