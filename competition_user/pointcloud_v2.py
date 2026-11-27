"""Versioned sensor-only point-cloud processing shared by every pipeline.

No function in this module accepts TF, odometry, world state, camera
projection, or calibration truth.  The same implementation is intended for
runtime perception, real-bag extraction, simulation training and annotation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable

import numpy as np


PIPELINE_VERSION = "pointcloud_v2.1.0"


@dataclass(frozen=True)
class PointCloudV2Config:
    min_range_m: float = 1.0
    max_range_m: float = 30.0
    dedup_voxel_m: float = 0.08
    plane_tolerance_m: float = 0.12
    plane_min_points: int = 100
    plane_min_fraction: float = 0.20
    plane_trials: int = 128
    cluster_base_m: float = 0.16
    cluster_range_slope: float = 0.014
    cluster_max_m: float = 0.48
    min_cluster_points: int = 8
    max_cluster_extent_m: float = 8.0
    split_extent_m: float = 2.0
    split_radius_factor: float = 0.70
    echo_voxel_m: float = 0.10
    echo_max_range_m: float = 2.0
    echo_min_presence: float = 0.85
    echo_min_frames: int = 8

    def to_dict(self) -> dict[str, object]:
        return {"version": PIPELINE_VERSION, **asdict(self)}


@dataclass(frozen=True)
class ClusterV2:
    indices: np.ndarray
    features: dict[str, object]


@dataclass(frozen=True)
class ProcessedCloudV2:
    points: np.ndarray
    source_indices: np.ndarray
    clusters: tuple[ClusterV2, ...]
    audit: dict[str, object]


class FixedEchoMaskBuilder:
    """Learn persistent near-sensor occupied cells from LiDAR frames only."""

    def __init__(self, config: PointCloudV2Config = PointCloudV2Config()) -> None:
        self.config = config
        self.frames = 0
        self.counts: dict[tuple[int, int, int], int] = {}

    def observe(self, points: np.ndarray) -> None:
        xyz = _finite_range(points, 0.05, self.config.echo_max_range_m)
        cells = {tuple(row) for row in np.floor(xyz / self.config.echo_voxel_m).astype(np.int32)}
        for cell in cells:
            self.counts[cell] = self.counts.get(cell, 0) + 1
        self.frames += 1

    def frozen_mask(self) -> frozenset[tuple[int, int, int]]:
        if self.frames < self.config.echo_min_frames:
            return frozenset()
        minimum = int(np.ceil(self.frames * self.config.echo_min_presence))
        return frozenset(cell for cell, count in self.counts.items() if count >= minimum)


def _finite_range(points: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    xyz = np.asarray(points, dtype=float).reshape((-1, 3))
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    ranges = np.linalg.norm(xyz, axis=1)
    return xyz[(ranges >= minimum) & (ranges <= maximum)]


def _deduplicate(points: np.ndarray, voxel_m: float) -> tuple[np.ndarray, np.ndarray]:
    if not len(points):
        return points, np.empty(0, dtype=int)
    _, first = np.unique(np.floor(points / voxel_m).astype(np.int32), axis=0, return_index=True)
    first = np.sort(first)
    return points[first], first


def _remove_echo(points: np.ndarray, mask: frozenset[tuple[int, int, int]], config: PointCloudV2Config) -> tuple[np.ndarray, int, np.ndarray]:
    if not mask or not len(points):
        return points, 0, np.arange(len(points))
    cells = np.floor(points / config.echo_voxel_m).astype(np.int32)
    remove = np.asarray([tuple(cell) in mask for cell in cells], dtype=bool)
    keep = np.flatnonzero(~remove)
    return points[keep], int(np.sum(remove)), keep


def _remove_plane(points: np.ndarray, config: PointCloudV2Config) -> tuple[np.ndarray, int, np.ndarray]:
    if len(points) < config.plane_min_points:
        return points, 0, np.arange(len(points))
    rng = np.random.default_rng(17)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(config.plane_trials):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length <= 1e-9:
            continue
        normal /= length
        candidate = np.abs((points - sample[0]) @ normal) <= config.plane_tolerance_m
        if np.sum(candidate) > np.sum(best):
            best = candidate
    threshold = max(config.plane_min_points, int(config.plane_min_fraction * len(points)))
    if np.sum(best) >= threshold:
        keep = np.flatnonzero(~best)
        return points[keep], int(np.sum(best)), keep
    return points, 0, np.arange(len(points))


def _radius(ranges: np.ndarray, config: PointCloudV2Config, factor: float = 1.0) -> np.ndarray:
    return np.minimum(config.cluster_max_m, config.cluster_base_m + config.cluster_range_slope * ranges) * factor


def _graph_components(points: np.ndarray, config: PointCloudV2Config, *, radius_factor: float = 1.0) -> list[np.ndarray]:
    if not len(points):
        return []
    ranges = np.linalg.norm(points, axis=1)
    radii = _radius(ranges, config, radius_factor)
    cell_m = config.cluster_max_m * radius_factor
    keys = np.floor(points / cell_m).astype(np.int32)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(keys):
        buckets.setdefault(tuple(key), []).append(index)
    neighbours: list[list[int]] = [[] for _ in points]
    offsets = tuple(product((-1, 0, 1), repeat=3))
    for index, key_array in enumerate(keys):
        key = tuple(key_array)
        for offset in offsets:
            bucket = buckets.get((key[0] + offset[0], key[1] + offset[1], key[2] + offset[2]), ())
            for other in bucket:
                if other <= index:
                    continue
                # Mutual-radius connection plus a real Euclidean distance test
                # prevents adjacent empty/bridge voxels from merging objects.
                if np.linalg.norm(points[index] - points[other]) <= min(radii[index], radii[other]):
                    neighbours[index].append(other); neighbours[other].append(index)
    unseen = set(range(len(points))); groups: list[np.ndarray] = []
    while unseen:
        start = unseen.pop(); pending = [start]; component = [start]
        while pending:
            current = pending.pop()
            for other in neighbours[current]:
                if other in unseen:
                    unseen.remove(other); pending.append(other); component.append(other)
        if len(component) >= config.min_cluster_points:
            groups.append(np.asarray(component, dtype=int))
    return groups


def _split_large(points: np.ndarray, indices: np.ndarray, config: PointCloudV2Config) -> list[np.ndarray]:
    cloud = points[indices]
    if np.ptp(cloud, axis=0).max() <= config.split_extent_m:
        return [indices]
    pieces = _graph_components(cloud, config, radius_factor=config.split_radius_factor)
    valid = [indices[piece] for piece in pieces if len(piece) >= config.min_cluster_points]
    return valid if len(valid) >= 2 else [indices]


def cluster_features(cloud: np.ndarray) -> dict[str, object]:
    center = np.mean(cloud, axis=0); low, high = cloud.min(axis=0), cloud.max(axis=0); extent = high - low
    covariance = np.cov(cloud, rowvar=False) if len(cloud) > 1 else np.zeros((3, 3))
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = np.maximum(values[order], 0.0), vectors[:, order]
    principal = vectors[:, 0]
    if principal[np.argmax(np.abs(principal))] < 0:
        principal = -principal
    total = max(float(np.sum(values)), 1e-12); ratios = values / total
    centroid_covariance = covariance / max(len(cloud), 1)
    center_standard_error = float(
        np.sqrt(max(float(np.trace(centroid_covariance)), 0.0))
    )
    radial = center / max(float(np.linalg.norm(center)), 1e-12)
    projections = cloud @ radial; radial_size = float(np.ptp(projections))
    tangential = cloud - np.outer(projections, radial); tangential_size = float(np.linalg.norm(np.ptp(tangential, axis=0)))
    volume = max(float(np.prod(np.maximum(extent, 0.05))), 1e-6)
    range_m = float(np.linalg.norm(center))
    ordered = np.sort(extent)[::-1]
    return {
        "points": int(len(cloud)), "center_m": center.round(4).tolist(), "extent_m": extent.round(4).tolist(),
        "range_m": round(range_m, 4), "pca_ratios": ratios.round(6).tolist(),
        "pca_eigenvalues_m2": values.round(6).tolist(),
        "centroid_covariance_diag_m2": np.maximum(
            np.diag(centroid_covariance), 0.0
        ).round(8).tolist(),
        "center_standard_error_m": round(center_standard_error, 6),
        "pca_principal_axis": principal.round(6).tolist(),
        "linearity": round(float((values[0] - values[1]) / max(values[0], 1e-12)), 6),
        "pca_linearity": round(float((values[0] - values[1]) / max(values[0], 1e-12)), 6),
        "planarity": round(float((values[1] - values[2]) / max(values[0], 1e-12)), 6),
        "scattering": round(float(values[2] / max(values[0], 1e-12)), 6),
        "elongation": round(float(ordered[0] / max(ordered[1], 0.05)), 6),
        "radial_size_m": round(radial_size, 6), "tangential_size_m": round(tangential_size, 6),
        "range_normalized_density": round(float(len(cloud) * range_m * range_m / volume), 6),
    }


def process_pointcloud_v2(points: np.ndarray, config: PointCloudV2Config = PointCloudV2Config(), *,
                          echo_mask: frozenset[tuple[int, int, int]] = frozenset()) -> ProcessedCloudV2:
    xyz = np.asarray(points, dtype=float).reshape((-1, 3)); raw = len(xyz)
    valid = np.isfinite(xyz).all(axis=1); ranges = np.linalg.norm(np.where(valid[:, None], xyz, 0.0), axis=1)
    source_indices = np.flatnonzero(valid & (ranges >= config.min_range_m) & (ranges <= config.max_range_m))
    filtered = xyz[source_indices]
    deduplicated, keep = _deduplicate(filtered, config.dedup_voxel_m); source_indices = source_indices[keep]
    echo_removed, echo_count, keep = _remove_echo(deduplicated, echo_mask, config); source_indices = source_indices[keep]
    plane_removed, plane_count, keep = _remove_plane(echo_removed, config); source_indices = source_indices[keep]
    initial = _graph_components(plane_removed, config)
    groups = [piece for group in initial for piece in _split_large(plane_removed, group, config)]
    groups = [group for group in groups if np.ptp(plane_removed[group], axis=0).max() <= config.max_cluster_extent_m]
    clusters = tuple(ClusterV2(group, cluster_features(plane_removed[group])) for group in groups)
    audit = {"pipeline_version": PIPELINE_VERSION, "raw_points": raw, "range_filtered_points": int(len(filtered)),
             "deduplicated_points": int(len(deduplicated)), "self_echo_removed_points": echo_count,
             "dominant_plane_removed_points": plane_count, "initial_components": len(initial),
             "final_clusters": len(clusters), "config": config.to_dict()}
    return ProcessedCloudV2(plane_removed, source_indices, clusters, audit)
