"""Organized range-image clustering for the competition LiDAR streams.

The implementation consumes only the PointCloud2 samples themselves.  It
preserves the recorded ring/column topology and combines the incidence-angle
test used by depth_clustering/LeGO-LOAM with an explicit 3-D distance guard.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from competition_user.pointcloud_v2 import ClusterV2, ProcessedCloudV2, cluster_features


PIPELINE_VERSION = "pointcloud_v3.2.0_range_image"


@dataclass(frozen=True)
class PointCloudV3Config:
    min_range_m: float = 1.0
    max_range_m: float = 30.0
    plane_tolerance_m: float = 0.12
    plane_min_points: int = 100
    plane_min_fraction: float = 0.20
    plane_trials: int = 128
    surface_angle_deg: float = 35.0
    max_ray_gap: int = 1
    max_neighbor_angle_deg: float = 5.0
    link_base_m: float = 0.18
    link_range_slope: float = 0.045
    link_max_m: float = 1.55
    min_cluster_points: int = 5
    min_cluster_rows: int = 2
    large_cluster_points: int = 30
    max_cluster_extent_m: float = 4.0
    echo_voxel_m: float = 0.10

    def to_dict(self) -> dict[str, object]:
        return {"version": PIPELINE_VERSION, **asdict(self)}


def _dominant_plane_mask(points: np.ndarray, valid: np.ndarray,
                         config: PointCloudV3Config) -> np.ndarray:
    source = np.flatnonzero(valid.ravel())
    cloud = points.reshape((-1, 3))[source]
    if len(cloud) < config.plane_min_points:
        return np.zeros(valid.shape, dtype=bool)
    rng = np.random.default_rng(17)
    best = np.zeros(len(cloud), dtype=bool)
    for _ in range(config.plane_trials):
        sample = cloud[rng.choice(len(cloud), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length <= 1e-9:
            continue
        normal /= length
        candidate = np.abs((cloud - sample[0]) @ normal) <= config.plane_tolerance_m
        if np.sum(candidate) > np.sum(best):
            best = candidate
    threshold = max(config.plane_min_points, int(config.plane_min_fraction * len(cloud)))
    result = np.zeros(valid.size, dtype=bool)
    if np.sum(best) >= threshold:
        result[source[best]] = True
    return result.reshape(valid.shape)


def _surface_connected(first: np.ndarray, second: np.ndarray, config: PointCloudV3Config) -> bool:
    range1, range2 = float(np.linalg.norm(first)), float(np.linalg.norm(second))
    cosine = float(np.dot(first, second) / max(range1 * range2, 1e-12))
    alpha = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    if alpha <= 1e-7 or alpha > np.deg2rad(config.max_neighbor_angle_deg):
        return False
    near, far = min(range1, range2), max(range1, range2)
    beta = float(np.arctan2(near * np.sin(alpha), far - near * np.cos(alpha)))
    distance_limit = min(config.link_max_m, config.link_base_m + config.link_range_slope * near)
    return beta > np.deg2rad(config.surface_angle_deg) and np.linalg.norm(first - second) <= distance_limit


def _range_components(points: np.ndarray, valid: np.ndarray,
                      config: PointCloudV3Config) -> list[np.ndarray]:
    height, width = valid.shape
    unseen = set(np.flatnonzero(valid.ravel()).tolist())
    groups: list[np.ndarray] = []
    while unseen:
        start = unseen.pop()
        pending, component = [start], [start]
        while pending:
            current = pending.pop()
            row, col = divmod(current, width)
            for gap in range(1, config.max_ray_gap + 1):
                candidates = ((row - gap, col), (row + gap, col),
                              (row, (col - gap) % width), (row, (col + gap) % width))
                for other_row, other_col in candidates:
                    if other_row < 0 or other_row >= height:
                        continue
                    other = other_row * width + other_col
                    if other not in unseen:
                        continue
                    if _surface_connected(points[row, col], points[other_row, other_col], config):
                        unseen.remove(other)
                        pending.append(other)
                        component.append(other)
        indices = np.asarray(component, dtype=int)
        rows = np.unique(indices // width).size
        if len(indices) >= config.large_cluster_points or (
                len(indices) >= config.min_cluster_points and rows >= config.min_cluster_rows):
            groups.append(indices)
    return groups


def process_pointcloud_v3(points: np.ndarray, config: PointCloudV3Config = PointCloudV3Config(), *,
                          echo_mask: frozenset[tuple[int, int, int]] = frozenset()) -> ProcessedCloudV2:
    organized = np.asarray(points, dtype=float)
    if organized.ndim != 3 or organized.shape[2] != 3:
        raise ValueError("V3 requires organized points shaped (rings, columns, 3)")
    flat = organized.reshape((-1, 3))
    finite = np.isfinite(organized).all(axis=2)
    ranges = np.linalg.norm(np.where(finite[:, :, None], organized, 0.0), axis=2)
    valid = finite & (ranges >= config.min_range_m) & (ranges <= config.max_range_m)
    range_filtered = int(np.sum(valid))
    echo_count = 0
    if echo_mask:
        cells = np.floor(np.where(finite[:, :, None], organized, 0.0) / config.echo_voxel_m).astype(np.int32)
        remove = np.zeros(valid.shape, dtype=bool)
        for row, col in zip(*np.nonzero(valid)):
            remove[row, col] = tuple(cells[row, col]) in echo_mask
        echo_count = int(np.sum(remove))
        valid &= ~remove
    plane = _dominant_plane_mask(organized, valid, config)
    valid &= ~plane
    groups = _range_components(organized, valid, config)
    groups = [group for group in groups
              if np.ptp(flat[group], axis=0).max() <= config.max_cluster_extent_m]
    source_indices = np.flatnonzero(valid.ravel())
    compact_lookup = np.full(flat.shape[0], -1, dtype=int)
    compact_lookup[source_indices] = np.arange(len(source_indices))
    compact = flat[source_indices]
    clusters = tuple(
        ClusterV2(compact_lookup[group], cluster_features(flat[group]))
        for group in groups
    )
    audit = {
        "pipeline_version": PIPELINE_VERSION,
        "raw_points": int(flat.shape[0]),
        "organized_shape": [int(organized.shape[0]), int(organized.shape[1])],
        "range_filtered_points": range_filtered,
        "self_echo_removed_points": echo_count,
        "dominant_plane_removed_points": int(np.sum(plane)),
        "final_clusters": len(clusters),
        "config": config.to_dict(),
    }
    return ProcessedCloudV2(compact, source_indices, clusters, audit)
