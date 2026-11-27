"""Resolution-aware scan-line clustering for organized competition LiDAR.

V4 treats a scan-line run as the smallest mergeable unit.  Adjacent rings are
merged only when several beams support the same surface, which prevents one
accidental bridge from joining long background ribbons.  Every threshold is
defined in sensor geometry; no TF, odometry, camera projection, or world state
is accepted by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from competition_user.pointcloud_v2 import ClusterV2, ProcessedCloudV2, cluster_features


PIPELINE_VERSION = "pointcloud_v4.7.0_strong_per_ring_support"


@dataclass(frozen=True)
class PointCloudV4Config:
    min_range_m: float = 1.0
    max_range_m: float = 30.0
    plane_tolerance_m: float = 0.12
    plane_min_points: int = 100
    plane_min_fraction: float = 0.20
    plane_trials: int = 128
    surface_angle_deg: float = 35.0
    max_neighbor_angle_deg: float = 5.0
    beam_link_base_m: float = 0.08
    beam_spacing_factor: float = 1.45
    beam_link_max_m: float = 1.10
    cross_row_column_slack: int = 1
    cross_row_min_links: int = 2
    cross_row_min_support: float = 0.34
    single_link_max_run_points: int = 0
    min_cluster_points: int = 5
    min_cluster_rows: int = 2
    min_points_per_supported_row: int = 2
    single_row_min_points: int = 24
    single_row_max_extent_m: float = 1.5
    split_trigger_extent_m: float = 2.5
    split_gap_base_m: float = 0.12
    split_gap_range_slope: float = 0.012
    balanced_split_extent_m: float = 2.0
    max_cluster_extent_m: float = 4.0
    echo_voxel_m: float = 0.10

    def to_dict(self) -> dict[str, object]:
        return {"version": PIPELINE_VERSION, **asdict(self)}


@dataclass(frozen=True)
class _Run:
    row: int
    indices: np.ndarray
    columns: np.ndarray


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=int)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, item: int) -> int:
        root = item
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while int(self.parent[item]) != item:
            parent = int(self.parent[item])
            self.parent[item] = root
            item = parent
        return root

    def union(self, first: int, second: int) -> None:
        left, right = self.find(first), self.find(second)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def _dominant_plane_mask(
    points: np.ndarray, valid: np.ndarray, config: PointCloudV4Config
) -> np.ndarray:
    source = np.flatnonzero(valid.ravel())
    cloud = points.reshape((-1, 3))[source]
    if len(cloud) < config.plane_min_points:
        return np.zeros(valid.shape, dtype=bool)
    rng = np.random.default_rng(17)
    best = np.zeros(len(cloud), dtype=bool)
    for _ in range(config.plane_trials):
        sample = cloud[rng.choice(len(cloud), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = float(np.linalg.norm(normal))
        if length <= 1e-9:
            continue
        normal /= length
        candidate = np.abs((cloud - sample[0]) @ normal) <= config.plane_tolerance_m
        if int(np.sum(candidate)) > int(np.sum(best)):
            best = candidate
    threshold = max(config.plane_min_points, int(config.plane_min_fraction * len(cloud)))
    result = np.zeros(valid.size, dtype=bool)
    if int(np.sum(best)) >= threshold:
        result[source[best]] = True
    return result.reshape(valid.shape)


def _beam_connected(
    first: np.ndarray, second: np.ndarray, config: PointCloudV4Config
) -> bool:
    range1, range2 = float(np.linalg.norm(first)), float(np.linalg.norm(second))
    cosine = float(np.dot(first, second) / max(range1 * range2, 1e-12))
    alpha = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    if alpha <= 1e-7 or alpha > np.deg2rad(config.max_neighbor_angle_deg):
        return False
    near, far = min(range1, range2), max(range1, range2)
    beta = float(np.arctan2(near * np.sin(alpha), far - near * np.cos(alpha)))
    nominal_spacing = 2.0 * near * np.sin(0.5 * alpha)
    distance_limit = min(
        config.beam_link_max_m,
        config.beam_link_base_m + config.beam_spacing_factor * nominal_spacing,
    )
    return (
        beta > np.deg2rad(config.surface_angle_deg)
        and float(np.linalg.norm(first - second)) <= distance_limit
    )


def _scanline_runs(
    points: np.ndarray, valid: np.ndarray, config: PointCloudV4Config
) -> list[_Run]:
    height, width = valid.shape
    result: list[_Run] = []
    for row in range(height):
        row_runs: list[list[int]] = []
        current: list[int] = []
        for column in range(width):
            if not valid[row, column]:
                if current:
                    row_runs.append(current)
                    current = []
                continue
            if current and not _beam_connected(
                points[row, current[-1]], points[row, column], config
            ):
                row_runs.append(current)
                current = []
            current.append(column)
        if current:
            row_runs.append(current)
        if (
            len(row_runs) >= 2
            and row_runs[0][0] == 0
            and row_runs[-1][-1] == width - 1
            and _beam_connected(
                points[row, row_runs[-1][-1]], points[row, row_runs[0][0]], config
            )
        ):
            row_runs[0] = row_runs[-1] + row_runs[0]
            row_runs.pop()
        for columns in row_runs:
            column_array = np.asarray(columns, dtype=int)
            result.append(
                _Run(row, row * width + column_array, column_array)
            )
    return result


def _cross_row_support(
    first: _Run,
    second: _Run,
    points: np.ndarray,
    config: PointCloudV4Config,
) -> tuple[int, float]:
    second_columns = {int(column) for column in second.columns}
    linked_first: set[int] = set()
    linked_second: set[int] = set()
    width = points.shape[1]
    for column in first.columns:
        column = int(column)
        for delta in range(-config.cross_row_column_slack, config.cross_row_column_slack + 1):
            other = (column + delta) % width
            if other not in second_columns:
                continue
            if _beam_connected(points[first.row, column], points[second.row, other], config):
                linked_first.add(column)
                linked_second.add(other)
    denominator = max(1, min(len(first.indices), len(second.indices)))
    distinct_links = min(len(linked_first), len(linked_second))
    support = distinct_links / denominator
    return distinct_links, float(support)


def _merge_runs(
    runs: list[_Run], points: np.ndarray, config: PointCloudV4Config
) -> tuple[list[np.ndarray], int]:
    by_row: dict[int, list[int]] = {}
    for index, run in enumerate(runs):
        by_row.setdefault(run.row, []).append(index)
    sets = _DisjointSet(len(runs))
    merge_count = 0
    for row, first_ids in by_row.items():
        for first_id in first_ids:
            first = runs[first_id]
            for second_id in by_row.get(row + 1, ()):
                second = runs[second_id]
                links, support = _cross_row_support(first, second, points, config)
                supported = (
                    links >= config.cross_row_min_links
                    and support >= config.cross_row_min_support
                )
                compact_single_link = (
                    links >= 1
                    and max(len(first.indices), len(second.indices))
                    <= config.single_link_max_run_points
                )
                if supported or compact_single_link:
                    sets.union(first_id, second_id)
                    merge_count += 1
    groups: dict[int, list[np.ndarray]] = {}
    for index, run in enumerate(runs):
        groups.setdefault(sets.find(index), []).append(run.indices)
    return [np.concatenate(items) for items in groups.values()], merge_count


def _valid_piece(
    indices: np.ndarray, width: int, points: np.ndarray, config: PointCloudV4Config
) -> bool:
    if len(indices) < config.min_cluster_points:
        return False
    rows = np.unique(indices // width).size
    if rows >= config.min_cluster_rows:
        return True
    extent = float(np.ptp(points.reshape((-1, 3))[indices], axis=0).max())
    return (
        len(indices) >= config.single_row_min_points
        and extent <= config.single_row_max_extent_m
    )


def _prune_weak_rows(
    indices: np.ndarray, width: int, config: PointCloudV4Config
) -> np.ndarray:
    rows = indices // width
    unique, counts = np.unique(rows, return_counts=True)
    supported = unique[counts >= config.min_points_per_supported_row]
    return indices[np.isin(rows, supported)]


def _split_significant_gaps(
    indices: np.ndarray,
    points: np.ndarray,
    width: int,
    config: PointCloudV4Config,
) -> list[np.ndarray]:
    cloud = points.reshape((-1, 3))[indices]
    if len(indices) < 2 * config.min_cluster_points:
        return [indices]
    extent = float(np.ptp(cloud, axis=0).max())
    if extent <= min(config.split_trigger_extent_m, config.balanced_split_extent_m):
        return [indices]
    values, vectors = np.linalg.eigh(np.cov(cloud, rowvar=False))
    axis = vectors[:, int(np.argmax(values))]
    projection = cloud @ axis
    order = np.argsort(projection)
    gaps = np.diff(projection[order])
    if not len(gaps):
        return [indices]
    split = int(np.argmax(gaps)) + 1
    threshold = config.split_gap_base_m + config.split_gap_range_slope * float(
        np.median(np.linalg.norm(cloud, axis=1))
    )
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    if float(gaps[split - 1]) > threshold:
        candidates.append((indices[order[:split]], indices[order[split:]]))
    if extent > config.balanced_split_extent_m:
        # Autoware's mature voxel-grid clusterer bounds oversized components
        # by a principal-axis rank split.  We independently use the same
        # geometric principle here after first preferring a real spatial gap.
        middle = len(order) // 2
        candidates.append((indices[order[:middle]], indices[order[middle:]]))
        columns = indices % width
        unique = np.unique(columns)
        if len(unique) >= 2:
            circular_gaps = np.diff(np.r_[unique, unique[0] + width])
            start = int(unique[(int(np.argmax(circular_gaps)) + 1) % len(unique)])
            unwrapped = (columns - start) % width
            for boundary in np.unique(unwrapped)[:-1]:
                candidates.append(
                    (indices[unwrapped <= boundary], indices[unwrapped > boundary])
                )
    valid_candidates = [
        (left, right)
        for left, right in candidates
        if _valid_piece(left, width, points, config)
        and _valid_piece(right, width, points, config)
    ]
    if not valid_candidates:
        return [indices]
    left, right = min(
        valid_candidates,
        key=lambda pair: max(
            float(np.ptp(points.reshape((-1, 3))[part], axis=0).max()) for part in pair
        ),
    )
    child_extent = max(
        float(np.ptp(points.reshape((-1, 3))[part], axis=0).max())
        for part in (left, right)
    )
    if child_extent >= extent - 1e-6:
        return [indices]
    return _split_significant_gaps(left, points, width, config) + _split_significant_gaps(
        right, points, width, config
    )


def process_pointcloud_v4(
    points: np.ndarray,
    config: PointCloudV4Config = PointCloudV4Config(),
    *,
    echo_mask: frozenset[tuple[int, int, int]] = frozenset(),
) -> ProcessedCloudV2:
    organized = np.asarray(points, dtype=float)
    if organized.ndim != 3 or organized.shape[2] != 3:
        raise ValueError("V4 requires organized points shaped (rings, columns, 3)")
    height, width = organized.shape[:2]
    flat = organized.reshape((-1, 3))
    finite = np.isfinite(organized).all(axis=2)
    ranges = np.linalg.norm(np.where(finite[:, :, None], organized, 0.0), axis=2)
    valid = finite & (ranges >= config.min_range_m) & (ranges <= config.max_range_m)
    range_filtered = int(np.sum(valid))
    echo_count = 0
    if echo_mask:
        cells = np.floor(
            np.where(finite[:, :, None], organized, 0.0) / config.echo_voxel_m
        ).astype(np.int32)
        remove = np.zeros(valid.shape, dtype=bool)
        for row, column in zip(*np.nonzero(valid)):
            remove[row, column] = tuple(cells[row, column]) in echo_mask
        echo_count = int(np.sum(remove))
        valid &= ~remove
    plane = _dominant_plane_mask(organized, valid, config)
    valid &= ~plane

    runs = _scanline_runs(organized, valid, config)
    merged, merge_count = _merge_runs(runs, organized, config)
    supported_groups = []
    for group in merged:
        pruned = _prune_weak_rows(group, width, config)
        if len(pruned):
            supported_groups.append(pruned)
    groups = []
    for group in supported_groups:
        if not _valid_piece(group, width, organized, config):
            continue
        for candidate in _split_significant_gaps(group, organized, width, config):
            piece = _prune_weak_rows(candidate, width, config)
            if (
                _valid_piece(piece, width, organized, config)
                and float(np.ptp(flat[piece], axis=0).max()) <= config.max_cluster_extent_m
            ):
                groups.append(piece)

    source_indices = np.flatnonzero(valid.ravel())
    compact_lookup = np.full(flat.shape[0], -1, dtype=int)
    compact_lookup[source_indices] = np.arange(len(source_indices))
    compact = flat[source_indices]
    clusters = tuple(
        ClusterV2(compact_lookup[group], cluster_features(flat[group])) for group in groups
    )
    audit = {
        "pipeline_version": PIPELINE_VERSION,
        "raw_points": int(flat.shape[0]),
        "organized_shape": [int(height), int(width)],
        "range_filtered_points": range_filtered,
        "self_echo_removed_points": echo_count,
        "dominant_plane_removed_points": int(np.sum(plane)),
        "scanline_runs": len(runs),
        "cross_row_merges": merge_count,
        "merged_components": len(merged),
        "row_supported_components": len(supported_groups),
        "final_clusters": len(clusters),
        "config": config.to_dict(),
    }
    return ProcessedCloudV2(compact, source_indices, clusters, audit)
