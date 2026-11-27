import numpy as np

from competition_user.pointcloud_v4 import (
    PointCloudV4Config,
    _prune_weak_rows,
    process_pointcloud_v4,
)


def scan_point(azimuth_deg: float, elevation_deg: float, distance_m: float) -> np.ndarray:
    azimuth, elevation = np.deg2rad([azimuth_deg, elevation_deg])
    return distance_m * np.asarray(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )


def empty_scan(rows: int = 4, columns: int = 12) -> np.ndarray:
    return np.full((rows, columns, 3), np.nan)


def config(**changes) -> PointCloudV4Config:
    return PointCloudV4Config(plane_min_points=1000, **changes)


def test_v4_keeps_a_supported_sparse_far_surface() -> None:
    scan = empty_scan()
    for row in range(3):
        for column in (2, 3, 4):
            scan[row, column] = scan_point(column, row, 20.0)
    result = process_pointcloud_v4(scan, config())
    assert len(result.clusters) == 1
    assert result.clusters[0].features["points"] == 9


def test_v4_does_not_bridge_an_empty_beam() -> None:
    scan = empty_scan()
    for row in range(3):
        for column in (1, 2, 4, 5):
            scan[row, column] = scan_point(column, row, 12.0)
    result = process_pointcloud_v4(scan, config())
    assert sorted(cluster.features["points"] for cluster in result.clusters) == [6, 6]


def test_v4_splits_an_adjacent_depth_discontinuity() -> None:
    scan = empty_scan()
    for row in range(3):
        for column in range(2, 5):
            scan[row, column] = scan_point(column, row, 5.0)
        for column in range(5, 8):
            scan[row, column] = scan_point(column, row, 15.0)
    result = process_pointcloud_v4(scan, config())
    assert sorted(cluster.features["points"] for cluster in result.clusters) == [9, 9]


def test_v4_rejects_a_single_point_bridge_between_long_runs() -> None:
    scan = empty_scan(rows=2, columns=20)
    for column in range(2, 8):
        scan[0, column] = scan_point(column, 0.0, 12.0)
    for column in range(7, 13):
        scan[1, column] = scan_point(column, 1.0, 12.0)
    result = process_pointcloud_v4(
        scan,
        config(single_row_min_points=5, single_row_max_extent_m=2.0),
    )
    assert sorted(cluster.features["points"] for cluster in result.clusters) == [6, 6]


def test_v4_does_not_count_column_slack_as_multiple_beam_support() -> None:
    scan = empty_scan(rows=2, columns=20)
    scan[0, 7] = scan_point(7, 0.0, 12.0)
    for column in range(6, 15):
        scan[1, column] = scan_point(column, 1.0, 12.0)
    result = process_pointcloud_v4(
        scan,
        config(single_row_min_points=5, single_row_max_extent_m=2.0),
    )
    assert [cluster.features["points"] for cluster in result.clusters] == [9]


def test_v4_prunes_a_singleton_ring_from_a_split_component() -> None:
    indices = np.asarray([31 * 360 + 60, *[32 * 360 + column for column in range(60, 69)]])
    pruned = _prune_weak_rows(indices, 360, config())
    assert len(pruned) == 9
    assert np.unique(pruned // 360).tolist() == [32]


def test_v4_preserves_original_source_indices() -> None:
    scan = empty_scan()
    for row in range(3):
        for column in range(2, 5):
            scan[row, column] = scan_point(column, row, 5.0)
    result = process_pointcloud_v4(scan, config())
    recovered = result.source_indices[result.clusters[0].indices]
    assert sorted(recovered.tolist()) == [2, 3, 4, 14, 15, 16, 26, 27, 28]


def test_v4_balanced_split_bounds_a_smooth_oversized_ribbon() -> None:
    scan = empty_scan(rows=3, columns=40)
    for row in range(3):
        for column in range(5, 25):
            scan[row, column] = scan_point(column, row, 12.0)
    result = process_pointcloud_v4(scan, config())
    assert len(result.clusters) >= 2
    assert sum(cluster.features["points"] for cluster in result.clusters) == 60
    assert max(max(cluster.features["extent_m"]) for cluster in result.clusters) <= 2.0


def test_v4_finds_a_legal_unbalanced_column_boundary() -> None:
    scan = empty_scan(rows=3, columns=60)
    for row, columns in enumerate((range(20, 26), range(22, 26), range(24, 26))):
        for column in columns:
            scan[row, column] = scan_point(column, row, 28.0)
    result = process_pointcloud_v4(scan, config())
    assert sorted(cluster.features["points"] for cluster in result.clusters) == [6, 6]
    assert max(max(cluster.features["extent_m"]) for cluster in result.clusters) <= 2.0


def test_v4_rejects_flattened_input_that_lost_scan_topology() -> None:
    with np.testing.assert_raises_regex(ValueError, "organized"):
        process_pointcloud_v4(np.zeros((12, 3)))
