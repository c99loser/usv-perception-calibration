import numpy as np

from competition_user.pointcloud_v3 import PointCloudV3Config, process_pointcloud_v3


def scan_point(azimuth_deg: float, elevation_deg: float, distance_m: float) -> np.ndarray:
    azimuth, elevation = np.deg2rad([azimuth_deg, elevation_deg])
    return distance_m * np.asarray([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])


def empty_scan(rows: int = 4, columns: int = 12) -> np.ndarray:
    return np.full((rows, columns, 3), np.nan)


def test_range_image_keeps_sparse_far_surface_connected() -> None:
    scan = empty_scan()
    for row in range(3):
        for column in (2, 3, 4):
            scan[row, column] = scan_point(column, row, 20.0)
    config = PointCloudV3Config(plane_min_points=1000, min_cluster_points=6, min_cluster_rows=2)
    result = process_pointcloud_v3(scan, config)
    assert len(result.clusters) == 1
    assert result.clusters[0].features["points"] == 9


def test_range_image_does_not_bridge_an_empty_beam() -> None:
    scan = empty_scan()
    for row in range(3):
        for column in (1, 2, 4, 5):
            scan[row, column] = scan_point(column, row, 12.0)
    config = PointCloudV3Config(plane_min_points=1000, min_cluster_points=5, min_cluster_rows=2)
    result = process_pointcloud_v3(scan, config)
    assert sorted(cluster.features["points"] for cluster in result.clusters) == [6, 6]


def test_range_image_splits_adjacent_depth_discontinuity() -> None:
    scan = empty_scan()
    for row in range(3):
        for column in range(2, 5):
            scan[row, column] = scan_point(column, row, 5.0)
        for column in range(5, 8):
            scan[row, column] = scan_point(column, row, 15.0)
    config = PointCloudV3Config(plane_min_points=1000, min_cluster_points=6, min_cluster_rows=2)
    result = process_pointcloud_v3(scan, config)
    assert sorted(cluster.features["points"] for cluster in result.clusters) == [9, 9]


def test_range_image_preserves_original_source_indices() -> None:
    scan = empty_scan()
    for row in range(3):
        for column in range(2, 5):
            scan[row, column] = scan_point(column, row, 5.0)
    config = PointCloudV3Config(plane_min_points=1000, min_cluster_points=6, min_cluster_rows=2)
    result = process_pointcloud_v3(scan, config)
    recovered = result.source_indices[result.clusters[0].indices]
    assert sorted(recovered.tolist()) == [2, 3, 4, 14, 15, 16, 26, 27, 28]


def test_v3_rejects_flattened_input_that_lost_scan_topology() -> None:
    with np.testing.assert_raises_regex(ValueError, "organized"):
        process_pointcloud_v3(np.zeros((12, 3)))
