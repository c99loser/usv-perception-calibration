import numpy as np

from competition_user.pointcloud_v2 import (
    FixedEchoMaskBuilder,
    PIPELINE_VERSION,
    PointCloudV2Config,
    cluster_features,
    process_pointcloud_v2,
)


def test_real_distance_gate_keeps_nearby_objects_separate():
    rng = np.random.default_rng(4)
    cloud = np.vstack((rng.normal((5.0, 0.0, 0.5), 0.025, (20, 3)),
                       rng.normal((5.0, 0.48, 0.5), 0.025, (20, 3))))
    result = process_pointcloud_v2(cloud, PointCloudV2Config(dedup_voxel_m=0.01))
    assert len(result.clusters) == 2
    assert result.audit["pipeline_version"] == PIPELINE_VERSION


def test_adaptive_radius_preserves_sparse_far_object():
    # 0.30 m point spacing is too sparse for the near-field radius but valid at 20 m.
    cloud = np.asarray([[20.0, 0.0, 0.3 * index] for index in range(8)], dtype=float)
    result = process_pointcloud_v2(cloud)
    assert len(result.clusters) == 1
    assert result.clusters[0].features["points"] == 8


def test_fixed_echo_mask_is_sensor_only_and_requires_persistence():
    config = PointCloudV2Config(echo_min_frames=4, echo_min_presence=0.75)
    builder = FixedEchoMaskBuilder(config)
    fixed = np.asarray([[1.2, 0.1, 0.0], [1.22, 0.1, 0.0]])
    for frame in range(4):
        transient = np.asarray([[1.5, 0.5 + frame * 0.2, 0.0]])
        builder.observe(np.vstack((fixed, transient)))
    mask = builder.frozen_mask()
    assert tuple(np.floor(fixed[0] / config.echo_voxel_m).astype(int)) in mask
    processed = process_pointcloud_v2(np.vstack((fixed, np.repeat([[5.0, 0.0, 0.5]], 8, axis=0))), config,
                                      echo_mask=mask)
    assert processed.audit["self_echo_removed_points"] >= 1


def test_shape_ratios_are_rotation_invariant():
    rng = np.random.default_rng(9)
    cloud = rng.normal(size=(100, 3)) * np.asarray((1.0, 0.3, 0.1)) + np.asarray((8.0, 2.0, 1.0))
    angle = np.deg2rad(37.0)
    rotation = np.asarray(((np.cos(angle), -np.sin(angle), 0),
                           (np.sin(angle), np.cos(angle), 0), (0, 0, 1)))
    first, second = cluster_features(cloud), cluster_features(cloud @ rotation.T)
    assert np.allclose(first["pca_ratios"], second["pca_ratios"], atol=1e-6)
    assert abs(first["linearity"] - second["linearity"]) < 1e-6


def test_cluster_features_expose_finite_centroid_uncertainty():
    cloud = np.asarray(
        [[5.0, -0.1, 0.4], [5.0, 0.1, 0.4], [5.1, 0.0, 0.5]], dtype=float
    )
    features = cluster_features(cloud)
    assert features["center_standard_error_m"] > 0.0
    assert len(features["centroid_covariance_diag_m2"]) == 3
    assert all(value >= 0.0 for value in features["centroid_covariance_diag_m2"])
