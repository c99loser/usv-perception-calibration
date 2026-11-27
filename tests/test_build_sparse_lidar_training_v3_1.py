from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tools" / "natural_scene_calibration" / "build_sparse_lidar_training_v3_1.py"
SPEC = importlib.util.spec_from_file_location("build_sparse_lidar_training_v3_1", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_oriented_xy_indices_ignores_z_and_respects_yaw() -> None:
    points = np.asarray([[0.9, 0.0, 99.0], [0.0, 0.9, -99.0], [1.1, 0.0, 0.0]], dtype=np.float32)
    indices = MODULE.oriented_xy_indices(points, [0.0, 0.0, 0.0], [2.0, 1.0, 0.1], 0.0)
    assert indices.tolist() == [0]
    rotated = MODULE.oriented_xy_indices(points, [0.0, 0.0, 0.0], [2.0, 1.0, 0.1], np.pi / 2.0)
    assert rotated.tolist() == [1]


def test_propagation_requires_explicit_review() -> None:
    assert MODULE.source_kind({"auto": False}) == "direct_human"
    assert MODULE.source_kind({"propagation": {"reviewed": False}}) == "unreviewed_propagation"
    assert MODULE.source_kind({"propagation": {"reviewed": True}}) == "reviewed_propagation"


def test_sparse_representation_and_geometry_weight() -> None:
    assert MODULE.observation_type(0) == "no_return"
    assert MODULE.observation_type(1) == "point"
    assert MODULE.observation_type(3) == "sparse_cluster"
    assert MODULE.observation_type(4) == "cluster"
    assert 0.0 < MODULE.geometry_weight(1, "high") < MODULE.geometry_weight(4, "high")
    assert MODULE.geometry_weight(4, "medium") < MODULE.geometry_weight(4, "high")


def test_track_gate_requires_static_multi_frame_support() -> None:
    records = [
        {
            "train_eligible": True,
            "class_name": "pillar",
            "static_for_calibration": "yes",
            "support_count": 2,
            "jump_anomaly": False,
        }
        for _ in range(5)
    ]
    assert MODULE.track_gate(records) == (True, [])
    records[0]["jump_anomaly"] = True
    eligible, reasons = MODULE.track_gate(records)
    assert not eligible
    assert "consecutive_center_jump_gt_5m" in reasons
