from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).parents[1] / "tools" / "simulation" / "label_sim_lidar_multiclass_geometry.py"
SPEC = importlib.util.spec_from_file_location("label_sim_lidar_multiclass_geometry", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_collision_surface_distances() -> None:
    sphere = MODULE.surface_distance_sphere(np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), 1.0)
    assert np.allclose(sphere, [0.0, 1.0])
    box = MODULE.surface_distance_box(np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), np.asarray([2.0, 2.0, 2.0]))
    assert np.allclose(box, [0.0, 1.0, 1.0])
    cylinder = MODULE.surface_distance_cylinder(np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]), 1.0, 2.0)
    assert np.allclose(cylinder, [0.0, 0.0, 1.0])


def test_quaternion_identity() -> None:
    assert np.allclose(MODULE.quat_matrix([0.0, 0.0, 0.0, 1.0]), np.eye(3))
