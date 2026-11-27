from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tools" / "natural_scene_calibration" / "train_sparse_lidar_candidate_v3_1.py"
SPEC = importlib.util.spec_from_file_location("train_sparse_lidar_candidate_v3_1", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_point_features_are_finite_for_one_point() -> None:
    values = MODULE.point_features(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
    assert values.shape == (len(MODULE.FEATURE_NAMES),)
    assert np.isfinite(values).all()


def test_fit_rule_uses_episode_independent_positive_distance() -> None:
    positive = [[value] for value in np.linspace(0.0, 0.4, 5)] + [[value] for value in np.linspace(0.6, 1.0, 5)]
    negative = [[value] for value in np.linspace(5.0, 6.0, 10)]
    x = np.asarray(positive + negative, dtype=float)
    y = np.asarray([True] * 10 + [False] * 10)
    groups = np.asarray(["a"] * 5 + ["b"] * 5 + ["a"] * 10)
    rule = MODULE.fit_rule(x, y, groups, 0.65, 0.75)
    assert rule["positive_distance_threshold"] > 0.0
    predicted, positive_distance, negative_distance = MODULE.predict(rule, np.asarray([[0.15], [6.15]]))
    assert predicted.tolist() == [True, False]
    assert positive_distance[0] < negative_distance[0]
