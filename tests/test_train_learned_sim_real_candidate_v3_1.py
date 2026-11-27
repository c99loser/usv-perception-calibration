from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).parents[1] / "tools" / "natural_scene_calibration" / "train_learned_sim_real_candidate_v3_1.py"
SPEC = importlib.util.spec_from_file_location("train_learned_sim_real_candidate_v3_1", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_camera_one_hot_contract() -> None:
    value = MODULE.camera_one_hot("right2")
    assert value.shape == (6,)
    assert value.sum() == 1.0
    assert value[-1] == 1.0


def test_domain_weights_balance_each_domain_class() -> None:
    real = np.asarray([False, False, True, True, True])
    sim = np.asarray([False, True, True])
    weights = MODULE.domain_balanced_weights(real, sim)
    assert np.isclose(weights[: len(real)][real].sum(), weights[: len(real)][~real].sum())
    assert np.isclose(weights[len(real) :][sim].sum(), weights[len(real) :][~sim].sum())


def test_threshold_prefers_high_precision_feasible_solution() -> None:
    truth = np.asarray([True, True, True, False] * 6)
    probability = np.asarray([0.99, 0.95, 0.90, 0.60] * 6)
    cameras = np.repeat(np.asarray(MODULE.CAMERAS), 4)
    threshold, summary = MODULE.select_threshold(truth, probability, cameras)
    assert threshold > 0.60
    assert summary["inner_precision"] == 1.0
