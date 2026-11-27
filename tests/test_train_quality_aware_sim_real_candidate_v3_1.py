from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).parents[1] / "tools" / "natural_scene_calibration" / "train_quality_aware_sim_real_candidate_v3_1.py"
SPEC = importlib.util.spec_from_file_location("train_quality_aware_sim_real_candidate_v3_1", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_threshold_enforces_center_quality() -> None:
    truth = np.asarray([True, True, True, True, True, False])
    probability = np.asarray([0.99, 0.98, 0.97, 0.96, 0.95, 0.94])
    errors = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 2.0])
    threshold, metrics = MODULE.choose_one_threshold(truth, probability, errors)
    assert threshold > 0.94
    assert metrics["precision"] == 1.0
    assert metrics["true_positive_center_error_p90_m"] <= 1.0
