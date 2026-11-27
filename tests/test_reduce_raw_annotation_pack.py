from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "tools" / "natural_scene_calibration" / "reduce_raw_annotation_pack.py"
SPEC = importlib.util.spec_from_file_location("reduce_raw_annotation_pack", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stratified_selection_prefers_stable_frames_and_covers_duration() -> None:
    scores = np.ones(120, dtype=np.float64)
    scores[1::5] = 0.1
    selected = MODULE.select_stability_stratified(scores, maximum_frames=24, minimum_gap=2)
    assert len(selected) == 24
    assert selected == sorted(set(selected))
    assert selected[0] < 5
    assert selected[-1] >= 115
    assert np.median(scores[selected]) == 0.1


def test_motion_score_marks_changed_middle_frame() -> None:
    baseline = np.full(24, np.nan, dtype=np.float32)
    baseline[:12] = 5.0
    changed = baseline.copy()
    changed[:12] = 20.0
    scores = MODULE.compute_motion_scores([baseline, baseline, changed, changed])
    assert scores[0] == 0.0
    assert scores[1] > scores[0]
    assert scores[2] > scores[3]


def test_uniform_reference_indices_are_one_per_stratum() -> None:
    selected = MODULE.uniform_reference_indices(101, 16)
    assert len(selected) == 16
    assert selected == sorted(set(selected))
    assert selected[0] < 7
    assert selected[-1] >= 94
