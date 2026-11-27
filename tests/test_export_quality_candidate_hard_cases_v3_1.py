from __future__ import annotations

import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "tools" / "natural_scene_calibration" / "export_quality_candidate_hard_cases_v3_1.py"
SPEC = importlib.util.spec_from_file_location("export_quality_candidate_hard_cases_v3_1", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_failed_camera_gate() -> None:
    fold = {"by_camera": {"front": {"precision": 0.9, "tp": 5, "true_positive_center_error_p90_m": 0.9},
                          "back": {"precision": 0.7, "tp": 20, "true_positive_center_error_p90_m": 0.2},
                          "left": {"precision": 1.0, "tp": 2, "true_positive_center_error_p90_m": 0.2}}}
    assert MODULE.failed_cameras(fold) == {"back", "left"}
