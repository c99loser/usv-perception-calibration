import importlib.util
from pathlib import Path
import sys


def _module():
    path = Path("tools/yolo/evaluate_yolo_candidates.py")
    spec = importlib.util.spec_from_file_location("yolo_eval", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_class_aware_matching_and_center_error():
    module = _module()
    predictions = [module.Box(0, 10, 10, 30, 30, .9), module.Box(1, 10, 10, 30, 30, .8)]
    targets = [module.Box(0, 11, 11, 31, 31)]
    result = module.metric(predictions, targets, .2)
    assert result["matches"] == 1
    assert result["precision"] == .5
    assert result["median_center_error_px"] > 0
