import importlib.util
from pathlib import Path
import sys


def _module():
    path = Path("tools/annotation/evaluate_candidate_extractor.py")
    spec = importlib.util.spec_from_file_location("candidate_evaluator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_matching_and_metrics_are_one_to_one():
    module = _module()
    truth = [module.Box(10, 10, 20, 20)]
    predictions = [module.Box(11, 11, 20, 20, 0.8), module.Box(12, 12, 20, 20, 0.7)]
    result = module.metrics(predictions, truth)
    assert result["matches"] == 1
    assert result["precision"] == 0.5


def test_threshold_selection_requires_precision_and_match_count():
    module = _module()
    truth = [module.Box(10, 10, 20, 20)]
    predictions = [module.Box(10, 10, 20, 20, 0.9), module.Box(100, 100, 20, 20, 0.1)]
    assert module.choose_threshold(predictions, truth, 0.9, 1, 0.2) == 0.9
