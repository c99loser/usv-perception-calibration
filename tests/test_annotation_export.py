import importlib.util
from pathlib import Path
import sys


def _module():
    path = Path("tools/annotation/export_natural_scene_coco.py")
    spec = importlib.util.spec_from_file_location("annotation_export", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evenly_spaced_indices_cover_stream_ends_without_duplicates():
    indices = _module().evenly_spaced_indices(384, 50)
    assert len(indices) == 50
    assert min(indices) == 0
    assert max(indices) == 383


def test_evenly_spaced_indices_handles_short_stream():
    assert _module().evenly_spaced_indices(3, 10) == {0, 1, 2}
