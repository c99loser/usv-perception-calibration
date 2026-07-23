import importlib.util
from pathlib import Path
import sys


def _module():
    path = Path("tools/yolo/build_raw_fisheye_temporal_split.py")
    spec = importlib.util.spec_from_file_location("temporal_split", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_temporal_windows_have_explicit_guard_bands():
    split = _module().split_for_frame
    assert split(69) == "train"
    assert split(84) == "train"
    assert split(85) is None
    assert split(92) == "val"
    assert split(99) is None
    assert split(104) == "test"
