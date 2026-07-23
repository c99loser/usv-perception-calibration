import importlib.util
from pathlib import Path

import numpy as np


def module():
    path = Path("tools/simulation/semantic_labels_to_yolo.py")
    spec = importlib.util.spec_from_file_location("semantic_to_yolo", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_connected_semantic_regions_become_yolo_boxes():
    tool = module()
    labels = np.zeros((10, 20, 3), dtype=np.uint8)
    labels[2:5, 4:9] = 1
    boxes = tool.extract_boxes(labels, min_pixels=3)
    assert boxes == [(1, 4, 2, 5, 3)]
    assert tool.yolo_row(boxes[0], 20, 10) == "0 0.32500000 0.35000000 0.25000000 0.30000000"


def test_unknown_renderer_label_is_rejected():
    tool = module()
    labels = np.zeros((3, 3, 3), dtype=np.uint8)
    labels[1, 1] = 9
    try:
        tool.extract_boxes(labels, min_pixels=1)
    except ValueError as error:
        assert "unknown semantic IDs" in str(error)
    else:
        raise AssertionError("unknown labels must be rejected")
