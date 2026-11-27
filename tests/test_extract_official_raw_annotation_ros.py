import numpy as np

from tools.natural_scene_calibration.extract_official_raw_annotation_ros import nearest_cloud


def cloud(stamp: int):
    return stamp, np.asarray([[stamp, 0, 0]], dtype=np.float32), {"raw_slots": 1}


def test_nearest_cloud_selects_previous_or_current() -> None:
    previous, current = cloud(100), cloud(200)
    assert nearest_cloud(120, previous, current)[0] == 100
    assert nearest_cloud(180, previous, current)[0] == 200


def test_nearest_cloud_handles_stream_boundaries() -> None:
    assert nearest_cloud(50, None, cloud(100))[0] == 100
    assert nearest_cloud(250, cloud(200), None)[0] == 200
    assert nearest_cloud(0, None, None) is None
