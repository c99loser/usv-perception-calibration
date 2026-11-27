from pathlib import Path

import numpy as np

from tools.natural_scene_calibration.build_raw_full_fov_annotation_pack import write_xyz_binary_pcd


def test_raw_pcd_contains_only_packed_xyz(tmp_path: Path) -> None:
    output = tmp_path / "raw.pcd"
    points = np.asarray([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]], dtype=np.float32)
    write_xyz_binary_pcd(output, points)
    content = output.read_bytes()
    marker = b"DATA binary\n"
    payload = content[content.index(marker) + len(marker):]
    assert b"FIELDS x y z\n" in content
    assert b" r g b" not in content
    assert len(payload) == len(points) * 12
    np.testing.assert_array_equal(np.frombuffer(payload, dtype="<f4").reshape((-1, 3)), points)


def test_raw_pcd_rejects_nonfinite_points(tmp_path: Path) -> None:
    points = np.asarray([[np.nan, 0.0, 0.0]], dtype=np.float32)
    try:
        write_xyz_binary_pcd(tmp_path / "bad.pcd", points)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite raw point must be rejected")
