import os
from pathlib import Path

import numpy as np

from tools.natural_scene_calibration.build_full_fov_pointcloud_episodes import (
    _draft_annotation,
    _native_path,
    _write_binary_pcd,
)


def test_native_path_converts_wsl_mount() -> None:
    expected = Path("F:/Challenger program/file.json") if os.name == "nt" else Path("/mnt/f/Challenger program/file.json")
    assert _native_path("/mnt/f/Challenger program/file.json") == expected
    assert _native_path("F:/Challenger program/file.json") == expected


def test_binary_pcd_has_packed_rgb_records(tmp_path: Path) -> None:
    points = np.asarray([[1.0, 2.0, 3.0], [-1.0, 4.0, 0.5]], dtype=np.float32)
    colors = np.asarray([[255, 0, 7], [1, 2, 3]], dtype=np.uint8)
    output = tmp_path / "sample.pcd"
    _write_binary_pcd(output, points, colors)
    content = output.read_bytes()
    marker = b"DATA binary\n"
    data = content[content.index(marker) + len(marker) :]
    assert len(data) == 2 * 15
    assert data[12:15] == bytes((255, 0, 7))
    assert b"FIELDS x y z r g b" in content


def test_draft_annotation_is_explicitly_ambiguous_and_editable() -> None:
    frames = [
        {
            "frame_index": 0,
            "annotation_assists": [
                {
                    "track_id": 4,
                    "status": "observed",
                    "center_m": [2.0, 3.0, 0.5],
                    "extent_m": [0.8, 0.6, 1.0],
                    "heading_rad": 1.2,
                }
            ],
        },
        {
            "frame_index": 1,
            "annotation_assists": [
                {
                    "track_id": 4,
                    "status": "compensated",
                    "center_m": [2.1, 3.0, 0.5],
                    "extent_m": [0.8, 0.6, 1.0],
                    "heading_rad": 1.2,
                }
            ],
        },
    ]
    annotation = _draft_annotation("episode", frames)
    assert len(annotation["objects"]) == 1
    assert annotation["objects"][0]["classTitle"] == "ambiguous"
    assert len(annotation["frames"]) == 2
    geometry = annotation["frames"][1]["figures"][0]["geometry"]
    assert geometry["position"] == {"x": 2.1, "y": 3.0, "z": 0.5}
    assert geometry["rotation"]["z"] == 0.0
