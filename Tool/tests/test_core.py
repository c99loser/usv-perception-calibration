from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointcloud_annotator.core import (
    ViewTransform,
    discover_episodes,
    move_box,
    propagate_boxes,
    read_pcd,
    resize_box,
)


def sample_box(track_id: str = "track_001") -> dict:
    return {
        "id": "old-id",
        "class_name": "pillar",
        "track_id": track_id,
        "static_for_calibration": "yes",
        "reviewer_confidence": "high",
        "center_xyz": [0.0, 0.0, 0.0],
        "dimensions_xyz": [2.0, 2.0, 2.0],
        "yaw_rad": 0.0,
    }


class GeometryTests(unittest.TestCase):
    def test_transform_round_trip_and_cursor_zoom(self):
        view = ViewTransform(center_x=2.0, center_y=-1.0, pixels_per_meter=10.0)
        px, py = view.world_to_canvas(4.5, 3.0, 800, 600)
        self.assertTrue(np.allclose(view.canvas_to_world(px, py, 800, 600), (4.5, 3.0)))
        anchor_before = view.canvas_to_world(140, 220, 800, 600)
        view.zoom_at(1.7, 140, 220, 800, 600)
        self.assertTrue(np.allclose(view.canvas_to_world(140, 220, 800, 600), anchor_before))

    def test_move_and_corner_resize(self):
        moved = move_box(sample_box(), 1.25, -0.5)
        self.assertEqual(moved["center_xyz"][:2], [1.25, -0.5])
        resized = resize_box(sample_box(), "ne", 3.0, 2.0)
        self.assertEqual(resized["center_xyz"][:2], [1.0, 0.5])
        self.assertEqual(resized["dimensions_xyz"][:2], [4.0, 3.0])


class TrackingTests(unittest.TestCase):
    def test_follow_shift_and_audit(self):
        previous = np.array([[-.5, -.5, 0], [.5, -.5, 0], [-.5, .5, 0], [.5, .5, 0]], dtype=np.float32)
        shift = np.array([0.4, -0.2, 0], dtype=np.float32)
        current = previous + shift
        result = propagate_boxes(previous, current, [sample_box()], [], follow=True)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["center_xyz"][0], 0.4, places=2)
        self.assertAlmostEqual(result[0]["center_xyz"][1], -0.2, places=2)
        self.assertNotEqual(result[0]["id"], "old-id")
        self.assertEqual(result[0]["track_id"], "track_001")
        self.assertFalse(result[0]["propagation"]["reviewed"])

    def test_existing_track_is_not_overwritten(self):
        existing = sample_box()
        existing["id"] = "current-id"
        existing["center_xyz"] = [9.0, 9.0, 0.0]
        result = propagate_boxes(np.empty((0, 3)), np.empty((0, 3)), [sample_box()], [existing])
        self.assertEqual(result, [existing])


class IoTests(unittest.TestCase):
    def test_ascii_pcd(self):
        path = Path(__file__).resolve().parent / "fixtures" / "ascii_sample.pcd"
        xyz = read_pcd(path)
        self.assertTrue(np.array_equal(xyz, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)))

    def test_discovery_excludes_holdout_by_default(self):
        root = Path(__file__).resolve().parent / "fixtures" / "episodes"
        self.assertEqual([p.name for p in discover_episodes(root)], ["train_a"])
        self.assertEqual(len(discover_episodes(root, allow_holdout=True)), 2)


if __name__ == "__main__":
    unittest.main()
