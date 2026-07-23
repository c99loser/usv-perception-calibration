import unittest
import numpy as np
from perception.bev import build_bev
from perception.config import PerceptionConfig
from perception.contracts import PointCloudFrame
from perception.pipeline import PerceptionPipeline

class PerceptionTest(unittest.TestCase):
    def test_pipeline_generates_detection_track_and_bev(self):
        rng = np.random.default_rng(7)
        cloud = np.vstack([rng.normal([6, 1, .2], [.12, .12, .1], (18, 3)), rng.normal([-4, -2, .3], [.15, .15, .1], (18, 3))])
        pipe = PerceptionPipeline(PerceptionConfig(cluster_min_points=3, hull_length_m=1, hull_width_m=1))
        result = pipe.process(PointCloudFrame(1_000_000_000, "front", cloud))
        self.assertGreaterEqual(len(result.detections), 2); self.assertGreaterEqual(len(result.tracks), 2); self.assertEqual(result.bev.dtype, np.uint8)
    def test_nan_and_hull_points_are_removed(self):
        pipe = PerceptionPipeline(PerceptionConfig(cluster_min_points=2, hull_length_m=2, hull_width_m=2))
        points = np.array([[0, 0, 0], [np.nan, 1, 0], [5, 0, 0], [5.3, 0, 0]])
        result = pipe.process(PointCloudFrame(1, "front", points))
        self.assertEqual(len(result.filtered_points), 2)
if __name__ == "__main__": unittest.main()
