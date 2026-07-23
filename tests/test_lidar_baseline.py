import numpy as np

from competition_user.lidar_baseline import SingleLidarPipeline
from competition_user.offline_evaluation import SingleLidarEvaluator, TruthObject
from preprocessing.filters import PCD_DTYPE


TOPIC = "/wamv/sensors/lidars/front_lidar_sensor/points"


def raw_cloud(centers):
    rng = np.random.default_rng(12)
    xyz = np.vstack([rng.normal(center, 0.08, size=(18, 3)) for center in centers])
    raw = np.zeros(len(xyz), dtype=PCD_DTYPE)
    raw["x"], raw["y"], raw["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    raw["intensity"] = 1.0
    return raw


def test_single_lidar_pipeline_tracks_without_any_transform():
    pipeline = SingleLidarPipeline(min_cluster_points=4)
    first = pipeline.process(raw_cloud([(8.0, 2.0, 1.0), (13.0, -1.0, 1.2)]), TOPIC, 0)
    second = pipeline.process(raw_cloud([(8.2, 2.0, 1.0), (13.1, -1.0, 1.2)]), TOPIC, 100_000_000)
    assert len(first.detections) == 2
    assert [track.track_id for track in second.tracks] == [1, 2]
    assert first.bev.dtype == np.uint8
    assert first.timings_ms["total"] >= 0


def test_offline_evaluation_reports_detection_bev_and_tracking_metrics():
    pipeline = SingleLidarPipeline(min_cluster_points=4)
    result = pipeline.process(raw_cloud([(9.0, 0.0, 1.0)]), TOPIC, 0)
    evaluator = SingleLidarEvaluator()
    evaluator.update(result.detections, result.tracks,
                     [TruthObject(center=np.array([9.0, 0.0, 1.0]), label=result.detections[0].label)],
                     result.bev, result.bev.copy(), result.timings_ms)
    summary = evaluator.summary()
    assert summary["matches"] == 1
    assert summary["precision"] == 1.0
    assert summary["mean_bev_iou"] == 1.0
