"""The replaceable end-to-end perception pipeline."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .bev import build_bev
from .config import PerceptionConfig
from .contracts import Detection, PointCloudFrame, Track
from .detector import detect
from .lidar import cluster, preprocess, transform_points
from .tracker import MultiObjectTracker

@dataclass
class PerceptionResult:
    timestamp_ns: int
    filtered_points: np.ndarray
    detections: list[Detection]
    tracks: list[Track]
    bev: np.ndarray

class PerceptionPipeline:
    def __init__(self, config: PerceptionConfig) -> None:
        self.config, self.tracker = config, MultiObjectTracker(config.track_match_distance_m, config.track_max_misses)
    def process(self, frame: PointCloudFrame) -> PerceptionResult:
        base_points = transform_points(frame.points, self.config.transform_for(frame.sensor))
        filtered = preprocess(base_points, self.config)
        detections = detect(cluster(filtered, self.config.cluster_eps_m, self.config.cluster_min_points), frame.sensor, frame.timestamp_ns)
        tracks = self.tracker.update(detections, frame.timestamp_ns)
        return PerceptionResult(frame.timestamp_ns, filtered, detections, tracks, build_bev(filtered, detections, self.config))
