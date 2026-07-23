"""Single-LiDAR perception baseline with no extrinsic or fusion dependency.

All coordinates emitted by this module remain in the input PointCloud2 frame.
It intentionally has no transform lookup and never combines observations from
different LiDARs.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from preprocessing.filters import PreprocessingPipeline


@dataclass(frozen=True)
class Detection:
    center: np.ndarray
    size: np.ndarray
    point_count: int
    label: str


@dataclass(frozen=True)
class Track:
    track_id: int
    center: np.ndarray
    velocity: np.ndarray
    label: str
    age: int
    hits: int


@dataclass(frozen=True)
class FrameResult:
    filtered_xyz: np.ndarray
    clusters: tuple[np.ndarray, ...]
    detections: tuple[Detection, ...]
    tracks: tuple[Track, ...]
    bev: np.ndarray
    timings_ms: dict[str, float]


class NearestNeighborTracker:
    """Small deterministic tracker for one sensor frame only."""

    def __init__(self, gate_m: float = 2.5, max_missed: int = 5) -> None:
        self.gate_m = gate_m
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[int, Track] = {}
        self._last_stamp_ns: int | None = None

    def update(self, detections: Iterable[Detection], stamp_ns: int) -> tuple[Track, ...]:
        detections = list(detections)
        dt = 0.1 if self._last_stamp_ns is None else max(1e-3, (stamp_ns - self._last_stamp_ns) / 1e9)
        self._last_stamp_ns = stamp_ns
        unmatched_tracks = set(self._tracks)
        unmatched_detections = set(range(len(detections)))
        pairs: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            predicted = track.center + track.velocity * dt
            for index, detection in enumerate(detections):
                distance = float(np.linalg.norm(predicted[:2] - detection.center[:2]))
                if distance <= self.gate_m:
                    pairs.append((distance, track_id, index))
        for _, track_id, index in sorted(pairs):
            if track_id not in unmatched_tracks or index not in unmatched_detections:
                continue
            previous = self._tracks[track_id]
            detection = detections[index]
            velocity = (detection.center - previous.center) / dt
            self._tracks[track_id] = Track(track_id, detection.center, velocity, detection.label,
                                           previous.age + 1, previous.hits + 1)
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(index)
        for track_id in unmatched_tracks:
            previous = self._tracks[track_id]
            aged = Track(track_id, previous.center + previous.velocity * dt, previous.velocity,
                         previous.label, previous.age + 1, previous.hits)
            if aged.age - aged.hits > self.max_missed:
                del self._tracks[track_id]
            else:
                self._tracks[track_id] = aged
        for index in unmatched_detections:
            detection = detections[index]
            self._tracks[self._next_id] = Track(self._next_id, detection.center, np.zeros(3),
                                                detection.label, 1, 1)
            self._next_id += 1
        return tuple(sorted(self._tracks.values(), key=lambda item: item.track_id))


class SingleLidarPipeline:
    """Preprocess, cluster, classify, track, and rasterize one LiDAR frame."""

    def __init__(self, *, cluster_radius_m: float = 0.8, min_cluster_points: int = 5,
                 max_cluster_points: int = 5000, ship_half_extent_m: float = 1.5,
                 bev_extent_m: float = 30.0, bev_resolution_m: float = 0.25) -> None:
        self.preprocessing = PreprocessingPipeline()
        self.cluster_radius_m = cluster_radius_m
        self.min_cluster_points = min_cluster_points
        self.max_cluster_points = max_cluster_points
        self.ship_half_extent_m = ship_half_extent_m
        self.bev_extent_m = bev_extent_m
        self.bev_resolution_m = bev_resolution_m
        self.tracker = NearestNeighborTracker()

    def _remove_ship_and_water(self, xyz: np.ndarray) -> np.ndarray:
        # Conservative per-sensor cleanup: do not assume a common vessel frame.
        in_ship = (np.abs(xyz[:, 0]) < self.ship_half_extent_m) & (np.abs(xyz[:, 1]) < self.ship_half_extent_m)
        water_band = xyz[:, 2] < -1.0
        return xyz[~(in_ship | water_band)]

    def _clusters(self, xyz: np.ndarray) -> tuple[np.ndarray, ...]:
        if len(xyz) == 0:
            return ()
        tree = cKDTree(xyz[:, :2])
        visited = np.zeros(len(xyz), dtype=bool)
        result = []
        for seed in range(len(xyz)):
            if visited[seed]:
                continue
            visited[seed] = True
            queue = [seed]
            members = []
            while queue:
                index = queue.pop()
                members.append(index)
                for neighbor in tree.query_ball_point(xyz[index, :2], self.cluster_radius_m):
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            if self.min_cluster_points <= len(members) <= self.max_cluster_points:
                result.append(xyz[np.asarray(members, dtype=int)])
        return tuple(result)

    @staticmethod
    def _classify(cluster: np.ndarray) -> Detection:
        low, high = np.min(cluster, axis=0), np.max(cluster, axis=0)
        size = high - low
        center = np.mean(cluster, axis=0)
        horizontal = max(float(size[0]), float(size[1]))
        vertical = float(size[2])
        if horizontal < 1.1 and vertical >= 1.0:
            label = "pillar"
        elif horizontal < 1.2 and vertical < 1.0:
            label = "buoy"
        elif horizontal >= 3.0:
            label = "boat"
        elif horizontal >= 1.2:
            label = "block"
        else:
            label = "unknown"
        return Detection(center=center, size=size, point_count=len(cluster), label=label)

    def _bev(self, xyz: np.ndarray) -> np.ndarray:
        size = int(round(2 * self.bev_extent_m / self.bev_resolution_m))
        bev = np.zeros((size, size), dtype=np.uint8)
        if len(xyz) == 0:
            return bev
        indices = np.floor((xyz[:, :2] + self.bev_extent_m) / self.bev_resolution_m).astype(int)
        valid = np.all((indices >= 0) & (indices < size), axis=1)
        bev[indices[valid, 1], indices[valid, 0]] = 255
        return bev

    def process(self, raw_points: np.ndarray, topic: str, stamp_ns: int) -> FrameResult:
        start = perf_counter()
        x, y, z, _, _ = self.preprocessing.process(topic, raw_points)
        filtered = self._remove_ship_and_water(np.column_stack((x, y, z))) if len(x) else np.empty((0, 3))
        preprocessed = perf_counter()
        clusters = self._clusters(filtered)
        clustered = perf_counter()
        detections = tuple(self._classify(cluster) for cluster in clusters)
        tracks = self.tracker.update(detections, stamp_ns)
        tracked = perf_counter()
        bev = self._bev(filtered)
        finished = perf_counter()
        return FrameResult(filtered, clusters, detections, tracks, bev, {
            "preprocess": (preprocessed - start) * 1000,
            "cluster": (clustered - preprocessed) * 1000,
            "detect_track": (tracked - clustered) * 1000,
            "bev": (finished - tracked) * 1000,
            "total": (finished - start) * 1000,
        })
