"""Natural-scene observations for self-calibration.

This module deliberately has no ROS, TF, odometry, simulator-world or fixed
fixture dependency.  It turns one LiDAR cloud into attributable clusters,
tracks them in the sensor frame, and reports whether their geometry is rich
enough to justify an external-calibration update.  A negative report is a
successful safe outcome: the caller must retain ``nominal_unverified``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ClusterObservation:
    frame_stamp_ns: int
    local_id: int
    center: tuple[float, float, float]
    extent: tuple[float, float, float]
    point_count: int
    bearing_rad: float
    range_m: float


@dataclass
class Track:
    track_id: int
    observations: list[ClusterObservation] = field(default_factory=list)

    @property
    def latest(self) -> ClusterObservation:
        return self.observations[-1]

    @property
    def stable(self) -> bool:
        return len(self.observations) >= 3


@dataclass(frozen=True)
class ObservabilityReport:
    observable: bool
    reason: str
    stable_track_count: int
    bearing_span_deg: float
    range_span_m: float


def clean_points(points: np.ndarray, *, min_range_m: float = 0.3, max_range_m: float = 60.0,
                 voxel_m: float = 0.10) -> np.ndarray:
    """Finite/nonzero/range filter plus light voxel de-duplication only."""
    xyz = np.asarray(points, dtype=float).reshape((-1, 3))
    finite = xyz[np.isfinite(xyz).all(axis=1)]
    distance = np.linalg.norm(finite, axis=1)
    kept = finite[(distance >= min_range_m) & (distance <= max_range_m)]
    if not len(kept):
        return kept
    _, index = np.unique(np.floor(kept / voxel_m).astype(np.int32), axis=0, return_index=True)
    return kept[np.sort(index)]


def cluster_points(points: np.ndarray, *, cell_m: float = 0.50, min_points: int = 6) -> list[np.ndarray]:
    """Voxel-neighbour connected components; deterministic DBSCAN-like baseline."""
    if not len(points):
        return []
    keys = np.floor(points / cell_m).astype(np.int32)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(keys):
        buckets.setdefault(tuple(key), []).append(index)
    visited: set[tuple[int, int, int]] = set()
    groups: list[np.ndarray] = []
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for start in buckets:
        if start in visited:
            continue
        pending, component, indices = [start], set(), []
        visited.add(start)
        while pending:
            key = pending.pop()
            component.add(key)
            indices.extend(buckets[key])
            for dx, dy, dz in offsets:
                neighbour = (key[0] + dx, key[1] + dy, key[2] + dz)
                if neighbour in buckets and neighbour not in visited:
                    visited.add(neighbour)
                    pending.append(neighbour)
        if len(indices) >= min_points:
            groups.append(points[np.asarray(indices, dtype=int)])
    return groups


def observations_from_cloud(points: np.ndarray, stamp_ns: int) -> list[ClusterObservation]:
    result = []
    for local_id, group in enumerate(cluster_points(clean_points(points))):
        low, high = np.min(group, axis=0), np.max(group, axis=0)
        center = (low + high) / 2.0
        result.append(ClusterObservation(
            frame_stamp_ns=stamp_ns, local_id=local_id,
            center=tuple(float(x) for x in center), extent=tuple(float(x) for x in high - low),
            point_count=len(group), bearing_rad=float(atan2(center[1], center[0])),
            range_m=float(np.linalg.norm(center)),
        ))
    return result


class NaturalTrackManager:
    """Greedy, explainable sensor-frame tracker for static-candidate selection."""
    def __init__(self, *, max_center_jump_m: float = 1.5, max_extent_delta_m: float = 1.0) -> None:
        self.max_center_jump_m = max_center_jump_m
        self.max_extent_delta_m = max_extent_delta_m
        self.tracks: dict[int, Track] = {}
        self._next_track_id = 1

    def update(self, observations: Iterable[ClusterObservation]) -> list[tuple[int, ClusterObservation]]:
        assigned: set[int] = set()
        output = []
        for observation in observations:
            best_id, best_cost = None, float("inf")
            for track_id, track in self.tracks.items():
                if track_id in assigned:
                    continue
                center_delta = np.linalg.norm(np.subtract(observation.center, track.latest.center))
                extent_delta = np.linalg.norm(np.subtract(observation.extent, track.latest.extent))
                if center_delta <= self.max_center_jump_m and extent_delta <= self.max_extent_delta_m:
                    cost = center_delta + 0.25 * extent_delta
                    if cost < best_cost:
                        best_id, best_cost = track_id, cost
            if best_id is None:
                best_id = self._next_track_id
                self._next_track_id += 1
                self.tracks[best_id] = Track(best_id)
            self.tracks[best_id].observations.append(observation)
            assigned.add(best_id)
            output.append((best_id, observation))
        return output

    def observability(self, *, min_stable_tracks: int = 3, min_bearing_span_deg: float = 20.0,
                       min_range_span_m: float = 3.0) -> ObservabilityReport:
        stable = [track for track in self.tracks.values() if track.stable]
        if len(stable) < min_stable_tracks:
            return ObservabilityReport(False, "fewer_than_three_persistent_natural_clusters", len(stable), 0.0, 0.0)
        bearings = np.unwrap(np.asarray([track.latest.bearing_rad for track in stable]))
        ranges = np.asarray([track.latest.range_m for track in stable])
        bearing_span = float(np.degrees(np.ptp(bearings)))
        range_span = float(np.ptp(ranges))
        if bearing_span < min_bearing_span_deg:
            return ObservabilityReport(False, "insufficient_bearing_diversity", len(stable), bearing_span, range_span)
        if range_span < min_range_span_m:
            return ObservabilityReport(False, "insufficient_range_diversity", len(stable), bearing_span, range_span)
        return ObservabilityReport(True, "natural_scene_constraints_sufficient_for_candidate_optimization", len(stable), bearing_span, range_span)
