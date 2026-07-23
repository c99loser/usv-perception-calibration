"""Simple constant-velocity multi-object tracker with deterministic association."""
from __future__ import annotations
import numpy as np
from .contracts import Detection, Track


class MultiObjectTracker:
    def __init__(self, match_distance_m: float = 2.5, max_misses: int = 4) -> None:
        self.match_distance_m, self.max_misses, self._next_id = match_distance_m, max_misses, 1
        self.tracks: list[Track] = []

    def update(self, detections: list[Detection], timestamp_ns: int) -> list[Track]:
        pairs = sorted((float(np.linalg.norm(track.position[:2] - det.position[:2])), ti, di) for ti, track in enumerate(self.tracks) for di, det in enumerate(detections) if track.label == det.label)
        used_t: set[int] = set(); used_d: set[int] = set()
        for distance, ti, di in pairs:
            if distance > self.match_distance_m or ti in used_t or di in used_d: continue
            track, det = self.tracks[ti], detections[di]
            dt = max((timestamp_ns - track.last_timestamp_ns) / 1e9, 1e-3)
            track.velocity = 0.65 * track.velocity + 0.35 * (det.position - track.position) / dt
            track.position, track.size, track.confidence = det.position, det.size, det.confidence
            track.last_timestamp_ns, track.hits, track.misses = timestamp_ns, track.hits + 1, 0
            used_t.add(ti); used_d.add(di)
        for index, track in enumerate(self.tracks):
            if index not in used_t: track.misses += 1
        for index, det in enumerate(detections):
            if index not in used_d:
                self.tracks.append(Track(self._next_id, det.position.copy(), np.zeros(3), det.size.copy(), det.label, det.confidence, timestamp_ns, timestamp_ns)); self._next_id += 1
        self.tracks = [track for track in self.tracks if track.misses <= self.max_misses]
        return list(self.tracks)
