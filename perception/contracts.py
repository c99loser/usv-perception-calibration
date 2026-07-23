"""Small ROS-independent contracts shared by replay and perception stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen=True)
class PointCloudFrame:
    timestamp_ns: int
    sensor: str
    points: np.ndarray  # shape (N, >= 3), expressed in the configured base frame


@dataclass(frozen=True)
class Detection:
    position: np.ndarray
    size: np.ndarray
    label: str
    confidence: float
    source: str
    timestamp_ns: int
    point_count: int


@dataclass
class Track:
    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    size: np.ndarray
    label: str
    confidence: float
    first_timestamp_ns: int
    last_timestamp_ns: int
    hits: int = 1
    misses: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "confidence": round(float(self.confidence), 3),
            "position_m": [round(float(v), 3) for v in self.position],
            "velocity_mps": [round(float(v), 3) for v in self.velocity],
            "size_m": [round(float(v), 3) for v in self.size],
            "hits": self.hits,
            "timestamp_ns": self.last_timestamp_ns,
        }
