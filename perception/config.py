"""Configuration loading. JSON is deliberately used to keep replay dependency-free."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np


LIDAR_NAMES = ("front", "back", "left", "left2", "right", "right2", "forward")
LIDAR_TOPIC_TEMPLATE = "/wamv/sensors/lidars/{name}_lidar_sensor/points"


@dataclass
class PerceptionConfig:
    base_frame: str = "base_link"
    lidar_topics: dict[str, str] = field(default_factory=lambda: {n: LIDAR_TOPIC_TEMPLATE.format(name=n) for n in LIDAR_NAMES})
    # Identity is a safe bootstrap mode: it makes no claim of calibrated fusion.
    # Replace each matrix after the separate calibration work is available.
    sensor_to_base: dict[str, list[list[float]]] = field(default_factory=lambda: {n: np.eye(4).tolist() for n in LIDAR_NAMES})
    roi_min_m: float = 0.3
    roi_max_m: float = 60.0
    z_min_m: float = -2.0
    z_max_m: float = 5.0
    voxel_size_m: float = 0.1
    cluster_eps_m: float = 0.55
    cluster_min_points: int = 4
    hull_length_m: float = 4.0
    hull_width_m: float = 2.2
    bev_extent_m: float = 10.0
    bev_resolution_m: float = 0.1
    track_match_distance_m: float = 2.5
    track_max_misses: int = 4

    @classmethod
    def load(cls, path: Path | None) -> "PerceptionConfig":
        if path is None:
            return cls()
        values: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        allowed = {key: value for key, value in values.items() if key in cls.__dataclass_fields__}
        return cls(**allowed)

    def transform_for(self, sensor: str) -> np.ndarray:
        matrix = self.sensor_to_base.get(sensor)
        if matrix is None:
            return np.eye(4, dtype=np.float64)
        matrix_np = np.asarray(matrix, dtype=np.float64)
        if matrix_np.shape != (4, 4):
            raise ValueError(f"sensor_to_base[{sensor!r}] must be 4x4")
        return matrix_np
