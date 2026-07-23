"""Deterministic LiDAR filtering, voxelisation and density clustering."""
from __future__ import annotations

from collections import deque
import numpy as np

from .config import PerceptionConfig


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points[:, :3], np.ones(len(points))))
    with np.errstate(invalid="ignore"):
        return (homogeneous @ transform.T)[:, :3]


def preprocess(points: np.ndarray, config: PerceptionConfig) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)[:, :3]
    finite = points[np.isfinite(points).all(axis=1)]
    radii = np.hypot(finite[:, 0], finite[:, 1])
    keep = (radii >= config.roi_min_m) & (radii <= config.roi_max_m) & (finite[:, 2] >= config.z_min_m) & (finite[:, 2] <= config.z_max_m)
    filtered = finite[keep]
    hull = (np.abs(filtered[:, 0]) <= config.hull_length_m / 2) & (np.abs(filtered[:, 1]) <= config.hull_width_m / 2)
    filtered = filtered[~hull]
    if not len(filtered):
        return filtered
    keys = np.floor(filtered / config.voxel_size_m).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return filtered[np.sort(indices)]


def cluster(points: np.ndarray, eps: float, min_points: int) -> list[np.ndarray]:
    """Grid-accelerated Euclidean connected components, suitable for baseline use."""
    if not len(points):
        return []
    cells = np.floor(points[:, :2] / eps).astype(np.int64)
    grid: dict[tuple[int, int], list[int]] = {}
    for index, cell in enumerate(cells):
        grid.setdefault((int(cell[0]), int(cell[1])), []).append(index)
    assigned = np.zeros(len(points), dtype=bool)
    result: list[np.ndarray] = []
    for start in range(len(points)):
        if assigned[start]:
            continue
        assigned[start] = True
        group: list[int] = []
        queue: deque[int] = deque([start])
        while queue:
            index = queue.popleft(); group.append(index)
            cx, cy = cells[index]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for candidate in grid.get((int(cx + dx), int(cy + dy)), []):
                        if not assigned[candidate] and np.linalg.norm(points[candidate, :2] - points[index, :2]) <= eps:
                            assigned[candidate] = True; queue.append(candidate)
        if len(group) >= min_points:
            result.append(points[np.asarray(group)])
    return result
