"""Occupancy BEV generation; output is NumPy for ROS/image adapters to consume."""
from __future__ import annotations
import numpy as np
from .config import PerceptionConfig
from .contracts import Detection


def build_bev(points: np.ndarray, detections: list[Detection], config: PerceptionConfig) -> np.ndarray:
    extent, resolution = config.bev_extent_m, config.bev_resolution_m
    side = int(round(2 * extent / resolution)); image = np.full((side, side), 128, dtype=np.uint8)
    def pixels(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return ((extent - xy[:, 0]) / resolution).astype(int), ((xy[:, 1] + extent) / resolution).astype(int)
    valid = (np.abs(points[:, 0]) <= extent) & (np.abs(points[:, 1]) <= extent)
    if np.any(valid):
        rows, cols = pixels(points[valid, :2]); inside = (rows >= 0) & (rows < side) & (cols >= 0) & (cols < side); image[rows[inside], cols[inside]] = 255
    hull_x, hull_y = config.hull_length_m / 2, config.hull_width_m / 2
    rr, cc = pixels(np.array([[-hull_x, -hull_y], [hull_x, hull_y]])); image[min(rr):max(rr)+1, min(cc):max(cc)+1] = 200
    for detection in detections:
        row, col = pixels(detection.position[None, :2]);
        if 0 <= row[0] < side and 0 <= col[0] < side: image[max(0,row[0]-2):row[0]+3, max(0,col[0]-2):col[0]+3] = 255
    return image
