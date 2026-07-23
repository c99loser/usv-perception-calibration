"""LiDAR geometry detector. Model adapters can replace this stage without API changes."""
from __future__ import annotations
import numpy as np
from .contracts import Detection


def classify(size: np.ndarray) -> tuple[str, float]:
    length, width, height = size
    horizontal = max(length, width)
    if height > 1.2 and height > 1.8 * max(width, 0.1): return "pillar", 0.62
    if horizontal < 0.8 and height < 1.2: return "buoy", 0.55
    if 1.2 <= length <= 3.5 and 1.2 <= width <= 3.5 and height <= 1.1: return "block", 0.5
    if horizontal >= 1.5: return "boat", 0.45
    return "unknown", 0.35


def detect(clusters: list[np.ndarray], source: str, timestamp_ns: int) -> list[Detection]:
    output: list[Detection] = []
    for points in clusters:
        minimum, maximum = points.min(axis=0), points.max(axis=0)
        size = maximum - minimum
        label, confidence = classify(size)
        output.append(Detection((minimum + maximum) / 2, size, label, confidence, source, timestamp_ns, len(points)))
    return output
