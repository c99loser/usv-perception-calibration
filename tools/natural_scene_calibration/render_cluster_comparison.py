#!/usr/bin/env python3
"""Render same-frame cluster geometry from two sensor-only observation files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


COLORS = tuple((int((37 * i) % 205 + 40), int((97 * i) % 205 + 40), int((157 * i) % 205 + 40))
               for i in range(1, 128))


def cluster_points(frame: dict) -> list[np.ndarray]:
    return [np.asarray(item.get("points_sensor_m", []), dtype=float).reshape((-1, 3))
            for item in frame["clusters"]]


def draw_panel(canvas: np.ndarray, origin: tuple[int, int], size: tuple[int, int], clusters: list[np.ndarray],
               axes: tuple[int, int], bounds: tuple[float, float, float, float], title: str) -> None:
    left, top = origin; width, height = size
    cv2.rectangle(canvas, (left, top), (left + width, top + height), (45, 45, 45), 1)
    low_x, high_x, low_y, high_y = bounds
    scale_x, scale_y = (width - 30) / max(high_x - low_x, 1e-6), (height - 45) / max(high_y - low_y, 1e-6)
    for index, cloud in enumerate(clusters):
        if not len(cloud):
            continue
        values = cloud[:, axes]
        px = left + 15 + np.rint((values[:, 0] - low_x) * scale_x).astype(int)
        py = top + height - 15 - np.rint((values[:, 1] - low_y) * scale_y).astype(int)
        color = COLORS[index % len(COLORS)]
        for x, y in zip(px, py):
            if left <= x <= left + width and top <= y <= top + height:
                cv2.circle(canvas, (int(x), int(y)), 2, color, -1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, title, (left + 12, top + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2,
                cv2.LINE_AA)


def bounds_for(groups: list[list[np.ndarray]], axes: tuple[int, int]) -> tuple[float, float, float, float]:
    points = [cloud[:, axes] for clusters in groups for cloud in clusters if len(cloud)]
    merged = np.vstack(points)
    low = np.percentile(merged, 1, axis=0); high = np.percentile(merged, 99, axis=0)
    padding = np.maximum((high - low) * 0.08, 0.5)
    return float(low[0] - padding[0]), float(high[0] + padding[0]), float(low[1] - padding[1]), float(high[1] + padding[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    first, second = baseline["frames"][args.frame_index], candidate["frames"][args.frame_index]
    if (first["image_stamp_ns"], first["lidar_stamp_ns"]) != (second["image_stamp_ns"], second["lidar_stamp_ns"]):
        raise SystemExit("frames do not share identical sensor timestamps")
    groups = [cluster_points(first), cluster_points(second)]
    canvas = np.full((920, 1640, 3), 18, dtype=np.uint8)
    bev_bounds, side_bounds = bounds_for(groups, (0, 1)), bounds_for(groups, (0, 2))
    names = (baseline.get("pointcloud_pipeline_version", "baseline"),
             candidate.get("pointcloud_pipeline_version", "candidate"))
    for column, clusters in enumerate(groups):
        x = 20 + column * 810
        large = sum(max(map(float, item["extent_m"])) > 2.0 for item in (first, second)[column]["clusters"])
        suffix = f"clusters={len(clusters)}, >2m={large}"
        draw_panel(canvas, (x, 20), (790, 420), clusters, (0, 1), bev_bounds, f"{names[column]} BEV x-y; {suffix}")
        draw_panel(canvas, (x, 470), (790, 420), clusters, (0, 2), side_bounds, f"{names[column]} side x-z")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise SystemExit(f"cannot write {args.output}")
    print(args.output)


if __name__ == "__main__":
    main()
