"""Pure data, geometry, and tracking helpers for the annotation GUI.

This module deliberately has no Tk dependency so its behaviour can be tested
headlessly.  All saved coordinates remain in each LiDAR sensor's local XYZ
frame; view zoom and pan never alter annotation geometry.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import uuid

import numpy as np


CLASSES = ("pillar", "buoy", "block", "boat", "background", "ambiguous")
PCD_NUMPY_TYPES = {
    ("F", 4): "<f4", ("F", 8): "<f8",
    ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8",
    ("U", 1): "<u1", ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
}


def _parse_pcd_header(stream) -> tuple[dict[str, list[str]], str]:
    header: dict[str, list[str]] = {}
    while True:
        raw = stream.readline()
        if not raw:
            raise ValueError("PCD header ended before DATA")
        line = raw.decode("ascii", errors="strict").strip()
        if not line or line.startswith("#"):
            continue
        key, *values = line.split()
        key = key.upper()
        header[key] = values
        if key == "DATA":
            if not values:
                raise ValueError("PCD DATA mode is missing")
            return header, values[0].lower()


def read_pcd(path: Path) -> np.ndarray:
    """Read XYZ from an ASCII or uncompressed-binary PCD file.

    Organized NaN slots are preserved.  The GUI decides which finite points to
    render, matching the project's raw finite-XYZ annotation rule.
    """
    with Path(path).open("rb") as stream:
        header, mode = _parse_pcd_header(stream)
        fields = header.get("FIELDS") or header.get("FIELD")
        if not fields or not {"x", "y", "z"}.issubset(fields):
            raise ValueError(f"PCD lacks x/y/z fields: {path}")
        sizes = [int(v) for v in header.get("SIZE", ["4"] * len(fields))]
        types = header.get("TYPE", ["F"] * len(fields))
        counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]
        points = int((header.get("POINTS") or [str(int(header["WIDTH"][0]) * int(header.get("HEIGHT", ["1"])[0]))])[0])
        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise ValueError(f"Inconsistent PCD field metadata: {path}")

        if mode == "ascii":
            raw = np.loadtxt(stream, dtype=np.float64, ndmin=2, max_rows=points)
            offsets: dict[str, int] = {}
            column = 0
            for name, count in zip(fields, counts):
                offsets[name] = column
                column += count
            if raw.shape[1] < column:
                raise ValueError(f"PCD ASCII row has too few columns: {path}")
            return np.column_stack([raw[:, offsets[name]] for name in ("x", "y", "z")]).astype(np.float32)

        if mode != "binary":
            raise ValueError(f"Unsupported PCD DATA mode {mode!r}; use ascii or binary: {path}")
        dtype_fields = []
        for name, size, kind, count in zip(fields, sizes, types, counts):
            try:
                base = PCD_NUMPY_TYPES[(kind.upper(), size)]
            except KeyError as exc:
                raise ValueError(f"Unsupported PCD field type {kind}{size}: {path}") from exc
            dtype_fields.append((name, base) if count == 1 else (name, base, (count,)))
        raw = np.frombuffer(stream.read(), dtype=np.dtype(dtype_fields), count=points)
        return np.column_stack((raw["x"], raw["y"], raw["z"])).astype(np.float32)


@dataclass
class ViewTransform:
    """A 2-D BEV camera centered in LiDAR-local metres."""

    center_x: float = 0.0
    center_y: float = 0.0
    pixels_per_meter: float = 7.0
    min_pixels_per_meter: float = 0.4
    max_pixels_per_meter: float = 250.0

    def world_to_canvas(self, x: float, y: float, width: float, height: float) -> tuple[float, float]:
        return (
            width / 2.0 + (x - self.center_x) * self.pixels_per_meter,
            height / 2.0 - (y - self.center_y) * self.pixels_per_meter,
        )

    def canvas_to_world(self, px: float, py: float, width: float, height: float) -> tuple[float, float]:
        return (
            self.center_x + (px - width / 2.0) / self.pixels_per_meter,
            self.center_y - (py - height / 2.0) / self.pixels_per_meter,
        )

    def zoom_at(self, factor: float, px: float, py: float, width: float, height: float) -> None:
        before = self.canvas_to_world(px, py, width, height)
        self.pixels_per_meter = float(np.clip(
            self.pixels_per_meter * factor,
            self.min_pixels_per_meter,
            self.max_pixels_per_meter,
        ))
        after = self.canvas_to_world(px, py, width, height)
        self.center_x += before[0] - after[0]
        self.center_y += before[1] - after[1]

    def pan_pixels(self, dx: float, dy: float) -> None:
        self.center_x -= dx / self.pixels_per_meter
        self.center_y += dy / self.pixels_per_meter

    def fit_range(self, width: float, height: float, half_range_m: float = 60.0) -> None:
        self.center_x = self.center_y = 0.0
        self.pixels_per_meter = max(self.min_pixels_per_meter, min(width, height) / (2.0 * half_range_m))


def box_bounds(box: dict) -> tuple[float, float, float, float]:
    x, y, _ = box["center_xyz"]
    dx, dy, _ = box["dimensions_xyz"]
    return x - dx / 2.0, y - dy / 2.0, x + dx / 2.0, y + dy / 2.0


def point_in_box(x: float, y: float, box: dict) -> bool:
    x0, y0, x1, y1 = box_bounds(box)
    return x0 <= x <= x1 and y0 <= y <= y1


def box_handles(box: dict) -> dict[str, tuple[float, float]]:
    x0, y0, x1, y1 = box_bounds(box)
    return {"sw": (x0, y0), "nw": (x0, y1), "se": (x1, y0), "ne": (x1, y1)}


def move_box(box: dict, dx: float, dy: float) -> dict:
    result = deepcopy(box)
    result["center_xyz"][0] = round(float(result["center_xyz"][0] + dx), 3)
    result["center_xyz"][1] = round(float(result["center_xyz"][1] + dy), 3)
    return result


def resize_box(box: dict, handle: str, x: float, y: float, min_size: float = 0.05) -> dict:
    """Move one corner while keeping the opposite corner fixed."""
    if handle not in box_handles(box):
        raise ValueError(f"Unknown box handle: {handle}")
    handles = box_handles(box)
    opposite = {"sw": "ne", "nw": "se", "se": "nw", "ne": "sw"}[handle]
    ox, oy = handles[opposite]
    if abs(x - ox) < min_size:
        x = ox + (min_size if x >= ox else -min_size)
    if abs(y - oy) < min_size:
        y = oy + (min_size if y >= oy else -min_size)
    result = deepcopy(box)
    result["center_xyz"][0] = round((x + ox) / 2.0, 3)
    result["center_xyz"][1] = round((y + oy) / 2.0, 3)
    result["dimensions_xyz"][0] = round(abs(x - ox), 3)
    result["dimensions_xyz"][1] = round(abs(y - oy), 3)
    return result


def _finite_xy(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float32)
    xy = points[:, :2]
    return xy[np.isfinite(xy).all(axis=1)]


def _points_near_box(points: np.ndarray, box: dict, expansion: float) -> np.ndarray:
    xy = _finite_xy(points)
    if not len(xy):
        return xy
    x, y, _ = box["center_xyz"]
    dx, dy, _ = box["dimensions_xyz"]
    hx = max(0.15, dx * expansion / 2.0)
    hy = max(0.15, dy * expansion / 2.0)
    mask = (np.abs(xy[:, 0] - x) <= hx) & (np.abs(xy[:, 1] - y) <= hy)
    return xy[mask]


def estimate_box_translation(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    box: dict,
    max_shift_m: float = 3.0,
) -> tuple[float, float, float, int, int]:
    """Estimate a robust local XY translation for one box.

    This is intentionally a modest tracker, not a source of ground truth.  It
    compares robust local point-cluster centres in an expanded search window,
    caps implausible motion, and reports a quality value for audit/UI display.
    """
    previous = _points_near_box(previous_points, box, expansion=1.15)
    current = _points_near_box(current_points, box, expansion=2.5)
    if len(previous) < 2 or len(current) < 2:
        return 0.0, 0.0, 0.0, len(previous), len(current)
    prev_center = np.median(previous, axis=0)
    # Reject distant clutter using a radius based on box size around the old
    # robust centre, then use the median as a stable small-object estimate.
    dims = np.asarray(box["dimensions_xyz"][:2], dtype=float)
    radius = max(0.35, float(np.linalg.norm(dims)) * 1.4, max_shift_m)
    current = current[np.linalg.norm(current - prev_center, axis=1) <= radius]
    if len(current) < 2:
        return 0.0, 0.0, 0.0, len(previous), len(current)
    curr_center = np.median(current, axis=0)
    shift = curr_center - prev_center
    norm = float(np.linalg.norm(shift))
    if norm > max_shift_m > 0:
        shift *= max_shift_m / norm
    balance = min(len(previous), len(current)) / max(len(previous), len(current))
    support = min(1.0, min(len(previous), len(current)) / 8.0)
    quality = float(round(balance * support, 3))
    return float(shift[0]), float(shift[1]), quality, len(previous), len(current)


def propagate_boxes(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    previous_boxes: list[dict],
    current_boxes: list[dict] | None = None,
    *,
    follow: bool = True,
    max_shift_m: float = 3.0,
) -> list[dict]:
    """Copy missing tracks to a new frame, optionally following local points.

    Existing non-empty ``track_id`` values are never overwritten.  Each copied
    frame box receives a new object id but preserves its track id.
    """
    result = deepcopy(current_boxes or [])
    existing_tracks = {str(b.get("track_id", "")).strip() for b in result if str(b.get("track_id", "")).strip()}
    for source in previous_boxes:
        track_id = str(source.get("track_id", "")).strip()
        if track_id and track_id in existing_tracks:
            continue
        copied = deepcopy(source)
        copied["id"] = str(uuid.uuid4())
        copied["source_box_id"] = source.get("id")
        dx = dy = quality = 0.0
        prev_count = curr_count = 0
        if follow:
            dx, dy, quality, prev_count, curr_count = estimate_box_translation(
                previous_points, current_points, source, max_shift_m=max_shift_m
            )
            copied = move_box(copied, dx, dy)
        copied["propagation"] = {
            "method": "local_point_median" if follow else "copy",
            "delta_xy": [round(dx, 3), round(dy, 3)],
            "quality": quality,
            "previous_support": prev_count,
            "current_support": curr_count,
            "reviewed": False,
        }
        result.append(copied)
        if track_id:
            existing_tracks.add(track_id)
    return result


def new_annotation_document(episode_id: str) -> dict:
    return {
        "schema_version": 2,
        "annotation_format": "local_annotation_sidecar_v2",
        "coordinate_frame": "sensor_local_xyz",
        "episode_id": episode_id,
        "frames": {},
    }


def load_annotation_document(path: Path, episode_id: str) -> dict:
    if not path.exists():
        return new_annotation_document(episode_id)
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document.get("frames"), dict):
        raise ValueError(f"Invalid annotation frames object: {path}")
    # Backward-compatible in-memory migration from the original v1 sidecar.
    document.setdefault("annotation_format", "local_annotation_sidecar_v1")
    document.setdefault("coordinate_frame", "sensor_local_xyz")
    document.setdefault("episode_id", episode_id)
    return document


def discover_episodes(root: Path, allow_holdout: bool = False) -> list[Path]:
    """Find episode directories without accidentally opening holdout data."""
    root = Path(root)
    candidates = [root] if (root / "frame_pointcloud_map.json").exists() else [p for p in root.iterdir() if p.is_dir()]
    episodes = [p for p in candidates if (p / "frame_pointcloud_map.json").exists()]
    if not allow_holdout:
        episodes = [p for p in episodes if "holdout" not in p.name.lower()]
    return sorted(episodes, key=lambda p: p.name)

