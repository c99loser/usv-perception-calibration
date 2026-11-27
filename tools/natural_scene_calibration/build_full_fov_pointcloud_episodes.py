#!/usr/bin/env python3
"""Build sensor-only full-FOV Point Cloud Episodes for human review.

The raw organized LiDAR scan remains the primary annotation surface. V4.11
clusters/tracks are only rendered as coloured candidate overlays. Camera images
are synchronized context and deliberately contain no 2D--3D projection because
the extrinsic is the quantity that will later be estimated.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from competition_user.pointcloud_annotation_assist import (
    ASSIST_VERSION,
    PointCloudAnnotationAssistConfig,
    build_annotation_assists,
)


SCHEMA = "real-lidar-full-fov-episodes/v2"
PROHIBITED_INPUTS = ("/tf", "/tf_static", "odometry", "/world/*", "nominal_extrinsic_projection")
BACKGROUND_RGB = np.asarray((92, 105, 119), dtype=np.uint8)
PALETTE = np.asarray(
    [
        (255, 99, 71), (0, 220, 180), (255, 196, 61), (95, 165, 255),
        (225, 110, 255), (75, 230, 90), (255, 135, 35), (45, 205, 255),
        (245, 80, 145), (180, 235, 70), (160, 125, 255), (255, 225, 90),
    ],
    dtype=np.uint8,
)


def _native_path(path: str | Path) -> Path:
    """Accept either Windows or WSL absolute paths when running on Windows."""
    text = str(path)
    if os.name == "nt" and text.startswith("/mnt/") and len(text) > 6:
        return Path(f"{text[5].upper()}:/{text[7:]}")
    if os.name != "nt" and len(text) > 3 and text[1:3] == ":/":
        return Path(f"/mnt/{text[0].lower()}/{text[3:]}")
    return Path(text)


def _load_points(value: object) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float32)
    return points[np.isfinite(points).all(axis=1)]


def _track_color(track_id: int, eligible: bool) -> np.ndarray:
    color = PALETTE[(max(track_id, 1) - 1) % len(PALETTE)].astype(np.float32)
    if not eligible:
        color = 0.48 * color + 0.52 * np.asarray((128, 128, 128), dtype=np.float32)
    return np.rint(color).astype(np.uint8)


def _point_keys(points: np.ndarray, resolution: float = 0.01) -> np.ndarray:
    return np.rint(points / resolution).astype(np.int32)


def _overlay_raw_points(
    raw_points: np.ndarray,
    raw_frame: dict,
    tracked_frame: dict,
) -> tuple[np.ndarray, list[dict], int, int]:
    colors = np.tile(BACKGROUND_RGB, (len(raw_points), 1))
    raw_keys = _point_keys(raw_points)
    key_to_indices: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(raw_keys):
        key_to_indices[tuple(int(value) for value in key)].append(index)
    overlays: list[dict] = []
    candidate_points = matched_points = 0
    for cluster in tracked_frame.get("clusters", []):
        source_index = int(cluster.get("source_cluster_index", -1))
        if not 0 <= source_index < len(raw_frame.get("clusters", [])):
            continue
        source = raw_frame["clusters"][source_index]
        points = _load_points(source.get("points_sensor_m"))
        track = cluster.get("temporal_track", {})
        track_id = int(track.get("track_id", -1))
        if track_id < 0 or not len(points):
            continue
        eligible = cluster.get("calibration_quality", {}).get("eligible") is True
        color = _track_color(track_id, eligible)
        matched_indices: set[int] = set()
        for key in _point_keys(points):
            matched_indices.update(key_to_indices.get(tuple(int(value) for value in key), ()))
        if matched_indices:
            colors[np.fromiter(matched_indices, dtype=np.int64)] = color
        candidate_points += len(points)
        matched_points += len(matched_indices)
        overlays.append(
            {
                "track_id": track_id,
                "source_cluster_index": source_index,
                "eligible": eligible,
                "candidate_points": len(points),
                "matched_raw_points": len(matched_indices),
                "center_m": [round(float(value), 5) for value in source.get("center_m", ())],
                "extent_m": [round(float(value), 5) for value in cluster.get("reference_extent_m", source.get("extent_m", ()))],
                "rgb": [int(value) for value in color],
            }
        )
    return colors, overlays, candidate_points, matched_points


def _write_binary_pcd(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if points.shape != (len(colors), 3):
        raise ValueError("point/color shape mismatch")
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")],
        align=False,
    )
    if dtype.itemsize != 15:
        raise RuntimeError("unexpected PCD record size")
    records = np.empty(len(points), dtype=dtype)
    for axis in "xyz":
        records[axis] = points[:, "xyz".index(axis)]
    for channel in "rgb":
        records[channel] = colors[:, "rgb".index(channel)]
    header = "\n".join(
        [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS x y z r g b",
            "SIZE 4 4 4 1 1 1",
            "TYPE F F F U U U",
            "COUNT 1 1 1 1 1 1",
            f"WIDTH {len(points)}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {len(points)}",
            "DATA binary",
            "",
        ]
    ).encode("ascii")
    path.write_bytes(header + records.tobytes())


def _equal(ax, x: np.ndarray, y: np.ndarray, minimum: float = 1.0) -> None:
    if not len(x):
        return
    span = max(float(np.ptp(x)), float(np.ptp(y)), minimum)
    cx, cy = float(np.mean((np.min(x), np.max(x)))), float(np.mean((np.min(y), np.max(y))))
    ax.set_xlim(cx - span * 0.55, cx + span * 0.55)
    ax.set_ylim(cy - span * 0.55, cy + span * 0.55)
    ax.set_aspect("equal", adjustable="box")


def _draw_annotation_assists(ax, assists: list[dict]) -> None:
    """Draw observed/compensated cuboid footprints without altering raw PCD."""
    for assist in assists:
        if assist["status"] == "observed" and float(assist["confidence"]) < 0.99:
            continue
        center = np.asarray(assist["center_m"], dtype=float)
        extent = np.asarray(assist["extent_m"], dtype=float)
        if center.shape != (3,) or extent.shape != (3,):
            continue
        compensated = assist["status"] == "compensated"
        color = "#ffd43b" if compensated else "#40e07b"
        linestyle = "--" if compensated else ":"
        rectangle = plt.Rectangle(
            center[:2] - 0.5 * extent[:2], extent[0], extent[1],
            fill=False, edgecolor=color, linewidth=1.4, linestyle=linestyle,
        )
        ax.add_patch(rectangle)
        ax.scatter([center[0]], [center[1]], marker="x", s=42, c=color, linewidths=1.4)
        ax.text(center[0], center[1], f" {assist['track_id']}", color=color, fontsize=7)


def _render_review(path: Path, image_path: Path, points: np.ndarray, colors: np.ndarray, frame: dict) -> None:
    figure = plt.figure(figsize=(16, 10), dpi=120, facecolor="#0e1319")
    grid = figure.add_gridspec(2, 3, wspace=0.22, hspace=0.28)
    axes = [figure.add_subplot(grid[r, c]) for r in range(2) for c in range(3)]
    for ax in axes:
        ax.set_facecolor("#141b23")
        ax.tick_params(colors="#a9b5c2", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#394653")
        ax.grid(color="#33404c", alpha=0.35, linewidth=0.5)
    axes[0].axis("off")
    try:
        axes[0].imshow(mpimg.imread(image_path))
        axes[0].set_title("Same-time fisheye context — NO 2D–3D projection", color="#ffd166")
    except Exception:
        axes[0].text(0.5, 0.5, "image unavailable", color="#ffd166", ha="center")

    candidate = np.any(colors != BACKGROUND_RGB, axis=1)
    background = ~candidate
    axes[1].scatter(points[background, 0], points[background, 1], s=1.0, c="#61707e", alpha=0.34, linewidths=0)
    if np.any(candidate):
        axes[1].scatter(points[candidate, 0], points[candidate, 1], s=9, c=colors[candidate] / 255.0, alpha=0.95, linewidths=0)
    axes[1].scatter([0], [0], marker="*", s=100, c="#ffffff")
    _draw_annotation_assists(axes[1], frame.get("annotation_assists", []))
    axes[1].set_title("Full-FOV XY — gray raw scan, color V4.11 tracks", color="#e7edf3")
    axes[1].set_xlabel("x / m", color="#a9b5c2"); axes[1].set_ylabel("y / m", color="#a9b5c2")
    _equal(axes[1], points[:, 0], points[:, 1], 10.0)

    azimuth = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    planar_range = np.linalg.norm(points[:, :2], axis=1)
    axes[2].scatter(azimuth[background], planar_range[background], s=1.0, c="#61707e", alpha=0.32, linewidths=0)
    if np.any(candidate):
        axes[2].scatter(azimuth[candidate], planar_range[candidate], s=8, c=colors[candidate] / 255.0, linewidths=0)
    axes[2].set_title("Full-FOV azimuth–range", color="#e7edf3")
    axes[2].set_xlabel("azimuth / deg", color="#a9b5c2"); axes[2].set_ylabel("range / m", color="#a9b5c2")
    for assist in frame.get("annotation_assists", []):
        if assist["status"] != "compensated":
            continue
        center = np.asarray(assist["center_m"], dtype=float)
        axes[2].scatter(
            [np.degrees(np.arctan2(center[1], center[0]))],
            [np.linalg.norm(center[:2])], marker="x", s=55, c="#ffd43b", linewidths=1.6,
        )

    axes[3].scatter(planar_range[background], points[background, 2], s=1.0, c="#61707e", alpha=0.32, linewidths=0)
    if np.any(candidate):
        axes[3].scatter(planar_range[candidate], points[candidate, 2], s=8, c=colors[candidate] / 255.0, linewidths=0)
    axes[3].set_title("Range–Z profile", color="#e7edf3")
    axes[3].set_xlabel("range / m", color="#a9b5c2"); axes[3].set_ylabel("z / m", color="#a9b5c2")
    for assist in frame.get("annotation_assists", []):
        if assist["status"] != "compensated":
            continue
        center = np.asarray(assist["center_m"], dtype=float)
        axes[3].scatter(
            [np.linalg.norm(center[:2])], [center[2]], marker="x", s=55,
            c="#ffd43b", linewidths=1.6,
        )

    compensated_assists = [
        item for item in frame.get("annotation_assists", [])
        if item["status"] == "compensated"
    ]
    if np.any(candidate) or compensated_assists:
        cp = points[candidate]
        center = (
            np.asarray(compensated_assists[0]["center_m"], dtype=float)
            if compensated_assists else np.median(cp, axis=0)
        )
        radius = np.linalg.norm(points[:, :2] - center[:2], axis=1)
        local = radius <= 5.0
        axes[4].scatter(points[local & background, 0], points[local & background, 1], s=2, c="#61707e", alpha=0.35, linewidths=0)
        axes[4].scatter(points[local & candidate, 0], points[local & candidate, 1], s=14, c=colors[local & candidate] / 255.0, linewidths=0)
        nearby_assists = [
            item for item in frame.get("annotation_assists", [])
            if np.linalg.norm(np.asarray(item["center_m"], dtype=float)[:2] - center[:2]) <= 5.0
        ]
        _draw_annotation_assists(axes[4], nearby_assists)
        _equal(axes[4], points[local, 0], points[local, 1], 3.0)
        axes[4].set_title(
            "Compensated neighbourhood (5 m radius)" if compensated_assists
            else "Candidate neighbourhood (5 m radius)",
            color="#e7edf3",
        )
    else:
        axes[4].text(0.5, 0.5, "No tracked candidate in this frame", color="#ffd166", ha="center")
        axes[4].set_title("Candidate neighbourhood", color="#e7edf3")
    axes[4].set_xlabel("x / m", color="#a9b5c2"); axes[4].set_ylabel("y / m", color="#a9b5c2")

    axes[5].axis("off")
    overlay_lines = []
    for item in frame["overlays"]:
        overlay_lines.append(
            f"track {item['track_id']:02d}  {'ELIGIBLE' if item['eligible'] else 'context '}  "
            f"pts {item['matched_raw_points']:3d}/{item['candidate_points']:3d}  "
            f"center {item['center_m']}"
        )
    for item in frame.get("annotation_assists", []):
        if item["status"] != "compensated":
            continue
        overlay_lines.append(
            f"track {item['track_id']:02d}  COMPENSATED  "
            f"conf {item['confidence']:.2f}  center {item['center_m']}"
        )
    text = (
        f"frame: {frame['frame_index']:03d}\n"
        f"raw valid points: {frame['raw_valid_points']}\n"
        f"candidate match: {frame['matched_candidate_points']}/{frame['candidate_points']} "
        f"({100.0 * frame['overlay_match_rate']:.1f}%)\n"
        f"sync delta: {frame['sync_delta_ms']:.2f} ms\n\n"
        + ("\n".join(overlay_lines[:18]) if overlay_lines else "no V4.11 track overlay")
    )
    axes[5].text(0.02, 0.98, text, va="top", color="#d9e2ea", family="monospace", fontsize=9)
    axes[5].set_title("Sensor-only audit; green observed / yellow compensated", color="#e7edf3")
    figure.suptitle(frame["episode_id"], color="#f4f7fa", fontsize=14)
    figure.savefig(path, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def _write_project_meta(project: Path) -> None:
    classes = []
    for title, color in (
        ("pillar", "#FF6347"), ("buoy", "#00DCB4"), ("block", "#FFC43D"),
        ("boat", "#5FA5FF"), ("background", "#808B96"), ("ambiguous", "#E16EFF"),
    ):
        classes.append({"title": title, "shape": "cuboid_3d", "color": color})
    (project / "meta.json").write_text(
        json.dumps({"classes": classes, "tags": [], "projectType": "point_cloud_episode"}, indent=2),
        encoding="utf-8",
    )


def _draft_annotation(source_id: str, frames: list[dict]) -> dict:
    """Build editable ambiguous cuboids; every figure remains an explicit proposal."""
    track_ids = sorted(
        {int(item["track_id"]) for frame in frames for item in frame["annotation_assists"]}
    )
    objects = []
    object_keys: dict[int, str] = {}
    for track_id in track_ids:
        key = uuid.uuid5(uuid.NAMESPACE_URL, f"challenger:{source_id}:assist:{track_id}").hex
        object_keys[track_id] = key
        objects.append({"key": key, "classTitle": "ambiguous", "tags": []})
    annotated_frames = []
    for frame in frames:
        figures = []
        for item in frame["annotation_assists"]:
            track_id = int(item["track_id"])
            center = item["center_m"]
            extent = item["extent_m"]
            figures.append(
                {
                    "key": uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"challenger:{source_id}:assist:{track_id}:{frame['frame_index']}",
                    ).hex,
                    "objectKey": object_keys[track_id],
                    "geometryType": "cuboid_3d",
                    "geometry": {
                        "position": {"x": center[0], "y": center[1], "z": center[2]},
                        # V4.11 extents are axis-aligned sensor-frame extents;
                        # heading is audit metadata and must not rotate this box.
                        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "dimensions": {"x": extent[0], "y": extent[1], "z": extent[2]},
                    },
                }
            )
        if figures:
            annotated_frames.append({"index": frame["frame_index"], "figures": figures})
    return {
        "description": (
            "Machine-generated sensor-only ambiguous cuboid proposals. "
            "They are annotation aids, not ground truth; inspect, resize, relabel, or delete."
        ),
        "key": uuid.uuid5(uuid.NAMESPACE_URL, f"challenger:{source_id}").hex,
        "tags": [],
        "objects": objects,
        "frames": annotated_frames,
        "framesCount": len(frames),
    }


def _source_summary(
    source: dict,
    output: Path,
    *,
    min_range: float,
    max_range: float,
    with_ambiguous_proposals: bool,
) -> dict:
    source_id = source["id"]
    split = source["split"]
    cache_path = _native_path(source["organized_cache"])
    raw_path = _native_path(source["raw_observations"])
    tracked_path = _native_path(source["tracked_observations"])
    cache = np.load(cache_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    tracked = json.loads(tracked_path.read_text(encoding="utf-8"))
    xyz = cache["xyz"]
    if not (len(xyz) == len(raw["frames"]) == len(tracked["frames"])):
        raise ValueError(f"frame count mismatch for {source_id}")
    project = output / f"supervisely_{split}_project"
    episode = project / source_id
    pointcloud = episode / "pointcloud"
    related = episode / "related_images"
    review = output / "review_samples" / source_id
    metadata = output / "frame_metadata" / source_id
    for folder in (pointcloud, related, review, metadata):
        folder.mkdir(parents=True, exist_ok=True)
    frame_map: dict[str, str] = {}
    frames: list[dict] = []
    assist_frames, assist_summary = build_annotation_assists(
        tracked["frames"], PointCloudAnnotationAssistConfig()
    )
    review_candidates: list[tuple[int, int]] = []
    compensation_candidates: list[tuple[int, int]] = []
    for frame_index, (cloud, raw_frame, tracked_frame) in enumerate(zip(xyz, raw["frames"], tracked["frames"])):
        if int(cache["lidar_stamp_ns"][frame_index]) != int(raw_frame["lidar_stamp_ns"]):
            raise ValueError(f"LiDAR timestamp mismatch {source_id}:{frame_index}")
        flat = np.asarray(cloud, dtype=np.float32).reshape((-1, 3))
        ranges = np.linalg.norm(flat, axis=1)
        valid = np.isfinite(flat).all(axis=1) & (ranges >= min_range) & (ranges <= max_range)
        points = flat[valid]
        colors, overlays, candidate_count, matched_count = _overlay_raw_points(points, raw_frame, tracked_frame)
        stem = f"{frame_index:06d}_{int(raw_frame['lidar_stamp_ns'])}"
        pcd_name = f"{stem}.pcd"
        _write_binary_pcd(pointcloud / pcd_name, points, colors)
        frame_map[str(frame_index)] = pcd_name
        source_image = _native_path(raw_frame["image"])
        image_folder = related / stem
        image_folder.mkdir()
        image_name = source_image.name
        shutil.copy2(source_image, image_folder / image_name)
        image_timestamp = dt.datetime.fromtimestamp(
            int(raw_frame["image_stamp_ns"]) / 1_000_000_000,
            tz=dt.timezone.utc,
        ).isoformat().replace("+00:00", "Z")
        image_metadata = {
            "name": image_name,
            "deviceId": str(raw["camera"]),
            "timestamp": image_timestamp,
            "sensorsData": {},
        }
        (image_folder / f"{image_name}.json").write_text(json.dumps(image_metadata, indent=2), encoding="utf-8")
        frame_record = {
            "episode_id": source_id,
            "split": split,
            "frame_index": frame_index,
            "pcd": f"pointcloud/{pcd_name}",
            "image": f"related_images/{stem}/{image_name}",
            "lidar_stamp_ns": int(raw_frame["lidar_stamp_ns"]),
            "image_stamp_ns": int(raw_frame["image_stamp_ns"]),
            "sync_delta_ms": float(raw_frame["sync_delta_ms"]),
            "raw_valid_points": len(points),
            "candidate_points": candidate_count,
            "matched_candidate_points": matched_count,
            "overlay_match_rate": matched_count / candidate_count if candidate_count else 1.0,
            "overlays": overlays,
            "annotation_assists": assist_frames[frame_index],
            "annotation_assist_ground_truth": False,
            "projection_valid": False,
        }
        (metadata / f"{stem}.json").write_text(json.dumps(frame_record, ensure_ascii=False, indent=2), encoding="utf-8")
        frames.append(frame_record)
        review_candidates.append((sum(item["matched_raw_points"] for item in overlays if item["eligible"]), frame_index))
        compensation_candidates.append(
            (
                sum(
                    item["status"] == "compensated"
                    for item in frame_record["annotation_assists"]
                ),
                frame_index,
            )
        )
    (episode / "frame_pointcloud_map.json").write_text(json.dumps(frame_map, indent=2), encoding="utf-8")
    annotation = _draft_annotation(source_id, frames) if with_ambiguous_proposals else {
        "description": "Full-FOV raw LiDAR episode; colored V4.11 candidates; no extrinsic projection.",
        "key": uuid.uuid5(uuid.NAMESPACE_URL, f"challenger:{source_id}").hex,
        "tags": [], "objects": [], "frames": [], "framesCount": len(frames),
    }
    (episode / "annotation.json").write_text(
        json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (episode / "annotation_assist_audit.json").write_text(
        json.dumps(assist_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selected = {0, len(frames) // 2, len(frames) - 1}
    selected.update(index for _, index in sorted(review_candidates, reverse=True)[:3])
    selected.update(
        index
        for count, index in sorted(compensation_candidates, reverse=True)[:3]
        if count > 0
    )
    for frame_index in sorted(selected):
        frame = frames[frame_index]
        pcd_path = pointcloud / Path(frame["pcd"]).name
        raw_bytes = pcd_path.read_bytes()
        marker = b"DATA binary\n"
        offset = raw_bytes.index(marker) + len(marker)
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        records = np.frombuffer(raw_bytes[offset:], dtype=dtype)
        review_points = np.column_stack((records["x"], records["y"], records["z"]))
        review_colors = np.column_stack((records["r"], records["g"], records["b"]))
        image_path = episode / frame["image"]
        _render_review(review / f"frame_{frame_index:03d}.png", image_path, review_points, review_colors, frame)
    return {
        "id": source_id,
        "split": split,
        "camera": str(raw["camera"]),
        "frames": len(frames),
        "unique_lidar_messages": len(set(int(value) for value in cache["lidar_stamp_ns"])),
        "mean_raw_valid_points": float(np.mean([frame["raw_valid_points"] for frame in frames])),
        "candidate_points": sum(frame["candidate_points"] for frame in frames),
        "matched_candidate_points": sum(frame["matched_candidate_points"] for frame in frames),
        "overlay_match_rate": (
            sum(frame["matched_candidate_points"] for frame in frames) /
            max(1, sum(frame["candidate_points"] for frame in frames))
        ),
        "annotation_assist": assist_summary,
        "ambiguous_proposals_written": with_ambiguous_proposals,
        "projection_valid": False,
        "episode_path": str(episode),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-range-m", type=float, default=0.3)
    parser.add_argument("--max-range-m", type=float, default=60.0)
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="optional repeatable source id for a bounded preview build",
    )
    parser.add_argument(
        "--with-ambiguous-proposals",
        action="store_true",
        help="preload editable ambiguous cuboids from observed and short-gap compensated tracks",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    sources = spec["sources"]
    if args.source_id:
        requested = set(args.source_id)
        sources = [source for source in sources if source["id"] in requested]
        missing = requested - {source["id"] for source in sources}
        if missing:
            raise SystemExit(f"unknown source ids: {sorted(missing)}")
    if not sources:
        raise SystemExit("no sources selected")
    args.output.mkdir(parents=True)
    for split in ("train", "holdout"):
        project = args.output / f"supervisely_{split}_project"
        project.mkdir()
        _write_project_meta(project)
    summaries = [
        _source_summary(
            source,
            args.output,
            min_range=args.min_range_m,
            max_range=args.max_range_m,
            with_ambiguous_proposals=args.with_ambiguous_proposals,
        )
        for source in sources
    ]
    manifest = {
        "schema": SCHEMA,
        "purpose": "full_fov_sensor_only_pointcloud_episode_annotation_and_review",
        "prohibited_inputs": list(PROHIBITED_INPUTS),
        "range_filter_m": [args.min_range_m, args.max_range_m],
        "episodes": len(summaries),
        "frames": sum(item["frames"] for item in summaries),
        "by_split": dict(Counter(item["split"] for item in summaries)),
        "by_split_camera": dict(Counter(f"{item['split']}:{item['camera']}" for item in summaries)),
        "mean_overlay_match_rate": float(np.mean([item["overlay_match_rate"] for item in summaries])),
        "annotation_assist_version": ASSIST_VERSION,
        "ambiguous_proposals_written": args.with_ambiguous_proposals,
        "compensated_proposals": sum(
            item["annotation_assist"]["compensated_proposals"] for item in summaries
        ),
        "projection_valid": False,
        "sources": summaries,
    }
    (args.output / "full_fov_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "README_使用说明.md").write_text(
        "# 全视场 LiDAR Episode 标注包 V1\n\n"
        "这是当前主标注入口。每帧 PCD 包含该 LiDAR 的完整有效 360° 扫描；灰色为原始背景点，彩色为 V4.11 已跟踪候选。同一 track 在同一 episode 中保持同色，低亮度候选表示该帧未通过 V4.11 几何质量门控。\n\n"
        "V2 新增论文启发的纯 LiDAR 时序辅助：五自由度平面状态、观测/历史双似然和短时漏检补偿。该辅助仅填充同一生产 track 在后续帧重新出现的内部缺口，不外推轨迹末端，不修改 PCD，也不构成真值。\n\n"
        "## Supervisely 导入\n\n"
        "- 训练：压缩并导入 `supervisely_train_project/`，项目类型选 Point Cloud Episodes。\n"
        "- 留出：单独压缩并导入 `supervisely_holdout_project/`，不得与训练项目混合。\n"
        "- 每个数据集文件夹是一个连续 64 帧 episode；使用时间轴跨帧确认同一静态物体后再画 3D cuboid。\n"
        "- 相机图仅是同时间场景上下文。包内没有外参或投影参数，禁止把图像中的像素位置直接当作某个彩色点簇的真值对应。\n\n"
        "## 审查与标注注意\n\n"
        "1. 先看 `review_samples/`，确认灰色全场点、彩色候选和局部轮廓均可辨认。\n"
        "2. 彩色候选是提议而非真值；漏掉的物体仍可能存在于灰色原始点中。\n"
        "3. 水面、多径、自船结构和稀疏远距回波不得强行标成目标；不确定时标 `ambiguous`。\n"
        "4. 优先标多帧稳定的立柱、浮标、浮块；移动船只标 `boat` 且不得作为静态外参约束。\n"
        "5. holdout 只用于最终独立评估，不得用于调阈值、改特征或训练。\n"
        "6. `frame_metadata/` 记录每帧点数、track、颜色和候选回贴覆盖率；`checksums.sha256` 用于传输后完整性校验。\n"
        "7. 原有单轨迹聚合包继续作为二级形状复核，不再作为主标注入口。\n",
        encoding="utf-8",
    )
    checksums = []
    for path in sorted(item for item in args.output.rglob("*") if item.is_file() and item.name != "checksums.sha256"):
        checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(args.output).as_posix()}")
    (args.output / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
