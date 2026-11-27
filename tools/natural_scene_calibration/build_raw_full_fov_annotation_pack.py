#!/usr/bin/env python3
"""Export blind, raw-valid LiDAR episodes with synchronized camera frames.

Only finite XYZ samples, LiDAR/image timestamps and synchronized image paths
are consumed.  No V4/V4.11 cluster, track, semantic, TF, odometry, world-state
or nominal-extrinsic information is read into the generated annotation data.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


SCHEMA = "real-lidar-raw-full-fov-annotation-pack/v1"
PROHIBITED_INPUTS = (
    "V4/V4.11 clusters",
    "V4/V4.11 tracks",
    "/tf",
    "/tf_static",
    "odometry",
    "/world/*",
    "nominal_extrinsic_projection",
)


def native_path(value: str | Path) -> Path:
    text = str(value)
    if os.name == "nt" and text.startswith("/mnt/") and len(text) > 6:
        return Path(f"{text[5].upper()}:/{text[7:]}")
    if os.name != "nt" and len(text) > 3 and text[1:3] == ":/":
        return Path(f"/mnt/{text[0].lower()}/{text[3:]}")
    return Path(text)


def write_xyz_binary_pcd(path: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype="<f4")
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("PCD input must be finite Nx3 XYZ")
    header = "\n".join(
        [
            "# .PCD v0.7 - raw valid PointCloud2 XYZ",
            "VERSION 0.7",
            "FIELDS x y z",
            "SIZE 4 4 4",
            "TYPE F F F",
            "COUNT 1 1 1",
            f"WIDTH {len(points)}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {len(points)}",
            "DATA binary",
            "",
        ]
    ).encode("ascii")
    path.write_bytes(header + np.ascontiguousarray(points).tobytes())


def write_meta(project: Path) -> None:
    classes = [
        {"title": "pillar", "shape": "cuboid_3d", "color": "#FF6347"},
        {"title": "buoy", "shape": "cuboid_3d", "color": "#00DCB4"},
        {"title": "block", "shape": "cuboid_3d", "color": "#FFC43D"},
        {"title": "boat", "shape": "cuboid_3d", "color": "#5FA5FF"},
        {"title": "background", "shape": "cuboid_3d", "color": "#808B96"},
        {"title": "ambiguous", "shape": "cuboid_3d", "color": "#E16EFF"},
    ]
    tags = [
        {
            "name": "static_for_calibration",
            "value_type": "oneof_string",
            "color": "#18A999",
            "values": ["yes", "no", "uncertain"],
            "applicable_type": "objectsOnly",
            "target_type": "entitiesOnly",
        },
        {
            "name": "reviewer_confidence",
            "value_type": "oneof_string",
            "color": "#7B61FF",
            "values": ["high", "medium", "low"],
            "applicable_type": "objectsOnly",
            "target_type": "entitiesOnly",
        },
    ]
    (project / "meta.json").write_text(
        json.dumps({"classes": classes, "tags": tags, "projectType": "point_cloud_episode"}, indent=2),
        encoding="utf-8",
    )


def render_raw_review(path: Path, image_path: Path, points: np.ndarray, title: str, audit: dict) -> None:
    figure = plt.figure(figsize=(16, 9), dpi=120, facecolor="#0e1319")
    grid = figure.add_gridspec(2, 3, wspace=0.23, hspace=0.30)
    axes = [figure.add_subplot(grid[r, c]) for r in range(2) for c in range(3)]
    for ax in axes:
        ax.set_facecolor("#141b23")
        ax.tick_params(colors="#a9b5c2", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#394653")
        ax.grid(color="#33404c", alpha=0.35, linewidth=0.5)
    axes[0].axis("off")
    axes[0].imshow(mpimg.imread(image_path))
    axes[0].set_title("Same-time image context — NO projection", color="#ffd166")
    color = "#9bb3c8"
    azimuth = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    planar = np.linalg.norm(points[:, :2], axis=1)
    panels = (
        (axes[1], points[:, 0], points[:, 1], "Raw full-FOV XY", "x / m", "y / m"),
        (axes[2], azimuth, planar, "Raw azimuth–range", "azimuth / deg", "range / m"),
        (axes[3], planar, points[:, 2], "Raw range–Z", "range / m", "z / m"),
        (axes[4], points[:, 0], points[:, 2], "Raw X–Z", "x / m", "z / m"),
    )
    for ax, x, y, subtitle, xlabel, ylabel in panels:
        ax.scatter(x, y, s=1.25, c=color, alpha=0.52, linewidths=0)
        ax.set_title(subtitle, color="#e7edf3")
        ax.set_xlabel(xlabel, color="#a9b5c2")
        ax.set_ylabel(ylabel, color="#a9b5c2")
    axes[1].scatter([0], [0], marker="*", s=90, c="#ffffff")
    axes[1].set_aspect("equal", adjustable="box")
    axes[5].axis("off")
    axes[5].text(
        0.02,
        0.98,
        "BLIND RAW-VALID EXPORT\n\n"
        f"frame: {audit['frame_index']:03d}\n"
        f"finite XYZ points: {audit['raw_valid_points']}\n"
        f"range: {audit['range_min_m']:.3f} .. {audit['range_max_m']:.3f} m\n"
        f"sync delta: {audit['sync_delta_ms']:.2f} ms\n\n"
        "No cluster / track / semantic overlay\n"
        "No TF / odometry / world truth\n"
        "No nominal extrinsic projection",
        va="top",
        color="#d9e2ea",
        family="monospace",
        fontsize=10,
    )
    figure.suptitle(title, color="#f4f7fa", fontsize=14)
    figure.savefig(path, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def export_episode(source: dict, output: Path) -> tuple[dict, list[dict]]:
    source_id, split = source["id"], source["split"]
    cache_path = native_path(source["organized_cache"])
    reference_path = native_path(source["frame_reference"])
    cache = np.load(cache_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    frames = reference.get("frames", [])
    xyz = cache["xyz"]
    if len(xyz) != len(frames):
        raise ValueError(f"frame count mismatch: {source_id}")
    project = output / f"supervisely_{split}_raw_project"
    episode = project / source_id
    pointcloud_dir = episode / "pointcloud"
    related_dir = episode / "related_images"
    frame_meta_dir = output / "frame_metadata" / source_id
    review_dir = output / "review_samples" / source_id
    for folder in (pointcloud_dir, related_dir, frame_meta_dir, review_dir):
        folder.mkdir(parents=True, exist_ok=True)
    frame_map: dict[str, str] = {}
    lineage_rows: list[dict] = []
    review_indices = {0, len(frames) // 2, len(frames) - 1}
    for frame_index, (cloud, frame) in enumerate(zip(xyz, frames)):
        lidar_stamp = int(frame["lidar_stamp_ns"])
        image_stamp = int(frame["image_stamp_ns"])
        if int(cache["lidar_stamp_ns"][frame_index]) != lidar_stamp:
            raise ValueError(f"LiDAR timestamp mismatch: {source_id}:{frame_index}")
        flat = np.asarray(cloud, dtype=np.float32).reshape((-1, 3))
        points = flat[np.isfinite(flat).all(axis=1)]
        ranges = np.linalg.norm(points, axis=1)
        stem = f"{frame_index:06d}_{lidar_stamp}"
        pcd_name = f"{stem}.pcd"
        pcd_path = pointcloud_dir / pcd_name
        write_xyz_binary_pcd(pcd_path, points)
        source_image = native_path(frame["image"])
        image_dir = related_dir / stem
        image_dir.mkdir()
        image_path = image_dir / source_image.name
        shutil.copy2(source_image, image_path)
        timestamp = dt.datetime.fromtimestamp(image_stamp / 1_000_000_000, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")
        (image_dir / f"{source_image.name}.json").write_text(
            json.dumps({"name": source_image.name, "deviceId": str(reference["camera"]), "timestamp": timestamp, "sensorsData": {}}, indent=2),
            encoding="utf-8",
        )
        frame_map[str(frame_index)] = pcd_name
        record = {
            "split": split,
            "episode_id": source_id,
            "bag_id": source["bag_id"],
            "camera": str(reference["camera"]),
            "frame_index": frame_index,
            "lidar_stamp_ns": lidar_stamp,
            "image_stamp_ns": image_stamp,
            "sync_delta_ms": float(frame["sync_delta_ms"]),
            "raw_valid_points": len(points),
            "range_min_m": float(ranges.min()),
            "range_max_m": float(ranges.max()),
            "pcd_path": f"supervisely_{split}_raw_project/{source_id}/pointcloud/{pcd_name}",
            "image_path": f"supervisely_{split}_raw_project/{source_id}/related_images/{stem}/{source_image.name}",
            "raw_xyz_sha256": hashlib.sha256(np.ascontiguousarray(points.astype("<f4")).tobytes()).hexdigest(),
            "cluster_overlay": False,
            "projection_valid": False,
        }
        (frame_meta_dir / f"{stem}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        lineage_rows.append(record)
        if frame_index in review_indices:
            render_raw_review(review_dir / f"frame_{frame_index:03d}.png", image_path, points, source_id, record)
    (episode / "frame_pointcloud_map.json").write_text(json.dumps(frame_map, indent=2), encoding="utf-8")
    (episode / "annotation.json").write_text(
        json.dumps(
            {
                "description": "Blind raw-valid XYZ episode with synchronized unprojected image context.",
                "key": uuid.uuid5(uuid.NAMESPACE_URL, f"challenger-raw:{source_id}").hex,
                "tags": [],
                "objects": [],
                "frames": [],
                "framesCount": len(frames),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return (
        {
            "id": source_id,
            "split": split,
            "bag_id": source["bag_id"],
            "camera": str(reference["camera"]),
            "frames": len(frames),
            "unique_lidar_messages": len(set(int(value) for value in cache["lidar_stamp_ns"])),
            "raw_valid_points": sum(row["raw_valid_points"] for row in lineage_rows),
            "mean_raw_valid_points": float(np.mean([row["raw_valid_points"] for row in lineage_rows])),
            "cluster_overlay": False,
            "projection_valid": False,
        },
        lineage_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    for source in spec["sources"]:
        forbidden = set(source) & {"tracked_observations", "cluster_observations", "labels"}
        if forbidden:
            raise SystemExit(f"blind raw spec contains prohibited keys for {source['id']}: {sorted(forbidden)}")
    args.output.mkdir(parents=True)
    for split in ("train", "holdout"):
        project = args.output / f"supervisely_{split}_raw_project"
        project.mkdir()
        write_meta(project)
    summaries, lineage = [], []
    for source in spec["sources"]:
        summary, rows = export_episode(source, args.output)
        summaries.append(summary)
        lineage.extend(rows)
    fields = list(lineage[0])
    with (args.output / "frame_lineage.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(lineage)
    label_fields = [
        "split", "episode_id", "object_key", "human_label", "first_frame", "last_frame",
        "static_for_calibration", "reviewer_confidence", "review_status", "reviewer_id", "reviewer_note",
    ]
    with (args.output / "track_labels_template.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=label_fields).writeheader()
    manifest = {
        "schema": SCHEMA,
        "purpose": "blind_human_annotation_of_raw_valid_lidar_with_synchronized_images",
        "prohibited_inputs": list(PROHIBITED_INPUTS),
        "point_policy": "all finite XYZ from organized cache; no range, cluster, semantic or geometric filtering",
        "episodes": len(summaries),
        "frames": len(lineage),
        "raw_valid_points": sum(item["raw_valid_points"] for item in summaries),
        "by_split": dict(Counter(item["split"] for item in summaries)),
        "by_split_camera": dict(Counter(f"{item['split']}:{item['camera']}" for item in summaries)),
        "cluster_overlay": False,
        "projection_valid": False,
        "sources": summaries,
    }
    (args.output / "package_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "README_标注说明.md").write_text(
        "# 原始全视场点云盲标包\n\n"
        "本包是严格 raw-valid 盲标版本：每帧仅删除 NaN/Inf，保留其余全部 XYZ 点；不读取、不显示 V4/V4.11 候选、轨迹或类别，也不使用 TF、里程计、世界真值和名义外参。\n\n"
        "## 标注入口\n\n"
        "- `supervisely_train_raw_project/`：090317、090835，允许用于人工标注、训练与阈值选择。\n"
        "- `supervisely_holdout_raw_project/`：091054，只在模型、特征与阈值冻结后交给独立评估人员。\n"
        "- 每个 episode 的 `annotation.json` 是空白原生标注文件；`meta.json` 已定义类别和静态性/置信度标签。\n"
        "- `frame_lineage.csv` 提供每帧 PCD、图像、纳秒时间戳、同步差和点数的不可变对应。\n\n"
        "## 规则\n\n"
        "1. 在 Point Cloud Episodes 中跨帧标注 3D cuboid，并让同一物体保持同一 object/track。\n"
        "2. 图片只用于认识场景和类别；没有 2D–3D 投影，不得按图片像素位置猜测点云对应。\n"
        "3. 只标点云中确有可辨回波的对象。自船结构、水面/多径、严重粘连或过度稀疏目标标 background/ambiguous。\n"
        "4. 正式静态候选仅限 pillar/buoy/block 且 static_for_calibration=yes；boat 必须为 no。\n"
        "5. 不得先查看彩色 V4.11 候选包；真值冻结后才能用其统计 TP/FP/FN，避免确认偏差。\n"
        "6. Supervisely 导出后保留其 annotation.json；`track_labels_template.csv` 只用于额外汇总，不替代 3D 标注。\n",
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(item for item in args.output.rglob("*") if item.is_file() and item.name != "checksums.sha256"):
        checksum_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(args.output).as_posix()}")
    (args.output / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
