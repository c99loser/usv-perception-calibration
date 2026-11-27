#!/usr/bin/env python3
"""Build a temporally diverse, stability-biased subset of a raw LiDAR package.

The source package is never modified. Selection is performed independently for
every bag/camera episode, so train/holdout separation is preserved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np


PCD_MARKER = b"DATA binary\n"
SIGNATURE_AZIMUTH_BINS = 72
SIGNATURE_HEIGHT_BINS = 12
SIGNATURE_HEIGHT_MIN_M = -3.0
SIGNATURE_HEIGHT_MAX_M = 3.0


def read_xyz_binary_pcd(path: Path) -> np.ndarray:
    content = path.read_bytes()
    if PCD_MARKER not in content:
        raise ValueError(f"not a binary PCD: {path}")
    header, payload = content.split(PCD_MARKER, 1)
    if b"FIELDS x y z\n" not in header or b"FIELDS x y z r" in header:
        raise ValueError(f"not an XYZ-only PCD: {path}")
    match = re.search(rb"(?:^|\n)POINTS (\d+)(?:\n|$)", header)
    if not match:
        raise ValueError(f"missing POINTS: {path}")
    count = int(match.group(1))
    if len(payload) != count * 12:
        raise ValueError(f"payload length mismatch: {path}")
    points = np.frombuffer(payload, dtype="<f4").reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError(f"non-finite point in blind raw PCD: {path}")
    return points


def angular_height_signature(points: np.ndarray) -> np.ndarray:
    """Return a compact nearest-range signature without using any extrinsic."""
    ranges = np.linalg.norm(points, axis=1)
    keep = (ranges >= 0.5) & (ranges <= 60.0)
    if not np.any(keep):
        return np.full(SIGNATURE_AZIMUTH_BINS * SIGNATURE_HEIGHT_BINS, np.nan, dtype=np.float32)
    points = points[keep]
    ranges = ranges[keep]
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    azimuth_bin = np.floor((azimuth + math.pi) * SIGNATURE_AZIMUTH_BINS / (2.0 * math.pi)).astype(np.int32)
    azimuth_bin = np.clip(azimuth_bin, 0, SIGNATURE_AZIMUTH_BINS - 1)
    height_bin = np.floor(
        (points[:, 2] - SIGNATURE_HEIGHT_MIN_M)
        * SIGNATURE_HEIGHT_BINS
        / (SIGNATURE_HEIGHT_MAX_M - SIGNATURE_HEIGHT_MIN_M)
    ).astype(np.int32)
    height_bin = np.clip(height_bin, 0, SIGNATURE_HEIGHT_BINS - 1)
    flat_index = azimuth_bin * SIGNATURE_HEIGHT_BINS + height_bin
    signature = np.full(SIGNATURE_AZIMUTH_BINS * SIGNATURE_HEIGHT_BINS, np.inf, dtype=np.float32)
    np.minimum.at(signature, flat_index, ranges.astype(np.float32))
    signature[~np.isfinite(signature)] = np.nan
    return signature


def signature_change(left: np.ndarray, right: np.ndarray) -> float:
    left_valid = np.isfinite(left)
    right_valid = np.isfinite(right)
    union = left_valid | right_valid
    if not np.any(union):
        return 1.0
    overlap = left_valid & right_valid
    missing_fraction = 1.0 - float(np.count_nonzero(overlap)) / float(np.count_nonzero(union))
    if np.any(overlap):
        denominator = np.maximum(np.minimum(left[overlap], right[overlap]), 1.0)
        relative = np.clip(np.abs(left[overlap] - right[overlap]) / denominator, 0.0, 1.0)
        range_change = float(np.median(relative))
    else:
        range_change = 1.0
    return 0.70 * range_change + 0.30 * missing_fraction


def compute_motion_scores(signatures: list[np.ndarray]) -> np.ndarray:
    if not signatures:
        return np.empty(0, dtype=np.float64)
    if len(signatures) == 1:
        return np.zeros(1, dtype=np.float64)
    adjacent = np.asarray(
        [signature_change(signatures[index], signatures[index + 1]) for index in range(len(signatures) - 1)],
        dtype=np.float64,
    )
    scores = np.empty(len(signatures), dtype=np.float64)
    scores[0] = adjacent[0]
    scores[-1] = adjacent[-1]
    if len(signatures) > 2:
        scores[1:-1] = 0.5 * (adjacent[:-1] + adjacent[1:])
    return scores


def select_stability_stratified(
    quality_scores: np.ndarray,
    maximum_frames: int,
    minimum_gap: int = 2,
) -> list[int]:
    """Choose the best frame in each temporal stratum with a soft gap rule."""
    count = int(len(quality_scores))
    target = min(count, int(maximum_frames))
    if target <= 0:
        return []
    edges = np.linspace(0, count, target + 1, dtype=np.int64)
    selected: list[int] = []
    for start, stop in zip(edges[:-1], edges[1:]):
        candidates = list(range(int(start), int(stop)))
        candidates.sort(key=lambda index: (float(quality_scores[index]), index))
        allowed = [
            index
            for index in candidates
            if all(abs(index - previous) >= minimum_gap for previous in selected[-2:])
        ]
        selected.append((allowed or candidates)[0])
    return sorted(selected)


def uniform_reference_indices(count: int, target: int) -> list[int]:
    if target <= 0:
        return []
    edges = np.linspace(0, count, target + 1, dtype=np.int64)
    return [int((start + stop - 1) // 2) for start, stop in zip(edges[:-1], edges[1:])]


def copy_frame(
    source_root: Path,
    output_root: Path,
    row: dict[str, str],
    source_episode_id: str,
    output_episode_id: str,
    selected_index: int,
    source_index: int,
    motion_score: float,
    quality_score: float,
) -> dict[str, object]:
    split = row["split"]
    source_pcd = source_root / row["pcd_path"]
    source_image = source_root / row["image_path"]
    source_image_json = source_image.with_name(source_image.name + ".json")
    source_metadata = (
        source_root
        / "frame_metadata"
        / source_episode_id
        / f"{source_index:06d}_{row['lidar_stamp_ns']}.json"
    )
    stem = f"{selected_index:06d}_{row['lidar_stamp_ns']}"
    episode_root = output_root / f"supervisely_{split}_raw_project" / output_episode_id
    destination_pcd = episode_root / "pointcloud" / f"{stem}.pcd"
    destination_image_dir = episode_root / "related_images" / stem
    destination_image = destination_image_dir / source_image.name
    destination_metadata = output_root / "frame_metadata" / output_episode_id / f"{stem}.json"
    destination_pcd.parent.mkdir(parents=True, exist_ok=True)
    destination_image_dir.mkdir(parents=True, exist_ok=True)
    destination_metadata.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_pcd, destination_pcd)
    shutil.copy2(source_image, destination_image)
    shutil.copy2(source_image_json, destination_image.with_name(destination_image.name + ".json"))
    metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    metadata.update(
        {
            "episode_id": output_episode_id,
            "frame_index": selected_index,
            "source_episode_id": source_episode_id,
            "source_frame_index": source_index,
            "selection_motion_score": motion_score,
            "selection_quality_score": quality_score,
            "selection_policy": "full-duration temporal strata; prefer locally stable, complete, well-synchronized frames",
            "pcd_path": destination_pcd.relative_to(output_root).as_posix(),
            "image_path": destination_image.relative_to(output_root).as_posix(),
        }
    )
    destination_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_row: dict[str, object] = dict(row)
    output_row.update(
        {
            "episode_id": output_episode_id,
            "frame_index": selected_index,
            "source_episode_id": source_episode_id,
            "source_frame_index": source_index,
            "selection_motion_score": f"{motion_score:.9f}",
            "selection_quality_score": f"{quality_score:.9f}",
            "pcd_path": destination_pcd.relative_to(output_root).as_posix(),
            "image_path": destination_image.relative_to(output_root).as_posix(),
        }
    )
    return output_row


def percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile)) if len(values) else 0.0


def zip_project(project: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(item for item in project.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(project).as_posix())


def write_checksums(root: Path) -> int:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "checksums.sha256"):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-frames-per-episode", type=int, default=64)
    parser.add_argument("--minimum-gap", type=int, default=2)
    parser.add_argument("--write-zips", action="store_true")
    args = parser.parse_args()
    if args.max_frames_per_episode <= 0 or args.max_frames_per_episode >= 100:
        raise SystemExit("--max-frames-per-episode must be between 1 and 99")
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    source_manifest = json.loads((args.source / "package_manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("schema") != "real-lidar-raw-full-fov-annotation-pack/v2-official-direct":
        raise SystemExit("source must be the official-direct V2 package")
    source_rows = list(csv.DictReader((args.source / "frame_lineage.csv").open(encoding="utf-8-sig")))
    rows_by_episode: dict[str, list[dict[str, str]]] = {}
    for row in source_rows:
        rows_by_episode.setdefault(row["episode_id"], []).append(row)
    args.output.mkdir(parents=True)
    output_rows: list[dict[str, object]] = []
    episode_reports: list[dict[str, object]] = []

    for split in ("train", "holdout"):
        source_project = args.source / f"supervisely_{split}_raw_project"
        output_project = args.output / f"supervisely_{split}_raw_project"
        output_project.mkdir(parents=True)
        shutil.copy2(source_project / "meta.json", output_project / "meta.json")
        for source_episode in sorted(path for path in source_project.iterdir() if path.is_dir()):
            rows = sorted(rows_by_episode[source_episode.name], key=lambda row: int(row["frame_index"]))
            points_by_frame = [read_xyz_binary_pcd(args.source / row["pcd_path"]) for row in rows]
            signatures = [angular_height_signature(points) for points in points_by_frame]
            motion_scores = compute_motion_scores(signatures)
            point_counts = np.asarray([len(points) for points in points_by_frame], dtype=np.float64)
            median_points = max(float(np.median(point_counts)), 1.0)
            completeness_penalty = np.clip((median_points - point_counts) / median_points, 0.0, 1.0)
            sync_penalty = np.asarray([float(row["sync_delta_ms"]) / 60.0 for row in rows], dtype=np.float64)
            quality_scores = motion_scores + 0.10 * completeness_penalty + 0.03 * sync_penalty
            selected = select_stability_stratified(
                quality_scores,
                args.max_frames_per_episode,
                minimum_gap=max(1, args.minimum_gap),
            )
            uniform = uniform_reference_indices(len(rows), len(selected))
            output_episode_id = (
                source_episode.name.removesuffix("_all") + f"_stable{args.max_frames_per_episode}"
            )
            output_episode = output_project / output_episode_id
            frame_map: dict[str, str] = {}
            for selected_index, source_index in enumerate(selected):
                output_row = copy_frame(
                    args.source,
                    args.output,
                    rows[source_index],
                    source_episode.name,
                    output_episode_id,
                    selected_index,
                    source_index,
                    float(motion_scores[source_index]),
                    float(quality_scores[source_index]),
                )
                output_rows.append(output_row)
                frame_map[str(selected_index)] = Path(str(output_row["pcd_path"])).name
            (output_episode / "frame_pointcloud_map.json").write_text(
                json.dumps(frame_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            annotation = json.loads((source_episode / "annotation.json").read_text(encoding="utf-8"))
            annotation.update(
                {
                    "description": "Stability-biased, full-duration subset of official-direct raw XYZ; no candidates or projection.",
                    "key": hashlib.sha256(output_episode_id.encode("utf-8")).hexdigest()[:32],
                    "objects": [],
                    "frames": [],
                    "framesCount": len(selected),
                }
            )
            (output_episode / "annotation.json").write_text(
                json.dumps(annotation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            selected_motion = motion_scores[selected]
            uniform_motion = motion_scores[uniform]
            episode_reports.append(
                {
                    "split": split,
                    "source_episode_id": source_episode.name,
                    "episode_id": output_episode_id,
                    "bag_id": rows[0]["bag_id"],
                    "camera": rows[0]["camera"],
                    "source_frames": len(rows),
                    "selected_frames": len(selected),
                    "reduction_fraction": 1.0 - len(selected) / len(rows),
                    "source_frame_indices": selected,
                    "selected_time_start_ns": int(rows[selected[0]]["lidar_stamp_ns"]),
                    "selected_time_end_ns": int(rows[selected[-1]]["lidar_stamp_ns"]),
                    "source_time_start_ns": int(rows[0]["lidar_stamp_ns"]),
                    "source_time_end_ns": int(rows[-1]["lidar_stamp_ns"]),
                    "selected_motion_median": percentile(selected_motion, 0.5),
                    "selected_motion_p90": percentile(selected_motion, 0.9),
                    "uniform_motion_median": percentile(uniform_motion, 0.5),
                    "uniform_motion_p90": percentile(uniform_motion, 0.9),
                    "motion_median_improvement_fraction": (
                        0.0
                        if percentile(uniform_motion, 0.5) == 0.0
                        else 1.0 - percentile(selected_motion, 0.5) / percentile(uniform_motion, 0.5)
                    ),
                    "raw_valid_points": int(sum(int(rows[index]["raw_valid_points"]) for index in selected)),
                    "max_sync_delta_ms": max(float(rows[index]["sync_delta_ms"]) for index in selected),
                }
            )

    lineage_fields = list(source_rows[0].keys())
    insert_at = lineage_fields.index("frame_index") + 1
    for field in reversed(
        ["source_episode_id", "source_frame_index", "selection_motion_score", "selection_quality_score"]
    ):
        lineage_fields.insert(insert_at, field)
    with (args.output / "frame_lineage.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=lineage_fields)
        writer.writeheader()
        writer.writerows(output_rows)
    audit = {
        "schema": "raw-lidar-stability-selection-audit/v1",
        "source_package": str(args.source),
        "maximum_frames_per_bag_camera": args.max_frames_per_episode,
        "minimum_source_frame_gap": args.minimum_gap,
        "motion_metric": "adjacent change of sensor-local azimuth-height nearest-range signature; no extrinsic used",
        "selection_policy": "one low-score frame per full-duration temporal stratum; score combines motion, completeness and sync",
        "episodes": episode_reports,
    }
    (args.output / "selection_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(args.source / "track_labels_template.csv", args.output / "track_labels_template.csv")
    total_points = sum(int(row["raw_valid_points"]) for row in output_rows)
    manifest = {
        "schema": "real-lidar-raw-full-fov-annotation-pack/v3-stability-stratified",
        "purpose": "reduced-workload blind annotation with full-duration coverage and stability preference",
        "source_package": str(args.source),
        "source_schema": source_manifest["schema"],
        "source_policy": source_manifest["source_policy"],
        "point_policy": source_manifest["point_policy"],
        "selection_policy": audit["selection_policy"],
        "motion_metric": audit["motion_metric"],
        "max_frames_per_bag_camera": args.max_frames_per_episode,
        "camera_lidar_pairs": source_manifest["camera_lidar_pairs"],
        "prohibited_inputs": source_manifest["prohibited_inputs"],
        "episodes": len(episode_reports),
        "frames": len(output_rows),
        "source_frames": len(source_rows),
        "reduction_fraction": 1.0 - len(output_rows) / len(source_rows),
        "raw_valid_points": total_points,
        "cluster_overlay": False,
        "projection_valid": False,
        "splits": {
            split: sum(1 for row in output_rows if row["split"] == split) for split in ("train", "holdout")
        },
        "sources": episode_reports,
    }
    (args.output / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "README_标注说明.md").write_text(
        "# 官方原始点云精简标注包 V3\n\n"
        "- 每个 bag × 方位最多 64 帧；训练两 bag 合计每方位 128 帧，091054 留出每方位 64 帧。\n"
        "- 选帧覆盖每个 episode 的完整时间范围，并优先局部变化较小、点数完整、同步误差较低的帧。\n"
        "- 运动分数只由传感器坐标系原始 XYZ 的相邻帧签名计算；未使用外参、TF、里程计、候选或图像投影。\n"
        "- `source_frame_index` 可回溯完整 V2；`selection_audit.json` 给出逐 episode 的选帧和运动统计。\n"
        "- 先标 train ZIP。holdout ZIP 必须封存至模型、规则和阈值冻结后再由独立评估人员打开。\n"
        "- 不应删除所有运动/困难帧；本包降低其比例，但仍保持全时段覆盖。需要补充困难样本时从完整 V2 定向追加。\n",
        encoding="utf-8",
    )
    checksum_count = write_checksums(args.output)
    if args.write_zips:
        for split in ("train", "holdout"):
            zip_project(
                args.output / f"supervisely_{split}_raw_project",
                args.output.with_name(args.output.name + f"_supervisely_{split}.zip"),
            )
    result = dict(manifest)
    result["checksums"] = checksum_count
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
