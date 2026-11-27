#!/usr/bin/env python3
"""Convert V3 raw-LiDAR cuboids into leakage-safe sparse observations.

Only direct manual boxes and propagation boxes explicitly marked ``reviewed``
are admitted.  The source V3 package is read-only.  Z, support count and the
observation representation are derived from actual points inside the annotated
oriented XY footprint; the annotator-entered Z extent is retained only for
traceability and is never used to select points.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "real-lidar-sparse-training/v3.1"
PCD_MARKER = b"DATA binary\n"
CLASSES = ("background", "buoy", "pillar", "block", "boat")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
STATIC_CLASSES = frozenset(("buoy", "pillar", "block"))
OBSERVATION_TO_ID = {"point": 0, "sparse_cluster": 1, "cluster": 2}
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4}


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
        raise ValueError(f"non-finite point: {path}")
    return points.copy()


def oriented_xy_indices(
    points: np.ndarray,
    center_xyz: list[float],
    dimensions_xyz: list[float],
    yaw_rad: float,
) -> np.ndarray:
    """Return raw point indices inside an oriented XY rectangle; ignore Z."""
    center = np.asarray(center_xyz, dtype=np.float64)
    dimensions = np.asarray(dimensions_xyz, dtype=np.float64)
    if center.shape != (3,) or dimensions.shape != (3,):
        raise ValueError("center_xyz and dimensions_xyz must contain three values")
    if not np.isfinite(center).all() or not np.isfinite(dimensions).all() or not math.isfinite(yaw_rad):
        raise ValueError("box geometry must be finite")
    if np.any(dimensions <= 0.0):
        raise ValueError("box dimensions must be positive")
    delta = points[:, :2].astype(np.float64) - center[:2]
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    local_x = cosine * delta[:, 0] + sine * delta[:, 1]
    local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    keep = (np.abs(local_x) <= dimensions[0] / 2.0) & (np.abs(local_y) <= dimensions[1] / 2.0)
    return np.flatnonzero(keep).astype(np.int64)


def source_kind(box: dict[str, Any]) -> str:
    propagation = box.get("propagation")
    if propagation is None:
        return "direct_human"
    if isinstance(propagation, dict) and propagation.get("reviewed") is True:
        return "reviewed_propagation"
    return "unreviewed_propagation"


def observation_type(support_count: int) -> str:
    if support_count <= 0:
        return "no_return"
    if support_count == 1:
        return "point"
    if support_count <= 3:
        return "sparse_cluster"
    return "cluster"


def robust_geometry(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if len(points) == 0:
        return np.full(3, np.nan, dtype=np.float32), np.zeros(3, dtype=np.float32), "no_return"
    center = np.median(points, axis=0).astype(np.float32)
    if len(points) >= 6:
        lower = np.quantile(points, 0.05, axis=0)
        upper = np.quantile(points, 0.95, axis=0)
    else:
        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)
    extent = np.maximum(upper - lower, 0.0).astype(np.float32)
    quality = "single_point" if len(points) == 1 else ("observed_sparse" if len(points) <= 3 else "observed")
    return center, extent, quality


def geometry_weight(support_count: int, confidence: str) -> float:
    base = 0.0 if support_count <= 0 else (0.25 if support_count == 1 else (0.5 if support_count <= 3 else min(1.0, math.log2(support_count + 1) / 4.0)))
    return float(base * CONFIDENCE_WEIGHT.get(confidence, 0.4))


def track_gate(records: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    accepted = [record for record in records if record["train_eligible"]]
    if not accepted:
        return False, ["no_admitted_observation"]
    if accepted[0]["class_name"] not in STATIC_CLASSES or any(record["static_for_calibration"] != "yes" for record in accepted):
        reasons.append("not_static_candidate")
    if len(accepted) < 5:
        reasons.append("fewer_than_5_admitted_frames")
    if sum(record["support_count"] for record in accepted) < 10:
        reasons.append("fewer_than_10_total_support_points")
    if any(record["jump_anomaly"] for record in accepted):
        reasons.append("consecutive_center_jump_gt_5m")
    if len({record["class_name"] for record in accepted}) != 1:
        reasons.append("class_inconsistent")
    return not reasons, reasons


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_lineage(root: Path) -> dict[tuple[str, int], dict[str, str]]:
    rows = _read_csv(root / "frame_lineage.csv")
    result = {(row["episode_id"], int(row["frame_index"])): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("frame_lineage.csv contains duplicate episode/frame rows")
    return result


def _deconflict_frame(records: list[dict[str, Any]]) -> None:
    """Remove positive-owned points from negative boxes and ambiguous positive overlaps."""
    owners: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["manual_admitted"]:
            for point_index in record["support_indices_raw"]:
                owners[int(point_index)].append(record)
    removed: dict[str, set[int]] = defaultdict(set)
    for point_index, point_owners in owners.items():
        if len(point_owners) < 2:
            continue
        positives = [owner for owner in point_owners if owner["class_name"] in STATIC_CLASSES]
        negatives = [owner for owner in point_owners if owner["class_name"] not in STATIC_CLASSES]
        if positives:
            for owner in negatives:
                removed[owner["annotation_id"]].add(point_index)
        if len({owner["track_id"] for owner in positives}) > 1:
            for owner in positives:
                removed[owner["annotation_id"]].add(point_index)
                owner["positive_overlap_conflict"] = True
    for record in records:
        rejected = removed.get(record["annotation_id"], set())
        record["overlap_points_removed"] = len(rejected)
        if rejected:
            record["support_indices"] = np.asarray(
                [index for index in record["support_indices_raw"] if int(index) not in rejected], dtype=np.int64
            )
        else:
            record["support_indices"] = record["support_indices_raw"]


def build(source_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output already exists; refusing to overwrite: {output_root}")
    train_project = source_root / "supervisely_train_raw_project"
    holdout_project = source_root / "supervisely_holdout_raw_project"
    if not train_project.is_dir() or not holdout_project.is_dir():
        raise FileNotFoundError("source V3 train/holdout project is incomplete")
    lineage = _load_lineage(source_root)
    sidecars = sorted(train_project.glob("train_*_stable64.json"))
    if len(sidecars) != 12:
        raise ValueError(f"expected 12 training sidecars, found {len(sidecars)}")

    records: list[dict[str, Any]] = []
    sidecar_hashes: dict[str, str] = {}
    for sidecar in sidecars:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError(f"unsupported sidecar schema: {sidecar}")
        episode_id = data["episode_id"]
        sidecar_hashes[sidecar.name] = _hash(sidecar)
        episode_root = train_project / episode_id
        frame_map = json.loads((episode_root / "frame_pointcloud_map.json").read_text(encoding="utf-8"))
        for frame_key, boxes in data.get("frames", {}).items():
            frame_index = int(frame_key)
            row = lineage[(episode_id, frame_index)]
            mapped_name = frame_map[str(frame_index)]
            pcd_path = episode_root / "pointcloud" / mapped_name
            if Path(row["pcd_path"]).name != mapped_name:
                raise ValueError(f"lineage/frame map mismatch: {episode_id} frame {frame_index}")
            points = read_xyz_binary_pcd(pcd_path)
            frame_records: list[dict[str, Any]] = []
            for box_index, box in enumerate(boxes):
                annotation_id = str(box.get("id") or f"{episode_id}:{frame_index}:{box_index}")
                kind = source_kind(box)
                class_name = str(box.get("class_name", ""))
                if class_name not in CLASS_TO_ID:
                    raise ValueError(f"unknown class {class_name!r}: {annotation_id}")
                raw_indices = oriented_xy_indices(points, box["center_xyz"], box["dimensions_xyz"], float(box["yaw_rad"]))
                record: dict[str, Any] = {
                    "annotation_id": annotation_id,
                    "episode_id": episode_id,
                    "bag_id": row["bag_id"],
                    "camera": row["camera"],
                    "frame_index": frame_index,
                    "source_frame_index": int(row["source_frame_index"]),
                    "lidar_stamp_ns": int(row["lidar_stamp_ns"]),
                    "image_stamp_ns": int(row["image_stamp_ns"]),
                    "sync_delta_ms": float(row["sync_delta_ms"]),
                    "pcd_path": row["pcd_path"],
                    "image_path": row["image_path"],
                    "track_id": str(box.get("track_id", "")),
                    "class_name": class_name,
                    "class_id": CLASS_TO_ID[class_name],
                    "static_for_calibration": str(box.get("static_for_calibration", "uncertain")),
                    "reviewer_confidence": str(box.get("reviewer_confidence", "low")),
                    "source_kind": kind,
                    "manual_admitted": kind != "unreviewed_propagation",
                    "source_box_id": str(box.get("source_box_id", "")),
                    "original_center_x": float(box["center_xyz"][0]),
                    "original_center_y": float(box["center_xyz"][1]),
                    "original_center_z": float(box["center_xyz"][2]),
                    "original_size_x": float(box["dimensions_xyz"][0]),
                    "original_size_y": float(box["dimensions_xyz"][1]),
                    "original_size_z": float(box["dimensions_xyz"][2]),
                    "original_yaw_rad": float(box["yaw_rad"]),
                    "support_indices_raw": raw_indices,
                    "positive_overlap_conflict": False,
                    "jump_anomaly": False,
                    "source_dimension_anomaly": False,
                }
                frame_records.append(record)
                records.append(record)
            _deconflict_frame(frame_records)
            for record in frame_records:
                support_points = points[record["support_indices"]]
                center, extent, z_quality = robust_geometry(support_points)
                record.update(
                    {
                        "support_points": support_points.astype(np.float32),
                        "support_count_raw": len(record["support_indices_raw"]),
                        "support_count": len(support_points),
                        "observation_type": observation_type(len(support_points)),
                        "derived_center_x": float(center[0]),
                        "derived_center_y": float(center[1]),
                        "derived_center_z": float(center[2]),
                        "support_extent_x": float(extent[0]),
                        "support_extent_y": float(extent[1]),
                        "support_extent_z": float(extent[2]),
                        "z_quality": z_quality,
                    }
                )

    # Apply track-local anomaly flags only to adjacent selected frames.  Gaps are
    # not compared because the platform may have moved substantially meanwhile.
    admitted_tracks: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["manual_admitted"]:
            admitted_tracks[(record["episode_id"], record["track_id"])].append(record)
    for track_records in admitted_tracks.values():
        track_records.sort(key=lambda item: item["frame_index"])
        for left, right in zip(track_records, track_records[1:]):
            if right["frame_index"] != left["frame_index"] + 1:
                continue
            left_xy = np.asarray((left["original_center_x"], left["original_center_y"]))
            right_xy = np.asarray((right["original_center_x"], right["original_center_y"]))
            if float(np.linalg.norm(right_xy - left_xy)) > 5.0:
                left["jump_anomaly"] = right["jump_anomaly"] = True
            left_size = np.asarray((left["original_size_x"], left["original_size_y"], left["original_size_z"]))
            right_size = np.asarray((right["original_size_x"], right["original_size_y"], right["original_size_z"]))
            if float(np.max(np.abs(right_size - left_size))) > 2.0:
                left["source_dimension_anomaly"] = right["source_dimension_anomaly"] = True

    for record in records:
        record["train_eligible"] = bool(
            record["manual_admitted"]
            and record["support_count"] > 0
            and not record["positive_overlap_conflict"]
        )
        record["geometry_eligible"] = bool(
            record["train_eligible"] and record["support_count"] >= 4 and not record["jump_anomaly"]
        )
        record["static_candidate_target"] = int(
            record["class_name"] in STATIC_CLASSES and record["static_for_calibration"] == "yes"
        )
        record["geometry_weight"] = geometry_weight(record["support_count"], record["reviewer_confidence"])

    track_rows: list[dict[str, Any]] = []
    eligible_track_keys: set[tuple[str, str]] = set()
    for key, track_records in sorted(admitted_tracks.items()):
        eligible, reasons = track_gate(track_records)
        if eligible:
            eligible_track_keys.add(key)
        accepted = [record for record in track_records if record["train_eligible"]]
        track_rows.append(
            {
                "episode_id": key[0],
                "track_id": key[1],
                "class_name": track_records[0]["class_name"],
                "static_for_calibration": track_records[0]["static_for_calibration"],
                "admitted_frames": len(accepted),
                "total_support_points": sum(record["support_count"] for record in accepted),
                "point_observations": sum(record["observation_type"] == "point" for record in accepted),
                "sparse_cluster_observations": sum(record["observation_type"] == "sparse_cluster" for record in accepted),
                "cluster_observations": sum(record["observation_type"] == "cluster" for record in accepted),
                "calibration_eligible": eligible,
                "gate_reasons": "|".join(reasons),
            }
        )
    for record in records:
        record["calibration_eligible"] = bool(
            record["train_eligible"]
            and (record["episode_id"], record["track_id"]) in eligible_track_keys
        )
        record["calibration_weight"] = record["geometry_weight"] if record["calibration_eligible"] else 0.0

    observations = [record for record in records if record["train_eligible"]]
    suggestions = [record for record in records if record["source_kind"] == "unreviewed_propagation"]
    quarantine = [
        record for record in records
        if record["manual_admitted"] and not record["train_eligible"]
    ]
    output_root.mkdir(parents=True)

    points_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    offsets = [0]
    for row_index, record in enumerate(observations):
        record["row_index"] = row_index
        record["point_offset_start"] = offsets[-1]
        points_parts.append(record["support_points"])
        index_parts.append(record["support_indices"].astype(np.int64))
        offsets.append(offsets[-1] + record["support_count"])
        record["point_offset_end"] = offsets[-1]
    all_points = np.concatenate(points_parts, axis=0) if points_parts else np.empty((0, 3), np.float32)
    all_indices = np.concatenate(index_parts, axis=0) if index_parts else np.empty(0, np.int64)
    np.savez_compressed(
        output_root / "observations.npz",
        points_xyz=all_points,
        raw_point_indices=all_indices,
        point_offsets=np.asarray(offsets, dtype=np.int64),
        centers_xyz=np.asarray([[r["derived_center_x"], r["derived_center_y"], r["derived_center_z"]] for r in observations], dtype=np.float32),
        support_extents_xyz=np.asarray([[r["support_extent_x"], r["support_extent_y"], r["support_extent_z"]] for r in observations], dtype=np.float32),
        class_ids=np.asarray([r["class_id"] for r in observations], dtype=np.int8),
        static_candidate_targets=np.asarray([r["static_candidate_target"] for r in observations], dtype=np.int8),
        observation_type_ids=np.asarray([OBSERVATION_TO_ID[r["observation_type"]] for r in observations], dtype=np.int8),
        geometry_eligible=np.asarray([r["geometry_eligible"] for r in observations], dtype=np.bool_),
        calibration_eligible=np.asarray([r["calibration_eligible"] for r in observations], dtype=np.bool_),
        geometry_weights=np.asarray([r["geometry_weight"] for r in observations], dtype=np.float32),
        calibration_weights=np.asarray([r["calibration_weight"] for r in observations], dtype=np.float32),
    )

    observation_fields = [
        "row_index", "annotation_id", "episode_id", "bag_id", "camera", "frame_index", "source_frame_index",
        "lidar_stamp_ns", "image_stamp_ns", "sync_delta_ms", "pcd_path", "image_path", "track_id", "class_name",
        "class_id", "static_for_calibration", "static_candidate_target", "reviewer_confidence", "source_kind",
        "source_box_id", "observation_type", "support_count_raw", "overlap_points_removed", "support_count",
        "point_offset_start", "point_offset_end", "derived_center_x", "derived_center_y", "derived_center_z",
        "support_extent_x", "support_extent_y", "support_extent_z", "z_quality", "original_center_x",
        "original_center_y", "original_center_z", "original_size_x", "original_size_y", "original_size_z",
        "original_yaw_rad", "jump_anomaly", "source_dimension_anomaly", "geometry_eligible", "geometry_weight",
        "calibration_eligible", "calibration_weight",
    ]
    _write_csv(output_root / "observations.csv", observations, observation_fields)
    suggestion_fields = [field for field in observation_fields if field not in {"row_index", "point_offset_start", "point_offset_end"}]
    _write_csv(output_root / "unreviewed_propagation_suggestions.csv", suggestions, suggestion_fields)
    _write_csv(output_root / "quarantine.csv", quarantine, suggestion_fields)
    _write_csv(
        output_root / "track_summary.csv",
        track_rows,
        ["episode_id", "track_id", "class_name", "static_for_calibration", "admitted_frames", "total_support_points", "point_observations", "sparse_cluster_observations", "cluster_observations", "calibration_eligible", "gate_reasons"],
    )

    by_episode = Counter(record["episode_id"] for record in observations)
    positive_by_episode = Counter(record["episode_id"] for record in observations if record["static_candidate_target"])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "source_root": str(source_root),
        "source_sidecars": len(sidecars),
        "source_figures": len(records),
        "direct_human_figures": sum(record["source_kind"] == "direct_human" for record in records),
        "reviewed_propagation_figures": sum(record["source_kind"] == "reviewed_propagation" for record in records),
        "unreviewed_propagation_excluded": len(suggestions),
        "training_observations": len(observations),
        "quarantined_manual_observations": len(quarantine),
        "support_points": int(len(all_points)),
        "observation_types": dict(sorted(Counter(record["observation_type"] for record in observations).items())),
        "classes": dict(sorted(Counter(record["class_name"] for record in observations).items())),
        "static_candidate_observations": sum(record["static_candidate_target"] for record in observations),
        "geometry_eligible_observations": sum(record["geometry_eligible"] for record in observations),
        "calibration_eligible_observations": sum(record["calibration_eligible"] for record in observations),
        "calibration_eligible_tracks": len(eligible_track_keys),
        "manual_observations_with_jump_flag": sum(record["jump_anomaly"] for record in observations),
        "manual_observations_with_source_dimension_flag": sum(record["source_dimension_anomaly"] for record in observations),
        "positive_overlap_conflicts_quarantined": sum(record["positive_overlap_conflict"] for record in quarantine),
        "overlap_points_removed": sum(record["overlap_points_removed"] for record in records if record["manual_admitted"]),
        "by_episode": {
            episode: {"training_observations": by_episode[episode], "static_candidate_observations": positive_by_episode[episode]}
            for episode in sorted(by_episode)
        },
        "gates": {
            "source_v3_read_only_contract": True,
            "unreviewed_propagation_excluded_from_supervision": not any(
                record["source_kind"] == "unreviewed_propagation" for record in observations
            ),
            "z_recomputed_from_actual_support": True,
            "holdout_kept_sealed": True,
            "all_12_train_episodes_have_at_least_20_observations": len(by_episode) == 12 and min(by_episode.values()) >= 20,
            "all_12_train_episodes_have_at_least_20_static_candidates": len(positive_by_episode) == 12 and min(positive_by_episode.values()) >= 20,
        },
    }
    report["ready_for_first_pass_training"] = all(report["gates"].values())
    report["ready_for_holdout_evaluation"] = False
    (output_root / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema": SCHEMA,
        "source_root": str(source_root),
        "source_sidecar_sha256": sidecar_hashes,
        "class_to_id": CLASS_TO_ID,
        "observation_type_to_id": OBSERVATION_TO_ID,
        "admission_rule": "direct_human OR propagation.reviewed==true",
        "point_selection": "inside annotated oriented XY footprint; annotation Z ignored",
        "split_policy": "both official training bags remain grouped by bag; official holdout is sealed",
        "prohibited_inputs": ["TF", "odometry", "world state", "ground truth", "extrinsic projection"],
        "files": ["observations.csv", "observations.npz", "track_summary.csv", "unreviewed_propagation_suggestions.csv", "quarantine.csv", "audit_report.json"],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = f"""# 真实稀疏 LiDAR 训练资产 V3.1\n\n此目录由 V3 原始全视场点云和人工框无损派生；源 V3 未被修改。\n\n- 监督训练只纳入直接人工框和 `propagation.reviewed=true` 的框。\n- 未复核传播框仅位于 `unreviewed_propagation_suggestions.csv`，不得作为标签。\n- 点支持仅由人工框的旋转 XY 范围选取；人工填写的 Z 不参与点筛选。\n- 1 点为 `point`，2–3 点为 `sparse_cluster`，4 点及以上为 `cluster`。\n- `observations.csv` 与 `observations.npz` 按 `row_index`/`point_offsets` 一一对应。\n- `geometry_eligible=false` 时禁止监督尺寸和朝向；点与稀疏簇仍可监督中心、类别或目标性。\n- `calibration_eligible` 只是严格轨迹门禁结果，不等同于已完成外参标定验证。\n- 官方第三 bag/holdout 未读取标注，当前不可宣称通过留出验证。\n\n本次构建：{len(observations)} 条训练观测，{len(all_points)} 个支持点，{len(suggestions)} 条未复核传播建议被隔离。\n"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")

    checksum_lines = []
    for path in sorted(item for item in output_root.iterdir() if item.is_file() and item.name != "checksums.sha256"):
        checksum_lines.append(f"{_hash(path)}  {path.name}")
    (output_root / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.source_root.resolve(), args.output_root.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
