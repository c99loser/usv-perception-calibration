#!/usr/bin/env python3
"""Build a leakage-safe, track-aware real-LiDAR annotation package.

The package joins V4 raw clusters (which retain bounded sensor-frame points)
to V4.11 tracked clusters by timestamp and ``source_cluster_index``.  It never
uses TF, odometry, world state, ground truth, or an extrinsic projection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "real-lidar-manual-annotation-pack/v4.11"
LABELS = ("pillar", "buoy", "block", "boat", "background", "ambiguous")
STATIC_VALUES = ("yes", "no", "uncertain")
PROHIBITED_INPUTS = ("/tf", "/tf_static", "odometry", "/world/*")


def _bag_id(value: object) -> str:
    return Path(str(value).replace("\\", "/")).name


def _evenly_spaced(items: list[dict], limit: int) -> list[dict]:
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    indices = {
        round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)
    }
    return [items[index] for index in sorted(indices)]


def _representatives(candidates: list[dict], max_per_track: int) -> list[dict]:
    by_track: dict[int, list[dict]] = defaultdict(list)
    for item in candidates:
        by_track[item["track_id"]].append(item)
    selected: list[dict] = []
    for track_items in by_track.values():
        track_items.sort(key=lambda item: item["frame_index"])
        eligible = [item for item in track_items if item["geometry_eligible"]]
        rejected = [item for item in track_items if not item["geometry_eligible"]]
        chosen: list[dict] = []
        if eligible:
            chosen.extend(_evenly_spaced(eligible, max(1, max_per_track - bool(rejected))))
        if rejected and len(chosen) < max_per_track:
            chosen.extend(_evenly_spaced(rejected, max_per_track - len(chosen)))
        if len(chosen) < max_per_track:
            remaining = [item for item in track_items if item not in chosen]
            chosen.extend(_evenly_spaced(remaining, max_per_track - len(chosen)))
        selected.extend(chosen[:max_per_track])
    return selected


def _select(candidates: list[dict], limit: int, max_per_track: int) -> list[dict]:
    representatives = _representatives(candidates, max_per_track)
    eligible = sorted(
        (item for item in representatives if item["geometry_eligible"]),
        key=lambda item: (-item["cluster_points"], item["item_id"]),
    )
    rejected = sorted(
        (item for item in representatives if not item["geometry_eligible"]),
        key=lambda item: (-item["cluster_points"], item["item_id"]),
    )
    eligible_target = int(math.ceil(0.75 * limit))
    chosen = eligible[:eligible_target] + rejected[: limit - eligible_target]
    if len(chosen) < limit:
        used = {item["item_id"] for item in chosen}
        remainder = [item for item in eligible + rejected if item["item_id"] not in used]
        chosen.extend(remainder[: limit - len(chosen)])
    return sorted(chosen, key=lambda item: (item["frame_index"], item["track_id"], item["cluster_index"]))


def _write_ply(path: Path, points: list[list[float]]) -> None:
    header = "\n".join(
        (
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "end_header",
            "",
        )
    )
    body = "\n".join(f"{float(x):.4f} {float(y):.4f} {float(z):.4f}" for x, y, z in points)
    path.write_text(header + body + "\n", encoding="ascii")


def _load_source(spec: dict, root: Path) -> tuple[list[dict], dict]:
    raw_path = Path(spec["raw_observations"])
    tracked_path = Path(spec["tracked_observations"])
    if not raw_path.is_absolute():
        raw_path = root / raw_path
    if not tracked_path.is_absolute():
        tracked_path = root / tracked_path
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    tracked = json.loads(tracked_path.read_text(encoding="utf-8"))
    if raw.get("purpose") != "natural_scene_sensor_only_observation_extraction":
        raise ValueError(f"{raw_path}: not a sensor-only extraction")
    if not raw.get("cluster_points_exported_for_manual_annotation"):
        raise ValueError(f"{raw_path}: bounded cluster points were not exported")
    if raw.get("camera") != tracked.get("camera") or _bag_id(raw.get("bag")) != _bag_id(tracked.get("bag")):
        raise ValueError(f"{spec['id']}: raw/tracked camera or bag mismatch")
    raw_frames = raw.get("frames", [])
    tracked_frames = tracked.get("frames", [])
    if len(raw_frames) != len(tracked_frames):
        raise ValueError(f"{spec['id']}: raw/tracked frame count mismatch")
    candidates: list[dict] = []
    for frame_index, (raw_frame, tracked_frame) in enumerate(zip(raw_frames, tracked_frames)):
        raw_key = (int(raw_frame["image_stamp_ns"]), int(raw_frame["lidar_stamp_ns"]))
        tracked_key = (int(tracked_frame["image_stamp_ns"]), int(tracked_frame["lidar_stamp_ns"]))
        if raw_key != tracked_key:
            raise ValueError(f"{spec['id']}: timestamp mismatch at frame {frame_index}")
        for cluster in tracked_frame.get("clusters", []):
            source_index = int(cluster.get("source_cluster_index", -1))
            if not 0 <= source_index < len(raw_frame.get("clusters", [])):
                raise ValueError(f"{spec['id']}: invalid source_cluster_index")
            raw_cluster = raw_frame["clusters"][source_index]
            points = raw_cluster.get("points_sensor_m", [])
            if len(points) < int(spec.get("min_sampled_points", 12)):
                continue
            track = cluster.get("temporal_track", {})
            quality = cluster.get("calibration_quality", {})
            track_id = int(track["track_id"])
            item_id = f"{spec['id']}__t{track_id:04d}__f{frame_index:03d}__c{source_index:03d}"
            candidates.append(
                {
                    "item_id": item_id,
                    "source_id": spec["id"],
                    "split_role": spec["split"],
                    "bag_id": _bag_id(raw.get("bag")),
                    "camera": raw["camera"],
                    "window_start_index": raw.get("start_index"),
                    "frame_index": frame_index,
                    "cluster_index": source_index,
                    "track_id": track_id,
                    "image_stamp_ns": raw_key[0],
                    "lidar_stamp_ns": raw_key[1],
                    "sync_delta_ms": raw_frame["sync_delta_ms"],
                    "source_image": raw_frame["image"],
                    "points_sensor_m": points,
                    "sampled_points": len(points),
                    "cluster_points": int(raw_cluster["points"]),
                    "raw_center_m": raw_cluster["center_m"],
                    "calibration_center_m": cluster.get("calibration_center_m"),
                    "reference_extent_m": cluster.get("reference_extent_m", cluster.get("extent_m")),
                    "range_m": cluster.get("range_m"),
                    "pca_linearity": raw_cluster.get("pca_linearity"),
                    "center_standard_error_m": quality.get("center_standard_error_m"),
                    "geometry_eligible": quality.get("eligible") is True,
                    "geometry_rejection_reasons": quality.get("reasons", []),
                    "raw_observations": str(raw_path),
                    "tracked_observations": str(tracked_path),
                }
            )
    summary = {
        "id": spec["id"],
        "split": spec["split"],
        "bag_id": _bag_id(raw.get("bag")),
        "camera": raw["camera"],
        "candidate_items": len(candidates),
        "candidate_tracks": len({item["track_id"] for item in candidates}),
        "eligible_candidates": sum(item["geometry_eligible"] for item in candidates),
        "raw_observations": str(raw_path),
        "tracked_observations": str(tracked_path),
    }
    return candidates, summary


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec.get("schema") != "real-lidar-annotation-pack-build-spec/v1":
        raise SystemExit("unsupported build spec")
    sources = spec.get("sources", [])
    if not sources or {item.get("split") for item in sources} != {"train", "holdout"}:
        raise SystemExit("both train and holdout sources are required")
    source_ids = [item.get("id") for item in sources]
    if len(source_ids) != len(set(source_ids)):
        raise SystemExit("source ids must be unique")

    args.output.mkdir(parents=True)
    (args.output / "annotations").mkdir()
    selected_all: list[dict] = []
    source_summaries = []
    for source in sources:
        candidates, summary = _load_source(source, args.spec.parent)
        selected = _select(
            candidates,
            int(source.get("max_items", spec.get("max_items_per_source", 80))),
            int(source.get("max_per_track", spec.get("max_per_track", 4))),
        )
        summary.update(
            {
                "selected_items": len(selected),
                "selected_tracks": len({item["track_id"] for item in selected}),
                "selected_geometry_eligible": sum(item["geometry_eligible"] for item in selected),
            }
        )
        selected_all.extend(selected)
        source_summaries.append(summary)

    train_bags = {item["bag_id"] for item in selected_all if item["split_role"] == "train"}
    holdout_bags = {item["bag_id"] for item in selected_all if item["split_role"] == "holdout"}
    if train_bags & holdout_bags:
        raise SystemExit(f"bag leakage across splits: {sorted(train_bags & holdout_bags)}")
    train_stamps = {(item["image_stamp_ns"], item["lidar_stamp_ns"]) for item in selected_all if item["split_role"] == "train"}
    holdout_stamps = {(item["image_stamp_ns"], item["lidar_stamp_ns"]) for item in selected_all if item["split_role"] == "holdout"}
    if train_stamps & holdout_stamps:
        raise SystemExit("timestamp leakage across train/holdout")

    manifest_rows: list[dict] = []
    for item in selected_all:
        split, source_id = item["split_role"], item["source_id"]
        cloud_rel = Path("clouds") / split / source_id / f"{item['item_id']}.ply"
        image_source = Path(str(item["source_image"]))
        image_rel = Path("images") / split / source_id / image_source.name
        cloud_path, image_path = args.output / cloud_rel, args.output / image_rel
        cloud_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        _write_ply(cloud_path, item["points_sensor_m"])
        if not image_path.exists():
            shutil.copy2(image_source, image_path)
        manifest_rows.append(
            {
                "item_id": item["item_id"],
                "source_id": source_id,
                "split_role": split,
                "use_for_training": "yes" if split == "train" else "no",
                "use_for_threshold_selection": "yes" if split == "train" else "no",
                "use_for_final_evaluation": "yes" if split == "holdout" else "no",
                "bag_id": item["bag_id"],
                "camera": item["camera"],
                "window_start_index": item["window_start_index"],
                "frame_index": item["frame_index"],
                "cluster_index": item["cluster_index"],
                "track_group_id": f"{source_id}__track_{item['track_id']:04d}",
                "image_stamp_ns": item["image_stamp_ns"],
                "lidar_stamp_ns": item["lidar_stamp_ns"],
                "sync_delta_ms": item["sync_delta_ms"],
                "cloud_path": cloud_rel.as_posix(),
                "image_path": image_rel.as_posix(),
                "sampled_points": item["sampled_points"],
                "cluster_points": item["cluster_points"],
                "raw_center_m": json.dumps(item["raw_center_m"], separators=(",", ":")),
                "calibration_center_m": json.dumps(item["calibration_center_m"], separators=(",", ":")),
                "reference_extent_m": json.dumps(item["reference_extent_m"], separators=(",", ":")),
                "range_m": item["range_m"],
                "pca_linearity": item["pca_linearity"],
                "center_standard_error_m": item["center_standard_error_m"],
                "geometry_eligible": "yes" if item["geometry_eligible"] else "no",
                "geometry_rejection_reasons": "|".join(item["geometry_rejection_reasons"]),
            }
        )

    manifest_fields = list(manifest_rows[0])
    _write_csv(args.output / "manifest.csv", manifest_rows, manifest_fields)
    annotation_fields = manifest_fields + [
        "human_label",
        "static_for_calibration",
        "review_status",
        "reviewer_confidence",
        "reviewer_id",
        "reviewer_note",
    ]
    for split in ("train", "holdout"):
        rows = []
        for row in manifest_rows:
            if row["split_role"] != split:
                continue
            annotated = dict(row)
            annotated.update(
                {
                    "human_label": "",
                    "static_for_calibration": "",
                    "review_status": "unreviewed",
                    "reviewer_confidence": "",
                    "reviewer_id": "",
                    "reviewer_note": "",
                }
            )
            rows.append(annotated)
        _write_csv(args.output / "annotations" / f"{split}_labels.csv", rows, annotation_fields)

    leakage = {
        "passed": True,
        "split_unit": "entire_source_bag",
        "train_bags": sorted(train_bags),
        "holdout_bags": sorted(holdout_bags),
        "bag_intersection": [],
        "timestamp_pair_intersection_count": 0,
        "holdout_use": "final_evaluation_only_after_model_and_threshold_freeze",
        "row_random_split_prohibited": True,
    }
    (args.output / "leakage_audit.json").write_text(
        json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    package = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "real_mcap_manual_lidar_semantic_annotation_for_l5",
        "source_spec": str(args.spec),
        "pointcloud_contract": "V4.11 geometry quality; labels never alter geometry eligibility",
        "prohibited_inputs": list(PROHIBITED_INPUTS),
        "labels": list(LABELS),
        "static_for_calibration_values": list(STATIC_VALUES),
        "source_summaries": source_summaries,
        "counts": {
            "total": len(manifest_rows),
            "train": sum(row["split_role"] == "train" for row in manifest_rows),
            "holdout": sum(row["split_role"] == "holdout" for row in manifest_rows),
            "geometry_eligible": sum(row["geometry_eligible"] == "yes" for row in manifest_rows),
            "by_split_camera": dict(
                Counter(f"{row['split_role']}:{row['camera']}" for row in manifest_rows)
            ),
        },
        "l5_thresholds": {"precision": 0.90, "recall": 0.70, "matches": 20, "per_bag_precision": 0.90},
        "publication_allowed": False,
    }
    (args.output / "package_manifest.json").write_text(
        json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "README.md").write_text(
        "# Real LiDAR V4.11 annotation pack\n\nRead `README_标注说明.md` before editing labels.\n",
        encoding="utf-8",
    )
    (args.output / "README_标注说明.md").write_text(
        """# 真实 LiDAR V4.11 配套标注包

本包用于 L5 真实域语义验证；它不能直接产生或接受外参。请编辑 `annotations/train_labels.csv`，并由独立评估人员保管 `annotations/holdout_labels.csv`。

## 标注规则

1. `human_label` 只能填 `pillar`、`buoy`、`block`、`boat`、`background`、`ambiguous`。
2. `static_for_calibration` 只能填 `yes`、`no`、`uncertain`。船只必须为 `no`；看不清时不要猜，填 `ambiguous/uncertain`。
3. `review_status` 完成时填 `reviewed`；正类、低置信度和 ambiguous 建议二审后填 `adjudicated`。
4. PLY 是 LiDAR 传感器坐标系中的稀疏簇；同时间图像只提供场景上下文，并未做外参投影。禁止凭名义外参把点投到图像后反推标签。
5. 禁止查看或使用 TF、里程计、`/world/*`、仿真真值或固定靶标信息。

## 防泄漏纪律

- train 只来自 090317、090835，可用于训练和阈值选择。
- holdout 只来自 091054；在模型、特征、阈值全部冻结前，开发者不得查看其统计结果。
- 同一 `track_group_id` 的多行高度相关，禁止随机按行拆分或把重复帧当独立证据；评估必须同时报告逐 track 结果。
- `geometry_eligible=no` 是审计样本，不能通过人工标签改成几何合格；L5 正式主指标以 V4.11 合格池为范围。

## 使用前质检

- 保留 UTF-8 CSV 表头和全部谱系列，只填写最后六个标注列。
- 不要让表格软件自动改写 19 位纳秒时间戳；若使用 Excel，必须把全部原始列按文本导入。更稳妥的做法是只在支持 UTF-8 且不自动格式化的 CSV 编辑器中填写最后六列。
- `ambiguous`、`uncertain`、未完成二审的争议样本必须排除训练与正式评价。
- 若 holdout 中合格静态正样本不足 20，必须从同一第三 bag 的新连续窗口追加数据，不能降低门槛或借用训练 bag。
- L5 通过后仍需跨 bag 外参修正重复性和三对闭环；本包结果不得把外参直接标为 accepted。

## 校验和说明

`checksums.sha256` 用于确认**开始标注前**的包没有损坏。填写两个 labels CSV 后，这两个文件的原始校验和按预期会变化；不要用旧校验和覆盖已填写标签。完成后应另存一个版本化目录，运行项目内 `validate_lidar_annotations_v4_11.py --require-complete`，再为完成版重新生成校验和。
""",
        encoding="utf-8",
    )
    (args.output / "annotations" / "HOLDOUT_FINAL_EVALUATION_ONLY.txt").write_text(
        "091054 holdout labels are final-evaluation-only. Do not use them for training, feature design, threshold selection, or error-driven tuning.\n",
        encoding="utf-8",
    )

    checksum_lines = []
    for path in sorted(item for item in args.output.rglob("*") if item.is_file() and item.name != "checksums.sha256"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {path.relative_to(args.output).as_posix()}")
    (args.output / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps(package["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
