#!/usr/bin/env python3
"""Export bounded hard cases from failed outer-bag quality-aware folds."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


QUALITY_PATH = Path(__file__).with_name("train_quality_aware_sim_real_candidate_v3_1.py")
SPEC = importlib.util.spec_from_file_location("quality_candidate", QUALITY_PATH)
assert SPEC and SPEC.loader
QUALITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALITY)
LEARNED = QUALITY.LEARNED
BASE = QUALITY.BASE


def failed_cameras(fold: dict[str, Any]) -> set[str]:
    result = set()
    for camera, metrics in fold["by_camera"].items():
        center = metrics["true_positive_center_error_p90_m"]
        if metrics["precision"] < 0.80 or metrics["tp"] < 5 or center is None or center > 1.0:
            result.add(camera)
    return result


def select_bounded(rows: list[dict[str, Any]], limits: dict[str, int]) -> list[dict[str, Any]]:
    selected = []
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["bag_id"], row["camera"], row["review_reason"])].append(row)
    for (_, _, reason), items in sorted(by_key.items()):
        if reason == "false_positive_negative":
            items.sort(key=lambda item: (-item["probability"], -item["support_count"], item["annotation_id"]))
        elif reason == "center_bad_true_positive":
            items.sort(key=lambda item: (-item["center_error_m"], -item["probability"], item["annotation_id"]))
        else:
            items.sort(key=lambda item: (-item["probability"], item["annotation_id"]))
        selected.extend(items[: limits[reason]])
    return sorted(selected, key=lambda item: (item["bag_id"], item["camera"], item["frame_index"], item["annotation_id"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--simulation-training-csv", type=Path, required=True)
    parser.add_argument("--source-v3-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    asset_root = args.asset_root.resolve(); source_root = args.source_v3_root.resolve(); output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    report = json.loads(args.model_report.read_text(encoding="utf-8"))
    rows = BASE.load_rows(asset_root / "observations.csv")
    archive = np.load(asset_root / "observations.npz", allow_pickle=False)
    base_features = BASE.feature_matrix(rows, archive)[:, LEARNED.REAL_FEATURE_INDICES]
    cameras = np.asarray([row["camera"] for row in rows])
    x = np.vstack([np.r_[values, LEARNED.camera_one_hot(camera)] for values, camera in zip(base_features, cameras)])
    static_truth = np.asarray([row["static_candidate_target"] == "1" for row in rows])
    bags = np.asarray([row["bag_id"] for row in rows])
    original_xy = np.asarray([[float(row["original_center_x"]), float(row["original_center_y"])] for row in rows])
    derived_xy = np.asarray([[float(row["derived_center_x"]), float(row["derived_center_y"])] for row in rows])
    errors = np.linalg.norm(derived_xy - original_xy, axis=1)
    quality_target = static_truth & (errors <= 1.0)
    x_sim, y_sim, _ = LEARNED.load_sim(args.simulation_training_csv.resolve(), "train")

    candidates: list[dict[str, Any]] = []
    for fold in report["real_cross_bag_folds"]:
        evaluation_bag = fold["evaluation_bag"]
        failing = failed_cameras(fold)
        training = bags != evaluation_bag; evaluation_indices = np.flatnonzero(bags == evaluation_bag)
        params = fold["training_only_parameters"]
        model = LEARNED.fit_model(x[training], quality_target[training], x_sim, y_sim, params)
        probability = model.predict_proba(x[evaluation_indices])[:, 1]
        for local_index, row_index in enumerate(evaluation_indices):
            row = rows[row_index]; camera = row["camera"]
            if camera not in failing:
                continue
            threshold = float(params["thresholds"][camera]); predicted = probability[local_index] >= threshold
            reason = None
            if predicted and not static_truth[row_index]:
                reason = "false_positive_negative"
            elif predicted and static_truth[row_index] and errors[row_index] > 1.0:
                reason = "center_bad_true_positive"
            elif not predicted and static_truth[row_index] and errors[row_index] <= 1.0:
                reason = "missed_center_good_positive"
            if reason is None:
                continue
            candidates.append(
                {
                    "review_reason": reason,
                    "bag_id": row["bag_id"], "episode_id": row["episode_id"], "camera": camera,
                    "frame_index": int(row["frame_index"]), "source_frame_index": int(row["source_frame_index"]),
                    "lidar_stamp_ns": row["lidar_stamp_ns"], "image_stamp_ns": row["image_stamp_ns"],
                    "annotation_id": row["annotation_id"], "track_id": row["track_id"], "class_name": row["class_name"],
                    "static_for_calibration": row["static_for_calibration"], "observation_type": row["observation_type"],
                    "support_count": int(row["support_count"]), "probability": round(float(probability[local_index]), 8),
                    "frozen_threshold": threshold, "center_error_m": round(float(errors[row_index]), 6),
                    "pcd_path": str(source_root / row["pcd_path"]), "image_path": str(source_root / row["image_path"]),
                    "recommended_action": {
                        "false_positive_negative": "confirm negative class/box; add as hard negative only if correct",
                        "center_bad_true_positive": "tighten XY footprint around actual returns or mark no reliable center",
                        "missed_center_good_positive": "confirm class/static flag and retain as positive hard example",
                    }[reason],
                }
            )
    selected = select_bounded(candidates, {"false_positive_negative": 30, "center_bad_true_positive": 30, "missed_center_good_positive": 15})
    output.mkdir(parents=True)
    write_csv(output / "review_queue.csv", selected)
    by_episode: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in selected:
        by_episode[row["episode_id"]][str(row["frame_index"])].append(
            {key: row[key] for key in ("annotation_id", "track_id", "class_name", "review_reason", "probability", "frozen_threshold", "center_error_m", "recommended_action")}
        )
    sidecar_dir = output / "review_sidecars"; sidecar_dir.mkdir()
    for episode, frames in sorted(by_episode.items()):
        payload = {"schema": "real-lidar-targeted-review/v1", "episode_id": episode, "frames": frames}
        (sidecar_dir / f"{episode}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": "real-lidar-targeted-hard-case-review/v1", "source_asset": str(asset_root),
        "source_model_report": str(args.model_report.resolve()), "official_third_bag_used": False,
        "selection_limits_per_bag_camera": {"false_positive_negative": 30, "center_bad_true_positive": 30, "missed_center_good_positive": 15},
        "failed_bag_cameras": sorted({f"{row['bag_id']}::{row['camera']}" for row in selected}),
        "selected_rows": len(selected), "by_reason": dict(Counter(row["review_reason"] for row in selected)),
        "by_bag_camera": dict(Counter(f"{row['bag_id']}::{row['camera']}" for row in selected)),
        "source_files_copied": False,
    }
    (output / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# V3.1 定向难例复核包\n\n本包不复制点云和图像，`review_queue.csv` 给出 V3 源文件绝对路径。"
        "只复核失败方位：假阳性确认无误后才能作为负样本；中心异常正例应收紧 XY 框或标记为无可靠中心；"
        "漏检正例只需确认类别与静态性。每个 bag/方位最多 75 条，第三 bag 未使用。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
