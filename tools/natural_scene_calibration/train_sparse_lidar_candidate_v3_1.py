#!/usr/bin/env python3
"""Train and cross-bag test a conservative sparse-LiDAR candidate rule.

The official third bag remains sealed.  Each diagnostic fold trains on one
official training bag and evaluates on the other; thresholds and normalization
are computed from the training fold only.  The final two-bag rule is candidate
only and cannot approve calibration by itself.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_NAMES = (
    "log_points", "range_m", "log_extent_x", "log_extent_y", "log_extent_z",
    "pca_ratio_1", "pca_ratio_2", "pca_ratio_3", "linearity", "planarity",
    "scattering", "verticality", "radial_size_m", "tangential_size_m",
)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def point_features(points: np.ndarray) -> np.ndarray:
    center = np.mean(points, axis=0)
    extent = np.ptp(points, axis=0) if len(points) > 1 else np.zeros(3)
    range_m = float(np.linalg.norm(center))
    if len(points) > 1:
        covariance = np.cov(points, rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        values = np.maximum(values[order], 0.0)
        principal = vectors[:, order[0]]
        ratios = values / max(float(np.sum(values)), 1e-12)
        linearity = float((values[0] - values[1]) / max(values[0], 1e-12))
        planarity = float((values[1] - values[2]) / max(values[0], 1e-12))
        scattering = float(values[2] / max(values[0], 1e-12))
        verticality = float(abs(principal[2]))
    else:
        ratios = np.zeros(3)
        linearity = planarity = scattering = verticality = 0.0
    radial = center / max(range_m, 1e-12)
    projection = points @ radial
    radial_size = float(np.ptp(projection)) if len(points) > 1 else 0.0
    tangential = points - np.outer(projection, radial)
    tangential_size = float(np.linalg.norm(np.ptp(tangential, axis=0))) if len(points) > 1 else 0.0
    return np.asarray(
        [
            math.log1p(len(points)), range_m, *np.log1p(np.maximum(extent, 0.0)),
            *ratios, linearity, planarity, scattering, verticality, radial_size, tangential_size,
        ],
        dtype=np.float64,
    )


def feature_matrix(rows: list[dict[str, str]], archive: Any) -> np.ndarray:
    offsets = archive["point_offsets"]
    points = archive["points_xyz"]
    return np.vstack([point_features(points[offsets[index] : offsets[index + 1]]) for index in range(len(rows))])


def nearest_distances(query: np.ndarray, prototypes: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    result = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), chunk_size):
        chunk = query[start : start + chunk_size]
        squared = np.sum((chunk[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
        result[start : start + len(chunk)] = np.sqrt(np.min(squared, axis=1))
    return result


def fit_rule(
    x: np.ndarray,
    y: np.ndarray,
    episode_groups: np.ndarray,
    distance_quantile: float,
    max_positive_negative_ratio: float,
) -> dict[str, Any]:
    center = np.median(x, axis=0)
    scale = np.maximum(np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0), 1e-4)
    z = (x - center) / scale
    positive = z[y]
    negative = z[~y]
    positive_groups = episode_groups[y]
    if len(positive) < 10 or len(negative) < 10:
        raise ValueError("insufficient positive or negative observations")
    leave_episode_out = []
    for index, row in enumerate(positive):
        allowed = positive_groups != positive_groups[index]
        if not np.any(allowed):
            allowed = np.arange(len(positive)) != index
        leave_episode_out.append(float(np.linalg.norm(positive[allowed] - row, axis=1).min()))
    threshold = float(np.quantile(leave_episode_out, distance_quantile))
    return {
        "center": center,
        "scale": scale,
        "positive": positive,
        "negative": negative,
        "positive_distance_threshold": threshold,
        "max_positive_negative_ratio": max_positive_negative_ratio,
    }


def predict(rule: dict[str, Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = (x - rule["center"]) / rule["scale"]
    positive_distance = nearest_distances(z, rule["positive"])
    negative_distance = nearest_distances(z, rule["negative"])
    predicted = (
        (positive_distance <= rule["positive_distance_threshold"])
        & (positive_distance < rule["max_positive_negative_ratio"] * negative_distance)
    )
    return predicted, positive_distance, negative_distance


def metrics(truth: np.ndarray, predicted: np.ndarray, center_errors: np.ndarray) -> dict[str, Any]:
    true_positive = truth & predicted
    tp = int(np.sum(true_positive))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    errors = center_errors[true_positive]
    return {
        "positives": int(np.sum(truth)),
        "predicted_candidates": int(np.sum(predicted)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "true_positive_center_error_median_m": float(np.median(errors)) if len(errors) else None,
        "true_positive_center_error_p90_m": float(np.quantile(errors, 0.90)) if len(errors) else None,
    }


def serializable_rule(rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "robust_scaled_1nn_static_candidate",
        "center": rule["center"].tolist(),
        "scale": rule["scale"].tolist(),
        "positive_prototypes": rule["positive"].tolist(),
        "negative_prototypes": rule["negative"].tolist(),
        "positive_distance_threshold": rule["positive_distance_threshold"],
        "max_positive_negative_ratio": rule["max_positive_negative_ratio"],
    }


def train(
    asset_root: Path,
    output_dir: Path,
    distance_quantile: float = 0.65,
    max_positive_negative_ratio: float = 0.75,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    rows = load_rows(asset_root / "observations.csv")
    archive = np.load(asset_root / "observations.npz", allow_pickle=False)
    x = feature_matrix(rows, archive)
    y = np.asarray([row["static_candidate_target"] == "1" for row in rows], dtype=bool)
    bags = np.asarray([row["bag_id"] for row in rows])
    episodes = np.asarray([row["episode_id"] for row in rows])
    cameras = np.asarray([row["camera"] for row in rows])
    original_xy = np.asarray([[float(row["original_center_x"]), float(row["original_center_y"])] for row in rows])
    derived_xy = np.asarray([[float(row["derived_center_x"]), float(row["derived_center_y"])] for row in rows])
    center_errors = np.linalg.norm(derived_xy - original_xy, axis=1)
    unique_bags = sorted(set(bags))
    if len(unique_bags) != 2:
        raise ValueError(f"expected exactly two official training bags, found {unique_bags}")

    folds = []
    for evaluation_bag in unique_bags:
        train_mask = bags != evaluation_bag
        eval_mask = bags == evaluation_bag
        rule = fit_rule(x[train_mask], y[train_mask], episodes[train_mask], distance_quantile, max_positive_negative_ratio)
        predicted, _, _ = predict(rule, x[eval_mask])
        fold_metrics = metrics(y[eval_mask], predicted, center_errors[eval_mask])
        by_camera: dict[str, Any] = {}
        for camera in sorted(set(cameras[eval_mask])):
            camera_mask = cameras[eval_mask] == camera
            by_camera[camera] = metrics(y[eval_mask][camera_mask], predicted[camera_mask], center_errors[eval_mask][camera_mask])
        fold_pass = bool(
            fold_metrics["precision"] >= 0.90
            and fold_metrics["tp"] >= 20
            and fold_metrics["true_positive_center_error_p90_m"] is not None
            and fold_metrics["true_positive_center_error_p90_m"] <= 1.0
            and all(value["precision"] >= 0.80 and value["tp"] >= 5 for value in by_camera.values())
        )
        folds.append(
            {
                "training_bags": [bag for bag in unique_bags if bag != evaluation_bag],
                "evaluation_bag": evaluation_bag,
                "threshold_from_training_only": rule["positive_distance_threshold"],
                "overall": fold_metrics,
                "by_camera": by_camera,
                "passed": fold_pass,
            }
        )

    final_rule = fit_rule(x, y, episodes, distance_quantile, max_positive_negative_ratio)
    cross_bag_pass = all(fold["passed"] for fold in folds)
    report = {
        "schema": "real-lidar-sparse-candidate-cross-bag-evaluation/v1",
        "asset_root": str(asset_root),
        "features": FEATURE_NAMES,
        "target": "pillar|buoy|block AND static_for_calibration=yes",
        "split_policy": "leave-one-official-training-bag-out; no random frame split",
        "official_holdout_used": False,
        "parameters_frozen_before_evaluation": {
            "distance_quantile": distance_quantile,
            "max_positive_negative_ratio": max_positive_negative_ratio,
        },
        "gates": {
            "overall_precision_min": 0.90,
            "per_camera_precision_min": 0.80,
            "overall_true_positive_min": 20,
            "per_camera_true_positive_min": 5,
            "true_positive_center_error_p90_max_m": 1.0,
        },
        "folds": folds,
        "cross_bag_passed": cross_bag_pass,
        "next_action": (
            "candidate-only runtime integration on sealed third bag; no calibration acceptance"
            if cross_bag_pass
            else "do not integrate; move to simulation pretraining plus real sparse fine-tuning"
        ),
    }
    model = {
        "schema": "real-lidar-sparse-candidate-rule/v1",
        "asset_root": str(asset_root),
        "features": FEATURE_NAMES,
        "candidate_only": True,
        "runtime_allowed": cross_bag_pass,
        "calibration_acceptance_allowed": False,
        "official_holdout_used": False,
        "class_contract": {"positive": "static pillar|buoy|block", "negative": "boat|background|non-static candidate class"},
        "rule": serializable_rule(final_rule),
    }
    output_dir.mkdir(parents=True)
    (output_dir / "cross_bag_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "candidate_rule.json").write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output_dir / "features.npz", features=x.astype(np.float32), targets=y, center_errors_m=center_errors.astype(np.float32))
    (output_dir / "README.md").write_text(
        "# V3.1 稀疏 LiDAR 首轮候选模型\n\n"
        "这是候选提取规则，不是外参标定结果。`runtime_allowed` 只有在两次跨 bag 诊断均过门禁时才为 true；"
        "官方第三 bag 未用于训练或调参。若未通过，应按报告转入仿真预训练加真实稀疏小样本微调。\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distance-quantile", type=float, default=0.65)
    parser.add_argument("--max-positive-negative-ratio", type=float, default=0.75)
    args = parser.parse_args()
    report = train(args.asset_root.resolve(), args.output_dir.resolve(), args.distance_quantile, args.max_positive_negative_ratio)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
