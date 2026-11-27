#!/usr/bin/env python3
"""Simulation-prior + real sparse fine-tuning with leakage-safe nested tuning."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


BASE_PATH = Path(__file__).with_name("train_sparse_lidar_candidate_v3_1.py")
SPEC = importlib.util.spec_from_file_location("sparse_candidate_base", BASE_PATH)
assert SPEC and SPEC.loader
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

FEATURE_NAMES = BASE.FEATURE_NAMES[:11]


def sim_features(row: dict[str, str]) -> np.ndarray:
    values = np.maximum(np.asarray([float(row["pca1"]), float(row["pca2"]), float(row["pca3"])], dtype=float), 0.0)
    ratios = values / max(float(np.sum(values)), 1e-12)
    pca1 = max(values[0], 1e-12)
    return np.asarray(
        [
            np.log1p(float(row["points"])), float(row["range_m"]),
            np.log1p(float(row["extent_x"])), np.log1p(float(row["extent_y"])), np.log1p(float(row["extent_z"])),
            *ratios, float(row["linearity"]), float((values[1] - values[2]) / pca1), float(values[2] / pca1),
        ],
        dtype=np.float64,
    )


def load_sim_train(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    # Dataset construction already deduplicates source clusters.  Only exact
    # semantic pillar/background labels are admitted; val rows remain sealed.
    rows = [row for row in rows if row["label"] in {"pillar", "background"}]
    x = np.vstack([sim_features(row) for row in rows])
    y = np.asarray([row["label"] == "pillar" for row in rows], dtype=bool)
    if np.sum(y) < 10 or np.sum(~y) < 10:
        raise ValueError("simulation training split is too small")
    return x, y


def fit_augmented(
    x_real: np.ndarray,
    y_real: np.ndarray,
    groups: np.ndarray,
    x_sim: np.ndarray,
    y_sim: np.ndarray,
    quantile: float,
    ratio: float,
    sim_positive_penalty: float,
    sim_negative_penalty: float,
) -> dict[str, Any]:
    rule = fit_core(x_real, y_real, groups, x_sim, y_sim, sim_positive_penalty, sim_negative_penalty)
    rule["positive_distance_threshold"] = float(np.quantile(rule.pop("training_positive_distances"), quantile))
    rule["max_positive_negative_ratio"] = ratio
    return rule


def fit_core(
    x_real: np.ndarray,
    y_real: np.ndarray,
    groups: np.ndarray,
    x_sim: np.ndarray,
    y_sim: np.ndarray,
    sim_positive_penalty: float,
    sim_negative_penalty: float,
) -> dict[str, Any]:
    center = np.median(x_real, axis=0)
    scale = np.maximum(np.percentile(x_real, 75, axis=0) - np.percentile(x_real, 25, axis=0), 1e-4)
    z = (x_real - center) / scale
    positive = z[y_real]; negative = z[~y_real]; positive_groups = groups[y_real]
    if len(positive) < 10 or len(negative) < 10:
        raise ValueError("insufficient real positive or negative observations")
    training_distances = np.empty(len(positive), dtype=np.float64)
    for group in sorted(set(positive_groups)):
        query = positive_groups == group
        prototypes = positive_groups != group
        if not np.any(prototypes):
            raise ValueError("need at least two real episode groups")
        training_distances[query] = BASE.nearest_distances(positive[query], positive[prototypes])
    z_sim = (x_sim - center) / scale
    return {
        "center": center, "scale": scale, "positive": positive, "negative": negative,
        "training_positive_distances": training_distances,
        "sim_positive": z_sim[y_sim], "sim_negative": z_sim[~y_sim],
        "sim_positive_penalty": sim_positive_penalty, "sim_negative_penalty": sim_negative_penalty,
    }


def distances(rule: dict[str, Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = (x - rule["center"]) / rule["scale"]
    real_positive = BASE.nearest_distances(z, rule["positive"])
    real_negative = BASE.nearest_distances(z, rule["negative"])
    sim_positive = BASE.nearest_distances(z, rule["sim_positive"]) * rule["sim_positive_penalty"]
    sim_negative = BASE.nearest_distances(z, rule["sim_negative"]) * rule["sim_negative_penalty"]
    return np.minimum(real_positive, sim_positive), np.minimum(real_negative, sim_negative)


def predict(rule: dict[str, Any], x: np.ndarray) -> np.ndarray:
    positive, negative = distances(rule, x)
    return (positive <= rule["positive_distance_threshold"]) & (positive < rule["max_positive_negative_ratio"] * negative)


def tune(
    x: np.ndarray,
    y: np.ndarray,
    episodes: np.ndarray,
    cameras: np.ndarray,
    x_sim: np.ndarray,
    y_sim: np.ndarray,
) -> dict[str, float]:
    sim_positive_penalty = 1.50; sim_negative_penalty = 0.75
    cached = []
    for camera in sorted(set(cameras)):
        train_mask = cameras != camera; eval_mask = cameras == camera
        core = fit_core(x[train_mask], y[train_mask], episodes[train_mask], x_sim, y_sim, sim_positive_penalty, sim_negative_penalty)
        positive_distance, negative_distance = distances(core, x[eval_mask])
        cached.append((y[eval_mask], positive_distance, negative_distance, core["training_positive_distances"]))
    candidates = []
    for quantile in (0.35, 0.50, 0.65):
        for ratio in (0.25, 0.35, 0.50):
            tp = fp = fn = 0; camera_tp = []; camera_precision = []
            for truth, positive_distance, negative_distance, training_distances in cached:
                threshold = float(np.quantile(training_distances, quantile))
                predicted = (positive_distance <= threshold) & (positive_distance < ratio * negative_distance)
                one_tp = int(np.sum(predicted & truth)); one_fp = int(np.sum(predicted & ~truth)); one_fn = int(np.sum(~predicted & truth))
                tp += one_tp; fp += one_fp; fn += one_fn; camera_tp.append(one_tp)
                camera_precision.append(one_tp / (one_tp + one_fp) if one_tp + one_fp else 0.0)
            precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
            viable = min(camera_tp) >= 3 and min(camera_precision) >= 0.75
            score = (int(viable), precision, recall, tp)
            candidates.append((score, {"distance_quantile": quantile, "max_positive_negative_ratio": ratio,
                                       "sim_positive_penalty": sim_positive_penalty, "sim_negative_penalty": sim_negative_penalty,
                                       "inner_precision": precision, "inner_recall": recall,
                                       "inner_min_camera_precision": min(camera_precision), "inner_min_camera_tp": min(camera_tp)}))
    return max(candidates, key=lambda item: item[0])[1]


def rule_to_json(rule: dict[str, Any]) -> dict[str, Any]:
    payload = BASE.serializable_rule(rule)
    payload["kind"] = "robust_scaled_1nn_sim_prior_real_finetune"
    payload.update(
        {
            "simulation_positive_prototypes": rule["sim_positive"].tolist(),
            "simulation_negative_prototypes": rule["sim_negative"].tolist(),
            "simulation_positive_distance_penalty": rule["sim_positive_penalty"],
            "simulation_negative_distance_penalty": rule["sim_negative_penalty"],
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--simulation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    asset_root = args.asset_root.resolve(); output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite {output_dir}")
    rows = BASE.load_rows(asset_root / "observations.csv")
    archive = np.load(asset_root / "observations.npz", allow_pickle=False)
    x = BASE.feature_matrix(rows, archive)[:, : len(FEATURE_NAMES)]
    y = np.asarray([row["static_candidate_target"] == "1" for row in rows])
    bags = np.asarray([row["bag_id"] for row in rows])
    episodes = np.asarray([row["episode_id"] for row in rows])
    cameras = np.asarray([row["camera"] for row in rows])
    original_xy = np.asarray([[float(row["original_center_x"]), float(row["original_center_y"])] for row in rows])
    derived_xy = np.asarray([[float(row["derived_center_x"]), float(row["derived_center_y"])] for row in rows])
    center_errors = np.linalg.norm(derived_xy - original_xy, axis=1)
    x_sim, y_sim = load_sim_train(args.simulation_csv.resolve())
    unique_bags = sorted(set(bags))
    if len(unique_bags) != 2:
        raise SystemExit("expected exactly two real training bags")
    folds = []
    for evaluation_bag in unique_bags:
        training = bags != evaluation_bag; evaluation = ~training
        params = tune(x[training], y[training], episodes[training], cameras[training], x_sim, y_sim)
        rule = fit_augmented(x[training], y[training], episodes[training], x_sim, y_sim,
                             params["distance_quantile"], params["max_positive_negative_ratio"],
                             params["sim_positive_penalty"], params["sim_negative_penalty"])
        predicted = predict(rule, x[evaluation])
        overall = BASE.metrics(y[evaluation], predicted, center_errors[evaluation])
        by_camera = {}
        for camera in sorted(set(cameras[evaluation])):
            mask = cameras[evaluation] == camera
            by_camera[camera] = BASE.metrics(y[evaluation][mask], predicted[mask], center_errors[evaluation][mask])
        passed = bool(overall["precision"] >= 0.90 and overall["tp"] >= 20
                      and overall["true_positive_center_error_p90_m"] is not None
                      and overall["true_positive_center_error_p90_m"] <= 1.0
                      and all(value["precision"] >= 0.80 and value["tp"] >= 5 for value in by_camera.values()))
        folds.append({"training_bags": [bag for bag in unique_bags if bag != evaluation_bag], "evaluation_bag": evaluation_bag,
                      "training_only_nested_parameters": params, "overall": overall, "by_camera": by_camera, "passed": passed})
    final_params = tune(x, y, episodes, cameras, x_sim, y_sim)
    final_rule = fit_augmented(x, y, episodes, x_sim, y_sim,
                               final_params["distance_quantile"], final_params["max_positive_negative_ratio"],
                               final_params["sim_positive_penalty"], final_params["sim_negative_penalty"])
    passed = all(fold["passed"] for fold in folds)
    report = {
        "schema": "sim-real-sparse-candidate-cross-bag-evaluation/v1",
        "asset_root": str(asset_root), "simulation_csv": str(args.simulation_csv.resolve()),
        "simulation_rows_used": int(len(x_sim)), "simulation_val_rows_used": 0,
        "features": FEATURE_NAMES,
        "method": "simulation geometric prototypes as penalized prior; real observations fine-tune scaler, threshold and primary prototypes",
        "split_policy": "outer leave-one-real-bag-out; parameters selected by inner leave-one-camera-out on training bag only",
        "official_third_bag_used": False,
        "gates": {"overall_precision_min": 0.90, "per_camera_precision_min": 0.80,
                  "overall_true_positive_min": 20, "per_camera_true_positive_min": 5,
                  "true_positive_center_error_p90_max_m": 1.0},
        "folds": folds, "cross_bag_passed": passed,
        "next_action": "candidate-only sealed-bag evaluation" if passed else "do not integrate; require richer simulation classes and/or learned detector",
    }
    model = {
        "schema": "sim-real-sparse-candidate-rule/v1", "features": FEATURE_NAMES,
        "candidate_only": True, "runtime_allowed": passed, "calibration_acceptance_allowed": False,
        "official_third_bag_used": False, "final_training_only_nested_parameters": final_params,
        "rule": rule_to_json(final_rule),
    }
    output_dir.mkdir(parents=True)
    (output_dir / "cross_bag_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "candidate_rule.json").write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# 仿真预训练 + 真实稀疏微调候选模型\n\n只读取仿真 CSV 的 `train` 行；仿真 val 与官方第三 bag 均保持封存。"
        "模型只允许输出候选，不能直接接受外参标定。是否允许进入密封 bag 实验以 `runtime_allowed` 为准。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
