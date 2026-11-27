#!/usr/bin/env python3
"""Train a nonlinear simulation+real sparse LiDAR candidate classifier.

Outer evaluation leaves one complete real bag out.  Model parameters and the
decision threshold are chosen only by leave-one-camera-out validation inside
the outer training bag.  The official third bag is never read.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


BASE_PATH = Path(__file__).with_name("train_sparse_lidar_candidate_v3_1.py")
SPEC = importlib.util.spec_from_file_location("sparse_candidate_base", BASE_PATH)
assert SPEC and SPEC.loader
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

CAMERAS = ("front", "back", "left", "left2", "right", "right2")
GEOMETRY_FEATURES = (
    "log_points", "range_m", "log_extent_x", "log_extent_y", "log_extent_z",
    "pca_ratio_1", "pca_ratio_2", "pca_ratio_3", "linearity", "planarity",
    "scattering", "radial_size_m", "tangential_size_m",
)
FEATURES = GEOMETRY_FEATURES + tuple(f"camera_{camera}" for camera in CAMERAS)
REAL_FEATURE_INDICES = tuple(range(11)) + (12, 13)


def camera_one_hot(camera: str) -> np.ndarray:
    if camera not in CAMERAS:
        raise ValueError(f"unknown camera: {camera}")
    return np.asarray([float(camera == item) for item in CAMERAS], dtype=np.float64)


def sim_geometry_features(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [
            np.log1p(float(row["points"])), float(row["range_m"]),
            np.log1p(float(row["extent_x"])), np.log1p(float(row["extent_y"])), np.log1p(float(row["extent_z"])),
            float(row["pca1"]), float(row["pca2"]), float(row["pca3"]),
            float(row["linearity"]), float(row["planarity"]), float(row["scattering"]),
            float(row["radial_size_m"]), float(row["tangential_size_m"]),
        ],
        dtype=np.float64,
    )


def load_sim(path: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    x = np.vstack([np.r_[sim_geometry_features(row), camera_one_hot(row["camera"])] for row in rows])
    y = np.asarray([row["label"] in {"buoy", "pillar", "block"} for row in rows], dtype=bool)
    cameras = np.asarray([row["camera"] for row in rows])
    return x, y, cameras


def domain_balanced_weights(y_real: np.ndarray, y_sim: np.ndarray, real_mass: float = 0.8) -> np.ndarray:
    weights = []
    for labels, mass in ((y_real, real_mass), (y_sim, 1.0 - real_mass)):
        one = np.zeros(len(labels), dtype=np.float64)
        for value in (False, True):
            count = int(np.sum(labels == value))
            if count:
                one[labels == value] = mass / (2.0 * count)
        weights.append(one)
    combined = np.concatenate(weights)
    return combined * len(combined) / max(float(np.sum(combined)), 1e-12)


def fit_model(x_real: np.ndarray, y_real: np.ndarray, x_sim: np.ndarray, y_sim: np.ndarray, params: dict[str, Any]) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=160,
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        early_stopping=False,
        random_state=17,
    )
    model.fit(np.vstack((x_real, x_sim)), np.concatenate((y_real, y_sim)), sample_weight=domain_balanced_weights(y_real, y_sim))
    return model


def select_threshold(truth: np.ndarray, probabilities: np.ndarray, cameras: np.ndarray) -> tuple[float, dict[str, float]]:
    best: tuple[tuple[float, ...], float, dict[str, float]] | None = None
    for threshold in np.linspace(0.50, 0.995, 100):
        predicted = probabilities >= threshold
        tp = int(np.sum(predicted & truth)); fp = int(np.sum(predicted & ~truth)); fn = int(np.sum(~predicted & truth))
        precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
        per_camera_precision = []; per_camera_tp = []
        for camera in sorted(set(cameras)):
            mask = cameras == camera; one_tp = int(np.sum(predicted[mask] & truth[mask])); one_fp = int(np.sum(predicted[mask] & ~truth[mask]))
            per_camera_tp.append(one_tp); per_camera_precision.append(one_tp / (one_tp + one_fp) if one_tp + one_fp else 0.0)
        viable = min(per_camera_tp) >= 3 and min(per_camera_precision) >= 0.75
        score = (float(viable), float(precision >= 0.90), precision, recall, float(tp))
        summary = {"inner_precision": precision, "inner_recall": recall,
                   "inner_min_camera_precision": min(per_camera_precision), "inner_min_camera_tp": min(per_camera_tp)}
        if best is None or score > best[0]:
            best = (score, float(threshold), summary)
    assert best is not None
    return best[1], best[2]


def tune(x: np.ndarray, y: np.ndarray, cameras: np.ndarray, x_sim: np.ndarray, y_sim: np.ndarray) -> dict[str, Any]:
    candidates = []
    grid = (
        {"max_leaf_nodes": 7, "min_samples_leaf": 10, "l2_regularization": 5.0},
        {"max_leaf_nodes": 15, "min_samples_leaf": 10, "l2_regularization": 5.0},
        {"max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 10.0},
        {"max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 10.0},
    )
    for params in grid:
        truths = []; probabilities = []; camera_rows = []
        for camera in CAMERAS:
            evaluation = cameras == camera; training = ~evaluation
            model = fit_model(x[training], y[training], x_sim, y_sim, params)
            truths.append(y[evaluation]); probabilities.append(model.predict_proba(x[evaluation])[:, 1]); camera_rows.append(cameras[evaluation])
        truth = np.concatenate(truths); probability = np.concatenate(probabilities); validation_cameras = np.concatenate(camera_rows)
        threshold, summary = select_threshold(truth, probability, validation_cameras)
        score = (float(summary["inner_min_camera_tp"] >= 3 and summary["inner_min_camera_precision"] >= 0.75),
                 float(summary["inner_precision"] >= 0.90), summary["inner_precision"], summary["inner_recall"])
        candidates.append((score, {**params, "threshold": threshold, **summary}))
    return max(candidates, key=lambda item: item[0])[1]


def evaluate(truth: np.ndarray, probabilities: np.ndarray, cameras: np.ndarray, center_errors: np.ndarray | None, threshold: float) -> dict[str, Any]:
    predicted = probabilities >= threshold
    overall = BASE.metrics(truth, predicted, center_errors if center_errors is not None else np.zeros(len(truth)))
    by_camera = {}
    for camera in sorted(set(cameras)):
        mask = cameras == camera
        by_camera[camera] = BASE.metrics(truth[mask], predicted[mask], center_errors[mask] if center_errors is not None else np.zeros(np.sum(mask)))
    return {"threshold": threshold, "overall": overall, "by_camera": by_camera}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--simulation-training-csv", type=Path, required=True)
    parser.add_argument("--simulation-holdout-csv", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    asset_root = args.asset_root.resolve(); output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    rows = BASE.load_rows(asset_root / "observations.csv")
    archive = np.load(asset_root / "observations.npz", allow_pickle=False)
    base_features = BASE.feature_matrix(rows, archive)[:, REAL_FEATURE_INDICES]
    real_cameras = np.asarray([row["camera"] for row in rows])
    x_real = np.vstack([np.r_[values, camera_one_hot(camera)] for values, camera in zip(base_features, real_cameras)])
    y_real = np.asarray([row["static_candidate_target"] == "1" for row in rows])
    bags = np.asarray([row["bag_id"] for row in rows])
    original_xy = np.asarray([[float(row["original_center_x"]), float(row["original_center_y"])] for row in rows])
    derived_xy = np.asarray([[float(row["derived_center_x"]), float(row["derived_center_y"])] for row in rows])
    center_errors = np.linalg.norm(derived_xy - original_xy, axis=1)
    x_sim, y_sim, _ = load_sim(args.simulation_training_csv.resolve(), "train")
    unique_bags = sorted(set(bags))
    if len(unique_bags) != 2:
        raise SystemExit("expected exactly two official real training bags")

    folds = []
    for evaluation_bag in unique_bags:
        training = bags != evaluation_bag; evaluation = ~training
        params = tune(x_real[training], y_real[training], real_cameras[training], x_sim, y_sim)
        model = fit_model(x_real[training], y_real[training], x_sim, y_sim, params)
        result = evaluate(y_real[evaluation], model.predict_proba(x_real[evaluation])[:, 1], real_cameras[evaluation], center_errors[evaluation], params["threshold"])
        overall = result["overall"]; by_camera = result["by_camera"]
        passed = bool(overall["precision"] >= 0.90 and overall["tp"] >= 20
                      and overall["true_positive_center_error_p90_m"] is not None
                      and overall["true_positive_center_error_p90_m"] <= 1.0
                      and all(value["precision"] >= 0.80 and value["tp"] >= 5 for value in by_camera.values()))
        folds.append({"training_bags": [bag for bag in unique_bags if bag != evaluation_bag], "evaluation_bag": evaluation_bag,
                      "training_only_parameters": params, **result, "passed": passed})

    final_params = tune(x_real, y_real, real_cameras, x_sim, y_sim)
    final_model = fit_model(x_real, y_real, x_sim, y_sim, final_params)
    sim_holdouts = []
    for path in args.simulation_holdout_csv:
        x_holdout, y_holdout, cameras_holdout = load_sim(path.resolve(), "val")
        result = evaluate(y_holdout, final_model.predict_proba(x_holdout)[:, 1], cameras_holdout, None, final_params["threshold"])
        sim_holdouts.append({"csv": str(path.resolve()), **result})
    simulation_precision_pass = all(item["overall"]["precision"] >= 0.90 for item in sim_holdouts)
    real_cross_bag_pass = all(fold["passed"] for fold in folds)
    runtime_allowed = bool(real_cross_bag_pass and simulation_precision_pass)
    report = {
        "schema": "learned-sim-real-sparse-candidate-evaluation/v1",
        "asset_root": str(asset_root), "features": FEATURES,
        "model": "HistGradientBoostingClassifier",
        "training_contract": "simulation domain mass 0.2; real direct/reviewed domain mass 0.8; class-balanced within each domain",
        "split_policy": "outer leave-one-real-bag-out; inner leave-one-camera-out for parameters and threshold",
        "official_third_bag_used": False,
        "simulation_training_rows": int(len(x_sim)), "simulation_training_positives": int(np.sum(y_sim)),
        "gates": {"overall_precision_min": 0.90, "per_camera_precision_min": 0.80,
                  "overall_true_positive_min": 20, "per_camera_true_positive_min": 5,
                  "true_positive_center_error_p90_max_m": 1.0, "simulation_each_world_precision_min": 0.90},
        "real_cross_bag_folds": folds, "simulation_holdouts": sim_holdouts,
        "real_cross_bag_passed": real_cross_bag_pass, "simulation_holdouts_precision_passed": simulation_precision_pass,
        "runtime_allowed": runtime_allowed,
        "next_action": "sealed-third-bag candidate-only experiment" if runtime_allowed else "do not integrate; collect targeted real/simulation hard negatives and class coverage",
    }
    output.mkdir(parents=True)
    joblib.dump(final_model, output / "candidate_model.joblib")
    model_contract = {
        "schema": "learned-sim-real-sparse-candidate-model/v1", "features": FEATURES,
        "candidate_only": True, "runtime_allowed": runtime_allowed, "calibration_acceptance_allowed": False,
        "official_third_bag_used": False, "decision_threshold": final_params["threshold"], "parameters": final_params,
        "model_file": "candidate_model.joblib",
    }
    (output / "model_contract.json").write_text(json.dumps(model_contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "cross_bag_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# 学习型仿真+真实稀疏候选模型\n\n外层按完整真实 bag 留出，内层只在训练 bag 中按相机留出选择模型和阈值。"
        "模型只能输出候选；`runtime_allowed=false` 时禁止加载。官方第三 bag 未使用。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
