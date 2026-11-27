#!/usr/bin/env python3
"""Train a center-quality-aware candidate model with grouped inner OOF gates."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold


LEARNED_PATH = Path(__file__).with_name("train_learned_sim_real_candidate_v3_1.py")
SPEC = importlib.util.spec_from_file_location("learned_candidate", LEARNED_PATH)
assert SPEC and SPEC.loader
LEARNED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEARNED)
BASE = LEARNED.BASE


def choose_one_threshold(truth: np.ndarray, probability: np.ndarray, errors: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates = []
    for threshold in np.linspace(0.30, 0.995, 140):
        predicted = probability >= threshold
        one = BASE.metrics(truth, predicted, errors)
        center_ok = one["true_positive_center_error_p90_m"] is not None and one["true_positive_center_error_p90_m"] <= 1.0
        viable = one["precision"] >= 0.90 and one["tp"] >= 5 and center_ok
        fallback = one["precision"] >= 0.80 and one["tp"] >= 3 and center_ok
        score = (float(viable), float(fallback), one["precision"], one["recall"], float(one["tp"]))
        candidates.append((score, float(threshold), one))
    _, threshold, metrics = max(candidates, key=lambda item: item[0])
    return threshold, metrics


def evaluate_thresholds(
    truth: np.ndarray,
    probability: np.ndarray,
    cameras: np.ndarray,
    errors: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    predicted = np.asarray([probability[index] >= thresholds[camera] for index, camera in enumerate(cameras)])
    overall = BASE.metrics(truth, predicted, errors)
    by_camera = {}
    for camera in LEARNED.CAMERAS:
        mask = cameras == camera
        by_camera[camera] = {"threshold": thresholds[camera], **BASE.metrics(truth[mask], predicted[mask], errors[mask])}
    return {"overall": overall, "by_camera": by_camera}


def tune_grouped_oof(
    x: np.ndarray,
    quality_target: np.ndarray,
    evaluation_truth: np.ndarray,
    cameras: np.ndarray,
    groups: np.ndarray,
    errors: np.ndarray,
    x_sim: np.ndarray,
    y_sim: np.ndarray,
) -> dict[str, Any]:
    grid = (
        {"max_leaf_nodes": 7, "min_samples_leaf": 10, "l2_regularization": 5.0},
        {"max_leaf_nodes": 15, "min_samples_leaf": 10, "l2_regularization": 5.0},
        {"max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 10.0},
        {"max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 10.0},
    )
    candidates = []
    splitter = GroupKFold(n_splits=3)
    for params in grid:
        probability = np.full(len(x), np.nan, dtype=np.float64)
        for training_indices, validation_indices in splitter.split(x, quality_target, groups):
            model = LEARNED.fit_model(x[training_indices], quality_target[training_indices], x_sim, y_sim, params)
            probability[validation_indices] = model.predict_proba(x[validation_indices])[:, 1]
        if not np.isfinite(probability).all():
            raise ValueError("grouped OOF prediction incomplete")
        thresholds = {}; inner_by_camera = {}
        for camera in LEARNED.CAMERAS:
            mask = cameras == camera
            threshold, one = choose_one_threshold(evaluation_truth[mask], probability[mask], errors[mask])
            thresholds[camera] = threshold; inner_by_camera[camera] = one
        result = evaluate_thresholds(evaluation_truth, probability, cameras, errors, thresholds)
        overall = result["overall"]
        camera_gate_count = sum(
            value["precision"] >= 0.80 and value["tp"] >= 5
            and value["true_positive_center_error_p90_m"] is not None
            and value["true_positive_center_error_p90_m"] <= 1.0
            for value in result["by_camera"].values()
        )
        score = (camera_gate_count, float(overall["precision"] >= 0.90),
                 float(overall["true_positive_center_error_p90_m"] is not None and overall["true_positive_center_error_p90_m"] <= 1.0),
                 overall["precision"], overall["recall"])
        candidates.append((score, {**params, "thresholds": thresholds, "inner_oof": result}))
    return max(candidates, key=lambda item: item[0])[1]


def gate(result: dict[str, Any]) -> bool:
    overall = result["overall"]
    return bool(
        overall["precision"] >= 0.90 and overall["tp"] >= 20
        and overall["true_positive_center_error_p90_m"] is not None
        and overall["true_positive_center_error_p90_m"] <= 1.0
        and all(
            value["precision"] >= 0.80 and value["tp"] >= 5
            and value["true_positive_center_error_p90_m"] is not None
            and value["true_positive_center_error_p90_m"] <= 1.0
            for value in result["by_camera"].values()
        )
    )


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
    base_features = BASE.feature_matrix(rows, archive)[:, LEARNED.REAL_FEATURE_INDICES]
    cameras = np.asarray([row["camera"] for row in rows])
    x = np.vstack([np.r_[values, LEARNED.camera_one_hot(camera)] for values, camera in zip(base_features, cameras)])
    static_truth = np.asarray([row["static_candidate_target"] == "1" for row in rows])
    bags = np.asarray([row["bag_id"] for row in rows])
    groups = np.asarray([f"{row['episode_id']}::{row['track_id']}" for row in rows])
    original_xy = np.asarray([[float(row["original_center_x"]), float(row["original_center_y"])] for row in rows])
    derived_xy = np.asarray([[float(row["derived_center_x"]), float(row["derived_center_y"])] for row in rows])
    errors = np.linalg.norm(derived_xy - original_xy, axis=1)
    quality_target = static_truth & (errors <= 1.0)
    x_sim, y_sim, _ = LEARNED.load_sim(args.simulation_training_csv.resolve(), "train")
    unique_bags = sorted(set(bags))
    if len(unique_bags) != 2:
        raise SystemExit("expected exactly two real training bags")

    folds = []
    for evaluation_bag in unique_bags:
        training = bags != evaluation_bag; evaluation = ~training
        params = tune_grouped_oof(x[training], quality_target[training], static_truth[training], cameras[training],
                                  groups[training], errors[training], x_sim, y_sim)
        model = LEARNED.fit_model(x[training], quality_target[training], x_sim, y_sim, params)
        result = evaluate_thresholds(static_truth[evaluation], model.predict_proba(x[evaluation])[:, 1], cameras[evaluation],
                                     errors[evaluation], params["thresholds"])
        folds.append({"training_bags": [bag for bag in unique_bags if bag != evaluation_bag], "evaluation_bag": evaluation_bag,
                      "training_only_parameters": params, **result, "passed": gate(result)})

    final_params = tune_grouped_oof(x, quality_target, static_truth, cameras, groups, errors, x_sim, y_sim)
    final_model = LEARNED.fit_model(x, quality_target, x_sim, y_sim, final_params)
    sim_holdouts = []
    for path in args.simulation_holdout_csv:
        x_holdout, truth_holdout, cameras_holdout = LEARNED.load_sim(path.resolve(), "val")
        zero_errors = np.zeros(len(truth_holdout))
        result = evaluate_thresholds(truth_holdout, final_model.predict_proba(x_holdout)[:, 1], cameras_holdout,
                                     zero_errors, final_params["thresholds"])
        sim_holdouts.append({"csv": str(path.resolve()), **result})
    real_pass = all(fold["passed"] for fold in folds)
    sim_pass = all(item["overall"]["precision"] >= 0.90 for item in sim_holdouts)
    runtime_allowed = bool(real_pass and sim_pass)
    report = {
        "schema": "quality-aware-sim-real-sparse-candidate-evaluation/v1",
        "asset_root": str(asset_root), "features": LEARNED.FEATURES,
        "model": "HistGradientBoostingClassifier",
        "real_training_target": "static pillar|buoy|block AND derived-to-manual XY center error <=1m",
        "evaluation_truth": "all static pillar|buoy|block; center quality enforced as a separate P90 gate",
        "split_policy": "outer complete bag holdout; inner 3-fold episode+track grouped OOF; per-camera thresholds from training only",
        "official_third_bag_used": False,
        "real_quality_positive_rows": int(np.sum(quality_target)), "real_static_rows": int(np.sum(static_truth)),
        "gates": {"overall_precision_min": 0.90, "per_camera_precision_min": 0.80,
                  "overall_true_positive_min": 20, "per_camera_true_positive_min": 5,
                  "overall_and_per_camera_center_error_p90_max_m": 1.0,
                  "simulation_each_world_precision_min": 0.90},
        "real_cross_bag_folds": folds, "simulation_holdouts": sim_holdouts,
        "real_cross_bag_passed": real_pass, "simulation_holdouts_precision_passed": sim_pass,
        "runtime_allowed": runtime_allowed,
        "next_action": "sealed-third-bag candidate-only experiment" if runtime_allowed else "do not integrate; export targeted review/hard-negative package from failed cameras",
    }
    output.mkdir(parents=True)
    joblib.dump(final_model, output / "candidate_model.joblib")
    contract = {"schema": "quality-aware-sim-real-sparse-candidate-model/v1", "features": LEARNED.FEATURES,
                "candidate_only": True, "runtime_allowed": runtime_allowed, "calibration_acceptance_allowed": False,
                "official_third_bag_used": False, "camera_thresholds": final_params["thresholds"],
                "model_file": "candidate_model.joblib"}
    (output / "model_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "cross_bag_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# 中心质量感知候选模型\n\n训练目标同时要求静态类别正确和实际支持中心误差不超过 1 m。"
        "阈值按相机在训练 bag 内的 track 分组 OOF 预测上冻结；第三 bag 未使用。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
