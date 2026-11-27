#!/usr/bin/env python3
"""Fit a conservative pillar-cluster kNN rule and blind-test random worlds.

This is a simulation-training gate only.  Training parameters are computed from
the declared ``train`` rows alone; every supplied evaluation CSV contributes
only its declared ``val`` rows and cannot alter the rule.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FEATURES = ("points", "range_m", "extent_x", "extent_y", "extent_z", "pca1", "pca2", "pca3", "linearity")


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def matrix(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in FEATURES] for row in rows], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-csv", type=Path, required=True)
    parser.add_argument("--evaluation-csv", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance-quantile", type=float, default=0.90)
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--min-per-root-precision", type=float, default=0.90,
                        help="each independent simulation world must meet this precision")
    parser.add_argument("--min-validation-positives", type=int, default=20)
    parser.add_argument("--max-positive-negative-ratio", type=float, default=1.0,
                        help="require d_positive < ratio * d_negative for a conservative safety margin")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if not 0 < args.distance_quantile <= 1:
        raise SystemExit("distance-quantile must be in (0,1]")
    if not 0 <= args.min_precision <= 1 or not 0 <= args.min_per_root_precision <= 1:
        raise SystemExit("precision gates must be in [0,1]")
    if not 0 < args.max_positive_negative_ratio <= 1:
        raise SystemExit("max-positive-negative-ratio must be in (0,1]")
    train_rows = [row for row in load(args.training_csv) if row["split"] == "train"]
    eval_rows = [row for path in args.evaluation_csv for row in load(path) if row["split"] == "val"]
    if not train_rows or not eval_rows:
        raise SystemExit("missing declared train or evaluation rows")
    x_train = matrix(train_rows); y_train = np.asarray([int(row["label_id"]) == 2 for row in train_rows])
    if np.sum(y_train) < 3 or np.sum(~y_train) < 3:
        raise SystemExit("insufficient positive or background training clusters")
    center = np.median(x_train, axis=0)
    scale = np.maximum(np.percentile(x_train, 75, axis=0) - np.percentile(x_train, 25, axis=0), 1e-4)
    z_train = (x_train - center) / scale
    positive, negative = z_train[y_train], z_train[~y_train]
    # Leave-one-positive-out distances make the train-derived acceptance
    # radius meaningful rather than trivially zero.
    pairwise = np.linalg.norm(positive[:, None] - positive[None, :], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    threshold = float(np.quantile(np.min(pairwise, axis=1), args.distance_quantile))
    x_eval = matrix(eval_rows); z_eval = (x_eval - center) / scale
    d_positive = np.linalg.norm(z_eval[:, None] - positive[None, :], axis=2).min(axis=1)
    d_negative = np.linalg.norm(z_eval[:, None] - negative[None, :], axis=2).min(axis=1)
    truth = np.asarray([int(row["label_id"]) == 2 for row in eval_rows])
    predicted = ((d_positive <= threshold) &
                 (d_positive < args.max_positive_negative_ratio * d_negative))
    tp, fp, fn = int(np.sum(predicted & truth)), int(np.sum(predicted & ~truth)), int(np.sum(~predicted & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    by_root = {}
    for root in sorted({row["source_root"] for row in eval_rows}):
        indices = np.asarray([row["source_root"] == root for row in eval_rows])
        one_tp, one_fp, one_fn = int(np.sum(predicted[indices] & truth[indices])), int(np.sum(predicted[indices] & ~truth[indices])), int(np.sum(~predicted[indices] & truth[indices]))
        by_root[root] = {"positive_clusters": int(np.sum(truth[indices])), "tp": one_tp, "fp": one_fp, "fn": one_fn,
                         "precision": one_tp / (one_tp + one_fp) if one_tp + one_fp else 0.0,
                         "recall": one_tp / (one_tp + one_fn) if one_tp + one_fn else 0.0}
    root_precision_ok = all(metrics["precision"] >= args.min_per_root_precision for metrics in by_root.values())
    accepted = (precision >= args.min_precision and root_precision_ok and
                int(np.sum(truth)) >= args.min_validation_positives)
    report = {"purpose": "simulation_training_only_pillar_candidate_gate", "runtime_allowed": False,
              "features": FEATURES, "training_csv": str(args.training_csv), "evaluation_csvs": [str(path) for path in args.evaluation_csv],
              "rule": {"kind": "scaled_1nn_positive_vs_negative", "distance_quantile_from_train_only": args.distance_quantile,
                       "positive_distance_threshold": threshold,
                       "max_positive_negative_ratio": args.max_positive_negative_ratio,
                       "center": center.tolist(), "scale": scale.tolist(),
                       "positive_prototypes": positive.tolist(), "negative_prototypes": negative.tolist()},
              "evaluation": {"positive_clusters": int(np.sum(truth)), "tp": tp, "fp": fp, "fn": fn,
                             "precision": precision, "recall": recall, "by_root": by_root},
              "gates": {"min_precision": args.min_precision, "min_per_root_precision": args.min_per_root_precision,
                        "per_root_precision_pass": root_precision_ok, "min_validation_positives": args.min_validation_positives},
              "accepted_for_real_mcap_candidate_stage": accepted,
              "next_required": "real-MCAP cross-bag repeatability; never use simulation geometry in runtime optimization"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"accepted": accepted, **report["evaluation"]}))


if __name__ == "__main__":
    main()
