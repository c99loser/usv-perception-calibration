#!/usr/bin/env python3
"""Audit frozen pillar rules at independent world/camera-pair granularity."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


FEATURES = ("points", "range_m", "extent_x", "extent_y", "extent_z", "pca1", "pca2", "pca3", "linearity")


def local_path(value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    if os.name == "nt" and value.startswith("/mnt/") and len(value) > 6:
        drive = value[5].upper()
        return Path(f"{drive}:\\{value[7:].replace('/', os.sep)}")
    return path


def load_rows(paths: list[str]) -> list[dict[str, str]]:
    rows = []
    for value in paths:
        with local_path(value).open(newline="", encoding="utf-8") as handle:
            rows.extend(row for row in csv.DictReader(handle) if row["split"] == "val")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    report = json.loads(args.rule.read_text(encoding="utf-8"))
    rows = load_rows(report["evaluation_csvs"])
    params = report["rule"]
    if tuple(report["features"]) != FEATURES:
        raise SystemExit("unsupported feature contract")
    matrix = np.asarray([[float(row[name]) for name in FEATURES] for row in rows])
    center, scale = np.asarray(params["center"]), np.asarray(params["scale"])
    positive = np.asarray(params["positive_prototypes"])
    negative = np.asarray(params["negative_prototypes"])
    normalized = (matrix - center) / scale
    dp = np.linalg.norm(normalized[:, None] - positive[None, :], axis=2).min(axis=1)
    dn = np.linalg.norm(normalized[:, None] - negative[None, :], axis=2).min(axis=1)
    ratio = float(params.get("max_positive_negative_ratio", 1.0))
    predicted = ((dp <= float(params["positive_distance_threshold"])) &
                 (dp < ratio * dn))
    truth = np.asarray([int(row["label_id"]) == 2 for row in rows])

    pairs: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        pairs[(row["source_root"], row["camera"])].append(index)
    details = []
    for (root, camera), indices_list in sorted(pairs.items()):
        indices = np.asarray(indices_list, dtype=int)
        has_target = bool(np.any(truth[indices]))
        tp = int(np.sum(predicted[indices] & truth[indices]))
        fp = int(np.sum(predicted[indices] & ~truth[indices]))
        details.append(
            {
                "source_root": root,
                "camera": camera,
                "has_target": has_target,
                "target_covered": bool(tp > 0),
                "false_candidate_present": bool(fp > 0),
                "tp_clusters": tp,
                "fp_clusters": fp,
                "fn_clusters": int(np.sum(~predicted[indices] & truth[indices])),
            }
        )
    positive_pairs = [item for item in details if item["has_target"]]
    coverage = sum(item["target_covered"] for item in positive_pairs)
    false_pairs = sum(item["false_candidate_present"] for item in details)
    audit = {
        "purpose": "simulation_training_only_world_camera_pair_coverage",
        "runtime_allowed": False,
        "rule": str(args.rule),
        "pairs": len(details),
        "positive_pairs": len(positive_pairs),
        "covered_positive_pairs": coverage,
        "pair_recall": coverage / len(positive_pairs) if positive_pairs else 0.0,
        "pairs_with_false_candidate": false_pairs,
        "false_pair_fraction": false_pairs / len(details) if details else 0.0,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({key: audit[key] for key in (
        "pairs", "positive_pairs", "covered_positive_pairs", "pair_recall",
        "pairs_with_false_candidate", "false_pair_fraction"
    )}))


if __name__ == "__main__":
    main()
