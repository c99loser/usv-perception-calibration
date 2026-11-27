#!/usr/bin/env python3
"""Validate a V3.1 sparse real-LiDAR training asset without mutating it."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate(root: Path) -> dict[str, object]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((root / "audit_report.json").read_text(encoding="utf-8"))
    observations = read_csv(root / "observations.csv")
    suggestions = read_csv(root / "unreviewed_propagation_suggestions.csv")
    quarantine = read_csv(root / "quarantine.csv")
    tracks = read_csv(root / "track_summary.csv")
    archive = np.load(root / "observations.npz", allow_pickle=False)

    checks: dict[str, bool] = {}
    checks["schema"] = manifest.get("schema") == "real-lidar-sparse-training/v3.1" == report.get("schema")
    checks["row_index_contiguous"] = [int(row["row_index"]) for row in observations] == list(range(len(observations)))
    offsets = archive["point_offsets"]
    support_counts = np.asarray([int(row["support_count"]) for row in observations], dtype=np.int64)
    checks["offset_shape"] = offsets.shape == (len(observations) + 1,)
    checks["offsets_match_csv"] = bool(
        checks["offset_shape"]
        and offsets[0] == 0
        and np.array_equal(np.diff(offsets), support_counts)
        and offsets[-1] == len(archive["points_xyz"]) == len(archive["raw_point_indices"])
    )
    vector_keys = (
        "centers_xyz", "support_extents_xyz", "class_ids", "static_candidate_targets",
        "observation_type_ids", "geometry_eligible", "calibration_eligible",
        "geometry_weights", "calibration_weights",
    )
    checks["array_rows_match_csv"] = all(len(archive[key]) == len(observations) for key in vector_keys)
    checks["finite_training_arrays"] = bool(
        np.isfinite(archive["points_xyz"]).all()
        and np.isfinite(archive["centers_xyz"]).all()
        and np.isfinite(archive["support_extents_xyz"]).all()
        and np.isfinite(archive["geometry_weights"]).all()
        and np.isfinite(archive["calibration_weights"]).all()
    )
    checks["no_zero_support_training_row"] = all(int(row["support_count"]) > 0 for row in observations)
    checks["no_unreviewed_supervision"] = all(row["source_kind"] != "unreviewed_propagation" for row in observations)
    checks["suggestion_pool_is_unreviewed_only"] = all(row["source_kind"] == "unreviewed_propagation" for row in suggestions)
    checks["holdout_not_referenced"] = not any("holdout" in row["pcd_path"].lower() or "holdout" in row["image_path"].lower() for row in observations)
    checks["report_counts_match"] = (
        int(report["training_observations"]) == len(observations)
        and int(report["unreviewed_propagation_excluded"]) == len(suggestions)
        and int(report["quarantined_manual_observations"]) == len(quarantine)
        and int(report["support_points"]) == len(archive["points_xyz"])
        and int(report["calibration_eligible_tracks"]) == sum(row["calibration_eligible"].lower() == "true" for row in tracks)
    )
    source_root = Path(manifest["source_root"])
    source_project = source_root / "supervisely_train_raw_project"
    checks["source_sidecars_unchanged"] = all(
        (source_project / name).is_file() and sha256(source_project / name) == expected
        for name, expected in manifest["source_sidecar_sha256"].items()
    )

    expected_checksums: dict[str, str] = {}
    for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            expected, name = line.split("  ", 1)
            expected_checksums[name] = expected
    checks["output_checksums"] = all((root / name).is_file() and sha256(root / name) == expected for name, expected in expected_checksums.items())
    result: dict[str, object] = {
        "schema": "real-lidar-sparse-training-validation/v1",
        "root": str(root),
        "checks": checks,
        "passed": all(checks.values()),
        "counts": {
            "observations": len(observations),
            "support_points": len(archive["points_xyz"]),
            "suggestions": len(suggestions),
            "quarantine": len(quarantine),
            "tracks": len(tracks),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = validate(args.root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
