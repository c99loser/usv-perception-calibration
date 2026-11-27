#!/usr/bin/env python3
"""Validate filled annotation CSVs from the V4.11 real-LiDAR pack."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


LABELS = {"pillar", "buoy", "block", "boat", "background", "ambiguous"}
STATIC_VALUES = {"yes", "no", "uncertain"}
FINISHED = {"reviewed", "adjudicated"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    package = json.loads((args.package / "package_manifest.json").read_text(encoding="utf-8"))
    if package.get("schema") != "real-lidar-manual-annotation-pack/v4.11":
        raise SystemExit("unsupported annotation package")
    manifest_rows = list(csv.DictReader((args.package / "manifest.csv").open(encoding="utf-8-sig")))
    expected = {row["item_id"]: row for row in manifest_rows}
    errors: list[str] = []
    counts = Counter()
    reviewed_tracks: dict[str, set[str]] = {"train": set(), "holdout": set()}
    for split in ("train", "holdout"):
        path = args.package / "annotations" / f"{split}_labels.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        seen = set()
        for row in rows:
            item_id = row.get("item_id", "")
            if item_id in seen:
                errors.append(f"{split}:{item_id}:duplicate")
            seen.add(item_id)
            if item_id not in expected or expected[item_id]["split_role"] != split:
                errors.append(f"{split}:{item_id}:not_in_immutable_manifest")
                continue
            for field, value in expected[item_id].items():
                if row.get(field) != value:
                    errors.append(f"{split}:{item_id}:immutable_field_changed:{field}")
                    break
            status = row.get("review_status", "")
            if status not in {"unreviewed", "needs_second_review", *FINISHED}:
                errors.append(f"{split}:{item_id}:invalid_review_status")
            if status in FINISHED:
                label = row.get("human_label", "")
                static = row.get("static_for_calibration", "")
                if label not in LABELS:
                    errors.append(f"{split}:{item_id}:invalid_label")
                if static not in STATIC_VALUES:
                    errors.append(f"{split}:{item_id}:invalid_static_value")
                if label == "boat" and static != "no":
                    errors.append(f"{split}:{item_id}:boat_must_be_nonstatic")
                if label == "ambiguous" and static != "uncertain":
                    errors.append(f"{split}:{item_id}:ambiguous_must_be_uncertain")
                counts[f"{split}:reviewed"] += 1
                counts[f"{split}:label:{label}"] += 1
                if row.get("geometry_eligible") == "yes":
                    counts[f"{split}:eligible_reviewed"] += 1
                    if label in {"pillar", "buoy", "block"} and static == "yes":
                        counts[f"{split}:eligible_static_positive"] += 1
                reviewed_tracks[split].add(row["track_group_id"])
            elif args.require_complete:
                errors.append(f"{split}:{item_id}:not_reviewed")
        expected_split = {item_id for item_id, row in expected.items() if row["split_role"] == split}
        if seen != expected_split:
            errors.append(f"{split}:row_set_mismatch")
    report = {
        "schema": "real-lidar-annotation-validation/v1",
        "passed": not errors,
        "require_complete": args.require_complete,
        "errors": errors[:200],
        "counts": dict(counts),
        "reviewed_track_groups": {key: len(value) for key, value in reviewed_tracks.items()},
        "warnings": [
            "ambiguous/uncertain rows must be excluded from training and evaluation",
            "holdout rows must never be used for model or threshold selection",
            "at least 20 eligible static-positive holdout matches are required for L5",
        ],
    }
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
