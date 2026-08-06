#!/usr/bin/env python3
"""Create a train/validation YOLO config with a strict three-bag boundary.

The two-bag human training dataset stays immutable.  All labelled images from
the third bag are listed only as ``val``/``test``.  This prevents legacy
per-frame train/val/test folders from reintroducing temporal leakage.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


CLASSES = ("buoy", "pillar", "block", "boat")
TIMESTAMP = re.compile(r"(\d{18,20})")


def timestamps(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        match = TIMESTAMP.search(path.stem)
        if not match:
            raise SystemExit(f"cannot derive timestamp from {path.name}")
        result.add(match.group(1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--holdout-dataset", type=Path, required=True)
    parser.add_argument("--holdout-bag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    train_report_path = args.train_dataset / "lineage_report.json"
    train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
    if args.holdout_bag in train_report.get("allowed_bags", []):
        raise SystemExit("declared holdout bag appears in training lineage")
    train_images = sorted((args.train_dataset / "images" / "train").glob("*"))
    if not train_images:
        raise SystemExit("training images are missing")
    manifest_path = args.holdout_dataset / "manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        holdout_rows = list(csv.DictReader(handle))
    if not holdout_rows:
        raise SystemExit("holdout manifest is empty")
    holdout_images: list[Path] = []
    records: list[dict[str, str]] = []
    for row in holdout_rows:
        old_split = row.get("split", "")
        source_name = Path(row.get("source_image", "")).name
        camera = row.get("camera", "")
        # CVAT's manifest records the source frame name, while the dataset
        # builder prefixes copied files with ``<camera>__`` to avoid collisions.
        image = args.holdout_dataset / "images" / old_split / f"{camera}__{source_name}"
        label = args.holdout_dataset / "labels" / old_split / f"{image.stem}.txt"
        if not image.is_file() or not label.is_file():
            raise SystemExit(f"missing holdout image or label for {source_name}")
        holdout_images.append(image)
        records.append({"split": "val", "source_bag": args.holdout_bag,
                        "camera": camera, "legacy_split": old_split,
                        "source_image": str(image), "source_label": str(label)})
    train_stamps, holdout_stamps = timestamps(train_images), timestamps(holdout_images)
    overlap = train_stamps & holdout_stamps
    if overlap:
        raise SystemExit(f"timestamp overlap across training and holdout: {sorted(overlap)[:3]}")
    args.output.mkdir(parents=True)
    list_path = args.output / "holdout_images.txt"
    list_path.write_text("\n".join(path.as_posix() for path in holdout_images) + "\n", encoding="utf-8")
    yaml = "\n".join((
        f"path: {args.output.as_posix()}", f"train: {((args.train_dataset / 'images' / 'train')).as_posix()}",
        "val: holdout_images.txt", "test: holdout_images.txt", "names:",
        *[f"  {index}: {name}" for index, name in enumerate(CLASSES)], "",
    ))
    (args.output / "dataset.yaml").write_text(yaml, encoding="utf-8")
    with (args.output / "holdout_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    audit = {
        "purpose": "raw_fisheye_yolo_formal_three_bag_validation",
        "formal_bag_holdout": True,
        "train_bags": train_report.get("allowed_bags", []),
        "holdout_bag": args.holdout_bag,
        "train_images": len(train_images), "holdout_images": len(holdout_images),
        "holdout_per_camera": dict(Counter(row["camera"] for row in records)),
        "legacy_holdout_splits_reclassified_as_val": dict(Counter(row["legacy_split"] for row in records)),
        "timestamp_overlap": 0,
        "rule": "holdout images may not be used for training, threshold selection, or model selection",
    }
    (args.output / "holdout_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
