#!/usr/bin/env python3
"""Build a deliberately conservative single-bag temporal YOLO prototype split.

This tool is not a substitute for bag-level validation.  It uses separated
contiguous frame-id windows and records the discarded guard frames so its
metrics cannot be misreported as an independent three-bag result.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path


FRAME = re.compile(r"__frame_(\d+)_")
NAMES = {0: "buoy", 1: "pillar", 2: "block", 3: "boat"}


def split_for_frame(frame_id: int) -> str | None:
    """Separated time windows; other labelled frames are guard bands."""
    if 69 <= frame_id <= 84:
        return "train"
    if 92 <= frame_id <= 98:
        return "val"
    if 104 <= frame_id <= 108:
        return "test"
    return None


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-bag", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    records: list[dict] = []
    skipped: Counter[str] = Counter()
    for image in sorted((args.source / "images").glob("*/*.jpg")):
        match = FRAME.search(image.name)
        if not match:
            skipped["unparseable_name"] += 1
            continue
        split = split_for_frame(int(match.group(1)))
        if split is None:
            skipped["guard_band"] += 1
            continue
        old_split = image.parent.name
        label = args.source / "labels" / old_split / f"{image.stem}.txt"
        if not label.exists():
            skipped["missing_label"] += 1
            continue
        image_target = args.output / "images" / split / image.name
        label_target = args.output / "labels" / split / label.name
        hardlink_or_copy(image, image_target)
        hardlink_or_copy(label, label_target)
        records.append({
            "split": split, "camera": image.name.split("__", 1)[0],
            "frame_id": int(match.group(1)), "source_bag": args.source_bag,
            "source_image": str(image), "image": str(image_target), "label": str(label_target),
        })
    if not records:
        raise SystemExit("no image-label pairs selected")

    yaml = "\n".join((
        f"path: {args.output.as_posix()}", "train: images/train", "val: images/val", "test: images/test", "names:",
        *[f"  {index}: {name}" for index, name in NAMES.items()], "",
    ))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dataset.yaml").write_text(yaml, encoding="utf-8")
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    per_split = Counter(row["split"] for row in records)
    audit = {
        "dataset": str(args.output), "source_dataset": str(args.source), "source_bag": args.source_bag,
        "formal_bag_holdout": False,
        "purpose": "raw_fisheye_yolo_prototype_only",
        "reason": "all human labels come from one bag; separated frame windows prevent direct temporal overlap but cannot establish independent bag generalisation",
        "windows": {"train": "69-84", "val": "92-98", "test": "104-108", "guard_bands": "85-91,99-103"},
        "images": dict(per_split), "skipped": dict(skipped),
        "next_required_validation": "train on two independently labelled/simulation bags and hold out a third bag",
    }
    (args.output / "split_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
