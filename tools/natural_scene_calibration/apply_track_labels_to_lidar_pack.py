#!/usr/bin/env python3
"""Propagate reviewed track-level labels to the immutable row-level pack."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABELS = {"pillar", "buoy", "block", "boat", "background", "ambiguous"}
STATIC_VALUES = {"yes", "no", "uncertain"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--track-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    payload = json.loads(args.track_labels.read_text(encoding="utf-8-sig"))
    if payload.get("schema") != "real-lidar-track-labels/v1":
        raise SystemExit("unsupported track-label schema")
    split = payload.get("split")
    if split not in {"train", "holdout"}:
        raise SystemExit("track-label split must be train or holdout")
    labels = payload.get("items", {})
    rows = list(
        csv.DictReader(
            (args.package / "annotations" / f"{split}_labels.csv").open(
                encoding="utf-8-sig"
            )
        )
    )
    expected_tracks = {row["track_group_id"] for row in rows}
    unknown = set(labels) - expected_tracks
    if unknown:
        raise SystemExit(f"unknown track ids: {sorted(unknown)[:5]}")
    missing = sorted(expected_tracks - set(labels))
    if args.require_complete and missing:
        raise SystemExit(f"{len(missing)} tracks are missing labels")
    completed = 0
    for row in rows:
        value = labels.get(row["track_group_id"], {})
        label = value.get("human_label", "")
        static = value.get("static_for_calibration", "")
        if not label:
            continue
        if label not in LABELS or static not in STATIC_VALUES:
            raise SystemExit(f"invalid label/static pair for {row['track_group_id']}")
        if label == "boat" and static != "no":
            raise SystemExit(f"boat must be non-static: {row['track_group_id']}")
        if label == "ambiguous" and static != "uncertain":
            raise SystemExit(f"ambiguous must be uncertain: {row['track_group_id']}")
        row.update(
            {
                "human_label": label,
                "static_for_calibration": static,
                "review_status": "reviewed",
                "reviewer_confidence": value.get("reviewer_confidence", ""),
                "reviewer_id": args.reviewer_id,
                "reviewer_note": value.get("reviewer_note", ""),
            }
        )
        completed += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "split": split,
                "rows": len(rows),
                "tracks": len(expected_tracks),
                "missing_tracks": len(missing),
                "labeled_rows": completed,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
