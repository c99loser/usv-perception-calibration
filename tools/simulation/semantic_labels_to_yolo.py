#!/usr/bin/env python3
"""Convert training-only Gazebo semantic label maps into raw-fisheye YOLO boxes.

The input labels are renderer output, not projected world poses.  A simulation
collector must assign semantic IDs 1..4 to buoy/pillar/block/boat respectively.
The converter refuses unknown non-background labels, so a scene configuration
mistake cannot silently contaminate a dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


CLASSES = {1: "buoy", 2: "pillar", 3: "block", 4: "boat"}


def extract_boxes(rgb_labels: np.ndarray, min_pixels: int) -> list[tuple[int, int, int, int, int]]:
    """Return (class_id, x, y, width, height) for connected semantic regions."""
    if rgb_labels.ndim != 3 or rgb_labels.shape[2] != 3:
        raise ValueError("label map must be an HxWx3 RGB raster")
    if not (np.array_equal(rgb_labels[..., 0], rgb_labels[..., 1])
            and np.array_equal(rgb_labels[..., 1], rgb_labels[..., 2])):
        raise ValueError("expected Gazebo labels_map grayscale RGB encoding")
    values = rgb_labels[..., 0]
    unknown = sorted(int(x) for x in np.unique(values) if int(x) not in {0, *CLASSES})
    if unknown:
        raise ValueError(f"unknown semantic IDs: {unknown}; check scene label plugins")
    boxes: list[tuple[int, int, int, int, int]] = []
    for class_id in CLASSES:
        count, _, stats, _ = cv2.connectedComponentsWithStats((values == class_id).astype(np.uint8), 8)
        for index in range(1, count):
            x, y, width, height, pixels = (int(v) for v in stats[index])
            if pixels >= min_pixels:
                boxes.append((class_id, x, y, width, height))
    return boxes


def yolo_row(box: tuple[int, int, int, int, int], image_width: int, image_height: int) -> str:
    class_id, x, y, width, height = box
    return (f"{class_id - 1} {(x + width / 2) / image_width:.8f} {(y + height / 2) / image_height:.8f} "
            f"{width / image_width:.8f} {height / image_height:.8f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-root", type=Path, required=True,
                        help="directories containing rgb.png, labels_map.png, pair.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--min-pixels", type=int, default=20)
    parser.add_argument("--source-scene", required=True)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    pairs = sorted(path for path in args.pairs_root.iterdir() if path.is_dir())
    if not pairs:
        raise SystemExit("no pair directories found")
    images_dir, labels_dir = args.output / "images" / args.split, args.output / "labels" / args.split
    records: list[dict[str, object]] = []
    for pair in pairs:
        rgb_path, map_path, meta_path = pair / "rgb.png", pair / "labels_map.png", pair / "pair.json"
        if not all(path.exists() for path in (rgb_path, map_path, meta_path)):
            raise SystemExit(f"incomplete pair: {pair}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("runtime_allowed") is not False:
            raise SystemExit(f"missing training-only guard in {meta_path}")
        labels_bgr = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
        if labels_bgr is None:
            raise SystemExit(f"cannot read {map_path}")
        labels_rgb = cv2.cvtColor(labels_bgr, cv2.COLOR_BGR2RGB)
        height, width = labels_rgb.shape[:2]
        boxes = extract_boxes(labels_rgb, args.min_pixels)
        stem = pair.name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        image_out, label_out = images_dir / f"{stem}.png", labels_dir / f"{stem}.txt"
        shutil.copy2(rgb_path, image_out)
        label_out.write_text("\n".join(yolo_row(box, width, height) for box in boxes) + ("\n" if boxes else ""), encoding="utf-8")
        records.append({
            "split": args.split, "source_kind": "gazebo_semantic_render_training_only",
            "source_scene": args.source_scene, "seed": args.seed, "pair": str(pair),
            "timestamp_ns": metadata["timestamp_ns"], "image": str(image_out), "label": str(label_out),
            "box_count": len(boxes),
        })
    (args.output / "dataset.yaml").write_text("\n".join((
        f"path: {args.output.as_posix()}", "train: images/train", "val: images/val", "names:",
        *[f"  {index - 1}: {name}" for index, name in CLASSES.items()], "",
    )), encoding="utf-8")
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    report = {"purpose": "simulation_training_only", "runtime_allowed": False, "images": len(records),
              "boxes": sum(int(row["box_count"]) for row in records), "classes": CLASSES,
              "source_scene": args.source_scene, "seed": args.seed}
    (args.output / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
