#!/usr/bin/env python3
"""Convert CVAT 1.1 XML annotations to YOLO with strict bag-level lineage.

This importer intentionally accepts only XML files stored alongside their source
images as ``<root>/images/<bag>/<camera>/annotations.xml``.  It refuses labels
from an unapproved bag and writes a manifest for the later leakage audit.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


CLASSES = ("buoy", "pillar", "block", "boat")
LABEL_ALIASES = {"bouy": "buoy"}  # Existing CVAT typo; recorded in the report.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def canonical_bag(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def as_float(element: ET.Element, key: str) -> float:
    try:
        return float(element.attrib[key])
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"invalid CVAT box attribute {key!r}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--allowed-bag", action="append", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    allowed = {canonical_bag(item) for item in args.allowed_bag}
    image_dir = args.output / "images" / args.split
    label_dir = args.output / "labels" / args.split
    records: list[dict[str, object]] = []
    aliases = Counter()
    clipped_boxes = 0
    skipped_boxes = 0

    for xml_path in args.xml:
        if xml_path.name != "annotations.xml" or not xml_path.is_file():
            raise SystemExit(f"expected an annotations.xml file, received {xml_path}")
        camera_dir = xml_path.parent
        camera = camera_dir.name
        bag = canonical_bag(camera_dir.parent.name)
        if bag not in allowed:
            raise SystemExit(f"XML {xml_path} belongs to unapproved bag {bag!r}")
        root = ET.parse(xml_path).getroot()
        if root.tag != "annotations":
            raise SystemExit(f"{xml_path} is not a CVAT annotations XML")
        for image in root.findall("image"):
            image_name = Path(image.attrib.get("name", ""))
            if image_name.name != str(image_name) or image_name.suffix.lower() not in IMAGE_EXTENSIONS:
                raise SystemExit(f"unsafe or unsupported image name {image_name!s} in {xml_path}")
            source = camera_dir / image_name
            if not source.is_file():
                raise SystemExit(f"XML image is not in its source directory: {source}")
            try:
                width, height = int(image.attrib["width"]), int(image.attrib["height"])
            except (KeyError, ValueError) as exc:
                raise SystemExit(f"invalid image dimensions in {xml_path}") from exc
            if width <= 0 or height <= 0:
                raise SystemExit(f"non-positive image dimensions in {xml_path}")
            rows: list[str] = []
            for box in image.findall("box"):
                raw_label = box.attrib.get("label", "")
                label = LABEL_ALIASES.get(raw_label, raw_label)
                if label not in CLASSES:
                    raise SystemExit(f"unknown label {raw_label!r} in {xml_path}")
                if label != raw_label:
                    aliases[raw_label] += 1
                x1, y1, x2, y2 = (as_float(box, key) for key in ("xtl", "ytl", "xbr", "ybr"))
                clipped = (max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))
                if clipped != (x1, y1, x2, y2):
                    clipped_boxes += 1
                x1, y1, x2, y2 = clipped
                if x2 <= x1 or y2 <= y1:
                    skipped_boxes += 1
                    continue
                class_id = CLASSES.index(label)
                rows.append(
                    f"{class_id} {(x1 + x2) / (2 * width):.8f} {(y1 + y2) / (2 * height):.8f} "
                    f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}"
                )
            if not rows:
                continue
            target_stem = f"{bag}_{camera}_{image_name.stem}"
            target_image = image_dir / f"{target_stem}{image_name.suffix.lower()}"
            target_label = label_dir / f"{target_stem}.txt"
            if target_image.exists() or target_label.exists():
                raise SystemExit(f"non-unique output name {target_stem}")
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target_image)
            target_label.write_text("\n".join(rows) + "\n", encoding="utf-8")
            records.append({"split": args.split, "source_kind": "human_cvat_xml", "source_bag": bag,
                            "camera": camera, "source_xml": str(xml_path), "source_image": str(source),
                            "image": str(target_image), "label": str(target_label), "box_count": len(rows)})
    if not records:
        raise SystemExit("no usable annotated images after validation")
    yaml = (f"path: {args.output.as_posix()}\n{args.split}: images/{args.split}\nnames:\n" +
            "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASSES)))
    (args.output / "dataset.yaml").write_text(yaml, encoding="utf-8")
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    report = {"purpose": "human_real_training_only", "split": args.split, "allowed_bags": sorted(allowed),
              "images": len(records), "boxes": sum(int(row["box_count"]) for row in records),
              "per_bag": dict(Counter(str(row["source_bag"]) for row in records)),
              "per_camera": dict(Counter(str(row["camera"]) for row in records)),
              "label_aliases": dict(aliases), "clipped_boxes": clipped_boxes, "skipped_boxes": skipped_boxes}
    (args.output / "lineage_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
