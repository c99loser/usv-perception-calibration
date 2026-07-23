#!/usr/bin/env python3
"""Convert CVAT COCO annotations to YOLO while enforcing bag-level lineage."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


CLASSES = ("buoy", "pillar", "block", "boat")


def canonical_bag(value: str) -> str:
    value = value.replace("\\", "/").rstrip("/")
    return value.rsplit("/", 1)[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--allowed-bag", action="append", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    allowed = {canonical_bag(item) for item in args.allowed_bag}
    payload = json.loads(args.coco.read_text(encoding="utf-8"))
    categories = {int(item["id"]): item["name"] for item in payload.get("categories", [])}
    if set(categories.values()) != set(CLASSES):
        raise SystemExit(f"expected exactly classes {CLASSES}, received {tuple(categories.values())}")
    id_to_class = {cid: CLASSES.index(name) for cid, name in categories.items()}
    images = {int(item["id"]): item for item in payload.get("images", [])}
    grouped: dict[int, list[dict]] = {image_id: [] for image_id in images}
    for annotation in payload.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id not in images or int(annotation["category_id"]) not in id_to_class:
            raise SystemExit("annotation references an unknown image or category")
        grouped[image_id].append(annotation)
    if not any(grouped.values()):
        raise SystemExit("COCO contains no annotations; refusing to create an empty training dataset")

    image_dir, label_dir = args.output / "images" / args.split, args.output / "labels" / args.split
    records: list[dict[str, object]] = []
    for image_id, image in sorted(images.items()):
        source_bag = canonical_bag(str(image.get("source_bag", "")))
        if source_bag not in allowed:
            raise SystemExit(f"image {image_id} source bag {source_bag!r} is not allowed")
        source = args.images_root / image["file_name"]
        if not source.is_file():
            raise SystemExit(f"missing image {source}")
        width, height = int(image["width"]), int(image["height"])
        rows: list[str] = []
        for annotation in grouped[image_id]:
            x, y, box_width, box_height = (float(v) for v in annotation["bbox"])
            if box_width <= 0 or box_height <= 0 or x < 0 or y < 0 or x + box_width > width or y + box_height > height:
                raise SystemExit(f"invalid bbox in image {image_id}")
            class_id = id_to_class[int(annotation["category_id"])]
            rows.append(f"{class_id} {(x + box_width / 2) / width:.8f} {(y + box_height / 2) / height:.8f} {box_width / width:.8f} {box_height / height:.8f}")
        if not rows:
            continue
        image_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True)
        target_image = image_dir / Path(image["file_name"]).name
        target_label = label_dir / f"{target_image.stem}.txt"
        if target_image.exists():
            raise SystemExit(f"non-unique output name {target_image.name}")
        shutil.copy2(source, target_image); target_label.write_text("\n".join(rows) + "\n", encoding="utf-8")
        records.append({"split": args.split, "source_kind": "human_cvat_coco", "source_bag": source_bag,
                        "camera": image.get("camera", ""), "timestamp_ns": image.get("timestamp_ns", ""),
                        "source_image": str(source), "image": str(target_image), "label": str(target_label),
                        "box_count": len(rows)})
    if not records:
        raise SystemExit("no annotated images after validation")
    (args.output / "dataset.yaml").write_text("\n".join((f"path: {args.output.as_posix()}", f"{args.split}: images/{args.split}", "names:", *[f"  {i}: {name}" for i, name in enumerate(CLASSES)], "")), encoding="utf-8")
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)
    report = {"purpose": "human_real_training_only", "split": args.split, "allowed_bags": sorted(allowed),
              "images": len(records), "boxes": sum(int(row["box_count"]) for row in records),
              "per_bag": dict(Counter(str(row["source_bag"]) for row in records))}
    (args.output / "lineage_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
