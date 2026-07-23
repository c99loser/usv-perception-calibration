"""Convert the project's CVAT XML annotations into an auditable YOLO dataset.

The converter deliberately requires the source images.  XML annotations alone
cannot train a detector.  It also normalizes two observed spelling variants:
``bouy`` -> ``buoy`` and ``pillaor`` -> ``pillar``.

Example
-------
python tools/prepare_yolo_dataset.py \
  --annotations-root "F:/Challenger program/datasets" \
  --images-root "F:/Challenger program/datasets/cvat_export_images" \
  --image-space raw_fisheye \
  --output "F:/Challenger program/datasets/usv_yolo_v1"
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import xml.etree.ElementTree as etree
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


CLASS_NAMES = ("buoy", "pillar", "block", "boat")
CLASS_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
LABEL_ALIASES = {"bouy": "buoy", "pillaor": "pillar"}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class Annotation:
    camera: str
    source_xml: Path
    image_name: str
    width: int
    height: int
    boxes: tuple[tuple[str, float, float, float, float], ...]


def normalize_label(label: str, source: Path) -> str:
    normalized = LABEL_ALIASES.get(label, label)
    if normalized not in CLASS_ID:
        raise ValueError(f"{source}: unsupported class {label!r}")
    return normalized


def parse_xml(path: Path) -> list[Annotation]:
    camera = path.parent.name
    root = etree.parse(path).getroot()
    result: list[Annotation] = []
    for image in root.findall("image"):
        width, height = int(image.attrib["width"]), int(image.attrib["height"])
        boxes = []
        for box in image.findall("box"):
            label = normalize_label(box.attrib["label"], path)
            xtl, ytl = float(box.attrib["xtl"]), float(box.attrib["ytl"])
            xbr, ybr = float(box.attrib["xbr"]), float(box.attrib["ybr"])
            # Clamp tiny CVAT rounding overshoots but reject degenerate boxes.
            xtl, ytl = max(0.0, xtl), max(0.0, ytl)
            xbr, ybr = min(float(width), xbr), min(float(height), ybr)
            if not (xtl < xbr and ytl < ybr):
                raise ValueError(f"{path}: invalid box in {image.attrib['name']}")
            boxes.append((label, xtl, ytl, xbr, ybr))
        result.append(Annotation(camera, path, image.attrib["name"], width, height, tuple(boxes)))
    return result


def build_image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.name in index:
                raise ValueError(
                    f"duplicate image basename {path.name!r}: {index[path.name]} and {path}; "
                    "use a unique CVAT export image directory"
                )
            index[path.name] = path
    return index


def frame_group(name: str) -> str:
    """Group same-timestamp multi-camera frames to avoid cross-split leakage."""
    if name.startswith("frame_"):
        return name.split("_", 2)[1]
    return name


def choose_split(group: str) -> str:
    # Deterministic 70/15/15 split by capture frame group.  All cameras for a
    # group share a split, preventing same instant views leaking into validation.
    value = sum((index + 1) * ord(char) for index, char in enumerate(group)) % 20
    return "train" if value < 14 else ("val" if value < 17 else "test")


def write_label(path: Path, item: Annotation) -> None:
    rows = []
    for label, xtl, ytl, xbr, ybr in item.boxes:
        cx = ((xtl + xbr) / 2.0) / item.width
        cy = ((ytl + ybr) / 2.0) / item.height
        width = (xbr - xtl) / item.width
        height = (ybr - ytl) / item.height
        rows.append(f"{CLASS_ID[label]} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}")
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument(
        "--image-space",
        choices=("raw_fisheye", "undistorted"),
        required=True,
        help="Geometry of the images used while annotating in CVAT.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    xmls = sorted(args.annotations_root.rglob("*.xml"))
    if not xmls:
        raise SystemExit(f"No XML annotations found below {args.annotations_root}")
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}; choose a new versioned path")

    annotations = [item for path in xmls for item in parse_xml(path)]
    image_index = build_image_index(args.images_root)
    missing = sorted({item.image_name for item in annotations} - set(image_index))
    if missing:
        preview = "\n  ".join(missing[:10])
        raise SystemExit(
            f"Missing {len(missing)} annotated images below {args.images_root}. "
            f"Export images from CVAT with the XML annotations first. Examples:\n  {preview}"
        )

    args.output.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (args.output / "images" / split).mkdir(parents=True)
        (args.output / "labels" / split).mkdir(parents=True)

    manifest_rows = []
    class_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for item in annotations:
        split = choose_split(frame_group(item.image_name))
        # Prefix camera name because CVAT tasks can share frame basenames.
        target_name = f"{item.camera}__{item.image_name}"
        image_dst = args.output / "images" / split / target_name
        label_dst = args.output / "labels" / split / (Path(target_name).stem + ".txt")
        shutil.copy2(image_index[item.image_name], image_dst)
        write_label(label_dst, item)
        class_counts.update(label for label, *_ in item.boxes)
        split_counts[split] += 1
        manifest_rows.append((split, item.camera, item.image_name, str(item.source_xml), len(item.boxes)))

    yaml = "\n".join((
        f"path: {args.output.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
        "",
    ))
    (args.output / "dataset.yaml").write_text(yaml, encoding="utf-8")
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split", "camera", "source_image", "source_xml", "box_count"))
        writer.writerows(manifest_rows)
    geometry_note = (
        "This dataset is in raw fisheye pixel coordinates. Train and infer on raw "
        "fisheye frames only; its boxes cannot be applied to undistorted frames."
        if args.image_space == "raw_fisheye"
        else "This dataset is in undistorted pixel coordinates. Train and infer on "
        "the same undistortion output only; its boxes cannot be applied to raw fisheye frames."
    )
    readme = "\n".join((
        "# USV YOLO dataset v1",
        "",
        "Generated from CVAT XML annotations.  `bouy` was normalized to `buoy`; "
        "`pillaor` was normalized to `pillar`.",
        "",
        f"Images: {len(annotations)}",
        f"Splits: {dict(split_counts)}",
        f"Class instances: {dict(class_counts)}",
        "",
        f"Image space: {args.image_space}",
        geometry_note,
        "",
    ))
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(f"Created {args.output}")
    print(readme)
    return 0


if __name__ == "__main__":
    sys.exit(main())
