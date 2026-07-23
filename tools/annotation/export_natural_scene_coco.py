#!/usr/bin/env python3
"""Export raw-fisheye natural-scene images to an annotation-ready COCO set.

This is deliberately a data-preparation tool, never a runtime input.  It reads
only the six permitted image streams and records their original timestamps;
truth, TF, odometry and fixed-target coordinates are neither read nor written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "camera_fisheye"))
from camera_fisheye_pipeline import (  # noqa: E402
    CAMERA_NAMES,
    convert_image_message,
    iter_ros_messages,
    parse_metadata,
    select_camera_topics,
)


CATEGORIES = (
    {"id": 1, "name": "buoy"},
    {"id": 2, "name": "pillar"},
    {"id": 3, "name": "block"},
    {"id": 4, "name": "boat"},
)


def evenly_spaced_indices(total: int, wanted: int) -> set[int]:
    """Return up to ``wanted`` deterministic, inclusive indices over a stream."""
    if total <= 0 or wanted <= 0:
        return set()
    if wanted >= total:
        return set(range(total))
    return {round(index * (total - 1) / (wanted - 1)) for index in range(wanted)}


def export_bag(bag_dir: Path, output_root: Path, samples_per_camera: int,
               image_id_start: int) -> tuple[list[dict], int]:
    import cv2

    summary = parse_metadata(bag_dir / "metadata.yaml")
    selections = select_camera_topics(summary)
    topic_to_selection = {item.image_topic: item for item in selections if item.image_topic}
    desired = {
        item.camera: evenly_spaced_indices(item.image_count, samples_per_camera)
        for item in selections
    }
    seen = {camera: 0 for camera in CAMERA_NAMES}
    images: list[dict] = []
    image_id = image_id_start
    for topic, message, timestamp in iter_ros_messages(bag_dir, list(topic_to_selection)):
        selection = topic_to_selection[topic]
        camera = selection.camera
        frame_index = seen[camera]
        seen[camera] += 1
        if frame_index not in desired[camera]:
            continue
        image = convert_image_message(message)
        relative = Path("images") / bag_dir.name / camera / f"{timestamp}.jpg"
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"Unable to write {destination}")
        images.append({
            "id": image_id,
            "file_name": relative.as_posix(),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "camera": camera,
            "timestamp_ns": int(timestamp),
            "source_bag": str(bag_dir),
            "source_topic": topic,
            "frame_index": frame_index,
        })
        image_id += 1
    missing = [camera for camera in CAMERA_NAMES if not any(x["camera"] == camera for x in images)]
    if missing:
        raise RuntimeError(f"No selected images exported for: {', '.join(missing)}")
    return images, image_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-camera", type=int, default=50)
    args = parser.parse_args()
    if args.samples_per_camera < 1:
        parser.error("--samples-per-camera must be positive")

    images: list[dict] = []
    image_id = 1
    for bag_dir in args.bag_dir:
        exported, image_id = export_bag(bag_dir, args.output_root, args.samples_per_camera, image_id)
        images.extend(exported)
    payload = {
        "info": {
            "description": "Natural-scene raw-fisheye annotation set; no truth/TF/odometry included",
            "annotation_state": "unannotated",
            "purpose": "offline_candidate_extractor_validation_only",
        },
        "licenses": [],
        "categories": list(CATEGORIES),
        "images": images,
        "annotations": [],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "annotations.coco.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    per_camera = {camera: sum(item["camera"] == camera for item in images) for camera in CAMERA_NAMES}
    (args.output_root / "README.md").write_text(
        "# 自然场景人工标注集\n\n"
        "- 输入：仅六路原始鱼眼图像；不含真值、TF、里程计或固定靶标坐标。\n"
        "- 请标注 `buoy`、`pillar`、`block`、`boat` 的 bbox 或中心点。\n"
        "- `annotations.coco.json` 初始为空；标注工具导出后请保留 `camera`、`timestamp_ns` 和 `source_topic` 字段。\n"
        "- 本集仅用于离线验证候选提取器，不能把任何标注或真值送入运行时节点。\n\n"
        + "|相机|帧数|\n|---|---:|\n"
        + "\n".join(f"|{camera}|{count}|" for camera, count in per_camera.items())
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"images": len(images), "per_camera": per_camera}, ensure_ascii=False))


if __name__ == "__main__":
    main()
