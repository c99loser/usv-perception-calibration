#!/usr/bin/env python3
"""Extract continuous, conservatively deduplicated camera clips from ROS2 MCAPs.

The output is intended as an initial annotation/training candidate set. Frames
remain in the raw (unrectified) camera domain. Every saved image is traceable to
its original bag timestamp and per-topic frame index through manifest.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from camera_fisheye_pipeline import (
    CAMERA_NAMES,
    IMAGE_TOPIC_TEMPLATE,
    convert_image_message,
    discover_bag_dirs,
    iter_ros_messages,
    parse_metadata,
    select_camera_topics,
)


@dataclass(frozen=True)
class ClipPlan:
    bag_dir: Path
    bag_id: str
    split: str
    clip_id: str
    start_index: int
    scan_count: int
    target_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frames-per-clip", type=int, default=40)
    parser.add_argument("--scan-per-clip", type=int, default=80)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--near-duplicate-mad",
        type=float,
        default=0.25,
        help="Reject adjacent candidates whose mean absolute pixel difference is <= this value.",
    )
    return parser.parse_args()


def evenly_spaced_starts(frame_count: int, scan_count: int, clip_count: int) -> list[int]:
    max_start = max(0, frame_count - scan_count)
    if clip_count == 1:
        return [max_start // 2]
    return [round(max_start * (i + 1) / (clip_count + 1)) for i in range(clip_count)]


def build_plans(bag_dirs: list[Path], frames_per_clip: int, scan_per_clip: int) -> list[ClipPlan]:
    if len(bag_dirs) < 3:
        raise RuntimeError(f"Expected at least 3 MCAP bag directories, found {len(bag_dirs)}")

    # Entire clips, rather than adjacent frames, are assigned to splits. This
    # prevents temporal leakage between train, validation, and test sets.
    assignments = (("train", 4), ("val", 1), ("test", 1))
    plans: list[ClipPlan] = []
    for bag_dir, (split, clip_count) in zip(bag_dirs[:3], assignments):
        summary = parse_metadata(bag_dir / "metadata.yaml")
        selections = select_camera_topics(summary)
        counts = [item.image_count for item in selections if item.image_topic]
        if len(counts) != len(CAMERA_NAMES) or min(counts) < frames_per_clip:
            raise RuntimeError(f"Six usable camera topics were not found in {bag_dir}")
        shared_count = min(counts)
        actual_scan = min(scan_per_clip, shared_count)
        for number, start in enumerate(
            evenly_spaced_starts(shared_count, actual_scan, clip_count), start=1
        ):
            plans.append(
                ClipPlan(
                    bag_dir=bag_dir,
                    bag_id=bag_dir.name,
                    split=split,
                    clip_id=f"{split}_{bag_dir.name}_{number:02d}",
                    start_index=start,
                    scan_count=actual_scan,
                    target_count=frames_per_clip,
                )
            )
    return plans


def frame_digest(image: np.ndarray) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def mean_abs_difference(left: np.ndarray, right: np.ndarray | None) -> float | None:
    if right is None or left.shape != right.shape:
        return None
    return float(cv2.absdiff(left, right).mean())


def write_jpeg(path: Path, image: np.ndarray, quality: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Failed to encode {path}")
    path.write_bytes(encoded.tobytes())
    return hashlib.sha256(encoded.tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"Output directory must be empty or absent: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    bag_dirs = discover_bag_dirs(args.data_root)
    plans = build_plans(bag_dirs, args.frames_per_clip, args.scan_per_clip)
    plans_by_bag: dict[Path, list[ClipPlan]] = {}
    for plan in plans:
        plans_by_bag.setdefault(plan.bag_dir, []).append(plan)

    rows: list[dict[str, Any]] = []
    rejection_counts = {"exact_duplicate": 0, "near_duplicate": 0, "decode_error": 0}
    seen_hashes = {camera: set() for camera in CAMERA_NAMES}
    last_saved: dict[tuple[str, str], np.ndarray] = {}
    saved_per_clip_camera: dict[tuple[str, str], int] = {}

    for bag_dir, bag_plans in plans_by_bag.items():
        summary = parse_metadata(bag_dir / "metadata.yaml")
        selections = select_camera_topics(summary)
        topic_to_camera = {
            item.image_topic: item.camera for item in selections if item.image_topic is not None
        }
        topic_indices = {topic: -1 for topic in topic_to_camera}

        for topic, message, timestamp_ns in iter_ros_messages(bag_dir, list(topic_to_camera)):
            camera = topic_to_camera[topic]
            topic_indices[topic] += 1
            source_index = topic_indices[topic]

            for plan in bag_plans:
                key = (plan.clip_id, camera)
                if saved_per_clip_camera.get(key, 0) >= plan.target_count:
                    continue
                if not (plan.start_index <= source_index < plan.start_index + plan.scan_count):
                    continue
                try:
                    image = convert_image_message(message)
                except Exception:
                    rejection_counts["decode_error"] += 1
                    continue

                digest = frame_digest(image)
                if digest in seen_hashes[camera]:
                    rejection_counts["exact_duplicate"] += 1
                    continue
                mad = mean_abs_difference(image, last_saved.get(key))
                if mad is not None and mad <= args.near_duplicate_mad:
                    rejection_counts["near_duplicate"] += 1
                    continue

                clip_position = saved_per_clip_camera.get(key, 0)
                relative_path = Path("images") / camera / plan.split / plan.clip_id / (
                    f"frame_{source_index:06d}_{timestamp_ns}.jpg"
                )
                output_sha256 = write_jpeg(
                    args.output_root / relative_path, image, args.jpeg_quality
                )
                seen_hashes[camera].add(digest)
                last_saved[key] = image
                saved_per_clip_camera[key] = clip_position + 1
                rows.append(
                    {
                        "camera": camera,
                        "split": plan.split,
                        "clip_id": plan.clip_id,
                        "clip_position_0based": clip_position,
                        "source_bag": plan.bag_id,
                        "source_topic": topic,
                        "source_frame_index_0based": source_index,
                        "source_frame_number_1based": source_index + 1,
                        "timestamp_ns": timestamp_ns,
                        "timestamp_sec": timestamp_ns // 1_000_000_000,
                        "timestamp_nanosec": timestamp_ns % 1_000_000_000,
                        "width": int(image.shape[1]),
                        "height": int(image.shape[0]),
                        "adjacent_pixel_mad": "" if mad is None else f"{mad:.6f}",
                        "raw_pixel_sha256": digest,
                        "output_jpeg_sha256": output_sha256,
                        "relative_path": relative_path.as_posix(),
                    }
                )

            if all(
                saved_per_clip_camera.get((plan.clip_id, name), 0) >= plan.target_count
                for plan in bag_plans
                for name in CAMERA_NAMES
            ):
                break

    missing = []
    for plan in plans:
        for camera in CAMERA_NAMES:
            actual = saved_per_clip_camera.get((plan.clip_id, camera), 0)
            if actual != plan.target_count:
                missing.append(f"{plan.clip_id}/{camera}: {actual}/{plan.target_count}")
    if missing:
        raise RuntimeError("Incomplete clips: " + ", ".join(missing))

    rows.sort(key=lambda row: (row["camera"], row["split"], row["clip_id"], row["clip_position_0based"]))
    manifest_path = args.output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        counts.setdefault(row["camera"], {}).setdefault(row["split"], 0)
        counts[row["camera"]][row["split"]] += 1
    summary = {
        "raw_unrectified": True,
        "image_encoding": f"JPEG quality {args.jpeg_quality}",
        "frames_per_camera": sum(next(iter(counts.values())).values()),
        "total_images": len(rows),
        "counts_by_camera_and_split": counts,
        "deduplication": {
            "exact": "SHA-256 of decoded raw pixels, scoped per camera",
            "near_duplicate": f"adjacent accepted frame MAD <= {args.near_duplicate_mad}",
            "rejections": rejection_counts,
        },
        "clip_plans": [
            {
                "bag": item.bag_id,
                "split": item.split,
                "clip_id": item.clip_id,
                "source_start_index_0based": item.start_index,
                "scan_count": item.scan_count,
                "saved_per_camera": item.target_count,
            }
            for item in plans
        ],
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
