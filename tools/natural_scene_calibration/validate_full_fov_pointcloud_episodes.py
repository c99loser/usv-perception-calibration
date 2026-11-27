#!/usr/bin/env python3
"""Validate a generated full-FOV point-cloud episode package."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def _pcd_points_and_payload(path: Path) -> tuple[int, int]:
    content = path.read_bytes()
    marker = b"DATA binary\n"
    if marker not in content:
        raise ValueError(f"not a binary PCD: {path}")
    header, payload = content.split(marker, 1)
    if b"FIELDS x y z r g b" not in header:
        raise ValueError(f"unexpected PCD fields: {path}")
    match = re.search(rb"(?:^|\n)POINTS (\d+)(?:\n|$)", header)
    if not match:
        raise ValueError(f"missing POINTS: {path}")
    points = int(match.group(1))
    if len(payload) != points * 15:
        raise ValueError(f"payload size mismatch: {path}: {len(payload)} != {points}*15")
    return points, len(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--verify-checksums", action="store_true")
    args = parser.parse_args()
    root = args.package
    manifest = json.loads((root / "full_fov_manifest.json").read_text(encoding="utf-8"))
    schema = manifest.get("schema")
    if schema not in {
        "real-lidar-full-fov-episodes/v1",
        "real-lidar-full-fov-episodes/v2",
    }:
        raise SystemExit("unsupported manifest schema")
    if manifest.get("projection_valid") is not False:
        raise SystemExit("projection must remain disabled")
    episode_count = frame_count = point_count = payload_bytes = 0
    split_ids: dict[str, set[str]] = {"train": set(), "holdout": set()}
    for split in ("train", "holdout"):
        project = root / f"supervisely_{split}_project"
        if not (project / "meta.json").is_file():
            raise SystemExit(f"missing meta.json: {project}")
        for episode in sorted(path for path in project.iterdir() if path.is_dir()):
            episode_count += 1
            split_ids[split].add(episode.name)
            mapping = json.loads((episode / "frame_pointcloud_map.json").read_text(encoding="utf-8"))
            annotation = json.loads((episode / "annotation.json").read_text(encoding="utf-8"))
            if annotation.get("framesCount") != len(mapping):
                raise SystemExit(f"framesCount mismatch: {episode}")
            if list(mapping) != [str(index) for index in range(len(mapping))]:
                raise SystemExit(f"non-contiguous frame map: {episode}")
            object_keys = {item.get("key") for item in annotation.get("objects", [])}
            if None in object_keys or len(object_keys) != len(annotation.get("objects", [])):
                raise SystemExit(f"invalid or duplicate annotation object keys: {episode}")
            if any(item.get("classTitle") != "ambiguous" for item in annotation.get("objects", [])):
                raise SystemExit(f"draft proposal is not explicitly ambiguous: {episode}")
            figure_keys: set[str] = set()
            for annotated_frame in annotation.get("frames", []):
                index = int(annotated_frame.get("index", -1))
                if not 0 <= index < len(mapping):
                    raise SystemExit(f"annotation frame out of bounds: {episode}:{index}")
                for figure in annotated_frame.get("figures", []):
                    if figure.get("objectKey") not in object_keys:
                        raise SystemExit(f"orphan proposal figure: {episode}:{index}")
                    key = figure.get("key")
                    if not key or key in figure_keys:
                        raise SystemExit(f"invalid or duplicate figure key: {episode}:{index}")
                    figure_keys.add(key)
                    geometry = figure.get("geometry", {})
                    values = [
                        float(geometry[group][axis])
                        for group in ("position", "rotation", "dimensions")
                        for axis in ("x", "y", "z")
                    ]
                    if not all(math.isfinite(value) for value in values):
                        raise SystemExit(f"non-finite proposal geometry: {episode}:{index}")
                    if any(float(geometry["dimensions"][axis]) <= 0 for axis in ("x", "y", "z")):
                        raise SystemExit(f"non-positive proposal dimensions: {episode}:{index}")
            if schema == "real-lidar-full-fov-episodes/v2":
                audit_path = episode / "annotation_assist_audit.json"
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if audit.get("sensor_only") is not True or audit.get("ground_truth") is not False:
                    raise SystemExit(f"annotation assist safety contract mismatch: {episode}")
            for frame_index, pcd_name in mapping.items():
                pcd = episode / "pointcloud" / pcd_name
                points, size = _pcd_points_and_payload(pcd)
                point_count += points
                payload_bytes += size
                stem = Path(pcd_name).stem
                images = [path for path in (episode / "related_images" / stem).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
                if len(images) != 1 or not images[0].with_name(images[0].name + ".json").is_file():
                    raise SystemExit(f"related image mismatch: {episode}:{frame_index}")
                metadata = root / "frame_metadata" / episode.name / f"{stem}.json"
                record = json.loads(metadata.read_text(encoding="utf-8"))
                if record.get("projection_valid") is not False or record.get("frame_index") != int(frame_index):
                    raise SystemExit(f"metadata contract mismatch: {metadata}")
                if schema == "real-lidar-full-fov-episodes/v2" and record.get("annotation_assist_ground_truth") is not False:
                    raise SystemExit(f"annotation assist truth contract mismatch: {metadata}")
                frame_count += 1
    if split_ids["train"] & split_ids["holdout"]:
        raise SystemExit("train/holdout episode leakage")
    if episode_count != manifest["episodes"] or frame_count != manifest["frames"]:
        raise SystemExit("manifest count mismatch")
    verified = 0
    if args.verify_checksums:
        for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise SystemExit(f"checksum mismatch: {relative}")
            verified += 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "episodes": episode_count,
                "frames": frame_count,
                "points": point_count,
                "pcd_binary_payload_bytes": payload_bytes,
                "checksums_verified": verified,
                "train_holdout_overlap": 0,
                "projection_valid": False,
                "annotation_assist_ground_truth": False if schema.endswith("/v2") else None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
