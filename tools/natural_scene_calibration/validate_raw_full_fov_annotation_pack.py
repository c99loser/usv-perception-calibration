#!/usr/bin/env python3
"""Validate the blind raw-valid LiDAR annotation export."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def validate_pcd(path: Path) -> tuple[int, bytes]:
    content = path.read_bytes()
    marker = b"DATA binary\n"
    if marker not in content:
        raise ValueError(f"not binary PCD: {path}")
    header, payload = content.split(marker, 1)
    if b"FIELDS x y z\n" not in header or b"FIELDS x y z r" in header:
        raise ValueError(f"not blind XYZ-only PCD: {path}")
    match = re.search(rb"(?:^|\n)POINTS (\d+)(?:\n|$)", header)
    if not match:
        raise ValueError(f"missing POINTS: {path}")
    points = int(match.group(1))
    if len(payload) != points * 12:
        raise ValueError(f"payload mismatch: {path}")
    return points, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--verify-checksums", action="store_true")
    args = parser.parse_args()
    root = args.package
    manifest = json.loads((root / "package_manifest.json").read_text(encoding="utf-8"))
    supported_schemas = {
        "real-lidar-raw-full-fov-annotation-pack/v1",
        "real-lidar-raw-full-fov-annotation-pack/v2-official-direct",
        "real-lidar-raw-full-fov-annotation-pack/v3-stability-stratified",
    }
    if manifest.get("schema") not in supported_schemas:
        raise SystemExit("wrong schema")
    if manifest.get("cluster_overlay") is not False or manifest.get("projection_valid") is not False:
        raise SystemExit("blind-data contract violated")
    rows = list(csv.DictReader((root / "frame_lineage.csv").open(encoding="utf-8-sig")))
    seen = set()
    total_points = 0
    for row in rows:
        key = (row["split"], row["episode_id"], int(row["frame_index"]))
        if key in seen:
            raise SystemExit(f"duplicate frame lineage: {key}")
        seen.add(key)
        pcd = root / row["pcd_path"]
        image = root / row["image_path"]
        points, payload = validate_pcd(pcd)
        if points != int(row["raw_valid_points"]):
            raise SystemExit(f"point count mismatch: {pcd}")
        if hashlib.sha256(payload).hexdigest() != row["raw_xyz_sha256"]:
            raise SystemExit(f"raw XYZ hash mismatch: {pcd}")
        if not image.is_file() or not image.with_name(image.name + ".json").is_file():
            raise SystemExit(f"missing related image metadata: {image}")
        if row["cluster_overlay"].lower() != "false" or row["projection_valid"].lower() != "false":
            raise SystemExit(f"lineage blind-data contract violated: {key}")
        total_points += points
    episode_count = 0
    split_episode_ids = {"train": set(), "holdout": set()}
    for split in ("train", "holdout"):
        project = root / f"supervisely_{split}_raw_project"
        meta = json.loads((project / "meta.json").read_text(encoding="utf-8"))
        if meta.get("projectType") != "point_cloud_episode":
            raise SystemExit(f"wrong project type: {project}")
        if any("value_type" not in tag or "values" not in tag or "valueType" in tag for tag in meta.get("tags", [])):
            raise SystemExit(f"invalid Supervisely tag meta: {project}")
        for episode in (path for path in project.iterdir() if path.is_dir()):
            episode_count += 1
            split_episode_ids[split].add(episode.name)
            mapping = json.loads((episode / "frame_pointcloud_map.json").read_text(encoding="utf-8"))
            annotation = json.loads((episode / "annotation.json").read_text(encoding="utf-8"))
            if annotation.get("objects") or annotation.get("frames"):
                raise SystemExit(f"annotation is not blank: {episode}")
            if annotation.get("framesCount") != len(mapping):
                raise SystemExit(f"frame map mismatch: {episode}")
    if split_episode_ids["train"] & split_episode_ids["holdout"]:
        raise SystemExit("train/holdout episode overlap")
    if len(rows) != manifest["frames"] or episode_count != manifest["episodes"] or total_points != manifest["raw_valid_points"]:
        raise SystemExit("manifest totals mismatch")
    checksums = 0
    if args.verify_checksums:
        for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                raise SystemExit(f"checksum mismatch: {relative}")
            checksums += 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "episodes": episode_count,
                "frames": len(rows),
                "raw_valid_points": total_points,
                "checksums_verified": checksums,
                "train_holdout_overlap": 0,
                "cluster_overlay": False,
                "projection_valid": False,
            }
        )
    )


if __name__ == "__main__":
    main()
