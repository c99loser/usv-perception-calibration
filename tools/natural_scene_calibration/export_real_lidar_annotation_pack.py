#!/usr/bin/env python3
"""Export direct real-MCAP LiDAR clusters for human static-object review.

This tool deliberately consumes only a sensor-only extraction made with
``--include-cluster-points``.  It creates one small ASCII PLY per selected
cluster plus a CSV template.  The reviewer assigns ``pillar``, ``buoy``,
``block`` or ``background`` and may mark an item ambiguous; no TF, pose,
world state, image projection, or nominal extrinsic is used to create labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABELS = ("pillar", "buoy", "block", "background", "ambiguous")


def write_ply(path: Path, points: list[list[float]]) -> None:
    header = "\n".join(("ply", "format ascii 1.0", f"element vertex {len(points)}",
                        "property float x", "property float y", "property float z", "end_header", ""))
    body = "\n".join(f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in points)
    path.write_text(header + body + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-observation", type=int, default=60)
    parser.add_argument("--min-points", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.max_per_observation < 1 or args.min_points < 12:
        raise SystemExit("invalid selection limit")

    args.output.mkdir(parents=True)
    clouds = args.output / "clouds"; clouds.mkdir()
    rows: list[dict[str, object]] = []
    for source in args.observations:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("purpose") != "natural_scene_sensor_only_observation_extraction":
            raise SystemExit(f"not a direct sensor-only extraction: {source}")
        has_exported_points = any(
            cluster.get("points_sensor_m")
            for frame in payload.get("frames", [])
            for cluster in frame.get("clusters", [])
        )
        if not payload.get("cluster_points_exported_for_manual_annotation", has_exported_points):
            raise SystemExit(f"missing --include-cluster-points export: {source}")
        ranked = []
        for frame_index, frame in enumerate(payload["frames"]):
            for cluster_index, cluster in enumerate(frame["clusters"]):
                points = cluster.get("points_sensor_m", [])
                if len(points) >= args.min_points:
                    # Prefer well-sampled, medium-range objects.  This is only
                    # an annotation ordering, never an automatic class label.
                    score = min(len(points), 512) - 2.0 * float(cluster["range_m"])
                    ranked.append((score, frame_index, cluster_index, points))
        for _, frame_index, cluster_index, points in sorted(ranked, reverse=True)[:args.max_per_observation]:
            frame, cluster = payload["frames"][frame_index], payload["frames"][frame_index]["clusters"][cluster_index]
            item_id = f"{payload['camera']}_{frame['image_stamp_ns']}_{cluster_index:03d}"
            ply = clouds / f"{item_id}.ply"; write_ply(ply, points)
            rows.append({"item_id": item_id, "source_observations": str(source), "bag": payload["bag"],
                         "camera": payload["camera"], "frame_index": frame_index, "cluster_index": cluster_index,
                         "image_stamp_ns": frame["image_stamp_ns"], "lidar_stamp_ns": frame["lidar_stamp_ns"],
                         "sync_delta_ms": frame["sync_delta_ms"], "source_image": frame["image"],
                         "cloud_ply": str(ply), "sampled_points": len(points), "cluster_points": cluster["points"],
                         "center_m": json.dumps(cluster["center_m"]), "extent_m": json.dumps(cluster["extent_m"]),
                         "range_m": cluster["range_m"], "pca_linearity": cluster["pca_linearity"],
                         "human_label": "", "review_status": "unreviewed", "reviewer_note": ""})
    fields = list(rows[0]) if rows else []
    for filename in ("manifest.csv", "labels_template.csv"):
        with (args.output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (args.output / "README.md").write_text(
        "# Real LiDAR cluster manual-review pack\n\n"
        "Label only from each direct sensor-frame cloud (`clouds/*.ply`) and, if useful, its same-timestamp raw image. "
        "Do not use TF, odometry, `/world/*`, or any projected truth. Fill `human_label` with one of: "
        + ", ".join(LABELS) + ". Mark unclear objects `ambiguous`; they must not enter training.\n", encoding="utf-8")
    print(json.dumps({"purpose": "real_mcap_manual_lidar_annotation_only", "items": len(rows),
                      "labels": LABELS, "prohibited_inputs": ["/tf", "/tf_static", "odometry", "/world/*"]}))


if __name__ == "__main__":
    main()
