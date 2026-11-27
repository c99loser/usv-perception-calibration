#!/usr/bin/env python3
"""Directly export all six raw LiDAR/camera pairs from official MCAP bags.

The extractor performs one filtered streaming pass per MCAP.  Each direct
``image_raw`` message is paired to the nearest same-name LiDAR PointCloud2
message, and the PCD contains every finite XYZ sample from that message.  It
does not consume cached windows, cluster JSON, tracks, TF, odometry, world
truth, camera calibration or nominal extrinsics.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    from .build_raw_full_fov_annotation_pack import (
        PROHIBITED_INPUTS,
        native_path,
        render_raw_review,
        write_meta,
        write_xyz_binary_pcd,
    )
except ImportError:  # direct script execution
    from build_raw_full_fov_annotation_pack import (
        PROHIBITED_INPUTS,
        native_path,
        render_raw_review,
        write_meta,
        write_xyz_binary_pcd,
    )


CAMERAS = ("front", "back", "left", "left2", "right", "right2")
SCHEMA = "real-lidar-raw-full-fov-annotation-pack/v2-official-direct"


def image_topic(camera: str) -> str:
    return f"/wamv/sensors/cameras/{camera}_camera_sensor/image_raw"


def lidar_topic(camera: str) -> str:
    return f"/wamv/sensors/lidars/{camera}_lidar_sensor/points"


def decode_xyz(message) -> tuple[np.ndarray, dict]:
    fields = {field.name: field for field in message.fields}
    if not all(name in fields and fields[name].datatype == 7 for name in ("x", "y", "z")):
        raise RuntimeError("PointCloud2 requires float32 x/y/z fields")
    rows = np.frombuffer(message.data, np.uint8).reshape(message.height, message.width, message.point_step)
    organized = np.stack(
        [
            rows[:, :, fields[name].offset : fields[name].offset + 4]
            .copy()
            .view("<f4")
            .reshape(message.height, message.width)
            for name in ("x", "y", "z")
        ],
        axis=-1,
    )
    flat = organized.reshape((-1, 3))
    points = flat[np.isfinite(flat).all(axis=1)]
    audit = {
        "organized_height": int(message.height),
        "organized_width": int(message.width),
        "raw_slots": int(message.height * message.width),
        "point_step": int(message.point_step),
        "row_step": int(message.row_step),
        "finite_xyz_points": int(len(points)),
        "removed_nonfinite_points": int(len(flat) - len(points)),
        "fields": [str(field.name) for field in message.fields],
    }
    return points, audit


def decode_image(message) -> np.ndarray:
    encoding = str(message.encoding).lower()
    channels = 4 if encoding in {"rgba8", "bgra8"} else 3
    rows = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
    image = rows[:, : message.width * channels].reshape(message.height, message.width, channels)
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding == "bgr8":
        return image.copy()
    raise RuntimeError(f"unsupported image encoding: {message.encoding}")


def nearest_cloud(
    image_stamp: int,
    previous: tuple[int, np.ndarray, dict] | None,
    current: tuple[int, np.ndarray, dict] | None,
) -> tuple[int, np.ndarray, dict] | None:
    candidates = [item for item in (previous, current) if item is not None]
    return min(candidates, key=lambda item: abs(item[0] - image_stamp)) if candidates else None


def read_xyz_pcd(path: Path) -> np.ndarray:
    content = path.read_bytes()
    marker = b"DATA binary\n"
    payload = content[content.index(marker) + len(marker) :]
    return np.frombuffer(payload, dtype="<f4").reshape((-1, 3))


@dataclass
class EpisodeState:
    camera: str
    split: str
    bag_id: str
    output_root: Path
    max_sync_ns: int
    previous_lidar: tuple[int, np.ndarray, dict] | None = None
    pending_images: list[tuple[int, object]] = field(default_factory=list)
    frames: list[dict] = field(default_factory=list)

    @property
    def episode_id(self) -> str:
        short = self.bag_id.removeprefix("rosbag2_2026_07_02-").replace("-", "")
        return f"{self.split}_{short}_{self.camera}_all"

    @property
    def project(self) -> Path:
        return self.output_root / f"supervisely_{self.split}_raw_project"

    @property
    def episode(self) -> Path:
        return self.project / self.episode_id

    def prepare(self) -> None:
        for folder in (
            self.episode / "pointcloud",
            self.episode / "related_images",
            self.output_root / "frame_metadata" / self.episode_id,
            self.output_root / "review_samples" / self.episode_id,
        ):
            folder.mkdir(parents=True, exist_ok=True)

    def accept_image(self, stamp: int, message: object) -> None:
        self.pending_images.append((stamp, message))

    def accept_lidar(self, stamp: int, message: object) -> None:
        current_points, current_audit = decode_xyz(message)
        current = (stamp, current_points, current_audit)
        ready = [item for item in self.pending_images if item[0] <= stamp]
        self.pending_images = [item for item in self.pending_images if item[0] > stamp]
        for image_stamp, image_message in ready:
            selected = nearest_cloud(image_stamp, self.previous_lidar, current)
            if selected is not None:
                self.emit(image_stamp, image_message, selected)
        self.previous_lidar = current

    def flush(self) -> None:
        for image_stamp, image_message in self.pending_images:
            selected = nearest_cloud(image_stamp, self.previous_lidar, None)
            if selected is not None:
                self.emit(image_stamp, image_message, selected)
        self.pending_images.clear()

    def emit(self, image_stamp: int, image_message: object, lidar: tuple[int, np.ndarray, dict]) -> None:
        lidar_stamp, points, pointcloud_audit = lidar
        sync_delta_ns = abs(image_stamp - lidar_stamp)
        if sync_delta_ns > self.max_sync_ns:
            return
        frame_index = len(self.frames)
        stem = f"{frame_index:06d}_{lidar_stamp}"
        pcd_name = f"{stem}.pcd"
        pcd_path = self.episode / "pointcloud" / pcd_name
        write_xyz_binary_pcd(pcd_path, points)
        image_dir = self.episode / "related_images" / stem
        image_dir.mkdir()
        image_name = f"{image_stamp}.jpg"
        image_path = image_dir / image_name
        if not cv2.imwrite(str(image_path), decode_image(image_message), [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"cannot write {image_path}")
        timestamp = dt.datetime.fromtimestamp(image_stamp / 1_000_000_000, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")
        (image_dir / f"{image_name}.json").write_text(
            json.dumps({"name": image_name, "deviceId": self.camera, "timestamp": timestamp, "sensorsData": {}}, indent=2),
            encoding="utf-8",
        )
        ranges = np.linalg.norm(points, axis=1)
        record = {
            "split": self.split,
            "episode_id": self.episode_id,
            "bag_id": self.bag_id,
            "camera": self.camera,
            "frame_index": frame_index,
            "lidar_stamp_ns": lidar_stamp,
            "image_stamp_ns": image_stamp,
            "sync_delta_ms": sync_delta_ns / 1_000_000,
            "raw_slots": pointcloud_audit["raw_slots"],
            "raw_valid_points": len(points),
            "removed_nonfinite_points": pointcloud_audit["removed_nonfinite_points"],
            "range_min_m": float(ranges.min()),
            "range_max_m": float(ranges.max()),
            "organized_height": pointcloud_audit["organized_height"],
            "organized_width": pointcloud_audit["organized_width"],
            "point_step": pointcloud_audit["point_step"],
            "source_fields": "|".join(pointcloud_audit["fields"]),
            "pcd_path": f"supervisely_{self.split}_raw_project/{self.episode_id}/pointcloud/{pcd_name}",
            "image_path": f"supervisely_{self.split}_raw_project/{self.episode_id}/related_images/{stem}/{image_name}",
            "raw_xyz_sha256": hashlib.sha256(np.ascontiguousarray(points.astype("<f4")).tobytes()).hexdigest(),
            "source": "official_mcap_direct",
            "cluster_overlay": False,
            "projection_valid": False,
        }
        metadata = self.output_root / "frame_metadata" / self.episode_id / f"{stem}.json"
        metadata.write_text(json.dumps({**record, "pointcloud_audit": pointcloud_audit}, ensure_ascii=False, indent=2), encoding="utf-8")
        self.frames.append(record)

    def finalize(self) -> dict:
        if not self.frames:
            raise RuntimeError(f"no synchronized frames for {self.episode_id}")
        frame_map = {str(item["frame_index"]): Path(item["pcd_path"]).name for item in self.frames}
        (self.episode / "frame_pointcloud_map.json").write_text(json.dumps(frame_map, indent=2), encoding="utf-8")
        (self.episode / "annotation.json").write_text(
            json.dumps(
                {
                    "description": "All synchronized raw-valid XYZ frames directly extracted from official MCAP; no candidates or projection.",
                    "key": uuid.uuid5(uuid.NAMESPACE_URL, f"challenger-official-raw:{self.episode_id}").hex,
                    "tags": [],
                    "objects": [],
                    "frames": [],
                    "framesCount": len(self.frames),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        for index in sorted({0, len(self.frames) // 2, len(self.frames) - 1}):
            record = self.frames[index]
            pcd = self.output_root / record["pcd_path"]
            image = self.output_root / record["image_path"]
            render_raw_review(
                self.output_root / "review_samples" / self.episode_id / f"frame_{index:03d}.png",
                image,
                read_xyz_pcd(pcd),
                self.episode_id,
                record,
            )
        return {
            "id": self.episode_id,
            "split": self.split,
            "bag_id": self.bag_id,
            "camera": self.camera,
            "frames": len(self.frames),
            "raw_slots": sum(int(item["raw_slots"]) for item in self.frames),
            "raw_valid_points": sum(int(item["raw_valid_points"]) for item in self.frames),
            "mean_raw_valid_points": float(np.mean([item["raw_valid_points"] for item in self.frames])),
            "max_sync_delta_ms": max(float(item["sync_delta_ms"]) for item in self.frames),
            "source": "official_mcap_direct",
            "cluster_overlay": False,
            "projection_valid": False,
        }


def process_bag(source: dict, output: Path, max_sync_ms: float) -> tuple[list[dict], list[dict]]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    bag_path = native_path(source["bag"])
    states = {
        camera: EpisodeState(camera, source["split"], source["bag_id"], output, int(max_sync_ms * 1_000_000))
        for camera in CAMERAS
    }
    for state in states.values():
        state.prepare()
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    expected = {image_topic(camera) for camera in CAMERAS} | {lidar_topic(camera) for camera in CAMERAS}
    missing = sorted(expected - set(topic_types))
    if missing:
        raise RuntimeError(f"missing official topics in {source['bag_id']}: {missing}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(expected)))
    message_types = {topic: get_message(topic_types[topic]) for topic in expected}
    image_topics = {image_topic(camera): camera for camera in CAMERAS}
    lidar_topics = {lidar_topic(camera): camera for camera in CAMERAS}
    messages = 0
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        message = deserialize_message(raw, message_types[topic])
        if topic in image_topics:
            states[image_topics[topic]].accept_image(stamp, message)
        else:
            states[lidar_topics[topic]].accept_lidar(stamp, message)
        messages += 1
        if messages % 2000 == 0:
            print(json.dumps({"bag": source["bag_id"], "stream_messages": messages, "exported_frames": sum(len(s.frames) for s in states.values())}), flush=True)
    for state in states.values():
        state.flush()
    summaries = [states[camera].finalize() for camera in CAMERAS]
    frames = [record for camera in CAMERAS for record in states[camera].frames]
    return summaries, frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sync-ms", type=float, default=60.0)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True)
    for split in ("train", "holdout"):
        project = args.output / f"supervisely_{split}_raw_project"
        project.mkdir()
        write_meta(project)
    summaries, frames = [], []
    for source in spec["bags"]:
        bag_summaries, bag_frames = process_bag(source, args.output, args.max_sync_ms)
        summaries.extend(bag_summaries)
        frames.extend(bag_frames)
    fields = list(frames[0])
    with (args.output / "frame_lineage.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frames)
    with (args.output / "track_labels_template.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(
            handle,
            fieldnames=["split", "episode_id", "object_key", "human_label", "first_frame", "last_frame", "static_for_calibration", "reviewer_confidence", "review_status", "reviewer_id", "reviewer_note"],
        ).writeheader()
    manifest = {
        "schema": SCHEMA,
        "purpose": "blind_annotation_of_all_six_direct_official_mcap_lidar_camera_pairs",
        "source_policy": "direct single-pass extraction from the three official MCAP bags only",
        "point_policy": "all finite XYZ; no range, cluster, semantic or geometric filtering",
        "camera_lidar_pairs": list(CAMERAS),
        "excluded_unpaired_sensors": {"camera": ["overhead"], "lidar": ["forward"]},
        "prohibited_inputs": list(PROHIBITED_INPUTS),
        "max_sync_ms": args.max_sync_ms,
        "episodes": len(summaries),
        "frames": len(frames),
        "raw_slots": sum(int(item["raw_slots"]) for item in summaries),
        "raw_valid_points": sum(int(item["raw_valid_points"]) for item in summaries),
        "cluster_overlay": False,
        "projection_valid": False,
        "sources": summaries,
    }
    (args.output / "package_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "README_标注说明.md").write_text(
        "# 官方三份 MCAP 六路原始点云盲标包 V2\n\n"
        "本包直接一次遍历官方 `数据/` 中三份 MCAP，抽取 front/back/left/left2/right/right2 六组同名 LiDAR 与鱼眼相机的全部可同步帧。未使用旧 64 帧缓存、候选 JSON、轨迹、TF、里程计、世界真值或名义外参。\n\n"
        "- 每帧 PCD 只含原 PointCloud2 中全部有限 XYZ；只移除 NaN/Inf。\n"
        "- 每帧配 direct `image_raw` 同时间 JPEG；图片没有投影，只供场景/类别判断。\n"
        "- overhead camera 与 forward LiDAR 不构成冻结的六路同名外参对，未混入。\n"
        "- 先导入 `supervisely_train_raw_project/`；`supervisely_holdout_raw_project/` 必须封存至模型和阈值冻结。\n"
        "- 标注方法不变：跨帧 3D cuboid、同物体保持同一 object/track；静态约束仅限 pillar/buoy/block + static_for_calibration=yes。\n"
        "- 自船、水面、多径、粘连或严重稀疏对象标 background/ambiguous；图像中存在但点云无可辨回波的对象不得画 3D 框。\n",
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(item for item in args.output.rglob("*") if item.is_file() and item.name != "checksums.sha256"):
        checksum_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(args.output).as_posix()}")
    (args.output / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
