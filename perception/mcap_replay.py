"""Raw ROS2 MCAP discovery and streaming, with no ROS daemon required."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator
import numpy as np

LOCAL_PYDEPS = Path(__file__).resolve().parents[1] / ".codex_pydeps"
# A shared cache can be selected without baking a machine path into the source.
shared = Path(os.environ["PERCEPTION_PYDEPS"]) if os.environ.get("PERCEPTION_PYDEPS") else None
for dependency_dir in (shared, LOCAL_PYDEPS):
    if dependency_dir and dependency_dir.exists() and str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from .config import PerceptionConfig
from .contracts import PointCloudFrame


def discover_bag_dirs(data_root: Path) -> list[Path]:
    return sorted({item.parent for item in data_root.rglob("metadata.yaml")})


def metadata_topic_names(metadata_path: Path) -> list[str]:
    return [line.split(":", 1)[1].strip().strip("\"'") for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip().startswith("name:")]


def inspect_bag(bag_dir: Path, config: PerceptionConfig) -> dict[str, object]:
    names = set(metadata_topic_names(bag_dir / "metadata.yaml"))
    selected = {sensor: topic for sensor, topic in config.lidar_topics.items() if topic in names}
    return {"bag_dir": str(bag_dir), "available_lidar_topics": selected, "missing_lidars": sorted(set(config.lidar_topics) - set(selected))}


def _pointcloud_xyz(message: object) -> np.ndarray:
    """Decode ROS PointCloud2 binary data for arbitrary field order/point stride."""
    fields = {str(field.name): field for field in getattr(message, "fields")}
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError("PointCloud2 must expose x, y, z fields")
    endian = ">" if bool(getattr(message, "is_bigendian", False)) else "<"
    datatype = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    try:
        dtype = np.dtype({"names": ["x", "y", "z"], "formats": [endian + datatype[fields[n].datatype] for n in ("x", "y", "z")], "offsets": [fields[n].offset for n in ("x", "y", "z")], "itemsize": int(getattr(message, "point_step"))})
    except KeyError as exc:
        raise ValueError(f"unsupported PointCloud2 datatype: {exc}") from exc
    raw = np.asarray(getattr(message, "data"), dtype=np.uint8).tobytes()
    count = int(getattr(message, "width")) * int(getattr(message, "height"))
    structured = np.frombuffer(raw, dtype=dtype, count=count)
    return np.column_stack([structured[axis] for axis in ("x", "y", "z")]).astype(np.float64, copy=False)


def replay_pointclouds(bag_dir: Path, config: PerceptionConfig, max_frames: int | None = None) -> Iterator[PointCloudFrame]:
    try:
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise RuntimeError("rosbags is required for MCAP replay; install project dependencies first.") from exc
    topic_to_sensor = {topic: sensor for sensor, topic in config.lidar_topics.items()}
    typestore = get_typestore(Stores.ROS2_JAZZY)
    emitted = 0
    with Reader(bag_dir) as reader:
        connections = [conn for conn in reader.connections if conn.topic in topic_to_sensor and conn.msgtype == "sensor_msgs/msg/PointCloud2"]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            message = typestore.deserialize_cdr(rawdata, connection.msgtype)
            points = _pointcloud_xyz(message)
            yield PointCloudFrame(int(timestamp), topic_to_sensor[connection.topic], points)
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                return
