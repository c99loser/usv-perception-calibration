#!/usr/bin/env python3
"""Offline fisheye camera baseline for the WAM-V docking challenge.

The module keeps ROS/OpenCV imports lazy so metadata inspection and unit tests
can run on a plain Python environment, while full frame extraction runs inside
ROS2 Jazzy with rosbag2_py, cv_bridge, OpenCV, and NumPy available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

LOCAL_PYDEPS = Path(__file__).resolve().parents[2] / ".codex_pydeps"
if os.name == "nt" and LOCAL_PYDEPS.exists() and str(LOCAL_PYDEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_PYDEPS))

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - numpy is expected in tests/runtime
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None


CAMERA_NAMES = (
    "front",
    "back",
    "left",
    "left2",
    "right",
    "right2",
)

IMAGE_TOPIC_TEMPLATE = "/wamv/sensors/cameras/{name}_camera_sensor/image_raw"
OPTICAL_IMAGE_TOPIC_TEMPLATE = "/wamv/sensors/cameras/{name}_camera_sensor/optical/image_raw"
CAMERA_INFO_TOPIC_TEMPLATE = "/wamv/sensors/cameras/{name}_camera_sensor/camera_info"
OPTICAL_CAMERA_INFO_TOPIC_TEMPLATE = "/wamv/sensors/cameras/{name}_camera_sensor/optical/camera_info"

IMAGE_TYPE = "sensor_msgs/msg/Image"
CAMERA_INFO_TYPE = "sensor_msgs/msg/CameraInfo"


@dataclass(frozen=True)
class TopicInfo:
    name: str
    msg_type: str
    message_count: int


@dataclass(frozen=True)
class BagSummary:
    bag_dir: Path
    mcap_file: Optional[Path]
    duration_seconds: Optional[float]
    starting_time_ns: Optional[int]
    total_messages: Optional[int]
    topics: Tuple[TopicInfo, ...]


@dataclass(frozen=True)
class CameraTopicSelection:
    camera: str
    image_topic: Optional[str]
    camera_info_topic: Optional[str]
    image_count: int
    camera_info_count: int
    image_source: str
    camera_info_source: str


@dataclass
class CameraIntrinsics:
    camera: str
    image_topic: str
    camera_info_topic: Optional[str]
    width: int
    height: int
    distortion_model: str
    k: List[float]
    d: List[float]
    r: List[float]
    p: List[float]
    source: str
    notes: List[str] = field(default_factory=list)

    @property
    def camera_matrix(self) -> "np.ndarray":
        require_numpy()
        return np.asarray(self.k, dtype=np.float64).reshape(3, 3)

    @property
    def distortion_coefficients(self) -> "np.ndarray":
        require_numpy()
        return np.asarray(self.d, dtype=np.float64).reshape(-1, 1)


@dataclass
class Detection:
    x: int
    y: int
    width: int
    height: int
    area: float
    score: float


@dataclass
class RectificationPlan:
    intrinsics: CameraIntrinsics
    map1: Any
    map2: Any
    new_k: Any
    model: str


def require_numpy() -> None:
    if np is None:
        raise RuntimeError(f"NumPy is required: {_NUMPY_IMPORT_ERROR}")


def discover_bag_dirs(data_root: Path) -> List[Path]:
    """Return directories that look like ROS2 MCAP bag directories."""
    return sorted({path.parent for path in data_root.rglob("metadata.yaml")})


def parse_metadata(metadata_path: Path) -> BagSummary:
    """Parse the small subset of rosbag2 metadata needed by this stage.

    This intentionally avoids a PyYAML dependency because the Windows planning
    environment may not have it. It is tolerant of rosbag2's stable YAML shape.
    """

    text = metadata_path.read_text(encoding="utf-8")
    topics: List[TopicInfo] = []
    current_name: Optional[str] = None
    current_type: Optional[str] = None
    duration_ns: Optional[int] = None
    start_ns: Optional[int] = None
    total_messages: Optional[int] = None
    mcap_file: Optional[Path] = None
    in_topic = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("duration:"):
            continue
        if line.startswith("nanoseconds:") and duration_ns is None:
            duration_ns = _parse_int_value(line)
        elif line.startswith("nanoseconds_since_epoch:") and start_ns is None:
            start_ns = _parse_int_value(line)
        elif line.startswith("message_count:") and total_messages is None:
            total_messages = _parse_int_value(line)
        elif line == "- topic_metadata:":
            in_topic = True
            current_name = None
            current_type = None
        elif in_topic and line.startswith("name:"):
            current_name = _parse_string_value(line)
        elif in_topic and line.startswith("type:"):
            current_type = _parse_string_value(line)
        elif in_topic and line.startswith("message_count:") and current_name and current_type:
            topics.append(
                TopicInfo(
                    name=current_name,
                    msg_type=current_type,
                    message_count=_parse_int_value(line),
                )
            )
            in_topic = False
        elif (line.startswith("path:") or line.startswith("- path:")) and mcap_file is None and line.endswith(".mcap"):
            mcap_file = metadata_path.parent / _parse_string_value(line)

    return BagSummary(
        bag_dir=metadata_path.parent,
        mcap_file=mcap_file,
        duration_seconds=None if duration_ns is None else duration_ns / 1_000_000_000.0,
        starting_time_ns=start_ns,
        total_messages=total_messages,
        topics=tuple(topics),
    )


def _parse_int_value(line: str) -> int:
    return int(line.split(":", 1)[1].strip())


def _parse_string_value(line: str) -> str:
    value = line.split(":", 1)[1].strip()
    return value.strip("\"'")


def select_camera_topics(summary: BagSummary) -> List[CameraTopicSelection]:
    by_name = {topic.name: topic for topic in summary.topics}
    selections: List[CameraTopicSelection] = []

    for camera in CAMERA_NAMES:
        image_candidates = (
            (IMAGE_TOPIC_TEMPLATE.format(name=camera), "direct"),
            (OPTICAL_IMAGE_TOPIC_TEMPLATE.format(name=camera), "optical"),
        )
        info_candidates = (
            (CAMERA_INFO_TOPIC_TEMPLATE.format(name=camera), "direct"),
            (OPTICAL_CAMERA_INFO_TOPIC_TEMPLATE.format(name=camera), "optical"),
        )
        image_topic, image_source, image_count = _choose_topic(by_name, image_candidates, IMAGE_TYPE)
        info_topic, info_source, info_count = _choose_topic(by_name, info_candidates, CAMERA_INFO_TYPE)
        selections.append(
            CameraTopicSelection(
                camera=camera,
                image_topic=image_topic,
                camera_info_topic=info_topic,
                image_count=image_count,
                camera_info_count=info_count,
                image_source=image_source,
                camera_info_source=info_source,
            )
        )
    return selections


def _choose_topic(
    by_name: Mapping[str, TopicInfo],
    candidates: Sequence[Tuple[str, str]],
    expected_type: str,
) -> Tuple[Optional[str], str, int]:
    for topic_name, source in candidates:
        topic = by_name.get(topic_name)
        if topic and topic.msg_type == expected_type and topic.message_count > 0:
            return topic.name, source, topic.message_count
    for topic_name, source in candidates:
        topic = by_name.get(topic_name)
        if topic and topic.msg_type == expected_type:
            return topic.name, f"{source}-empty", topic.message_count
    return None, "missing", 0


def estimate_intrinsics_from_fov(
    camera: str,
    image_topic: str,
    camera_info_topic: Optional[str] = None,
    width: int = 1280,
    height: int = 720,
    hfov_degrees: float = 175.0,
) -> CameraIntrinsics:
    fx = width / (2.0 * math.tan(math.radians(hfov_degrees) / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return CameraIntrinsics(
        camera=camera,
        image_topic=image_topic,
        camera_info_topic=camera_info_topic,
        width=width,
        height=height,
        distortion_model="fisheye",
        k=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        d=[0.0, 0.0, 0.0, 0.0],
        r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        source="estimated_from_1280x720_hfov_175",
        notes=["Fallback intrinsics; replace with CameraInfo when available."],
    )


def intrinsics_from_camera_info(
    camera: str,
    image_topic: str,
    camera_info_topic: str,
    camera_info_msg: Any,
) -> CameraIntrinsics:
    width = int(getattr(camera_info_msg, "width", 0) or 1280)
    height = int(getattr(camera_info_msg, "height", 0) or 720)
    distortion_model = str(getattr(camera_info_msg, "distortion_model", "") or "unknown")
    k = [float(value) for value in getattr(camera_info_msg, "k", [])]
    d = [float(value) for value in getattr(camera_info_msg, "d", [])]
    r = [float(value) for value in getattr(camera_info_msg, "r", [])]
    p = [float(value) for value in getattr(camera_info_msg, "p", [])]

    fallback = estimate_intrinsics_from_fov(camera, image_topic, camera_info_topic, width, height)
    notes: List[str] = []
    if len(k) != 9:
        k = fallback.k
        notes.append("CameraInfo K missing or malformed; used HFOV estimate.")
    if not d:
        d = fallback.d
        notes.append("CameraInfo D missing; used zero distortion.")
    if len(r) != 9:
        r = fallback.r
    if len(p) != 12:
        p = fallback.p

    return CameraIntrinsics(
        camera=camera,
        image_topic=image_topic,
        camera_info_topic=camera_info_topic,
        width=width,
        height=height,
        distortion_model=distortion_model,
        k=k,
        d=d,
        r=r,
        p=p,
        source="mcap_camera_info",
        notes=notes,
    )


def write_intrinsics_yaml(path: Path, intrinsics: Sequence[CameraIntrinsics]) -> None:
    data = {
        "cameras": {
            item.camera: {
                "image_topic": item.image_topic,
                "camera_info_topic": item.camera_info_topic,
                "width": item.width,
                "height": item.height,
                "distortion_model": item.distortion_model,
                "K": item.k,
                "D": item.d,
                "R": item.r,
                "P": item.p,
                "source": item.source,
                "notes": item.notes,
            }
            for item in intrinsics
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_simple_yaml(data), encoding="utf-8")


def to_simple_yaml(value: Any, indent: int = 0) -> str:
    """Emit readable YAML for dictionaries/lists made of scalar values."""
    prefix = " " * indent
    if isinstance(value, dict):
        lines: List[str] = []
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(to_simple_yaml(child, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}{key}: {format_yaml_scalar(child)}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        if not value:
            return f"{prefix}[]\n"
        if all(not isinstance(item, (dict, list)) for item in value):
            return f"{prefix}[{', '.join(format_yaml_scalar(item) for item in value)}]\n"
        lines = []
        for item in value:
            lines.append(f"{prefix}-")
            lines.append(to_simple_yaml(item, indent + 2).rstrip())
        return "\n".join(lines) + "\n"
    return f"{prefix}{format_yaml_scalar(value)}\n"


def format_yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(float(value)) if isinstance(value, float) else str(value)
    if isinstance(value, str) and re.match(r"^[A-Za-z0-9_./:+-]+$", value):
        return value
    return json.dumps(value, ensure_ascii=False)


def write_stage_report(
    path: Path,
    summaries: Sequence[BagSummary],
    selections_by_bag: Mapping[Path, Sequence[CameraTopicSelection]],
    output_root: Optional[Path] = None,
) -> None:
    lines = ["# Camera Fisheye Stage Report", ""]
    if output_root:
        lines.extend([f"Output root: `{output_root}`", ""])
    for summary in summaries:
        lines.extend(
            [
                f"## {summary.bag_dir.name}",
                "",
                f"- MCAP: `{summary.mcap_file.name if summary.mcap_file else 'missing'}`",
                f"- Duration: `{summary.duration_seconds:.3f}s`" if summary.duration_seconds else "- Duration: `unknown`",
                f"- Total messages: `{summary.total_messages}`" if summary.total_messages is not None else "- Total messages: `unknown`",
                "",
                "| Camera | Image topic | Frames | CameraInfo topic | Frames | Status |",
                "|---|---:|---:|---|---:|---|",
            ]
        )
        for selection in selections_by_bag[summary.bag_dir]:
            status = "ok" if selection.image_topic and selection.camera_info_topic else "missing-input"
            lines.append(
                "| {camera} | `{image_topic}` | {image_count} | `{info_topic}` | {info_count} | {status} |".format(
                    camera=selection.camera,
                    image_topic=selection.image_topic or "-",
                    image_count=selection.image_count,
                    info_topic=selection.camera_info_topic or "-",
                    info_count=selection.camera_info_count,
                    status=status,
                )
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def read_first_camera_infos(
    bag_dir: Path,
    selections: Sequence[CameraTopicSelection],
) -> Dict[str, CameraIntrinsics]:
    info_topics = {
        selection.camera_info_topic: selection
        for selection in selections
        if selection.image_topic and selection.camera_info_topic
    }
    result: Dict[str, CameraIntrinsics] = {}
    for topic_name, msg, _timestamp in iter_ros_messages(bag_dir, list(info_topics.keys())):
        selection = info_topics[topic_name]
        if selection.camera in result or not selection.image_topic:
            continue
        result[selection.camera] = intrinsics_from_camera_info(
            selection.camera,
            selection.image_topic,
            topic_name,
            msg,
        )
        if len(result) == len(info_topics):
            break

    for selection in selections:
        if selection.camera in result or not selection.image_topic:
            continue
        result[selection.camera] = estimate_intrinsics_from_fov(
            selection.camera,
            selection.image_topic,
            selection.camera_info_topic,
        )
    return result


def read_first_camera_infos_rosbags(
    bag_dir: Path,
    selections: Sequence[CameraTopicSelection],
) -> Dict[str, CameraIntrinsics]:
    try:
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import Stores, get_typestore
    except Exception as exc:
        raise RuntimeError("rosbags is required for the non-ROS fallback path.") from exc

    typestore = get_typestore(Stores.ROS2_JAZZY)
    info_topics = {
        selection.camera_info_topic: selection
        for selection in selections
        if selection.image_topic and selection.camera_info_topic
    }
    result: Dict[str, CameraIntrinsics] = {}
    with Reader(bag_dir) as reader:
        connections = [conn for conn in reader.connections if conn.topic in info_topics]
        for conn, _timestamp, rawdata in reader.messages(connections=connections):
            selection = info_topics[conn.topic]
            if selection.camera in result or not selection.image_topic:
                continue
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            result[selection.camera] = intrinsics_from_camera_info(
                selection.camera,
                selection.image_topic,
                conn.topic,
                msg,
            )
            if len(result) == len(info_topics):
                break

    for selection in selections:
        if selection.camera in result or not selection.image_topic:
            continue
        result[selection.camera] = estimate_intrinsics_from_fov(
            selection.camera,
            selection.image_topic,
            selection.camera_info_topic,
        )
    return result


def iter_ros_messages(
    bag_dir: Path,
    topics: Sequence[str],
) -> Iterator[Tuple[str, Any, int]]:
    """Yield deserialized ROS messages for selected topics."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except Exception as exc:
        raise RuntimeError(
            "ROS2 Python dependencies are required. Run this inside a sourced "
            "ROS2 Jazzy environment with rosbag2_py and rclpy available."
        ) from exc

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    if hasattr(reader, "set_filter"):
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))

    type_map = {item.name: item.type for item in reader.get_all_topics_and_types()}
    msg_classes = {topic: get_message(type_map[topic]) for topic in topics if topic in type_map}
    topic_set = set(topics)

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic not in topic_set or topic not in msg_classes:
            continue
        yield topic, deserialize_message(data, msg_classes[topic]), int(timestamp)


def iter_raw_ros_messages(
    bag_dir: Path,
) -> Iterator[Tuple[str, bytes, int, Dict[str, str]]]:
    """Yield raw serialized messages plus the bag topic type map."""
    try:
        import rosbag2_py
    except Exception as exc:
        raise RuntimeError(
            "ROS2 Python dependencies are required. Run this inside a sourced "
            "ROS2 Jazzy environment with rosbag2_py available."
        ) from exc

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    type_map = {item.name: item.type for item in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        yield topic, data, int(timestamp), type_map


def create_rosbag_writer(output_bag_dir: Path, topic_types: Mapping[str, str]) -> Any:
    try:
        import rosbag2_py
    except Exception as exc:
        raise RuntimeError(
            "ROS2 Python dependencies are required. Run this inside a sourced "
            "ROS2 Jazzy environment with rosbag2_py available."
        ) from exc

    if output_bag_dir.exists():
        raise RuntimeError(f"Output bag directory already exists: {output_bag_dir}")
    output_bag_dir.parent.mkdir(parents=True, exist_ok=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output_bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    for topic_id, (name, msg_type) in enumerate(sorted(topic_types.items()), start=1):
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                id=topic_id,
                name=name,
                type=msg_type,
                serialization_format="cdr",
            )
        )
    return writer


def get_ros_message_classes(topic_types: Mapping[str, str]) -> Dict[str, Any]:
    try:
        from rosidl_runtime_py.utilities import get_message
    except Exception as exc:
        raise RuntimeError(
            "rosidl_runtime_py is required to deserialize and rewrite selected topics."
        ) from exc
    return {topic: get_message(msg_type) for topic, msg_type in topic_types.items()}


def convert_image_message(image_msg: Any) -> "np.ndarray":
    require_numpy()
    try:
        from cv_bridge import CvBridge
    except Exception:
        return convert_image_message_without_cv_bridge(image_msg)
    return CvBridge().imgmsg_to_cv2(image_msg, desired_encoding="bgr8")


def convert_image_message_without_cv_bridge(image_msg: Any) -> "np.ndarray":
    require_numpy()
    encoding = str(getattr(image_msg, "encoding", "")).lower()
    width = int(getattr(image_msg, "width"))
    height = int(getattr(image_msg, "height"))
    data = bytes(getattr(image_msg, "data"))
    if encoding in {"rgb8", "bgr8"}:
        image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        if encoding == "rgb8":
            image = image[:, :, ::-1]
        return image.copy()
    if encoding in {"mono8", "8uc1"}:
        gray = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        return np.repeat(gray[:, :, None], 3, axis=2)
    raise RuntimeError(f"Unsupported Image encoding without cv_bridge: {encoding}")


def undistort_image(image: "np.ndarray", intrinsics: CameraIntrinsics) -> Tuple["np.ndarray", str]:
    plan = build_rectification_plan(intrinsics)
    return apply_rectification(image, plan), plan.model


def build_rectification_plan(
    intrinsics: CameraIntrinsics,
    balance: float = 0.35,
    rectified_hfov_degrees: float = 120.0,
) -> RectificationPlan:
    require_numpy()
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for undistortion.") from exc

    k = intrinsics.camera_matrix
    d = intrinsics.distortion_coefficients
    size = (intrinsics.width, intrinsics.height)
    model = intrinsics.distortion_model.lower()
    if d.size == 0 or np.allclose(d, 0.0):
        return build_equidistant_fov_rectification_plan(intrinsics, rectified_hfov_degrees)

    if "fisheye" in model and d.size >= 4:
        d4 = d[:4].reshape(4, 1)
        new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            k,
            d4,
            size,
            np.eye(3),
            balance=balance,
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            k,
            d4,
            np.eye(3),
            new_k,
            size,
            cv2.CV_16SC2,
        )
        return RectificationPlan(intrinsics, map1, map2, new_k, "fisheye")

    new_k, _roi = cv2.getOptimalNewCameraMatrix(k, d, size, alpha=0.2, newImgSize=size)
    map1, map2 = cv2.initUndistortRectifyMap(k, d, None, new_k, size, cv2.CV_16SC2)
    return RectificationPlan(intrinsics, map1, map2, new_k, "pinhole")


def build_equidistant_fov_rectification_plan(
    intrinsics: CameraIntrinsics,
    rectified_hfov_degrees: float = 120.0,
) -> RectificationPlan:
    """Approximate a custom 175 degree fisheye as equidistant projection.

    The challenge CameraInfo currently advertises zero distortion, so OpenCV's
    calibrated undistort functions would become a no-op. This deterministic
    fallback uses the PDF-provided HFOV prior to create a useful rectilinear
    view for inspection and downstream baseline processing.
    """
    require_numpy()
    width = int(intrinsics.width)
    height = int(intrinsics.height)
    source_cx = width / 2.0
    source_cy = height / 2.0
    source_f = width / math.radians(175.0)
    target_f = width / (2.0 * math.tan(math.radians(rectified_hfov_degrees) / 2.0))
    target_cx = width / 2.0
    target_cy = height / 2.0

    yy, xx = np.indices((height, width), dtype=np.float32)
    x = (xx - target_cx) / target_f
    y = (yy - target_cy) / target_f
    radius = np.sqrt(x * x + y * y)
    theta = np.arctan(radius)
    scale = np.divide(
        source_f * theta,
        radius,
        out=np.zeros_like(radius, dtype=np.float32),
        where=radius > 1e-8,
    )
    map1 = (source_cx + x * scale).astype(np.float32)
    map2 = (source_cy + y * scale).astype(np.float32)
    new_k = np.asarray(
        [
            [target_f, 0.0, target_cx],
            [0.0, target_f, target_cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return RectificationPlan(intrinsics, map1, map2, new_k, "equidistant_fov_approx")


def apply_rectification(image: "np.ndarray", plan: RectificationPlan) -> "np.ndarray":
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for undistortion.") from exc
    return cv2.remap(image, plan.map1, plan.map2, interpolation=cv2.INTER_LINEAR)


def rectification_map_xy(plan: RectificationPlan) -> Tuple["np.ndarray", "np.ndarray"]:
    """Return source x/y coordinates from either float or OpenCV fixed-point maps."""
    require_numpy()
    map1 = np.asarray(plan.map1)
    map2 = np.asarray(plan.map2)
    if map1.ndim == 3 and map1.shape[2] >= 2:
        return map1[:, :, 0].astype(np.float32), map1[:, :, 1].astype(np.float32)
    return map1.astype(np.float32), map2.astype(np.float32)


def rectification_valid_mask(plan: RectificationPlan) -> "np.ndarray":
    source_width = int(plan.intrinsics.width)
    source_height = int(plan.intrinsics.height)
    source_x, source_y = rectification_map_xy(plan)
    return (
        (source_x >= 0.0)
        & (source_x < float(source_width))
        & (source_y >= 0.0)
        & (source_y < float(source_height))
    )


def compute_rectification_map_metrics(plan: RectificationPlan) -> Dict[str, float]:
    require_numpy()
    valid_mask = rectification_valid_mask(plan)
    total_pixels = int(valid_mask.size)
    valid_pixels = int(np.count_nonzero(valid_mask))
    if valid_pixels:
        ys, xs = np.where(valid_mask)
        roi_area = int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    else:
        roi_area = 0
    return {
        "valid_pixel_ratio": valid_pixels / total_pixels if total_pixels else 0.0,
        "invalid_pixel_ratio": 1.0 - (valid_pixels / total_pixels) if total_pixels else 0.0,
        "roi_area_ratio": roi_area / total_pixels if total_pixels else 0.0,
    }


def compute_rectified_image_metrics(rectified_image: "np.ndarray", black_threshold: int = 5) -> Dict[str, float]:
    require_numpy()
    image = np.asarray(rectified_image)
    if image.ndim == 2:
        black_mask = image <= black_threshold
    else:
        black_mask = np.all(image <= black_threshold, axis=2)
    return {
        "black_pixel_ratio": float(np.count_nonzero(black_mask) / black_mask.size)
        if black_mask.size
        else 0.0,
    }


def camera_info_for_rectified_image(camera_info_msg: Any, plan: RectificationPlan) -> Any:
    new_k = np.asarray(plan.new_k, dtype=np.float64)
    original_d = getattr(camera_info_msg, "d", [])
    original_k = getattr(camera_info_msg, "k", [])
    original_r = getattr(camera_info_msg, "r", [])
    original_p = getattr(camera_info_msg, "p", [])
    camera_info_msg.width = int(plan.intrinsics.width)
    camera_info_msg.height = int(plan.intrinsics.height)
    camera_info_msg.distortion_model = "plumb_bob"
    camera_info_msg.d = sequence_like(original_d, [0.0, 0.0, 0.0, 0.0, 0.0])
    camera_info_msg.k = sequence_like(original_k, [
        float(new_k[0, 0]),
        float(new_k[0, 1]),
        float(new_k[0, 2]),
        float(new_k[1, 0]),
        float(new_k[1, 1]),
        float(new_k[1, 2]),
        float(new_k[2, 0]),
        float(new_k[2, 1]),
        float(new_k[2, 2]),
    ])
    camera_info_msg.r = sequence_like(original_r, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    camera_info_msg.p = sequence_like(original_p, [
        float(new_k[0, 0]),
        float(new_k[0, 1]),
        float(new_k[0, 2]),
        0.0,
        float(new_k[1, 0]),
        float(new_k[1, 1]),
        float(new_k[1, 2]),
        0.0,
        float(new_k[2, 0]),
        float(new_k[2, 1]),
        float(new_k[2, 2]),
        0.0,
    ])
    return camera_info_msg


def sequence_like(original: Any, values: Sequence[float]) -> Any:
    if np is not None and hasattr(original, "dtype"):
        return np.asarray(values, dtype=original.dtype)
    return list(values)


def enhance_image(image: "np.ndarray") -> "np.ndarray":
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for enhancement.") from exc
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    gamma = 1.08
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(enhanced, table)


def detect_candidates(image: "np.ndarray") -> Tuple["np.ndarray", List[Detection]]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for candidate detection.") from exc

    enhanced = enhance_image(image)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    saturated = cv2.inRange(hsv, (0, 45, 45), (179, 255, 255))
    bright = cv2.inRange(hsv, (0, 0, 170), (179, 80, 255))
    mask = cv2.bitwise_or(saturated, bright)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: List[Detection] = []
    debug = image.copy()
    min_area = max(80, int(image.shape[0] * image.shape[1] * 0.00008))
    max_area = int(image.shape[0] * image.shape[1] * 0.25)

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / float(h or 1)
        if aspect < 0.08 or aspect > 10.0:
            continue
        score = min(1.0, area / float(max_area))
        detections.append(Detection(x=x, y=y, width=w, height=h, area=area, score=score))

    detections.sort(key=lambda item: item.area, reverse=True)
    for detection in detections[:50]:
        cv2.rectangle(
            debug,
            (detection.x, detection.y),
            (detection.x + detection.width, detection.y + detection.height),
            (0, 255, 255),
            2,
        )
    return debug, detections[:50]


def write_image(path: Path, image: "np.ndarray") -> None:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required to write images.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def run_extract_for_bag(
    bag_dir: Path,
    output_root: Path,
    samples_per_camera: int,
) -> Dict[str, List[Dict[str, Any]]]:
    summary = parse_metadata(bag_dir / "metadata.yaml")
    selections = select_camera_topics(summary)
    intrinsics_by_camera = read_first_camera_infos(bag_dir, selections)
    image_topics = {
        selection.image_topic: selection
        for selection in selections
        if selection.image_topic
    }
    written: Dict[str, List[Dict[str, Any]]] = {camera: [] for camera in CAMERA_NAMES}
    bag_name = bag_dir.name
    plans_by_camera = {
        camera: build_rectification_plan(intrinsics)
        for camera, intrinsics in intrinsics_by_camera.items()
    }

    for topic_name, msg, timestamp in iter_ros_messages(bag_dir, list(image_topics.keys())):
        selection = image_topics[topic_name]
        if len(written[selection.camera]) >= samples_per_camera:
            if all(len(written[camera]) >= samples_per_camera for camera in CAMERA_NAMES):
                break
            continue

        image = convert_image_message(msg)
        plan = plans_by_camera[selection.camera]
        raw_path = output_root / "raw_frames" / bag_name / selection.camera / f"{timestamp}.png"
        undistorted_path = output_root / "undistorted_frames" / bag_name / selection.camera / f"{timestamp}.png"
        debug_path = output_root / "debug_detections" / bag_name / selection.camera / f"{timestamp}.png"

        undistorted = apply_rectification(image, plan)
        map_metrics = compute_rectification_map_metrics(plan)
        image_metrics = compute_rectified_image_metrics(undistorted)
        debug, detections = detect_candidates(undistorted)
        write_image(raw_path, image)
        write_image(undistorted_path, undistorted)
        write_image(debug_path, debug)
        written[selection.camera].append(
            {
                "timestamp": timestamp,
                "raw": str(raw_path),
                "undistorted": str(undistorted_path),
                "debug": str(debug_path),
                "detections": len(detections),
                "undistort_model": plan.model,
                "valid_pixel_ratio": map_metrics["valid_pixel_ratio"],
                "black_pixel_ratio": image_metrics["black_pixel_ratio"],
            }
        )
    return written


def average_metric(items: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in items if key in item]
    return sum(values) / len(values) if values else 0.0


def run_evaluate_rectification_for_bag(
    bag_dir: Path,
    output_root: Path,
    samples_per_camera: int,
) -> Dict[str, Any]:
    summary = parse_metadata(bag_dir / "metadata.yaml")
    selections = select_camera_topics(summary)
    intrinsics_by_camera = read_first_camera_infos(bag_dir, selections)
    plans_by_camera = {
        camera: build_rectification_plan(intrinsics)
        for camera, intrinsics in intrinsics_by_camera.items()
    }
    image_topics = {
        selection.image_topic: selection
        for selection in selections
        if selection.image_topic and selection.camera in plans_by_camera
    }
    result: Dict[str, Any] = {
        "bag": str(bag_dir),
        "samples_per_camera": samples_per_camera,
        "cameras": {camera: {"frames": []} for camera in CAMERA_NAMES},
    }

    for camera, plan in plans_by_camera.items():
        result["cameras"][camera].update(
            {
                "model": plan.model,
                "intrinsics_source": plan.intrinsics.source,
                **compute_rectification_map_metrics(plan),
            }
        )

    for topic_name, msg, timestamp in iter_ros_messages(bag_dir, list(image_topics.keys())):
        selection = image_topics[topic_name]
        frames = result["cameras"][selection.camera]["frames"]
        if len(frames) >= samples_per_camera:
            if all(len(result["cameras"][camera]["frames"]) >= samples_per_camera for camera in CAMERA_NAMES):
                break
            continue

        plan = plans_by_camera[selection.camera]
        image = convert_image_message(msg)
        rectified = apply_rectification(image, plan)
        frames.append(
            {
                "timestamp": timestamp,
                **compute_rectified_image_metrics(rectified),
            }
        )

    for camera in CAMERA_NAMES:
        camera_result = result["cameras"][camera]
        frames = camera_result["frames"]
        camera_result["frame_count"] = len(frames)
        camera_result["mean_black_pixel_ratio"] = average_metric(frames, "black_pixel_ratio")

    output_dir = output_root / "rectification_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{bag_dir.name}_rectification_metrics.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_rectification_metrics_report(
        output_dir / f"{bag_dir.name}_rectification_metrics.md",
        result,
    )
    return result


def write_rectification_metrics_report(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Rectification Metrics",
        "",
        f"- Bag: `{result['bag']}`",
        f"- Samples per camera: {result['samples_per_camera']}",
        "",
        "| camera | model | source | frames | valid_pixel_ratio | invalid_pixel_ratio | roi_area_ratio | mean_black_pixel_ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    cameras = result["cameras"]
    for camera in CAMERA_NAMES:
        item = cameras[camera]
        lines.append(
            "| {camera} | {model} | {source} | {frames} | {valid:.4f} | {invalid:.4f} | {roi:.4f} | {black:.4f} |".format(
                camera=camera,
                model=item.get("model", "missing"),
                source=item.get("intrinsics_source", "missing"),
                frames=item.get("frame_count", 0),
                valid=float(item.get("valid_pixel_ratio", 0.0)),
                invalid=float(item.get("invalid_pixel_ratio", 0.0)),
                roi=float(item.get("roi_area_ratio", 0.0)),
                black=float(item.get("mean_black_pixel_ratio", 0.0)),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def corrected_bag_output_dir(input_bag_dir: Path, output_root: Path) -> Path:
    return output_root / "corrected_mcaps" / f"{input_bag_dir.name}_fisheye_corrected"


def write_corrected_bag(
    input_bag_dir: Path,
    output_bag_dir: Path,
    max_messages: Optional[int] = None,
) -> Dict[str, Any]:
    """Copy a bag while replacing six fisheye image topics with rectified images."""
    try:
        from rclpy.serialization import deserialize_message, serialize_message
    except Exception as exc:
        return write_corrected_bag_rosbags(input_bag_dir, output_bag_dir, max_messages, str(exc))

    summary = parse_metadata(input_bag_dir / "metadata.yaml")
    selections = select_camera_topics(summary)
    intrinsics_by_camera = read_first_camera_infos(input_bag_dir, selections)
    plans_by_camera = {
        camera: build_rectification_plan(intrinsics)
        for camera, intrinsics in intrinsics_by_camera.items()
    }
    image_topic_to_camera = {
        selection.image_topic: selection.camera
        for selection in selections
        if selection.image_topic and selection.camera in plans_by_camera
    }
    info_topic_to_camera = {
        selection.camera_info_topic: selection.camera
        for selection in selections
        if selection.camera_info_topic and selection.camera in plans_by_camera
    }

    raw_iter = iter_raw_ros_messages(input_bag_dir)
    try:
        first_topic, first_data, first_timestamp, topic_types = next(raw_iter)
    except StopIteration as exc:
        raise RuntimeError(f"Input bag has no messages: {input_bag_dir}") from exc

    writer = create_rosbag_writer(output_bag_dir, topic_types)
    msg_classes = get_ros_message_classes(topic_types)
    stats = {
        "input_bag": str(input_bag_dir),
        "output_bag": str(output_bag_dir),
        "image_topics_corrected": {camera: 0 for camera in CAMERA_NAMES},
        "camera_info_topics_corrected": {camera: 0 for camera in CAMERA_NAMES},
        "messages_copied": 0,
        "messages_written": 0,
        "max_messages": max_messages,
    }

    def process_one(topic: str, data: bytes, timestamp: int) -> None:
        output_data = data
        if topic in image_topic_to_camera:
            camera = image_topic_to_camera[topic]
            image_msg = deserialize_message(data, msg_classes[topic])
            image = convert_image_message(image_msg)
            rectified = apply_rectification(image, plans_by_camera[camera])
            output_msg = image_to_ros_message(rectified, image_msg)
            output_data = serialize_message(output_msg)
            stats["image_topics_corrected"][camera] += 1
        elif topic in info_topic_to_camera:
            camera = info_topic_to_camera[topic]
            info_msg = deserialize_message(data, msg_classes[topic])
            output_msg = camera_info_for_rectified_image(info_msg, plans_by_camera[camera])
            output_data = serialize_message(output_msg)
            stats["camera_info_topics_corrected"][camera] += 1
        else:
            stats["messages_copied"] += 1
        writer.write(topic, output_data, timestamp)
        stats["messages_written"] += 1

    process_one(first_topic, first_data, first_timestamp)
    for topic, data, timestamp, _topic_types in raw_iter:
        if max_messages is not None and stats["messages_written"] >= max_messages:
            break
        process_one(topic, data, timestamp)

    output_bag_dir.parent.mkdir(parents=True, exist_ok=True)
    stats_path = output_bag_dir.parent / f"{output_bag_dir.name}_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def write_corrected_bag_rosbags(
    input_bag_dir: Path,
    output_bag_dir: Path,
    max_messages: Optional[int] = None,
    ros_import_error: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        from rosbags.rosbag2 import CompressionFormat, CompressionMode, Reader, StoragePlugin, Writer
        from rosbags.typesys import Stores, get_typestore
    except Exception as exc:
        raise RuntimeError(
            "Neither ROS2 rclpy nor rosbags fallback is available for corrected MCAP output."
        ) from exc

    if output_bag_dir.exists():
        raise RuntimeError(f"Output bag directory already exists: {output_bag_dir}")
    summary = parse_metadata(input_bag_dir / "metadata.yaml")
    selections = select_camera_topics(summary)
    intrinsics_by_camera = read_first_camera_infos_rosbags(input_bag_dir, selections)
    plans_by_camera = {
        camera: build_rectification_plan(intrinsics)
        for camera, intrinsics in intrinsics_by_camera.items()
    }
    image_topic_to_camera = {
        selection.image_topic: selection.camera
        for selection in selections
        if selection.image_topic and selection.camera in plans_by_camera
    }
    info_topic_to_camera = {
        selection.camera_info_topic: selection.camera
        for selection in selections
        if selection.camera_info_topic and selection.camera in plans_by_camera
    }
    typestore = get_typestore(Stores.ROS2_JAZZY)
    output_bag_dir.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "input_bag": str(input_bag_dir),
        "output_bag": str(output_bag_dir),
        "writer": "rosbags",
        "ros_import_error": ros_import_error,
        "image_topics_corrected": {camera: 0 for camera in CAMERA_NAMES},
        "camera_info_topics_corrected": {camera: 0 for camera in CAMERA_NAMES},
        "messages_copied": 0,
        "messages_written": 0,
        "max_messages": max_messages,
    }

    writer = Writer(output_bag_dir, version=9, storage_plugin=StoragePlugin.MCAP)
    writer.set_compression(CompressionMode.STORAGE, CompressionFormat.ZSTD)
    stats["compression"] = "mcap_storage_zstd"
    with Reader(input_bag_dir) as reader, writer:
        connection_map = {}
        for conn in reader.connections:
            connection_map[conn.id] = writer.add_connection(
                conn.topic,
                conn.msgtype,
                msgdef=conn.msgdef.data,
                rihs01=conn.digest,
                serialization_format=conn.ext.serialization_format,
                offered_qos_profiles=conn.ext.offered_qos_profiles,
            )
        for conn, timestamp, rawdata in reader.messages():
            if max_messages is not None and stats["messages_written"] >= max_messages:
                break
            output_data = rawdata
            if conn.topic in image_topic_to_camera:
                camera = image_topic_to_camera[conn.topic]
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                image = convert_rosbags_image_message(msg)
                rectified = apply_rectification(image, plans_by_camera[camera])
                output_msg = image_to_rosbags_message(rectified, msg)
                output_data = typestore.serialize_cdr(output_msg, conn.msgtype)
                stats["image_topics_corrected"][camera] += 1
            elif conn.topic in info_topic_to_camera:
                camera = info_topic_to_camera[conn.topic]
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                output_msg = camera_info_for_rectified_image(msg, plans_by_camera[camera])
                output_data = typestore.serialize_cdr(output_msg, conn.msgtype)
                stats["camera_info_topics_corrected"][camera] += 1
            else:
                stats["messages_copied"] += 1
            writer.write(connection_map[conn.id], timestamp, output_data)
            stats["messages_written"] += 1

    stats_path = output_bag_dir.parent / f"{output_bag_dir.name}_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def image_to_ros_message(image: "np.ndarray", source_msg: Any) -> Any:
    try:
        from cv_bridge import CvBridge
    except Exception as exc:
        raise RuntimeError("cv_bridge is required to write corrected Image messages.") from exc
    out_msg = CvBridge().cv2_to_imgmsg(image, encoding="bgr8")
    out_msg.header = source_msg.header
    return out_msg


def convert_rosbags_image_message(image_msg: Any) -> "np.ndarray":
    require_numpy()
    encoding = str(getattr(image_msg, "encoding", "")).lower()
    width = int(getattr(image_msg, "width"))
    height = int(getattr(image_msg, "height"))
    data = np.asarray(getattr(image_msg, "data"), dtype=np.uint8)
    if encoding in {"rgb8", "bgr8"}:
        image = data.reshape(height, width, 3)
        if encoding == "rgb8":
            image = image[:, :, ::-1]
        return image.copy()
    if encoding in {"mono8", "8uc1"}:
        gray = data.reshape(height, width)
        return np.repeat(gray[:, :, None], 3, axis=2)
    raise RuntimeError(f"Unsupported Image encoding for rosbags fallback: {encoding}")


def image_to_rosbags_message(image: "np.ndarray", source_msg: Any) -> Any:
    source_msg.height = int(image.shape[0])
    source_msg.width = int(image.shape[1])
    source_msg.encoding = "bgr8"
    source_msg.is_bigendian = 0
    source_msg.step = int(image.shape[1] * 3)
    source_msg.data = np.asarray(image, dtype=np.uint8).reshape(-1)
    return source_msg


def write_corrected_bags_for_data_root(
    data_root: Path,
    output_root: Path,
    max_messages: Optional[int] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for bag_dir in discover_bag_dirs(data_root):
        output_bag_dir = corrected_bag_output_dir(bag_dir, output_root)
        result[bag_dir.name] = write_corrected_bag(bag_dir, output_bag_dir, max_messages)
    manifest_path = output_root / "corrected_mcaps" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def inspect_data_root(data_root: Path, output_root: Path) -> Tuple[List[BagSummary], Dict[Path, List[CameraTopicSelection]]]:
    bag_dirs = discover_bag_dirs(data_root)
    summaries = [parse_metadata(bag_dir / "metadata.yaml") for bag_dir in bag_dirs]
    selections_by_bag = {summary.bag_dir: select_camera_topics(summary) for summary in summaries}
    write_stage_report(output_root / "camera_stage_report.md", summaries, selections_by_bag, output_root)
    return summaries, selections_by_bag


def export_intrinsics_from_first_bag(data_root: Path, output_root: Path) -> None:
    bag_dirs = discover_bag_dirs(data_root)
    if not bag_dirs:
        raise RuntimeError(f"No metadata.yaml files found under {data_root}")
    summary = parse_metadata(bag_dirs[0] / "metadata.yaml")
    selections = select_camera_topics(summary)
    intrinsics = list(read_intrinsics_with_estimated_fallback(summary.bag_dir, selections).values())
    write_intrinsics_yaml(output_root / "camera_intrinsics.yaml", intrinsics)


def run_full_pipeline(data_root: Path, output_root: Path, samples_per_camera: int) -> None:
    summaries, selections_by_bag = inspect_data_root(data_root, output_root)
    if not summaries:
        raise RuntimeError(f"No MCAP bag metadata found under {data_root}")
    first_intrinsics = read_intrinsics_with_estimated_fallback(
        summaries[0].bag_dir,
        selections_by_bag[summaries[0].bag_dir],
    )
    write_intrinsics_yaml(output_root / "camera_intrinsics.yaml", list(first_intrinsics.values()))
    extraction_index: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for summary in summaries:
        extraction_index[summary.bag_dir.name] = run_extract_for_bag(
            summary.bag_dir,
            output_root,
            samples_per_camera,
        )
    (output_root / "extraction_index.json").write_text(
        json.dumps(extraction_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_intrinsics_with_estimated_fallback(
    bag_dir: Path,
    selections: Sequence[CameraTopicSelection],
) -> Dict[str, CameraIntrinsics]:
    try:
        return read_first_camera_infos(bag_dir, selections)
    except RuntimeError as exc:
        if "ROS2 Python dependencies are required" not in str(exc):
            raise
        result: Dict[str, CameraIntrinsics] = {}
        for selection in selections:
            if not selection.image_topic:
                continue
            item = estimate_intrinsics_from_fov(
                selection.camera,
                selection.image_topic,
                selection.camera_info_topic,
            )
            item.notes.append(
                "Generated without ROS2 dependencies; rerun in ROS2 Jazzy to read MCAP CameraInfo."
            )
            result[selection.camera] = item
        return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect camera topics from rosbag metadata.")
    inspect_parser.add_argument("--data-root", type=Path, default=Path("数据"))
    inspect_parser.add_argument("--output-root", type=Path, default=Path("output/camera_fisheye"))

    intrinsics_parser = subparsers.add_parser("intrinsics", help="Export camera_intrinsics.yaml from the first bag.")
    intrinsics_parser.add_argument("--data-root", type=Path, default=Path("数据"))
    intrinsics_parser.add_argument("--output-root", type=Path, default=Path("output/camera_fisheye"))

    extract_parser = subparsers.add_parser("extract", help="Extract raw, undistorted, and debug frames from one bag.")
    extract_parser.add_argument("--bag-dir", type=Path, required=True)
    extract_parser.add_argument("--output-root", type=Path, default=Path("output/camera_fisheye"))
    extract_parser.add_argument("--samples-per-camera", type=int, default=3)

    eval_parser = subparsers.add_parser(
        "evaluate-rectification",
        help="Sample one bag and write rectification quality metrics.",
    )
    eval_parser.add_argument("--bag-dir", type=Path, required=True)
    eval_parser.add_argument("--output-root", type=Path, default=Path("output/camera_fisheye"))
    eval_parser.add_argument("--samples-per-camera", type=int, default=3)

    correct_parser = subparsers.add_parser("correct-bag", help="Write a corrected MCAP bag for one input bag.")
    correct_parser.add_argument("--bag-dir", type=Path, required=True)
    correct_parser.add_argument("--output-root", type=Path, default=Path("output/camera_fisheye"))
    correct_parser.add_argument("--output-bag-dir", type=Path)
    correct_parser.add_argument("--max-messages", type=int)

    correct_all_parser = subparsers.add_parser("correct-all", help="Write corrected MCAP bags for all bags under data root.")
    correct_all_parser.add_argument("--data-root", type=Path, default=Path("数据"))
    correct_all_parser.add_argument("--output-root", type=Path, default=Path("output/camera_fisheye"))
    correct_all_parser.add_argument("--max-messages", type=int)

    run_parser = subparsers.add_parser("run", help="Run inspect, intrinsics export, and extraction for all bags.")
    run_parser.add_argument("--data-root", type=Path, default=Path("数据"))
    run_parser.add_argument("--output-root", type=Path, default=Path("output/camera_fisheye"))
    run_parser.add_argument("--samples-per-camera", type=int, default=3)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "inspect":
        inspect_data_root(args.data_root, args.output_root)
    elif args.command == "intrinsics":
        export_intrinsics_from_first_bag(args.data_root, args.output_root)
    elif args.command == "extract":
        run_extract_for_bag(args.bag_dir, args.output_root, args.samples_per_camera)
    elif args.command == "evaluate-rectification":
        run_evaluate_rectification_for_bag(args.bag_dir, args.output_root, args.samples_per_camera)
    elif args.command == "correct-bag":
        output_bag_dir = args.output_bag_dir or corrected_bag_output_dir(args.bag_dir, args.output_root)
        write_corrected_bag(args.bag_dir, output_bag_dir, args.max_messages)
    elif args.command == "correct-all":
        write_corrected_bags_for_data_root(args.data_root, args.output_root, args.max_messages)
    elif args.command == "run":
        run_full_pipeline(args.data_root, args.output_root, args.samples_per_camera)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
