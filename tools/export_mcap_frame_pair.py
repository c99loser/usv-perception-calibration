"""Export a PointCloud2 frame and its nearest front-camera image from an MCAP."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PCD_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("_pad0", "<f4"),
    ("intensity", "<f4"), ("_pad1", "<f4"), ("ring", "<u2"), ("_pad2", "V6"),
])


def import_reader() -> object:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".codex_pydeps"))
    from mcap_ros2.reader import read_ros2_messages
    return read_ros2_messages


def render_cloud(points: np.ndarray, output: Path, title: str) -> None:
    width, height, limit = 1200, 900, 60.0
    image = Image.new("RGB", (width, height), "#07111d")
    draw = ImageDraw.Draw(image)
    scale = min(width, height) / (2 * limit)
    cx, cy = width / 2, height / 2
    for tick in range(-60, 61, 10):
        px, py = cx + tick * scale, cy - tick * scale
        draw.line((px, 0, px, height), fill="#193149")
        draw.line((0, py, width, py), fill="#193149")
    draw.line((0, cy, width, cy), fill="#49657b")
    draw.line((cx, 0, cx, height), fill="#49657b")
    valid = np.isfinite(points["x"]) & np.isfinite(points["y"])
    sample = points[valid][::max(1, int(valid.sum() / 45000))]
    for x, y in zip(sample["x"], sample["y"]):
        px, py = int(cx + float(x) * scale), int(cy - float(y) * scale)
        if 0 <= px < width and 0 <= py < height:
            draw.point((px, py), fill="#48e5ec")
    draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill="#ffb703")
    draw.text((18, 16), title, fill="#e8f3ff")
    draw.text((18, 40), f"有效点：{len(points):,}；X 前方，Y 左方；范围 ±{limit:g}m", fill="#b8d4eb")
    image.save(output)


def decode_image(message: object) -> Image.Image:
    msg = message.ros_msg
    raw = bytes(msg.data)
    encoding = str(msg.encoding).lower()
    width, height, step = int(msg.width), int(msg.height), int(msg.step)
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, step)[:, :width * (4 if "a8" in encoding else 3)]
    if encoding in {"rgb8", "rgba8"}:
        mode = "RGBA" if "rgba" in encoding else "RGB"
        return Image.frombytes(mode, (width, height), rows.tobytes())
    if encoding in {"bgr8", "bgra8"}:
        channels = 4 if "bgra" in encoding else 3
        array = rows.reshape(height, width, channels)
        array = array[:, :, [2, 1, 0, 3]] if channels == 4 else array[:, :, ::-1]
        return Image.fromarray(array, "RGBA" if channels == 4 else "RGB")
    raise RuntimeError(f"Unsupported front camera encoding: {msg.encoding}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mcap", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame", type=int, default=10)
    parser.add_argument(
        "--point-topic",
        default="/wamv/sensors/lidars/front_lidar_sensor/points_step4",
        help="PointCloud2 topic to export (defaults to the Step4 front LiDAR topic).",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    point_topic = args.point_topic
    image_topic = "/wamv/sensors/cameras/front_camera_sensor/image_raw"
    read_ros2_messages = import_reader()
    point_message = None
    for count, message in enumerate(read_ros2_messages(str(args.mcap), topics=[point_topic]), start=1):
        if count == args.frame:
            point_message = message
            break
    if point_message is None:
        raise RuntimeError(f"Point cloud frame {args.frame} was not found")
    raw = np.frombuffer(bytes(point_message.ros_msg.data), dtype=PCD_DTYPE)
    valid = np.isfinite(raw["x"]) & np.isfinite(raw["y"]) & np.isfinite(raw["z"])
    label = "Front LiDAR raw points" if point_topic.endswith("/points") else "Front LiDAR Step4"
    suffix = "points_raw" if point_topic.endswith("/points") else "points_step4"
    point_output = args.output / f"frame_{args.frame:04d}_front_{suffix}.png"
    render_cloud(raw[valid], point_output, f"{label} — frame {args.frame}")

    nearest, nearest_delta = None, None
    target_stamp = point_message.log_time_ns
    for message in read_ros2_messages(str(args.mcap), topics=[image_topic]):
        delta = abs(message.log_time_ns - target_stamp)
        if nearest_delta is None or delta < nearest_delta:
            nearest, nearest_delta = message, delta
        if message.log_time_ns > target_stamp and nearest_delta is not None:
            break
    if nearest is None:
        raise RuntimeError("No front-camera image was found")
    image_output = args.output / f"frame_{args.frame:04d}_front_camera_nearest.png"
    decode_image(nearest).save(image_output)
    print(f"point_stamp_ns={target_stamp}")
    print(f"camera_stamp_ns={nearest.log_time_ns}")
    print(f"camera_offset_ms={nearest_delta / 1_000_000:.3f}")
    print(f"point_png={point_output}")
    print(f"camera_png={image_output}")


if __name__ == "__main__":
    main()
