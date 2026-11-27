"""Fuse numbered Step4 LiDAR frames with the MCAP's base_link TF transforms."""
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
SENSORS = ("front", "back", "left", "left2", "right", "right2", "forward")
COLORS = {
    "front": "#2ee8e8", "back": "#ff6b6b", "left": "#a7f542", "left2": "#54d39e",
    "right": "#9b7bff", "right2": "#ff85cf", "forward": "#ffbd45",
}


def rotation_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = (qx*qx + qy*qy + qz*qz + qw*qw) ** 0.5
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def render_panel(draw: ImageDraw.ImageDraw, origin: tuple[int, int], size: tuple[int, int],
                 fused: dict[str, np.ndarray], limit: float, title: str) -> None:
    ox, oy = origin
    width, height = size
    scale, cx, cy = min(width, height) / (2 * limit), ox + width / 2, oy + height / 2
    draw.rectangle((ox, oy, ox + width, oy + height), fill="#07111d", outline="#39556a")
    tick = 20 if limit > 60 else 10
    for value in range(-int(limit), int(limit) + 1, tick):
        px, py = cx + value * scale, cy - value * scale
        draw.line((px, oy, px, oy + height), fill="#193149")
        draw.line((ox, py, ox + width, py), fill="#193149")
    draw.line((ox, cy, ox + width, cy), fill="#587286")
    draw.line((cx, oy, cx, oy + height), fill="#587286")
    # A compact hull marker: bow points to +X (right).
    hull = [(cx + 7*scale, cy), (cx + 4*scale, cy - 2*scale), (cx - 7*scale, cy - 2*scale),
            (cx - 7*scale, cy + 2*scale), (cx + 4*scale, cy + 2*scale)]
    draw.polygon(hull, outline="#ffffff")
    for sensor, points in fused.items():
        shown = points[::max(1, len(points) // 35000)]
        for x, y in shown[:, :2]:
            px, py = int(cx + float(x) * scale), int(cy - float(y) * scale)
            if ox <= px < ox + width and oy <= py < oy + height:
                draw.point((px, py), fill=COLORS[sensor])
    draw.text((ox + 12, oy + 12), title, fill="#e8f3ff")
    draw.text((ox + 12, oy + 33), "base_link: +X bow, +Y port; white outline: vessel", fill="#b8d4eb")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mcap", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame", type=int, default=10)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".codex_pydeps"))
    from mcap_ros2.reader import read_ros2_messages

    topics = {f"/wamv/sensors/lidars/{name}_lidar_sensor/points_step4": name for name in SENSORS}
    counts = {name: 0 for name in SENSORS}
    raw_frames: dict[str, np.ndarray] = {}
    transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for message in read_ros2_messages(str(args.mcap), topics=["/tf", *topics]):
        if message.channel.topic == "/tf":
            for transform in message.ros_msg.transforms:
                child = transform.child_frame_id.rsplit("/", 1)[-1]
                if child.endswith("_lidar_sensor"):
                    sensor = child.removesuffix("_lidar_sensor")
                    if sensor in SENSORS:
                        t, q = transform.transform.translation, transform.transform.rotation
                        transforms[sensor] = (
                            np.array([t.x, t.y, t.z], dtype=np.float64),
                            rotation_from_quaternion(q.x, q.y, q.z, q.w),
                        )
        else:
            sensor = topics[message.channel.topic]
            counts[sensor] += 1
            if counts[sensor] == args.frame:
                raw = np.frombuffer(bytes(message.ros_msg.data), dtype=PCD_DTYPE)
                valid = np.isfinite(raw["x"]) & np.isfinite(raw["y"]) & np.isfinite(raw["z"])
                raw_frames[sensor] = np.column_stack((raw["x"][valid], raw["y"][valid], raw["z"][valid])).astype(np.float64)
        if len(raw_frames) == len(SENSORS) and len(transforms) == len(SENSORS):
            break
    missing = [name for name in SENSORS if name not in raw_frames or name not in transforms]
    if missing:
        raise RuntimeError("Missing frame or TF for: " + ", ".join(missing))
    fused = {name: points @ transforms[name][1].T + transforms[name][0] for name, points in raw_frames.items()}

    image = Image.new("RGB", (1800, 980), "#07111d")
    draw = ImageDraw.Draw(image)
    render_panel(draw, (20, 20), (860, 860), fused, 60.0, f"Frame {args.frame}: near-field fused LiDAR (±60 m)")
    render_panel(draw, (920, 20), (860, 860), fused, 200.0, f"Frame {args.frame}: full-range fused LiDAR (±200 m)")
    legend_x, legend_y = 30, 905
    draw.text((legend_x, legend_y), "Sources: ", fill="#e8f3ff")
    cursor = legend_x + 68
    for name in SENSORS:
        draw.text((cursor, legend_y), f"■ {name}", fill=COLORS[name])
        cursor += 105
    draw.text((30, 932), "Each source was transformed with its /tf base_link → *_lidar_sensor pose from this MCAP.", fill="#b8d4eb")
    image.save(args.output)
    print(f"output={args.output}")
    print("point_counts=" + ", ".join(f"{name}:{len(fused[name])}" for name in SENSORS))


if __name__ == "__main__":
    main()
