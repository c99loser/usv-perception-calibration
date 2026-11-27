"""Export the same numbered frame for each final (points_step4) LiDAR topic."""
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


def render(points: np.ndarray, output: Path, sensor: str, frame: int) -> None:
    width, height, limit = 800, 600, (200.0 if sensor == "forward" else 60.0)
    image = Image.new("RGB", (width, height), "#07111d")
    draw = ImageDraw.Draw(image)
    scale, cx, cy = min(width, height) / (2 * limit), width / 2, height / 2
    for tick in range(-int(limit), int(limit) + 1, 20 if limit > 60 else 10):
        px, py = cx + tick * scale, cy - tick * scale
        draw.line((px, 0, px, height), fill="#193149")
        draw.line((0, py, width, py), fill="#193149")
    draw.line((0, cy, width, cy), fill="#49657b")
    draw.line((cx, 0, cx, height), fill="#49657b")
    valid = np.isfinite(points["x"]) & np.isfinite(points["y"]) & np.isfinite(points["z"])
    shown = points[valid][::max(1, int(valid.sum() / 30000))]
    for x, y in zip(shown["x"], shown["y"]):
        px, py = int(cx + float(x) * scale), int(cy - float(y) * scale)
        if 0 <= px < width and 0 <= py < height:
            draw.point((px, py), fill="#48e5ec")
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill="#ffb703")
    draw.text((12, 12), f"{sensor} LiDAR Step4 | frame {frame}", fill="#e8f3ff")
    draw.text((12, 34), f"valid points: {int(valid.sum()):,}; XY view; range +/-{limit:g}m", fill="#b8d4eb")
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mcap", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".codex_pydeps"))
    from mcap_ros2.reader import read_ros2_messages

    topic_to_sensor = {f"/wamv/sensors/lidars/{name}_lidar_sensor/points_step4": name for name in SENSORS}
    counts = {name: 0 for name in SENSORS}
    captured: dict[str, np.ndarray] = {}
    for message in read_ros2_messages(str(args.mcap), topics=list(topic_to_sensor)):
        sensor = topic_to_sensor[message.channel.topic]
        counts[sensor] += 1
        if counts[sensor] == args.frame:
            captured[sensor] = np.frombuffer(bytes(message.ros_msg.data), dtype=PCD_DTYPE).copy()
        if len(captured) == len(SENSORS):
            break
    missing = [sensor for sensor in SENSORS if sensor not in captured]
    if missing:
        raise RuntimeError(f"Frame {args.frame} missing for: {', '.join(missing)}")

    tiles: list[Image.Image] = []
    for sensor in SENSORS:
        path = args.output / f"frame_{args.frame:04d}_{sensor}_points_step4.png"
        render(captured[sensor], path, sensor, args.frame)
        tiles.append(Image.open(path).resize((400, 300)))
        print(path)
    montage = Image.new("RGB", (1200, 900), "#07111d")
    for index, tile in enumerate(tiles):
        montage.paste(tile, ((index % 3) * 400, (index // 3) * 300))
    montage_path = args.output / f"frame_{args.frame:04d}_all_lidars_points_step4.png"
    montage.save(montage_path)
    print(montage_path)


if __name__ == "__main__":
    main()
