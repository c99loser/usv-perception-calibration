#!/usr/bin/env python3
"""Extract one image per permitted camera and test controlled checkerboard visibility."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

CAMERAS = ("front", "back", "left", "left2", "right", "right2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    expected = {f"/wamv/sensors/cameras/{name}_camera_sensor/image_raw": name for name in CAMERAS}
    bridge, results, frames, found_counts = CvBridge(), {}, {}, {name: 0 for name in CAMERAS}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        name = expected.get(topic)
        if name is None:
            continue
        message = deserialize_message(data, get_message(types[topic]))
        image = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 8x6 squares => 7x5 internal corners. SB is more robust for rendered images.
        found, corners = cv2.findChessboardCornersSB(gray, (7, 5), flags=cv2.CALIB_CB_EXHAUSTIVE)
        annotated = image.copy()
        if found:
            cv2.drawChessboardCorners(annotated, (7, 5), corners, found)
            found_counts[name] += 1
            # Keep a detected frame; it is preferable to an arbitrary first
            # frame while retaining only recorded camera data.
            frames[name] = annotated
            results[name] = {"checkerboard_found": True, "resolution": [int(image.shape[1]), int(image.shape[0])],
                             "corner_count": int(len(corners)), "detected_frames": found_counts[name]}
        elif name not in frames:
            frames[name] = annotated
            results[name] = {"checkerboard_found": False, "resolution": [int(image.shape[1]), int(image.shape[0])],
                             "corner_count": 0, "detected_frames": 0}
    for name, image in frames.items():
        cv2.imwrite(str(args.output / f"{name}.png"), image)
    # A compact montage is convenient for human review; missing frames appear black.
    tiles = []
    for name in CAMERAS:
        image = frames.get(name, np.zeros((720, 1280, 3), dtype=np.uint8))
        tile = cv2.resize(image, (640, 360))
        cv2.putText(tile, name, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        tiles.append(tile)
    cv2.imwrite(str(args.output / "camera_montage.png"), np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:]))))
    (args.output / "checkerboard_visibility.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
