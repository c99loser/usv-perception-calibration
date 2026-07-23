#!/usr/bin/env python3
"""Capture one synchronized RGB / semantic-label image pair from Gazebo.

This is deliberately a *training-data* collector.  The semantic topic must
never be bridged into, recorded by, or read by the competition runtime.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import gz.transport13 as gzt
from gz.msgs10.image_pb2 import Image


def stamp_ns(message: Image) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(getattr(stamp, "nsec", 0))


def as_rgb(message: Image) -> np.ndarray:
    raw = np.frombuffer(message.data, dtype=np.uint8)
    expected = int(message.width) * int(message.height) * 3
    if raw.size != expected:
        raise ValueError(f"expected {expected} RGB bytes, received {raw.size}")
    return raw.reshape((int(message.height), int(message.width), 3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rgb-topic", default="/preflight/rgb")
    parser.add_argument("--label-topic", default="/preflight/semantic/labels_map")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args()

    received: dict[str, dict[int, Image]] = {"rgb": {}, "labels": {}}
    def receive(key: str):
        def callback(message: Image) -> None:
            bucket = received[key]
            bucket[stamp_ns(message)] = message
            # A short history tolerates subscription order without pairing
            # frames from different simulation timestamps.
            while len(bucket) > 20:
                del bucket[min(bucket)]
        return callback
    node = gzt.Node()
    if not node.subscribe(Image, args.rgb_topic, receive("rgb")):
        raise RuntimeError(f"could not subscribe to {args.rgb_topic}")
    if not node.subscribe(Image, args.label_topic, receive("labels")):
        raise RuntimeError(f"could not subscribe to {args.label_topic}")

    deadline = time.monotonic() + args.timeout_seconds
    common: set[int] = set()
    while not common and time.monotonic() < deadline:
        common = set(received["rgb"]) & set(received["labels"])
        time.sleep(0.02)
    if not common:
        raise TimeoutError("no equal RGB / label timestamp observed")

    selected_stamp = max(common)
    rgb, labels = received["rgb"][selected_stamp], received["labels"][selected_stamp]
    if (rgb.width, rgb.height) != (labels.width, labels.height):
        raise ValueError("RGB and semantic label dimensions differ")
    rgb_pixels, label_pixels = as_rgb(rgb), as_rgb(labels)
    if stamp_ns(rgb) != stamp_ns(labels):
        raise ValueError(f"unsynchronized pair: rgb={stamp_ns(rgb)}, labels={stamp_ns(labels)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    # OpenCV writes BGR, while Gazebo publishes RGB.  The explicit conversion
    # preserves both raster images exactly when read back with cv2.imread.
    cv2.imwrite(str(args.output_root / "rgb.png"), cv2.cvtColor(rgb_pixels, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.output_root / "labels_map.png"), cv2.cvtColor(label_pixels, cv2.COLOR_RGB2BGR))
    metadata = {
        "purpose": "simulation_training_only",
        "runtime_allowed": False,
        "rgb_topic": args.rgb_topic,
        "label_topic": args.label_topic,
        "timestamp_ns": stamp_ns(rgb),
        "width": int(rgb.width),
        "height": int(rgb.height),
        "pixel_format": {"rgb": int(rgb.pixel_format_type), "labels": int(labels.pixel_format_type)},
    }
    (args.output_root / "pair.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
