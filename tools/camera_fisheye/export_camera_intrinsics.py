#!/usr/bin/env python3
"""Export camera_intrinsics.yaml from MCAP CameraInfo messages."""

from camera_fisheye_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main(["intrinsics", *(__import__("sys").argv[1:])]))
