#!/usr/bin/env python3
"""Extract raw, undistorted, and debug detection frames from one MCAP bag."""

from camera_fisheye_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main(["extract", *(__import__("sys").argv[1:])]))
