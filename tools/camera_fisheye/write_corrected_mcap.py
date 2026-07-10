#!/usr/bin/env python3
"""Write corrected MCAP bags with rectified fisheye camera image topics."""

from camera_fisheye_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main(["correct-bag", *(__import__("sys").argv[1:])]))
