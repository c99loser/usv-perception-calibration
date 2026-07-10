#!/usr/bin/env python3
"""Inspect MCAP camera topics and write camera_stage_report.md."""

from camera_fisheye_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main(["inspect", *(__import__("sys").argv[1:])]))
