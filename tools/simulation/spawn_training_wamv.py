#!/usr/bin/env python3
"""Spawn a training-only WAM-V using VRX's own SDF generation path."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from vrx_gz.model import Model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--xacro", type=Path, required=True)
    parser.add_argument("--name", default="wamv")
    parser.add_argument("--x", type=float, default=-710.0)
    parser.add_argument("--y", type=float, default=350.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    args = parser.parse_args()
    if not args.xacro.is_file():
        raise SystemExit(f"missing xacro: {args.xacro}")
    model = Model(args.name, "wam-v", [args.x, args.y, 0.0, 0.0, 0.0, args.yaw])
    model.set_urdf(str(args.xacro))
    # Matches the official perception-user large WAM-V path.
    model.deck_inject = True
    spawn = model.spawn_args()
    subprocess.run(["ros2", "run", "ros_gz_sim", "create", "-world", args.world, *spawn], check=True)


if __name__ == "__main__":
    main()
