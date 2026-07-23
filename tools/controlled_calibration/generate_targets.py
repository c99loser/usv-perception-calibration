#!/usr/bin/env python3
"""Generate versioned, known-geometry calibration targets for USVSim.

The generated SDF models are *calibration fixtures*, never competition inputs.
Each of the six camera/LiDAR pairs receives a static 8x6 checkerboard with a
collision backing and four non-coplanar corner posts.  The accompanying JSON
is the only source of target geometry used by the calibration tools.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


TARGET_VERSION = "controlled-targets-v1"
PAIRS = ("front", "back", "left", "left2", "right", "right2")
# ENU offsets from the fixed perception_user WAM-V start.  Boards face inward.
OFFSETS = {
    "front": (15.0, 0.0), "back": (-15.0, 0.0),
    "left": (5.0, 13.0), "left2": (-8.0, 13.0),
    "right": (5.0, -13.0), "right2": (-8.0, -13.0),
}


def target_sdf(name: str) -> str:
    """An upright 8x6 board. Local x/z are board axes; local y is normal."""
    square = 0.35
    cols, rows = 8, 6
    squares = []
    for row in range(rows):
        for col in range(cols):
            shade = "0.03 0.03 0.03 1" if (row + col) % 2 else "0.95 0.95 0.95 1"
            lx = (col - (cols - 1) / 2.0) * square
            lz = (row - (rows - 1) / 2.0) * square
            # Put a pattern on both faces.  The prior target placed squares
            # inside the 8 cm backing, where Gazebo correctly occluded them.
            squares.append(f'''<visual name="sq_front_{row}_{col}"><pose>{lx:.3f} 0.046 {lz:.3f} 0 0 0</pose><geometry><box><size>{square:.3f} 0.012 {square:.3f}</size></box></geometry><material><ambient>{shade}</ambient><diffuse>{shade}</diffuse></material></visual><visual name="sq_back_{row}_{col}"><pose>{lx:.3f} -0.046 {lz:.3f} 0 0 0</pose><geometry><box><size>{square:.3f} 0.012 {square:.3f}</size></box></geometry><material><ambient>{shade}</ambient><diffuse>{shade}</diffuse></material></visual>''')
    posts = []
    for idx, (px, pz) in enumerate(((-1.58, -1.23), (1.58, -1.23), (-1.58, 1.23), (1.58, 1.23))):
        # Different post depths make the target usable by LiDAR, not merely planar.
        depth = (0.16, 0.26, 0.36, 0.46)[idx]
        posts.append(f'''<link name="post_{idx}"><pose>{px} 0 {pz} 0 0 0</pose><collision name="c"><geometry><cylinder><radius>0.09</radius><length>{depth}</length></cylinder></geometry></collision><visual name="v"><geometry><cylinder><radius>0.09</radius><length>{depth}</length></cylinder></geometry><material><ambient>1 0.12 0.02 1</ambient><diffuse>1 0.12 0.02 1</diffuse></material></visual></link>''')
    return f'''<?xml version="1.0"?><sdf version="1.9"><model name="{name}"><static>true</static><link name="board"><collision name="backing"><pose>0 0 0 0 0 0</pose><geometry><box><size>3.2 0.08 2.5</size></box></geometry></collision><visual name="back"><geometry><box><size>3.2 0.08 2.5</size></box></geometry><material><ambient>0.35 0.35 0.35 1</ambient><diffuse>0.35 0.35 0.35 1</diffuse></material></visual>{''.join(squares)}</link>{''.join(posts)}</model></sdf>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wamv-x", type=float, default=-710.0)
    parser.add_argument("--wamv-y", type=float, default=350.0)
    parser.add_argument("--layout-index", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    targets, spawn_rows = [], []
    for pair in PAIRS:
        dx, dy = OFFSETS[pair]
        # Three repeatable static layouts vary target range, bearing and height
        # without moving the vessel or using any truth topic.
        bearing = math.radians(6.0 * args.layout_index)
        scale = 1.0 + 0.08 * args.layout_index
        dx, dy = scale * (dx * math.cos(bearing) - dy * math.sin(bearing)), scale * (dx * math.sin(bearing) + dy * math.cos(bearing))
        x, y = args.wamv_x + dx, args.wamv_y + dy
        # Board local +y normal must point from target back to WAM-V.
        inward = math.atan2(args.wamv_y - y, args.wamv_x - x)
        yaw = inward - math.pi / 2.0
        filename = f"target_{pair}.sdf"
        (args.output / filename).write_text(target_sdf(f"calib_{pair}"), encoding="utf-8")
        spawn_rows.append(f"{pair},{x:.6f},{y:.6f},2.600000,{yaw:.8f}")
        targets.append({"id": f"calib_{pair}", "pair": pair, "sdf": filename,
                        "pose_world": [x, y, 2.6, 0.0, 0.0, yaw],
                        "checkerboard": {"rows": 6, "columns": 8, "square_size_m": 0.35},
                        "corner_post_depths_m": [0.16, 0.26, 0.36, 0.46]})
    manifest = {"schema": "controlled-calibration-target-manifest/v1", "target_version": TARGET_VERSION,
                "layout_index": args.layout_index,
                "world": "perception_user", "wamv_reference": [args.wamv_x, args.wamv_y, 0.0], "targets": targets}
    (args.output / "targets.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output / "spawn.csv").write_text("\n".join(spawn_rows) + "\n", encoding="utf-8")
    print(args.output / "targets.json")


if __name__ == "__main__":
    main()
