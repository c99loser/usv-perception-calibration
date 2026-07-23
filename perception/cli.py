"""CLI for inspecting and replaying raw MCAP bags into BEV/tracking artefacts."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .config import PerceptionConfig
from .mcap_replay import discover_bag_dirs, inspect_bag, replay_pointclouds
from .pipeline import PerceptionPipeline

def write_pgm(path: Path, image: np.ndarray) -> None:
    """Write a dependency-free, directly viewable grayscale BEV image."""
    path.write_bytes(f"P5\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii") + image.tobytes())

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "replay"):
        item = sub.add_parser(name); item.add_argument("--data-root", type=Path, required=True); item.add_argument("--config", type=Path)
    replay = sub.choices["replay"]; replay.add_argument("--output-dir", type=Path, default=Path("output/perception")); replay.add_argument("--max-frames", type=int, default=20)
    args = parser.parse_args(argv); config = PerceptionConfig.load(args.config); bags = discover_bag_dirs(args.data_root)
    if args.command == "inspect": print(json.dumps([inspect_bag(bag, config) for bag in bags], ensure_ascii=False, indent=2)); return 0
    args.output_dir.mkdir(parents=True, exist_ok=True); pipeline = PerceptionPipeline(config); count = 0
    for bag in bags:
        for frame in replay_pointclouds(bag, config, max_frames=max(0, args.max_frames - count)):
            result = pipeline.process(frame); stem = f"{count:05d}_{frame.sensor}"; np.save(args.output_dir / f"{stem}_bev.npy", result.bev); write_pgm(args.output_dir / f"{stem}_bev.pgm", result.bev)
            (args.output_dir / f"{stem}_tracks.json").write_text(json.dumps([track.as_dict() for track in result.tracks], ensure_ascii=False, indent=2), encoding="utf-8"); count += 1
            if count >= args.max_frames: print(json.dumps({"frames_processed": count, "output_dir": str(args.output_dir)})); return 0
    print(json.dumps({"frames_processed": count, "output_dir": str(args.output_dir)})); return 0
if __name__ == "__main__": raise SystemExit(main())
