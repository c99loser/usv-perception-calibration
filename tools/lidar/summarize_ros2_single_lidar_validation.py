#!/usr/bin/env python3
"""Aggregate per-sensor real-MCAP ROS 2 integration reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for path in sorted(args.root.glob("*/report.json")):
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    summary = {
        "scope": "single_lidar_only_no_extrinsics_no_fusion",
        "sensor_count": len(reports),
        "passed_count": sum(report["passed"] for report in reports),
        "all_passed": bool(reports) and all(report["passed"] for report in reports),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
