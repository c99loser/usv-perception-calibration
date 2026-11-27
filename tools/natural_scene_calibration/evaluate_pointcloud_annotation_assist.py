#!/usr/bin/env python3
"""Evaluate patent-inspired annotation assists on existing tracked episodes."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from competition_user.pointcloud_annotation_assist import build_annotation_assists


def _native_path(value: str | Path) -> Path:
    text = str(value)
    if os.name == "nt" and text.startswith("/mnt/") and len(text) > 6:
        return Path(f"{text[5].upper()}:/{text[7:]}")
    if os.name != "nt" and len(text) > 3 and text[1:3] == ":/":
        return Path(f"/mnt/{text[0].lower()}/{text[3:]}")
    return Path(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-holdout",
        action="store_true",
        help="explicit final audit only; never use holdout to tune assist parameters",
    )
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    sources = []
    confidence_values: list[float] = []
    effective_values: list[float] = []
    for source in spec["sources"]:
        if source["split"] == "holdout" and not args.include_holdout:
            continue
        payload = json.loads(
            _native_path(source["tracked_observations"]).read_text(encoding="utf-8")
        )
        per_frame, summary = build_annotation_assists(payload["frames"])
        compensated = [
            item
            for items in per_frame
            for item in items
            if item["status"] == "compensated"
        ]
        confidence_values.extend(float(item["confidence"]) for item in compensated)
        effective_values.extend(float(item["effective_hypotheses"]) for item in compensated)
        sources.append(
            {
                "id": source["id"],
                "split": source["split"],
                "camera": str(payload.get("camera", "unknown")),
                "frames": len(payload["frames"]),
                **summary,
            }
        )
    report = {
        "schema": "pointcloud-annotation-assist-evaluation/v1",
        "purpose": "sensor_only_offline_annotation_assistance_not_ground_truth",
        "evaluation_scope": "train_and_holdout" if args.include_holdout else "train_only",
        "episodes": len(sources),
        "frames": sum(item["frames"] for item in sources),
        "by_split": dict(Counter(item["split"] for item in sources)),
        "observed_proposals": sum(item["observed_proposals"] for item in sources),
        "compensated_proposals": sum(item["compensated_proposals"] for item in sources),
        "rejected_low_confidence_gaps": sum(
            item["rejected_low_confidence_gaps"] for item in sources
        ),
        "rejected_ineligible_endpoint_gaps": sum(
            item["rejected_ineligible_endpoint_gaps"] for item in sources
        ),
        "compensation_confidence": {
            "minimum": float(np.min(confidence_values)) if confidence_values else None,
            "median": float(np.median(confidence_values)) if confidence_values else None,
            "maximum": float(np.max(confidence_values)) if confidence_values else None,
        },
        "effective_hypotheses": {
            "minimum": float(np.min(effective_values)) if effective_values else None,
            "median": float(np.median(effective_values)) if effective_values else None,
            "maximum": float(np.max(effective_values)) if effective_values else None,
        },
        "prohibited_inputs": ["/tf", "/tf_static", "odometry", "/world/*", "camera_projection"],
        "ground_truth": False,
        "sources": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
