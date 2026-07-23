#!/usr/bin/env python3
"""Export HSV/contour boxes as *unverified* annotation-review proposals.

The output intentionally is not COCO ``annotations``.  A human must accept or
correct proposals before they enter the separate COCO ground-truth file used by
the offline candidate-extractor evaluator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "camera_fisheye"))
from camera_fisheye_pipeline import detect_candidates  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import cv2

    payload = json.loads(args.coco.read_text(encoding="utf-8"))
    proposals = []
    for item in payload.get("images", []):
        image = cv2.imread(str(args.dataset_root / item["file_name"]))
        if image is None:
            raise RuntimeError(f"Missing image: {item['file_name']}")
        _debug, detections = detect_candidates(image)
        proposals.append({
            "image_id": int(item["id"]),
            "camera": item["camera"],
            "timestamp_ns": int(item["timestamp_ns"]),
            "proposals": [
                {"bbox": [detection.x, detection.y, detection.width, detection.height],
                 "area_proxy_score": detection.score,
                 "status": "unverified_hsv_contour_proposal"}
                for detection in detections
            ],
        })
    result = {
        "purpose": "human_annotation_assistance_only",
        "runtime_eligibility": "forbidden",
        "truth_eligibility": "forbidden_until_human_review",
        "source": "fixed_hsv_contour_baseline",
        "images": proposals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"images": len(proposals), "proposals": sum(len(x["proposals"]) for x in proposals)}))


if __name__ == "__main__":
    main()
