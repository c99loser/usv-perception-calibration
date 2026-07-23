#!/usr/bin/env python3
"""Generate a labelled *training-only* copy of a perception scene SDF.

The official random-world generator remains unmodified.  This tool runs it for
a fixed seed, then adds Gazebo semantic labels to the four object classes.  The
result must only be launched for dataset collection, never competition runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


LABELS = {"buoy_": 1, "pillar_": 2, "block_": 3, "boat": 4}


def label_for(name: str) -> int | None:
    return next((label for prefix, label in LABELS.items() if name.startswith(prefix)), None)


def inject_labels(tree: ET.ElementTree) -> Counter[int]:
    """Apply class labels to generated model/include elements and return counts."""
    counts: Counter[int] = Counter()
    for element in tree.getroot().iter():
        if element.tag not in {"model", "include"}:
            continue
        name = element.get("name") or element.findtext("name") or ""
        label = label_for(name)
        if label is None:
            continue
        if any(child.tag == "plugin" and child.get("name") == "gz::sim::systems::Label" for child in element):
            raise ValueError(f"label plugin already exists for {name}")
        plugin = ET.SubElement(element, "plugin", {
            "filename": "gz-sim-label-system", "name": "gz::sim::systems::Label"})
        ET.SubElement(plugin, "label").text = str(label)
        counts[label] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if not args.generator.is_file():
        raise SystemExit(f"missing generator: {args.generator}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".unlabelled.sdf")
    completed = subprocess.run([sys.executable, str(args.generator), str(args.seed), str(temporary)],
                               text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"generator failed ({completed.returncode}): {completed.stderr.strip()}")
    try:
        tree = ET.parse(temporary)
    finally:
        temporary.unlink(missing_ok=True)
    names = [element.get("name") or element.findtext("name") or "" for element in tree.getroot().iter()]
    forbidden = [name for name in names if any(token in name.lower() for token in ("checker", "board", "calibration"))]
    if forbidden:
        raise ValueError(f"controlled calibration assets forbidden in training scene: {forbidden[:5]}")
    counts = inject_labels(tree)
    missing = [name for label, name in ((1, "buoy"), (2, "pillar"), (3, "block"), (4, "boat")) if not counts[label]]
    if missing:
        raise ValueError(f"scene seed lacks required classes: {missing}")
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    report = {
        "purpose": "simulation_training_only", "runtime_allowed": False,
        "generator": str(args.generator), "generator_sha256": hashlib.sha256(args.generator.read_bytes()).hexdigest(),
        "seed": args.seed, "world": str(args.output), "semantic_labels": dict(LABELS),
        "instances": {str(key): value for key, value in sorted(counts.items())},
    }
    args.output.with_suffix(".lineage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
