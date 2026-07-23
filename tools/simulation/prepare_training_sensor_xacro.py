#!/usr/bin/env python3
"""Add six training-only semantic fisheye sensors to a copied WAM-V xacro."""
from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path


CAMERAS = ("front", "back", "left", "left2", "right", "right2")
XACRO_NS = "http://www.ros.org/wiki/xacro"


def augment(root: ET.Element) -> int:
    added = 0
    for camera in CAMERAS:
        source_name, target_name = f"{camera}_camera_sensor", f"{camera}_camera_training_semantic"
        source: ET.Element | None = None
        parent: ET.Element | None = None
        for gazebo in root.iter("gazebo"):
            for sensor in gazebo.findall("sensor"):
                if sensor.get("name") == source_name:
                    source, parent = sensor, gazebo
                if sensor.get("name") == target_name:
                    raise ValueError(f"duplicate training sensor {target_name}")
        if source is None or parent is None:
            raise ValueError(f"missing source sensor {source_name}")
        clone = copy.deepcopy(source)
        clone.set("name", target_name)
        clone.set("type", "segmentation")
        topic = ET.Element("topic"); topic.text = f"/training/semantic/{camera}"
        clone.insert(0, topic)
        camera_config = clone.find("camera")
        if camera_config is None:
            raise ValueError(f"missing camera config for {source_name}")
        segmentation_type = ET.Element("segmentation_type"); segmentation_type.text = "semantic"
        camera_config.insert(0, segmentation_type)
        parent.append(clone)
        added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-level-input", type=Path)
    parser.add_argument("--top-level-output", type=Path)
    args = parser.parse_args()
    if bool(args.top_level_input) != bool(args.top_level_output):
        parser.error("--top-level-input and --top-level-output must be supplied together")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.top_level_output and args.top_level_output.exists():
        raise SystemExit(f"refusing to overwrite {args.top_level_output}")
    if args.top_level_output and args.top_level_output.parent != args.output.parent:
        parser.error("top-level output and sensor-fragment output must share a directory")
    ET.register_namespace("xacro", XACRO_NS)
    tree = ET.parse(args.input)
    added = augment(tree.getroot())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    serialized = args.output.read_text(encoding="utf-8")
    # ElementTree chooses ns0 for the xacro namespace.  Keep the source's
    # explicit prefix because xacro's include processor expects `xacro:`.
    args.output.write_text(serialized.replace("xmlns:ns0=", "xmlns:xacro=").replace("ns0:", "xacro:"),
                           encoding="utf-8")
    top_level = None
    if args.top_level_input:
        original = "$(find wamv_gazebo)/urdf/wamv_large_gazebo.urdf.xacro"
        text = args.top_level_input.read_text(encoding="utf-8")
        if text.count(original) != 1:
            raise ValueError("expected exactly one wamv_large_gazebo include in top-level xacro")
        # Preserve the literal xacro: prefix.  XML serializers may rename it to
        # ns0, which is valid XML but not reliably expanded by the xacro CLI.
        args.top_level_output.write_text(text.replace(original, args.output.name), encoding="utf-8")
        top_level = str(args.top_level_output)
    print({"purpose": "simulation_training_only", "runtime_allowed": False, "added_sensors": added,
           "top_level": top_level})


if __name__ == "__main__":
    main()
