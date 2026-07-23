#!/usr/bin/env python3
"""Audit which declared calibration fixtures are visible to each LiDAR.

This is a capture-quality diagnostic.  It reads recorded point clouds plus the
published geometry of the calibration fixtures; it never reads simulator TF,
/world state or odometry.  Nominal sensor poses are used only to make a wide
ROI, so a failed audit instructs us to change fixture placement rather than to
silently fit unrelated background points.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation
from sensor_msgs_py import point_cloud2

POS = {'front': (7, 0, 2.9), 'back': (-7, 0, 2.9), 'left': (-6, 1.626, 2.9),
       'left2': (6, 1.626, 2.9), 'right': (-6, -1.626, 2.9),
       'right2': (6, -1.626, 2.9), 'forward': (7, 0, 2)}
RPY = {'front': (0, 115, 0), 'back': (0, 115, 180), 'left': (0, 115, 90),
       'left2': (0, 115, 90), 'right': (0, 115, -90),
       'right2': (0, 115, -90), 'forward': (0, 0, 0)}


def topic(name: str) -> str:
    return f'/wamv/sensors/lidars/{name}_lidar_sensor/points'


def sensor_to_base(name: str) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', RPY[name], degrees=True).as_matrix()
    transform[:3, 3] = POS[name]
    return transform


def to_base(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def fixtures() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    # Wide, known AABBs around the six boards and shared lattice boxes.
    result = {}
    for name, x, y in (('front', 13, 0), ('back', -13, 0), ('left', 4, 11),
                       ('left2', -7, 11), ('right', 4, -11), ('right2', -7, -11)):
        result[f'board_{name}'] = (np.array((x, y, 2.8)), np.array((4.5, 4.5, 3.2)))
    for index, (x, y, z, sx, sy, sz) in enumerate((
        (-12, -9, 2.0, 1.0, 1.0, 4.0), (-12, 9, 2.5, 1.5, 1.0, 5.0),
        (12, -9, 2.2, 1.0, 1.5, 4.5), (12, 9, 3.0, 1.8, 1.0, 5.5),
        (0, -14, 1.8, 1.2, 1.2, 3.6), (0, 14, 2.8, 1.0, 1.8, 5.2))):
        result[f'lattice_{index}'] = (np.array((x, y, z)), np.array((sx, sy, sz)) / 2 + 0.9)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(args.bag), storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    names = {topic(name): name for name in POS}
    counts = {name: {fixture: [] for fixture in fixtures()} for name in POS}
    regions = fixtures()
    while reader.has_next():
        name, data, _ = reader.read_next()
        sensor = names.get(name)
        if sensor is None:
            continue
        message = deserialize_message(data, get_message(types[name]))
        raw = point_cloud2.read_points(message, field_names=('x', 'y', 'z'), skip_nans=True)
        points = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(float)
        points = to_base(sensor_to_base(sensor), points[np.isfinite(points).all(axis=1)])
        for fixture, (center, half_extent) in regions.items():
            counts[sensor][fixture].append(int((np.abs(points - center) <= half_extent).all(axis=1).sum()))
    report = {'schema': 'controlled-fixture-visibility/v1', 'truth_usage': 'offline_evaluation_only',
              'fixture_source': 'declared calibration_user geometry',
              'sensors': {name: {fixture: {'median_points': int(np.median(values)) if values else 0,
                                             'frames': len(values)}
                                  for fixture, values in values_by_fixture.items()}
                          for name, values_by_fixture in counts.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
