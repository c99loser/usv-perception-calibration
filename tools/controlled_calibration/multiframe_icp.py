#!/usr/bin/env python3
"""Timestamp-paired, finite-point + voxel-only multi-frame LiDAR ICP audit.

The reader intentionally uses only the recorded ``PointCloud2`` messages.  A
nominal mechanical pose is permitted solely as an ICP initial guess; no TF,
world state or odometry is read.  Unlike the original diagnostic version,
every synchronized scan in every bag is evaluated, so stability means scan to
scan repeatability rather than agreement between three arbitrary first frames.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sensor_msgs_py import point_cloud2

POS = {'front': (7, 0, 2.9), 'back': (-7, 0, 2.9), 'left': (-6, 1.626, 2.9),
       'left2': (6, 1.626, 2.9), 'right': (-6, -1.626, 2.9),
       'right2': (6, -1.626, 2.9), 'forward': (7, 0, 2)}
RPY = {'front': (0, 115, 0), 'back': (0, 115, 180), 'left': (0, 115, 90),
       'left2': (0, 115, 90), 'right': (0, 115, -90),
       'right2': (0, 115, -90), 'forward': (0, 0, 0)}
NAMES = tuple(POS)
# Physical overlap graph: front-left-back-right-front is a real FOV loop.
# A direct front-back edge is intentionally absent because opposite-facing
# sensors need not observe the same target and ICP would only fit clutter.
EDGES = (('left', 'front'), ('right', 'front'), ('back', 'right'),
         ('forward', 'front'), ('left2', 'left'), ('right2', 'right'),
         ('back', 'left'))
# Each edge is restricted to the one declared board that the capture-quality
# audit confirms is in both sensors' field of view.  These are fixture IDs,
# not simulator truth topics or an externally observed pose.
EDGE_FIXTURE = {('left', 'front'): 'pair_front_left', ('right', 'front'): 'pair_front_right',
                ('back', 'right'): 'pair_back_right', ('forward', 'front'): 'board_front',
                ('left2', 'left'): 'pair_left_pair', ('right2', 'right'): 'pair_right_pair',
                ('back', 'left'): 'pair_back_left'}
FIXTURES = {
    'board_front': (np.array((13, 0, 2.8)), np.array((4.5, 4.5, 3.2))),
    'board_back': (np.array((-13, 0, 2.8)), np.array((4.5, 4.5, 3.2))),
    'board_left': (np.array((4, 11, 2.8)), np.array((4.5, 4.5, 3.2))),
    'board_left2': (np.array((-7, 11, 2.8)), np.array((4.5, 4.5, 3.2))),
    'board_right': (np.array((4, -11, 2.8)), np.array((4.5, 4.5, 3.2))),
    'board_right2': (np.array((-7, -11, 2.8)), np.array((4.5, 4.5, 3.2))),
    # Half extents cover the declared three-box pair fixtures, including a
    # small nominal-pose tolerance but not unrelated surrounding geometry.
    'pair_front_left': (np.array((10, 7, 2.4)), np.array((3.0, 3.0, 3.6))),
    'pair_front_right': (np.array((10, -7, 2.4)), np.array((3.0, 3.0, 3.6))),
    'pair_back_right': (np.array((-10, -7, 2.4)), np.array((3.0, 3.0, 3.6))),
    'pair_back_left': (np.array((-10, 7, 2.4)), np.array((3.0, 3.0, 3.6))),
    'pair_left_pair': (np.array((-1, 11, 2.4)), np.array((3.0, 3.0, 3.6))),
    'pair_right_pair': (np.array((-1, -11, 2.4)), np.array((3.0, 3.0, 3.6))),
}
MAX_PAIR_SKEW_S = 0.30
VOXEL_M = 0.10
MAX_CORRESPONDENCE_M = 0.75
MIN_CORRESPONDENCES = 40
MIN_FITNESS = 0.15
MAX_RMSE_M = 0.15
MAX_TRANSLATION_DELTA_M = 0.75
MAX_ROTATION_DELTA_DEG = 12.0


def topic(name: str) -> str:
    return f'/wamv/sensors/lidars/{name}_lidar_sensor/points'


def read_frames(bag: Path) -> dict[str, list[tuple[int, np.ndarray]]]:
    """Read every LiDAR scan, filtering only non-finite XYZ values."""
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    types = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    frames = {name: [] for name in NAMES}
    by_topic = {topic(name): name for name in NAMES}
    while reader.has_next():
        name, data, stamp = reader.read_next()
        sensor = by_topic.get(name)
        if sensor is None:
            continue
        msg = deserialize_message(data, get_message(types[name]))
        raw = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        xyz = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(float)
        frames[sensor].append((stamp, xyz[np.isfinite(xyz).all(axis=1)]))
    return frames


def synchronized_bundles(frames: dict[str, list[tuple[int, np.ndarray]]]) -> list[dict[str, np.ndarray]]:
    """Pair all sensors to each front scan, rejecting excessive time skew."""
    bundles = []
    for stamp, cloud in frames['front']:
        bundle = {'front': cloud}
        for name in NAMES:
            if name == 'front' or not frames[name]:
                continue
            nearest_stamp, nearest_cloud = min(frames[name], key=lambda item: abs(item[0] - stamp))
            if abs(nearest_stamp - stamp) <= MAX_PAIR_SKEW_S * 1e9:
                bundle[name] = nearest_cloud
        if len(bundle) == len(NAMES):
            bundles.append(bundle)
    return bundles


def vox(points: np.ndarray, side: float = VOXEL_M) -> np.ndarray:
    keys = np.floor(points / side).astype(np.int32)
    _, keep = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(keep)]


def nominal(source: str, target: str) -> np.ndarray:
    def sensor_to_base(name: str) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler('xyz', RPY[name], degrees=True).as_matrix()
        transform[:3, 3] = POS[name]
        return transform
    return np.linalg.inv(sensor_to_base(target)) @ sensor_to_base(source)


def sensor_to_base(name: str) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', RPY[name], degrees=True).as_matrix()
    transform[:3, 3] = POS[name]
    return transform


def apply(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def icp(source: np.ndarray, target: np.ndarray, initial: np.ndarray) -> tuple[np.ndarray, float, float, int]:
    """Point-to-point ICP; correspondence trimming is not point-cloud filtering."""
    source, target = vox(source), vox(target)
    if len(source) < MIN_CORRESPONDENCES or len(target) < MIN_CORRESPONDENCES:
        return initial.copy(), 0.0, float('inf'), 0
    transform = initial.copy()
    tree = cKDTree(target)
    for _ in range(30):
        transformed = apply(transform, source)
        distance, indices = tree.query(transformed, k=1)
        keep = distance < MAX_CORRESPONDENCE_M
        if int(keep.sum()) < MIN_CORRESPONDENCES:
            break
        p, q = transformed[keep], target[indices[keep]]
        centroid_p, centroid_q = p.mean(axis=0), q.mean(axis=0)
        u, _, vh = np.linalg.svd((p - centroid_p).T @ (q - centroid_q))
        rotation = vh.T @ u.T
        if np.linalg.det(rotation) < 0:
            vh[-1] *= -1
            rotation = vh.T @ u.T
        delta = np.eye(4)
        delta[:3, :3] = rotation
        delta[:3, 3] = centroid_q - rotation @ centroid_p
        transform = delta @ transform
        if np.linalg.norm(delta[:3, 3]) < 1e-4 and Rotation.from_matrix(rotation).magnitude() < 1e-4:
            break
    transformed = apply(transform, source)
    distance, _ = tree.query(transformed, k=1)
    keep = distance < MAX_CORRESPONDENCE_M
    matches = int(keep.sum())
    return transform, float(matches / len(source)), (float(np.sqrt(np.mean(distance[keep] ** 2))) if matches else float('inf')), matches


def fixture_points(sensor: str, points: np.ndarray, fixture: str) -> np.ndarray:
    """Keep points in a declared target's wide nominal base-frame ROI only."""
    center, half_extent = FIXTURES[fixture]
    approximate_base = apply(sensor_to_base(sensor), points)
    keep = (np.abs(approximate_base - center) <= half_extent).all(axis=1)
    return points[keep]


def robust_transform(transforms: list[np.ndarray]) -> np.ndarray:
    """Median translation plus quaternion-mean rotation for a diagnostic edge."""
    result = np.eye(4)
    result[:3, 3] = np.median([item[:3, 3] for item in transforms], axis=0)
    result[:3, :3] = Rotation.from_matrix([item[:3, :3] for item in transforms]).mean().as_matrix()
    return result


def edge_report(source: str, target: str, runs: list[tuple[np.ndarray, float, float, int]]) -> dict:
    transforms = [item[0] for item in runs]
    translations = np.asarray([item[:3, 3] for item in transforms])
    mean_rotation = Rotation.from_matrix([item[:3, :3] for item in transforms]).mean()
    rotation_delta = np.degrees(np.max((mean_rotation.inv() * Rotation.from_matrix(
        [item[:3, :3] for item in transforms])).magnitude()))
    translation_delta = float(np.max(np.linalg.norm(translations - np.median(translations, axis=0), axis=1)))
    fitness = float(np.median([item[1] for item in runs]))
    raw_rmse = float(np.median([item[2] for item in runs]))
    rmse = raw_rmse if math.isfinite(raw_rmse) else None
    stable = bool(fitness >= MIN_FITNESS and rmse is not None and rmse <= MAX_RMSE_M and
                  translation_delta <= MAX_TRANSLATION_DELTA_M and rotation_delta <= MAX_ROTATION_DELTA_DEG)
    transform = robust_transform(transforms)
    return {
        'source': source, 'target': target, 'fitness': fitness, 'rmse_m': rmse,
        'translation_delta_m': translation_delta, 'rotation_delta_deg': float(rotation_delta),
        'stable': stable, 'transform_source_to_target': transform.tolist(),
        'runs': [{'fitness': fit, 'rmse_m': (err if math.isfinite(err) else None), 'correspondences': count}
                 for _, fit, err, count in runs],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bags', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundles = [bundle for bag in args.bags for bundle in synchronized_bundles(read_frames(bag))]
    if not bundles:
        raise SystemExit('No seven-LiDAR timestamp-synchronized scan bundles found')
    edges = []
    for source, target in EDGES:
        fixture = EDGE_FIXTURE[(source, target)]
        runs = [icp(fixture_points(source, bundle[source], fixture),
                    fixture_points(target, bundle[target], fixture), nominal(source, target))
                for bundle in bundles]
        report = edge_report(source, target, runs)
        report['fixture'] = fixture
        edges.append(report)
    output = {
        'schema': 'controlled-multiframe-icp/v2',
        'preprocess': f'finite_points_only + declared_fixture_roi + {VOXEL_M:.2f}m_voxel_unique',
        'truth_usage': 'offline_evaluation_only',
        'synchronization': {'reference': 'front_lidar', 'max_pair_skew_s': MAX_PAIR_SKEW_S,
                            'bundles': len(bundles)},
        'thresholds': {'min_fitness': MIN_FITNESS, 'max_rmse_m': MAX_RMSE_M,
                       'max_translation_delta_m': MAX_TRANSLATION_DELTA_M,
                       'max_rotation_delta_deg': MAX_ROTATION_DELTA_DEG},
        'edges': edges,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'bundles': len(bundles), 'edges': edges}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
