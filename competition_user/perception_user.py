"""Minimal safe ROS 2 entry point for the organizer's perception task.

It deliberately subscribes only to the seven LiDAR point topics and six
fisheye image topics that the platform says will be available during scoring.
It is a readiness/diagnostic node, not a substitute for the detection,
calibration, BEV, or fusion algorithms still to be connected here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2

from competition_user.calibration_contract import ExtrinsicStatus
from competition_user.natural_calibration import NaturalTrackManager, observations_from_cloud


LIDARS = ("front", "back", "left", "left2", "right", "right2", "forward")
CAMERAS = ("front", "back", "left", "left2", "right", "right2")
RECORDED_SENSOR_QOS = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)


@dataclass
class InputStats:
    messages: int = 0
    last_stamp_ns: int = 0
    valid_points: int = 0
    nonfinite_or_zero_points: int = 0


def lidar_topic(name: str) -> str:
    return f"/wamv/sensors/lidars/{name}_lidar_sensor/points"


def camera_topic(name: str) -> str:
    return f"/wamv/sensors/cameras/{name}_camera_sensor/image_raw"


class PerceptionUser(Node):
    """Reports readiness of the permitted production inputs once per second."""

    def __init__(self) -> None:
        super().__init__("perception_user")
        self._seen: Set[str] = set()
        self._input_subscriptions = []
        self._stats: Dict[str, InputStats] = {}
        self._extrinsic = ExtrinsicStatus()
        self._natural_trackers = {name: NaturalTrackManager() for name in LIDARS}
        self._natural_reports = {}

        for name in LIDARS:
            topic = lidar_topic(name)
            self._input_subscriptions.append(
                self.create_subscription(
                    PointCloud2, topic, self._make_lidar_callback(name, topic), RECORDED_SENSOR_QOS
                )
            )
        for name in CAMERAS:
            topic = camera_topic(name)
            self._input_subscriptions.append(
                self.create_subscription(
                    Image, topic, self._make_image_callback(topic), RECORDED_SENSOR_QOS
                )
            )

        self._expected = {lidar_topic(name) for name in LIDARS} | {
            camera_topic(name) for name in CAMERAS
        }
        self._readiness_timer = self.create_timer(1.0, self._report_readiness)
        self._diagnostic_timer = self.create_timer(5.0, self._report_contract)
        self.get_logger().info(
            "Perception input guard started: subscribing to 7 LiDAR and 6 fisheye image streams only."
        )

    @staticmethod
    def _stamp_ns(message: PointCloud2 | Image) -> int:
        return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)

    def _record(self, topic: str, message: PointCloud2 | Image) -> InputStats:
        stats = self._stats.setdefault(topic, InputStats())
        stats.messages += 1
        stats.last_stamp_ns = self._stamp_ns(message)
        self._seen.add(topic)
        return stats

    def _make_image_callback(self, topic: str):
        def callback(message: Image) -> None:
            self._record(topic, message)
        return callback

    def _make_lidar_callback(self, sensor: str, topic: str):
        def callback(message: PointCloud2) -> None:
            stats = self._record(topic, message)
            # Runtime receives only PointCloud2: finite/zero accounting is
            # performed here, before any future clustering or calibration.
            try:
                raw = point_cloud2.read_points(message, field_names=("x", "y", "z"), skip_nans=False)
                xyz = np.column_stack((raw["x"], raw["y"], raw["z"])).astype(float)
                finite_nonzero = np.isfinite(xyz).all(axis=1) & np.any(xyz != 0.0, axis=1)
                stats.valid_points += int(finite_nonzero.sum())
                stats.nonfinite_or_zero_points += int((~finite_nonzero).sum())
                observations = observations_from_cloud(xyz[finite_nonzero], self._stamp_ns(message))
                self._natural_trackers[sensor].update(observations)
                self._natural_reports[sensor] = self._natural_trackers[sensor].observability()
            except Exception as exc:  # malformed clouds must degrade safely
                self.get_logger().warning(f"Unable to audit {topic}: {exc}")
        return callback

    def _report_contract(self) -> None:
        active_lidar = [self._stats.get(lidar_topic(name)) for name in LIDARS]
        active_camera = [self._stats.get(camera_topic(name)) for name in CAMERAS]
        lidar_times = [item.last_stamp_ns for item in active_lidar if item and item.last_stamp_ns]
        camera_times = [item.last_stamp_ns for item in active_camera if item and item.last_stamp_ns]
        skew_ms = None
        if lidar_times and camera_times:
            skew_ms = min(abs(lidar - camera) for lidar in lidar_times for camera in camera_times) / 1e6
        total_valid = sum(item.valid_points for item in active_lidar if item)
        total_invalid = sum(item.nonfinite_or_zero_points for item in active_lidar if item)
        valid_ratio = total_valid / max(1, total_valid + total_invalid)
        self.get_logger().info(
            "Calibration contract: "
            f"state={self._extrinsic.state.value}, pixel_fusion={self._extrinsic.allows_pixel_fusion}, "
            f"valid_lidar_ratio={valid_ratio:.3f}, nearest_image_lidar_skew_ms={skew_ms}"
        )
        observable = [name for name, report in self._natural_reports.items() if report.observable]
        rejected = {name: report.reason for name, report in self._natural_reports.items() if not report.observable}
        self.get_logger().info(
            f"Natural-scene calibration observations: observable_lidars={observable}; "
            f"rejected={rejected}. No external update is attempted without cross-sensor association."
        )

    def _report_readiness(self) -> None:
        missing = sorted(self._expected - self._seen)
        if not missing:
            self.get_logger().info("All 13 permitted perception inputs are active.")
            self._readiness_timer.cancel()
            return
        self.get_logger().info(
            f"Waiting for {len(missing)}/13 permitted perception inputs: {', '.join(missing)}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionUser()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
