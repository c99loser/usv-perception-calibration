"""ROS 2 adapter for the no-extrinsic single-LiDAR baseline."""
from __future__ import annotations

import json

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String

from competition_user.lidar_baseline import SingleLidarPipeline
from preprocessing.filters import PCD_DTYPE, validate_pointcloud2_layout


class SingleLidarPerceptionNode(Node):
    """Subscribes to exactly one PointCloud2 topic and preserves its frame_id."""

    def __init__(self) -> None:
        super().__init__("single_lidar_perception")
        self.declare_parameter("input_topic", "/wamv/sensors/lidars/forward_lidar_sensor/points")
        self.declare_parameter("output_prefix", "/perception/single_lidar")
        input_topic = self.get_parameter("input_topic").value
        prefix = self.get_parameter("output_prefix").value.rstrip("/")
        self.pipeline = SingleLidarPipeline()
        self._processed_frames = 0
        self.filtered_pub = self.create_publisher(PointCloud2, f"{prefix}/filtered_points", 10)
        self.detections_pub = self.create_publisher(PoseArray, f"{prefix}/detections", 10)
        self.bev_pub = self.create_publisher(Image, f"{prefix}/bev", 10)
        self.status_pub = self.create_publisher(String, f"{prefix}/status", 10)
        self.subscription = self.create_subscription(PointCloud2, input_topic, self._callback, 10)
        self.get_logger().info(f"single-LiDAR baseline active for {input_topic}; no transforms or fusion are used")

    @staticmethod
    def _pointcloud(header, xyz: np.ndarray) -> PointCloud2:
        message = PointCloud2()
        message.header = header
        message.height = 1
        message.width = len(xyz)
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        message.is_bigendian = False
        message.point_step = 12
        message.row_step = 12 * len(xyz)
        message.data = np.asarray(xyz, dtype=np.float32).tobytes()
        message.is_dense = True
        return message

    def _callback(self, message: PointCloud2) -> None:
        try:
            validate_pointcloud2_layout(message.point_step, message.row_step, message.width, message.height,
                                        message.is_bigendian, message.fields)
            raw = np.frombuffer(bytes(message.data), dtype=PCD_DTYPE)
            stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
            result = self.pipeline.process(raw, self.get_parameter("input_topic").value, stamp_ns)
            self.filtered_pub.publish(self._pointcloud(message.header, result.filtered_xyz))
            poses = PoseArray(header=message.header)
            for detection in result.detections:
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = map(float, detection.center)
                pose.orientation.w = 1.0
                poses.poses.append(pose)
            self.detections_pub.publish(poses)
            bev = Image(header=message.header, height=result.bev.shape[0], width=result.bev.shape[1],
                        encoding="mono8", is_bigendian=0, step=result.bev.shape[1], data=result.bev.tobytes())
            self.bev_pub.publish(bev)
            self.status_pub.publish(String(data=json.dumps({
                "scope": "single_lidar_only", "frame_id": message.header.frame_id,
                "detections": len(result.detections), "tracks": len(result.tracks), "timings_ms": result.timings_ms,
            })))
            self._processed_frames += 1
            if self._processed_frames == 1 or self._processed_frames % 20 == 0:
                self.get_logger().info(
                    f"processed={self._processed_frames} frame_id={message.header.frame_id} "
                    f"detections={len(result.detections)} tracks={len(result.tracks)}"
                )
        except Exception as exc:
            self.get_logger().warning(f"frame rejected; publishing no cross-sensor fallback: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SingleLidarPerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
