#!/usr/bin/env python3
"""Publish a few valid PointCloud2 frames for the ROS adapter smoke test."""
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField

from preprocessing.filters import PCD_DTYPE


class Publisher(Node):
    def __init__(self):
        super().__init__("single_lidar_smoke_publisher")
        self.publisher = self.create_publisher(PointCloud2, "/wamv/sensors/lidars/back_lidar_sensor/points", 10)

    def frame(self, index):
        rng = np.random.default_rng(index)
        xyz = np.vstack((rng.normal((8.0 + index * 0.1, 2.0, 1.0), 0.08, (20, 3)),
                         rng.normal((13.0, -1.0, 1.2), 0.08, (20, 3))))
        raw = np.zeros(len(xyz), dtype=PCD_DTYPE)
        raw["x"], raw["y"], raw["z"], raw["intensity"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], 1.0
        msg = PointCloud2()
        msg.header.frame_id = "smoke_back_lidar_frame"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height, msg.width = 1, len(raw)
        msg.fields = [PointField(name="x", offset=0, datatype=7, count=1), PointField(name="y", offset=4, datatype=7, count=1), PointField(name="z", offset=8, datatype=7, count=1), PointField(name="intensity", offset=16, datatype=7, count=1), PointField(name="ring", offset=24, datatype=4, count=1)]
        msg.point_step, msg.row_step, msg.is_bigendian, msg.is_dense = 32, 32 * len(raw), False, True
        msg.data = raw.tobytes()
        return msg


def main():
    rclpy.init(); node = Publisher()
    try:
        time.sleep(2.0)  # allow DDS discovery before first publish
        for index in range(20):
            node.publisher.publish(node.frame(index)); rclpy.spin_once(node, timeout_sec=0.05); time.sleep(0.2)
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__": main()
