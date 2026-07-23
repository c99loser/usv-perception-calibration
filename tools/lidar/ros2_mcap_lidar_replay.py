#!/usr/bin/env python3
"""Minimal, deterministic ROS 2 replay of one PointCloud2 topic from an MCAP bag.

It is a test fixture, not a production input.  It publishes the original CDR
messages unchanged, so an online node receives exactly the recorded
PointCloud2 layout and header/frame_id.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import PointCloud2


class McapTopicReplay(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("mcap_single_lidar_replay")
        self.publisher = self.create_publisher(PointCloud2, topic, 10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--max-messages", type=int, default=20)
    parser.add_argument("--rate-hz", type=float, default=2.0)
    args = parser.parse_args()
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(args.bag), storage_id="mcap"),
                ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    rclpy.init(); node = McapTopicReplay(args.topic)
    try:
        # DDS discovery needs a short quiet period before the first message.
        time.sleep(2.0)
        sent = 0
        while reader.has_next() and sent < args.max_messages:
            topic, data, _ = reader.read_next()
            if topic != args.topic:
                continue
            node.publisher.publish(deserialize_message(data, PointCloud2))
            rclpy.spin_once(node, timeout_sec=0.0)
            sent += 1
            time.sleep(1.0 / args.rate_hz)
        node.get_logger().info(f"published {sent} recorded PointCloud2 messages from {args.bag}")
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
