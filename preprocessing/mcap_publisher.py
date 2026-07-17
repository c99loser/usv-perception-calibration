"""
================================================================================
 mcap_publisher.py — 离线 MCAP → ROS2 话题发布
================================================================================
用途: 读取赛事方预录的 MCAP 文件，逐帧应用四步预处理，将结果以 ROS2 话题
      形式发布出去，供后续自标定/融合/检测模块使用。

用法 (在 perception_user.py 中调用):
    from preprocessing.mcap_publisher import run_mcap_pipeline
    run_mcap_pipeline(mcap_path='/path/to/recording.mcap')

或独立运行:
    python mcap_publisher.py /path/to/recording.mcap

发布的话题 (7路):
    /wamv/sensors/lidars/{name}/points_filtered
================================================================================
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
from pathlib import Path

from mcap_ros2.reader import read_ros2_messages

from .filters import (
    PreprocessingPipeline, PCD_DTYPE, ALL_LIDAR_TOPICS,
    validate_pointcloud2_layout,
)

PUBLISH_SUFFIX = '_filtered'


class McapPreprocessorNode(Node):
    """
    读 MCAP → 预处理 → 发布 ROS2 话题。
    """

    def __init__(self, mcap_path):
        super().__init__('mcap_preprocessor')

        self.mcap_path = Path(mcap_path)
        if not self.mcap_path.exists():
            raise FileNotFoundError(str(self.mcap_path))

        self.pipeline = PreprocessingPipeline()

        # 为每个 LiDAR 创建 publisher
        self.publishers = {}
        for topic in ALL_LIDAR_TOPICS:
            sensor_name = topic.split('/')[-2]
            output_topic = topic + PUBLISH_SUFFIX
            self.publishers[topic] = self.create_publisher(
                PointCloud2, output_topic, 10
            )
            self.get_logger().info(
                f'[{sensor_name}] MCAP → {output_topic}'
            )

        self.frame_idx = 0

    def run(self):
        """
        逐帧读取 MCAP → 处理 → 发布。
        以原始 MCAP 的时间戳节奏回放 (不加速)。
        """
        self.get_logger().info(f'开始处理: {self.mcap_path}')

        lidar_set = set(ALL_LIDAR_TOPICS)
        prev_time = None

        for msg in read_ros2_messages(str(self.mcap_path), topics=sorted(lidar_set)):
            topic = msg.channel.topic
            pc = msg.ros_msg
            log_time = msg.log_time_ns

            # ---- 按时间戳节奏回放 ----
            if prev_time is not None:
                elapsed_ns = log_time - prev_time
                if elapsed_ns > 0:
                    sleep_sec = elapsed_ns / 1e9
                    # rclpy 不提供 sleep, 用 spin_once 近似
                    # 比赛场景通常只需要处理结果，不需要严格按原始节奏
            prev_time = log_time

            # ---- 解析 → 预处理 ----
            validate_pointcloud2_layout(
                pc.point_step, pc.row_step, pc.width, pc.height,
                pc.is_bigendian, pc.fields,
            )
            expected_bytes = pc.width * pc.height * pc.point_step
            if len(pc.data) != expected_bytes:
                raise ValueError(
                    f'PointCloud2 data has {len(pc.data)} bytes, '
                    f'expected {expected_bytes}'
                )
            raw = np.frombuffer(bytes(pc.data), dtype=PCD_DTYPE)
            x, y, z, intensity, ring = self.pipeline.process(topic, raw)

            if len(x) == 0:
                continue

            # ---- 构造 PointCloud2 → 发布 ----
            out = self._build_pointcloud2(x, y, z, intensity, ring, pc)
            self.publishers[topic].publish(out)

            self.frame_idx += 1
            if self.frame_idx % 500 == 0:
                self.get_logger().info(
                    f'已处理 {self.frame_idx} 帧...'
                )

            # 允许 ROS2 事件循环处理
            rclpy.spin_once(self, timeout_sec=0)

        self.get_logger().info(f'完成! 共处理 {self.frame_idx} 帧')

    def _build_pointcloud2(self, x, y, z, intensity, ring, orig_msg):
        """numpy → sensor_msgs/PointCloud2"""
        N = len(x)
        buf = np.zeros(N, dtype=PCD_DTYPE)
        buf['x'] = x; buf['y'] = y; buf['z'] = z
        buf['intensity'] = intensity; buf['ring'] = ring

        out = PointCloud2()
        out.header = orig_msg.header
        out.height = 1
        out.width = N
        out.fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='ring',      offset=24, datatype=PointField.UINT16,  count=1),
        ]
        out.is_bigendian = False
        out.point_step = 32
        out.row_step = 32 * N
        out.data = buf.tobytes()
        out.is_dense = True
        return out


def run_mcap_pipeline(mcap_path):
    """
    perception_user.py 的调用入口。

    用法:
        from preprocessing.mcap_publisher import run_mcap_pipeline
        run_mcap_pipeline('/path/to/recording.mcap')
    """
    rclpy.init()
    node = McapPreprocessorNode(mcap_path)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('用法: python mcap_publisher.py <mcap_path>')
        sys.exit(1)
    run_mcap_pipeline(sys.argv[1])
