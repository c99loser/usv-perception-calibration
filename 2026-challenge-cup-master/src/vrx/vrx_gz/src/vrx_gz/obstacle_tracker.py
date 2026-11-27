#!/usr/bin/env python3
"""障碍物跟踪节点 — 按类型分话题发布。

ROS 输出:
  /world/obstacles/pillars   PoseArray (1Hz)  — 立柱位置
  /world/obstacles/buoys     PoseArray (3Hz)  — 浮球位置
  /world/obstacles/boats     PoseArray (3Hz)  — 游艇位置

速度通过相邻两帧位移差计算: v = Δpos / Δt
"""

import math, os, re, time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from std_msgs.msg import Header
import gz.transport13 as gzt
import gz.msgs10.pose_v_pb2 as pose_v_pb


class ObstacleTracker(Node):
    def __init__(self):
        super().__init__('obstacle_tracker')
        self.declare_parameter('world', 'berthing_test')
        self.declare_parameter('world_sdf', '')
        self.declare_parameter('dock_x', -533.0)
        self.declare_parameter('dock_y', 162.0)
        self.declare_parameter('dock_yaw', 0.0)
        world = self.get_parameter('world').value

        self._states = {}
        self._prev_states = {}
        self._pillars = []

        self._gz = gzt.Node()
        topic = f'/world/{world}/dynamic_pose/info'
        ok = self._gz.subscribe(pose_v_pb.Pose_V, topic, self._gz_cb)
        if ok:
            self.get_logger().info(f'订阅 gz: {topic}')
        else:
            self.get_logger().error(f'订阅失败: {topic}')

        self._load_pillars()

        self.pillar_pub = self.create_publisher(PoseArray, '/world/obstacles/pillars', 10)
        self.buoy_pub   = self.create_publisher(PoseArray, '/world/obstacles/buoys', 10)
        self.boat_pub   = self.create_publisher(PoseArray, '/world/obstacles/boats', 10)
        self.block_pub  = self.create_publisher(PoseArray, '/world/obstacles/blocks', 10)
        self.dock_pub   = self.create_publisher(PoseStamped, '/world/obstacles/dock', 10)

        self.create_timer(1.0, self._publish_pillars)
        self.create_timer(1.0 / 3.0, self._publish_dynamic)
        self.create_timer(2.0, self._publish_dock)

    def _load_pillars(self):
        sdf_path = self.get_parameter('world_sdf').value
        if not sdf_path or not os.path.exists(sdf_path):
            self.get_logger().warn('SDF 未找到, 立柱不可用')
            return
        try:
            with open(sdf_path) as f:
                content = f.read()
            p = re.compile(
                r'<model\s+name="(pillar_[^"]+)">[^<]*<static>true</static>\s*'
                r'<pose>([\-\d.]+)\s+([\-\d.]+)\s+([\-\d.]+)')
            self._pillars = [(float(m.group(2)), float(m.group(3)), float(m.group(4)))
                             for m in p.finditer(content)]
            self.get_logger().info(f'立柱 {len(self._pillars)} 个')
        except Exception as e:
            self.get_logger().error(f'解析 SDF 失败: {e}')

    def _gz_cb(self, msg):
        stamp = msg.header.stamp
        t = stamp.sec + getattr(stamp, 'nsec', getattr(stamp, 'nanosec', 0)) * 1e-9
        new_states = {}
        for pose in msg.pose:
            name = pose.name
            if not name:
                continue
            new_states[name] = {
                'pos': (pose.position.x, pose.position.y, pose.position.z),
                'quat': (pose.orientation.x, pose.orientation.y,
                         pose.orientation.z, pose.orientation.w),
                't': t,
            }
        self._prev_states = self._states
        self._states = new_states

    def _publish_pillars(self):
        if not self._pillars:
            return
        now = self.get_clock().now().to_msg()
        pa = PoseArray(header=Header(stamp=now, frame_id='map'))
        for x, y, z in self._pillars:
            p = Pose(); p.position.x, p.position.y, p.position.z = x, y, z
            p.orientation.w = 1.0
            pa.poses.append(p)
        self.pillar_pub.publish(pa)

    def _publish_dynamic(self):
        now = self.get_clock().now().to_msg()
        for prefix, pub in [('boat', self.boat_pub), ('buoy', self.buoy_pub),
                             ('block', self.block_pub)]:
            pool = {n: s for n, s in self._states.items() if n.startswith(prefix)}
            pa = PoseArray(header=Header(stamp=now, frame_id='map'))
            for name, st in sorted(pool.items()):
                x, y, z = st['pos']
                qx, qy, qz, qw = st['quat']
                p = Pose()
                p.position.x, p.position.y, p.position.z = x, y, z
                p.orientation.x, p.orientation.y = qx, qy
                p.orientation.z, p.orientation.w = qz, qw
                pa.poses.append(p)
            pub.publish(pa)

    def _publish_dock(self):
        """发布泊位位置, 2s 周期"""
        dx = self.get_parameter('dock_x').value
        dy = self.get_parameter('dock_y').value
        dyaw = self.get_parameter('dock_yaw').value
        now = self.get_clock().now().to_msg()
        msg = PoseStamped(header=Header(stamp=now, frame_id='map'))
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = dx, dy, 0.3
        msg.pose.orientation.z = math.sin(dyaw / 2)
        msg.pose.orientation.w = math.cos(dyaw / 2)
        self.dock_pub.publish(msg)


def main():
    rclpy.init()
    node = ObstacleTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
