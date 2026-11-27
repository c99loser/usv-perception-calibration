#!/usr/bin/env python3
"""障碍船推进器随机游走节点。

通过向每艘障碍船的左/右推进器发送 cmd_thrust 推力指令,
实现真实的物理动力学游走(有惯性、转向、速度连续变化)。

位姿回读: gz transport subscribe (独立内部线程回调, 稳定)
推力发布: gz transport advertise
控制逻辑: rclpy 定时器 (_tick)

控制原理(差速推进器):
  - 左右推力相同 → 直线前进
  - 左右推力不同 → 转向
  - 朝随机目标方向 + 15m范围约束

ROS 参数:
  - world: 世界名(默认 berthing_test)
  - wander_radius: 游走半径(米, 默认 15)
  - thrust_fwd: 前进推力(默认 8)
  - thrust_turn: 转向推力差(默认 6)
  - update_rate: 推力更新频率(Hz, 默认 5)
"""
import math
import random
import threading
import time

import rclpy
from rclpy.node import Node

import gz.transport13 as gzt
import gz.msgs10.double_pb2 as double_pb
import gz.msgs10.pose_v_pb2 as pose_v_pb


class BoatWander(Node):
    def __init__(self):
        super().__init__('boat_wander')
        self.declare_parameter('world', 'berthing_test')
        self.declare_parameter('wander_radius', 15.0)
        self.declare_parameter('thrust_fwd', 8.0)
        self.declare_parameter('thrust_turn', 6.0)
        self.declare_parameter('update_rate', 5.0)

        self._radius = self.get_parameter('wander_radius').value
        self._thrust_fwd = self.get_parameter('thrust_fwd').value
        self._thrust_turn = self.get_parameter('thrust_turn').value
        rate = self.get_parameter('update_rate').value
        self._world = self.get_parameter('world').value

        # gz transport: 一个 Node 同时 subscribe(位姿) + advertise(推力)
        # gz 的 subscribe 回调在其内部线程触发, 不阻塞 rclpy
        self._gz = gzt.Node()
        self._pubs = {}  # boat_name -> (left_pub, right_pub)

        # 船状态(线程安全: gz回调线程写, rclpy tick线程读)
        self._lock = threading.Lock()
        self._boats = {}  # name -> {cx,cy,x,y,yaw,tx,ty,last_switch}
        self._wandering = False

        # gz subscribe 位姿(持续回读, 内部线程回调)
        ok = self._gz.subscribe(
            pose_v_pb.Pose_V,
            f'/world/{self._world}/dynamic_pose/info',
            self._gz_pose_cb)
        self.get_logger().info(f'订阅 gz 位姿: ok={ok}')

        # rclpy 定时器: 延迟启动游走 + 定时发推力
        self.create_timer(2.0, self._try_start)
        self._tick_timer = None
        self.get_logger().info(
            f'boat_wander(推进器模式) 启动: radius={self._radius}m '
            f'thrust={self._thrust_fwd}/{self._thrust_turn}')

    def _gz_pose_cb(self, msg):
        """gz 内部线程回调: 持续更新所有 boat 的位姿。"""
        with self._lock:
            for pose in msg.pose:
                name = pose.name
                if not name or not name.startswith('boat'):
                    continue
                o = pose.orientation
                yaw = math.atan2(
                    2.0 * (o.w * o.z + o.x * o.y),
                    1.0 - 2.0 * (o.y * o.y + o.z * o.z))
                if name not in self._boats:
                    # 首次: 记录起始位置作为游走中心
                    self._boats[name] = {
                        'cx': pose.position.x, 'cy': pose.position.y,
                        'x': pose.position.x, 'y': pose.position.y, 'yaw': yaw,
                        'tx': pose.position.x, 'ty': pose.position.y,
                        'last_switch': 0.0,
                    }
                else:
                    self._boats[name]['x'] = pose.position.x
                    self._boats[name]['y'] = pose.position.y
                    self._boats[name]['yaw'] = yaw

    def _try_start(self):
        """rclpy 定时器: gz 数据到达后启动游走。"""
        with self._lock:
            if self._wandering or len(self._boats) == 0:
                return
            for st in self._boats.values():
                self._new_target(st)
            self._wandering = True
            n = len(self._boats)
        rate = self.get_parameter('update_rate').value
        self._tick_timer = self.create_timer(1.0 / max(rate, 0.1), self._tick)
        self.get_logger().info(f'障碍船开始推进器游走 ({n}艘)')

    def _ensure_pubs(self, name):
        if name in self._pubs:
            return self._pubs[name]
        lt = f'/model/{name}/joint/left_engine_propeller_joint/cmd_thrust'
        rt = f'/model/{name}/joint/right_engine_propeller_joint/cmd_thrust'
        self._pubs[name] = (
            self._gz.advertise(lt, double_pb.Double),
            self._gz.advertise(rt, double_pb.Double))
        return self._pubs[name]

    def _new_target(self, st):
        """在起始点 ±0.6*radius 范围内选随机目标。"""
        ang = random.uniform(0, 2 * math.pi)
        r = random.uniform(self._radius * 0.2, self._radius * 0.6)
        st['tx'] = st['cx'] + r * math.cos(ang)
        st['ty'] = st['cy'] + r * math.sin(ang)

    @staticmethod
    def _angle_diff(a, b):
        return (a - b + math.pi) % (2 * math.pi) - math.pi

    def _tick(self):
        """rclpy 定时器: 读位姿 + 计算推力 + 发布。"""
        with self._lock:
            if not self._wandering or not self._boats:
                return
            boats = {n: dict(s) for n, s in self._boats.items()}  # 快照

        now = time.time()
        for name, st in boats.items():
            dc = math.hypot(st['x'] - st['cx'], st['y'] - st['cy'])
            # 范围约束: 接近边界 → 拉回中心
            if dc > self._radius * 0.85:
                st['tx'] = st['cx']
                st['ty'] = st['cy']
                with self._lock:
                    if name in self._boats:
                        self._boats[name]['tx'] = st['cx']
                        self._boats[name]['ty'] = st['cy']
            # 定期切换目标(在范围内时)
            elif now - st['last_switch'] > random.uniform(5.0, 10.0) and dc < self._radius * 0.5:
                ang = random.uniform(0, 2 * math.pi)
                r = random.uniform(self._radius * 0.2, self._radius * 0.5)
                ntx = st['cx'] + r * math.cos(ang)
                nty = st['cy'] + r * math.sin(ang)
                with self._lock:
                    if name in self._boats:
                        self._boats[name]['tx'] = ntx
                        self._boats[name]['ty'] = nty
                        self._boats[name]['last_switch'] = now

            # 朝目标方向
            dx = st['tx'] - st['x']
            dy = st['ty'] - st['y']
            target_yaw = math.atan2(dy, dx)
            yaw_err = self._angle_diff(target_yaw, st['yaw'])

            # 差速控制
            turn = max(-1.0, min(1.0, yaw_err / math.pi * 2))
            left = self._thrust_fwd - turn * self._thrust_turn
            right = self._thrust_fwd + turn * self._thrust_turn
            # 拉回时加大推力
            if dc > self._radius * 0.85:
                left = right = self._thrust_fwd * 1.5

            lp, rp = self._ensure_pubs(name)
            lm = double_pb.Double(); lm.data = left; lp.publish(lm)
            rm = double_pb.Double(); rm.data = right; rp.publish(rm)


def main():
    rclpy.init()
    node = BoatWander()
    rclpy.spin(node)
    for name, (lp, rp) in node._pubs.items():
        z = double_pb.Double(); z.data = 0.0
        lp.publish(z); rp.publish(z)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
