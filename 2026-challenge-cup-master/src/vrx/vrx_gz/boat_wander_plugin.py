"""
Gazebo Python 系统插件 — 障碍船自主游荡。
在每个仿真步中随机调整推力，无需外部进程。
"""
import random
import math
import time

from gz.math7 import Vector3d
from gz.msgs.double_pb2 import Double
from gz.msgs.pose_pb2 import Pose
from gz.transport13 import Node


class BoatWander:
    """单个船的游荡控制器"""

    def __init__(self, model_name, safe_center, safe_radius):
        self.model = model_name
        self.safe_cx, self.safe_cy = safe_center
        self.safe_r = safe_radius
        self.left_cmd = 0.0
        self.right_cmd = 0.0
        self.last_update = time.time()
        self.update_interval = 2.0

    def update(self, px, py):
        """根据船当前位置决定推力"""
        now = time.time()
        if now - self.last_update < self.update_interval:
            return self.left_cmd, self.right_cmd
        self.last_update = now

        dist = math.hypot(px - self.safe_cx, py - self.safe_cy)
        if dist < self.safe_r:
            # 靠近安全区 — 加速远离
            thrust = 50 + 150 * (1 - dist / self.safe_r)
            self.left_cmd = self.right_cmd = thrust
        else:
            thrust = random.uniform(30, 150)
            self.left_cmd = thrust
            self.right_cmd = thrust * random.uniform(0.7, 1.3)
        return self.left_cmd, self.right_cmd


class WanderManager:
    """管理所有游荡船"""

    def __init__(self):
        self.boats = {}
        self.node = Node()
        self.last_pose_check = 0
        self.poses = {}  # 船位缓存

    def add_boat(self, name):
        self.boats[name] = BoatWander(name, (-710, 350), 40)

    def step(self):
        """每个仿真步调用"""
        now = time.time()
        # 每1秒更新一次船位
        if now - self.last_pose_check > 1.0:
            for name in self.boats:
                try:
                    # 通过 transport 获取位姿
                    req = Empty()
                    rep = Pose()
                    # 简化: 用缓存位置
                    pass
                except:
                    pass
            self.last_pose_check = now

        # 发送推力
        for name, boat in self.boats.items():
            left, right = boat.update(
                self.poses.get(name, (0, 0))[0],
                self.poses.get(name, (0, 0))[1])
            # 发布推力到模型topic
            msg = Double()
            msg.data = left
            self.node.publish(f'/model/{name}/thrusters/left/thrust', msg)
            msg2 = Double()
            msg2.data = right
            self.node.publish(f'/model/{name}/thrusters/right/thrust', msg2)
