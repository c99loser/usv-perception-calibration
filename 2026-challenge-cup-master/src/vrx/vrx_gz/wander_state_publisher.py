#!/usr/bin/env python3
"""
游荡船状态发布器 — 发布10艘船的位置、速度、航向到 gz topic。
用法: python3 wander_state_publisher.py
发布格式(gz.msgs.StringMsg): "x,y,yaw,vx,vy,speed,heading_deg"
"""

import subprocess
import time
import math
import signal
import sys

BOATS = ["w1","w2","w3","w4","w5","w6","boat1","boat2","boat3","boat4"]
RATE = 0.5  # 发布频率 2Hz

def gz_pose(boat):
    """通过 gz model --pose 获取世界坐标位姿"""
    try:
        result = subprocess.run(
            ["gz", "model", "-m", boat, "--pose"],
            capture_output=True, text=True, timeout=3)
        for line in result.stdout.splitlines():
            if "Pose" in line and "[" in line:
                # 格式: [-508.0 272.0 0.2] ...
                nums = []
                for token in line.replace("[","").replace("]","").split():
                    try:
                        nums.append(float(token))
                    except ValueError:
                        pass
                if len(nums) >= 6:
                    return nums[0], nums[1], nums[5]  # x, y, yaw
    except:
        pass
    return None, None, None

def gz_publish(topic, msg):
    subprocess.run(
        ["gz", "topic", "-t", topic, "-m", "gz.msgs.StringMsg",
         "-p", f"data: '{msg}'"],
        capture_output=True, timeout=2)

def yaw_from_quat(qw, qz):
    """从2D四元数提取 yaw (qw=cos(θ/2), qz=sin(θ/2))"""
    return 2 * math.atan2(qz, qw)

prev = {boat: {"x": 0, "y": 0, "t": 0} for boat in BOATS}

print(f"状态发布器启动 — {len(BOATS)}艘船, {1/RATE:.0f}Hz")
print(f"  topics: /wander/<boat>/state")

try:
    while True:
        t0 = time.time()
        for boat in BOATS:
            x, y, yaw = gz_pose(boat)
            if x is None:
                continue

            # 计算速度
            dt = t0 - prev[boat]["t"]
            vx = vy = speed = 0.0
            if dt > 0.01 and prev[boat]["t"] > 0:
                vx = (x - prev[boat]["x"]) / dt
                vy = (y - prev[boat]["y"]) / dt
                speed = math.sqrt(vx*vx + vy*vy)
            heading_deg = math.degrees(yaw) if yaw else 0

            prev[boat] = {"x": x, "y": y, "t": t0}

            msg = f"{x:.3f},{y:.3f},{yaw:.4f},{vx:.3f},{vy:.3f},{speed:.3f},{heading_deg:.1f}"
            gz_publish(f"/wander/{boat}/state", msg)

        elapsed = time.time() - t0
        sleep_time = max(0, RATE - elapsed)
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n状态发布器已停止。")
