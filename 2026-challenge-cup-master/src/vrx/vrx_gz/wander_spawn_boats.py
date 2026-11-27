#!/usr/bin/env python3
"""
靠泊仿真游荡脚本 — 6艘 spawn 区小船 + 3艘航道区障碍船。
通过 gz topic 直接向推进器发送推力指令。
随 ros2 launch 自动启动。
"""

import subprocess
import random
import time
import signal
import sys

# spawn 区游荡船 (WAM-V出生点附近)
SPAWN_BOATS = ["w1", "w2", "w3", "w4", "w5", "w6"]
# 航道区障碍船
CHANNEL_BOATS = ["boat1", "boat2", "boat3", "boat4"]

ALL_BOATS = SPAWN_BOATS + CHANNEL_BOATS

THRUST_FWD = 2.0
THRUST_BACK = -1.0
THRUST_TURN = 1.5

MIN_DURATION = 3.0
MAX_DURATION = 8.0

# 两区边界
ZONE_BOUNDS = {
    "spawn":   {"x_min": -520, "x_max": -496, "y_min": 244, "y_max": 276},
    "channel": {"x_min": -545, "x_max": -495, "y_min": 183, "y_max": 218},
}

def gz_thrust(boat, left, right):
    for side, thrust in [("left", left), ("right", right)]:
        cmd = ["gz", "topic", "-t", f"/{boat}/thrusters/{side}/thrust",
               "-m", "gz.msgs.Double", "-p", f"data: {thrust}"]
        subprocess.run(cmd, capture_output=True, timeout=2)

def gz_pose(boat):
    """通过 gz model --pose 获取世界坐标"""
    try:
        result = subprocess.run(
            ["gz", "model", "-m", boat, "--pose"],
            capture_output=True, text=True, timeout=3)
        for line in result.stdout.splitlines():
            if "Pose" in line and "[" in line:
                nums = []
                for token in line.replace("[","").replace("]","").split():
                    try:
                        nums.append(float(token))
                    except ValueError:
                        pass
                if len(nums) >= 2:
                    return nums[0], nums[1]
    except:
        pass
    return None, None

def stop_all():
    for boat in ALL_BOATS:
        gz_thrust(boat, 0, 0)
    print("\n所有游荡船已停止。")

def signal_handler(sig, frame):
    stop_all()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

BEHAVIORS = [
    ("forward",      THRUST_FWD,       THRUST_FWD),
    ("forward",      THRUST_FWD,       THRUST_FWD),
    ("reverse",      THRUST_BACK,      THRUST_BACK),
    ("turn_left",   -THRUST_TURN,      THRUST_TURN),
    ("turn_right",   THRUST_TURN,     -THRUST_TURN),
    ("gentle_left",  THRUST_FWD * 0.6, THRUST_FWD * 1.0),
    ("gentle_right", THRUST_FWD * 1.0, THRUST_FWD * 0.6),
    ("idle", 0, 0),
]

print(f"游荡船启动")
print(f"  spawn区({len(SPAWN_BOATS)}艘): {SPAWN_BOATS}  边界: x=[{ZONE_BOUNDS['spawn']['x_min']},{ZONE_BOUNDS['spawn']['x_max']}] y=[{ZONE_BOUNDS['spawn']['y_min']},{ZONE_BOUNDS['spawn']['y_max']}]")
print(f"  航道区({len(CHANNEL_BOATS)}艘): {CHANNEL_BOATS}  边界: x=[{ZONE_BOUNDS['channel']['x_min']},{ZONE_BOUNDS['channel']['x_max']}] y=[{ZONE_BOUNDS['channel']['y_min']},{ZONE_BOUNDS['channel']['y_max']}]")

states = {boat: {"behavior": "idle", "duration": 2.0, "start_time": time.time(),
                 "left": 0.0, "right": 0.0} for boat in ALL_BOATS}

# 确定每艘船的所属区域
def get_zone(boat):
    return "spawn" if boat in SPAWN_BOATS else "channel"

try:
    while True:
        for boat in ALL_BOATS:
            now = time.time()
            state = states[boat]

            if now - state["start_time"] >= state["duration"]:
                zone = get_zone(boat)
                bounds = ZONE_BOUNDS[zone]
                x, y = gz_pose(boat)

                near_left   = x is not None and x < bounds["x_min"] + 8
                near_right  = x is not None and x > bounds["x_max"] - 8
                near_bottom = y is not None and y < bounds["y_min"] + 8
                near_top    = y is not None and y > bounds["y_max"] - 8

                if near_left:
                    state["left"], state["right"] = THRUST_TURN, -THRUST_TURN
                elif near_right:
                    state["left"], state["right"] = -THRUST_TURN, THRUST_TURN
                elif near_bottom:
                    state["left"], state["right"] = THRUST_FWD, THRUST_FWD
                elif near_top:
                    state["left"], state["right"] = THRUST_BACK, THRUST_BACK
                else:
                    _, state["left"], state["right"] = random.choice(BEHAVIORS)

                state["duration"] = random.uniform(MIN_DURATION, MAX_DURATION)
                state["start_time"] = now
                gz_thrust(boat, state["left"], state["right"])

        time.sleep(1.0)

except KeyboardInterrupt:
    stop_all()
