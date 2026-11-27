#!/usr/bin/env python3
"""
碰撞检测器 — 监控 WAM-V 与浮标的碰撞。
订阅 gz contact topic，检测 WAM-V 是否撞到浮标。
用法: python3 collision_monitor.py
发布: /collision/wamv_buoy (gz.msgs.StringMsg) — 碰撞时发布浮标名
"""

import subprocess
import time
import json
import signal
import sys
import re

# 浮标名称模式
BUOY_PATTERNS = ["sp_", "gate_", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"]

def gz_publish(topic, msg):
    subprocess.run(
        ["gz", "topic", "-t", topic, "-m", "gz.msgs.StringMsg",
         "-p", f"data: '{msg}'"],
        capture_output=True, timeout=2)

def get_contacts():
    """获取当前接触列表"""
    try:
        result = subprocess.run(
            ["gz", "topic", "-e", "-t", "/vrx/contacts"],
            capture_output=True, text=True, timeout=2)
        return result.stdout
    except:
        return ""

def parse_contacts(text):
    """解析 contact 输出，提取碰撞对"""
    pairs = set()
    # 匹配 "collision1" 和 "collision2" 模式
    pattern = r"collision[12]\s*{\s*[^}]*?name:\s*\"([^\"]+)\""
    matches = re.findall(pattern, text)
    for i in range(0, len(matches)-1, 2):
        pairs.add((matches[i], matches[i+1]))
    return pairs

def is_buoy(name):
    for pat in BUOY_PATTERNS:
        if pat in name:
            return True
    return False

print("碰撞检测器启动 — 监控 WAM-V ↔ 浮标碰撞")
print(f"  浮标模式: {BUOY_PATTERNS}")
print(f"  topic: /collision/wamv_buoy")

last_contacts = set()

try:
    while True:
        text = get_contacts()
        if text:
            pairs = parse_contacts(text)
            new_pairs = pairs - last_contacts
            for c1, c2 in new_pairs:
                # 检查是否 WAM-V 与浮标碰撞
                if "wamv" in c1.lower() and is_buoy(c2):
                    buoy = c2.split("::")[0] if "::" in c2 else c2
                    msg = f"WAMV_COLLISION buoy={buoy} time={time.time():.1f}"
                    print(f"  ⚠ {msg}")
                    gz_publish("/collision/wamv_buoy", msg)
                elif "wamv" in c2.lower() and is_buoy(c1):
                    buoy = c1.split("::")[0] if "::" in c1 else c1
                    msg = f"WAMV_COLLISION buoy={buoy} time={time.time():.1f}"
                    print(f"  ⚠ {msg}")
                    gz_publish("/collision/wamv_buoy", msg)
            last_contacts = pairs
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n碰撞检测器已停止。")
