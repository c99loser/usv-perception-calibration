#!/usr/bin/env python3
"""
轻量游荡船 — 纯随机推力, 零轮询, 不卡。
"""
import subprocess, random, time, signal, sys

BOATS_P1 = ['boat1','boat2','boat3','boat4','boat5','boat6','boat7','boat8']
BOATS_P3 = ['boat20','boat21','boat22','boat23','boat24','boat25','boat26','boat27']
BOATS_P4 = ['boat30','boat31','boat32','boat33','boat34','boat35']

THRUST_LO, THRUST_HI = 0.8, 2.0   # P1小船: 小推力慢游荡
THRUST_FAR_LO, THRUST_FAR_HI = 2.0, 5.0  # P3/P4: 正常游荡

running = True
def sig_handler(sig, frame):
    global running; running = False
signal.signal(signal.SIGTERM, sig_handler)
signal.signal(signal.SIGINT, sig_handler)

print(f"Light wander: P1={len(BOATS_P1)} P3={len(BOATS_P3)} P4={len(BOATS_P4)} boats")

cycle = 0
while running:
    # P1: 小推力慢游荡
    for boat in BOATS_P1:
        left = random.uniform(THRUST_LO, THRUST_HI)
        right = left * random.uniform(0.5, 1.5)
        try:
            subprocess.run(['gz', 'topic', '-t', f'/{boat}/thrusters/left/thrust',
                '-m', 'gz.msgs.Double', '-p', f'data: {left:.1f}'],
                capture_output=True, timeout=1)
            subprocess.run(['gz', 'topic', '-t', f'/{boat}/thrusters/right/thrust',
                '-m', 'gz.msgs.Double', '-p', f'data: {right:.1f}'],
                capture_output=True, timeout=1)
        except:
            pass
    # P3+P4: 正常游荡
    for boat in BOATS_P3 + BOATS_P4:
        left = random.uniform(THRUST_FAR_LO, THRUST_FAR_HI)
        right = left * random.uniform(0.6, 1.4)
        try:
            subprocess.run(['gz', 'topic', '-t', f'/{boat}/thrusters/left/thrust',
                '-m', 'gz.msgs.Double', '-p', f'data: {left:.1f}'],
                capture_output=True, timeout=1)
            subprocess.run(['gz', 'topic', '-t', f'/{boat}/thrusters/right/thrust',
                '-m', 'gz.msgs.Double', '-p', f'data: {right:.1f}'],
                capture_output=True, timeout=1)
        except:
            pass
    cycle += 1
    if cycle % 10 == 0:
        print(f"  wander cycle {cycle}")
    time.sleep(2.0)

for boat in BOATS_P1 + BOATS_P3 + BOATS_P4:
    try:
        subprocess.run(['gz', 'topic', '-t', f'/{boat}/thrusters/left/thrust',
            '-m', 'gz.msgs.Double', '-p', 'data: 0'], capture_output=True, timeout=1)
        subprocess.run(['gz', 'topic', '-t', f'/{boat}/thrusters/right/thrust',
            '-m', 'gz.msgs.Double', '-p', 'data: 0'], capture_output=True, timeout=1)
    except:
        pass
print("Light wander stopped")
