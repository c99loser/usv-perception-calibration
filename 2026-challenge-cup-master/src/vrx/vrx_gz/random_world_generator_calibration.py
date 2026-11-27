#!/usr/bin/env python3
"""Generate a dedicated, static controlled-calibration world."""
import sys
from random_world_generator_perception import gen_world

if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out = sys.argv[2] if len(sys.argv) > 2 else '/tmp/calibration_user.sdf'
    layout = seed % 3
    gen_world(-710.0, 350.0, 0.0, out, world_name='calibration_user', calibration=True, calibration_layout=layout)
    print('WAMV=-710,350,0,0,0,0')
    print(f'WORLD={out}')
    print('CALIBRATION_TARGETS=6')
    print(f'CALIBRATION_LAYOUT={layout}')
