#!/usr/bin/env bash
# Training-only smoke test; run from repository root in the WSL Gazebo environment.
set -euo pipefail

result_dir="output/simulation_preflight_pair"
if [[ -e "$result_dir" ]]; then
  echo "Refusing to overwrite existing $result_dir" >&2
  exit 2
fi
gz sim -s -r --headless-rendering tools/simulation/fisheye_segmentation_preflight.sdf >/tmp/fisheye_segmentation_preflight.log 2>&1 &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT
sleep 5
python3 tools/simulation/capture_gz_training_pair.py --output-root "$result_dir"
