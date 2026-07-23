#!/usr/bin/env bash
# Training-only collector launch.  Never use this launch in a competition run.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <directory-containing-labelled-world-and-training-xacro>" >&2
  exit 2
fi
asset_dir=$(realpath "$1")
world=perception_seed_20260722_labelled
xacro_file="$asset_dir/wamv_large_training_semantic.urdf.xacro"
[[ -f "$asset_dir/$world.sdf" && -f "$xacro_file" ]] || { echo "Missing training assets" >&2; exit 2; }

echo "TRAINING ONLY: semantic topics are labels, never runtime inputs."
export GZ_SIM_RESOURCE_PATH="$asset_dir${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
gz sim -s -r "$asset_dir/$world.sdf" &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
sleep 5
python3 "$(dirname "$0")/spawn_training_wamv.py" --world perception_user --xacro "$xacro_file"
wait "$server_pid"
