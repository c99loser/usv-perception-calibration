#!/usr/bin/env bash
# Training-only collection of one synchronized semantic image / LiDAR sample
# per camera.  This script is explicitly not a competition-runtime launch.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <training-assets-dir> <empty-output-dir>" >&2
  exit 2
fi
asset_dir=$(realpath "$1")
out_dir=$(realpath -m "$2")
[[ -d "$asset_dir" ]] || { echo "Missing training assets: $asset_dir" >&2; exit 2; }
[[ ! -e "$out_dir" || -d "$out_dir" ]] || { echo "Output path is not a directory: $out_dir" >&2; exit 2; }
script_dir=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out_dir"

"$script_dir/launch_training_semantic_world.sh" "$asset_dir" >"${TMPDIR:-/tmp}/six_lidar_semantic.log" 2>&1 &
sim_pid=$!
cleanup() { kill "$sim_pid" 2>/dev/null || true; wait "$sim_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
sleep 38

for camera in front back left left2 right right2; do
  if [[ -e "$out_dir/${camera}_0001" ]]; then
    echo "Keeping existing capture: $out_dir/${camera}_0001"
    continue
  fi
  root="/world/perception_user/model/wamv/link/wamv/base_link/sensor"
  python3 "$script_dir/capture_gz_lidar_semantic_pair.py" \
    --output-root "$out_dir/${camera}_0001" \
    --rgb-topic "$root/${camera}_camera_sensor/image" \
    --label-topic "/training/semantic/${camera}/labels_map" \
    --lidar-topic "$root/${camera}_lidar_sensor/scan/points" \
    --pose-topic "/world/perception_user/pose/info" \
    --max-sync-ms 50 --timeout-seconds 15
done
