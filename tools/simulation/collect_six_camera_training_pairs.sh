#!/usr/bin/env bash
# Collect exactly one synchronized raw-fisheye RGB / semantic-label pair per camera.
# Training data only; it deliberately has no ROS runtime bridge.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <training-assets-dir> <empty-output-pairs-dir> [samples-per-camera]" >&2; exit 2
fi
asset_dir=$(realpath "$1")
out_dir=$(realpath -m "$2")
samples=${3:-1}
[[ "$samples" =~ ^[1-9][0-9]*$ ]] || { echo "samples-per-camera must be positive" >&2; exit 2; }
[[ ! -e "$out_dir" ]] || { echo "Refusing to overwrite $out_dir" >&2; exit 2; }
mkdir -p "$out_dir"

"$(dirname "$0")/launch_training_semantic_world.sh" "$asset_dir" &
launcher_pid=$!
cleanup() { kill "$launcher_pid" 2>/dev/null || true; wait "$launcher_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
sleep 38
for camera in front back left left2 right right2; do
  rgb="/world/perception_user/model/wamv/link/wamv/base_link/sensor/${camera}_camera_sensor/image"
  labels="/training/semantic/${camera}/labels_map"
  for ((index = 1; index <= samples; index++)); do
    pair=$(printf '%s_%04d' "$camera" "$index")
    python3 "$(dirname "$0")/capture_gz_training_pair.py" \
      --output-root "$out_dir/$pair" --rgb-topic "$rgb" --label-topic "$labels" --timeout-seconds 12
  done
done
