#!/usr/bin/env bash
# Apply the predeclared conservative semantic-neighborhood rule to six
# simulation-only LiDAR / camera pairs.  Runtime code never calls this script.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 <six-pair-root> <intrinsics-json> [semantic-neighborhood-px] [output-tag]" >&2
  exit 2
fi
root=$(realpath "$1")
intrinsics=$(realpath "$2")
script_dir=$(cd "$(dirname "$0")" && pwd)
neighborhood=${3:-10}
tag=${4:-near10}
[[ "$tag" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Unsafe output tag: $tag" >&2; exit 2; }
declare -A yaw=( [front]=0 [back]=180 [left]=90 [left2]=90 [right]=-90 [right2]=-90 )

for camera in front back left left2 right right2; do
  out="$root/${camera}_0001/labeled_points_${tag}.npz"
  if [[ -e "$out" ]]; then
    echo "Keeping existing label artifact: $out"
    continue
  fi
  python3 "$script_dir/label_sim_lidar_points.py" \
    --pair "$root/${camera}_0001" --intrinsics "$intrinsics" --camera "$camera" --output "$out" \
    --camera-rpy-deg 0 25 "${yaw[$camera]}" --lidar-rpy-deg 0 115 "${yaw[$camera]}" \
    --lidar-minus-camera-m 0 0 -0.1 --projection opencv_fisheye --semantic-neighborhood-px "$neighborhood"
done
