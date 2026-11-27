#!/usr/bin/env bash
# Synthetic-only exact pillar labels from the generated world's collision mesh.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <six-pair-root> <labelled-world-sdf>" >&2
  exit 2
fi
root=$(realpath "$1")
world=$(realpath "$2")
script_dir=$(cd "$(dirname "$0")" && pwd)
for camera in front back left left2 right right2; do
  out="$root/${camera}_0001/labeled_points_worldpillar.npz"
  if [[ -e "$out" ]]; then
    echo "Keeping existing label artifact: $out"
    continue
  fi
  python3 "$script_dir/label_sim_lidar_from_world_geometry.py" \
    --pair "$root/${camera}_0001" --world "$world" --camera "$camera" --output "$out" --surface-tolerance-m 0.12
done
