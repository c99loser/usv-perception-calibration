#!/usr/bin/env bash
# Training-only collector launch.  Never use this launch in a competition run.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <directory-containing-labelled-world-and-training-xacro>" >&2
  exit 2
fi
asset_dir=$(realpath "$1")
xacro_file="$asset_dir/wamv_large_training_semantic.urdf.xacro"
[[ -f "$xacro_file" ]] || { echo "Missing training WAM-V xacro: $xacro_file" >&2; exit 2; }

# Each synthetic split must use a separately generated labelled world.  The
# original hard-coded 20260722 filename silently reused one world for later
# seeds, invalidating the intended split.  Require exactly one copied world
# so the asset directory itself is the explicit training provenance.
world_files=("$asset_dir"/perception_seed_*_labelled.sdf)
[[ ${#world_files[@]} -eq 1 && -f "${world_files[0]}" ]] || {
  echo "Expected exactly one perception_seed_*_labelled.sdf in $asset_dir" >&2
  exit 2
}
world=$(basename "${world_files[0]}" .sdf)

echo "TRAINING ONLY: semantic topics are labels, never runtime inputs."
export GZ_SIM_RESOURCE_PATH="$asset_dir${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
gz sim -s -r "$asset_dir/$world.sdf" &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
sleep 5
python3 "$(dirname "$0")/spawn_training_wamv.py" --world perception_user --xacro "$xacro_file"
wait "$server_pid"
