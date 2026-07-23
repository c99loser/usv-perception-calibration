#!/usr/bin/env bash
# Spawn controlled targets after the official perception_user scene is running.
set -eo pipefail
PROJECT_ROOT="/mnt/f/Challenger program"
RUN_DIR="${1:?Usage: $0 /absolute/output/run_directory}"
LAYOUT_INDEX="${2:-0}"
WAMV_X="${3:--710}"
WAMV_Y="${4:-350}"
source /opt/ros/jazzy/setup.bash
source "$HOME/USVSim/install/setup.bash"
python3 "$PROJECT_ROOT/tools/controlled_calibration/generate_targets.py" --output "$RUN_DIR/targets" --layout-index "$LAYOUT_INDEX" --wamv-x "$WAMV_X" --wamv-y "$WAMV_Y"
while IFS=, read -r target x y z yaw; do
  ros2 run ros_gz_sim create -world perception_user -name "calib_${target}" \
    -x "$x" -y "$y" -z "$z" -Y "$yaw" \
    -file "$RUN_DIR/targets/target_${target}.sdf"
done < "$RUN_DIR/targets/spawn.csv"
echo "Spawned six controlled targets. Manifest: $RUN_DIR/targets/targets.json"
