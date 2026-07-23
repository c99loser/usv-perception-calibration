#!/usr/bin/env bash
# One reproducible static controlled-calibration capture.
# The launch runs in its own process group so every simulator child is removed.
set -eo pipefail
RUN_DIR="${1:?Usage: $0 /absolute/output/run_directory [seconds]}"
DURATION="${2:-20}"
LAYOUT_INDEX="${3:-0}"
WORLD="${4:-perception_user}"
PROJECT_ROOT="/mnt/f/Challenger program"
source /opt/ros/jazzy/setup.bash
source "$HOME/USVSim/install/setup.bash"
mkdir -p "$RUN_DIR"
setsid ros2 launch vrx_gz competition.launch.py world:="$WORLD" headless:=true \
  > "$RUN_DIR/launch.log" 2>&1 &
LAUNCH_PGID=$!
cleanup() {
  kill -INT -- "-$LAUNCH_PGID" 2>/dev/null || true
  wait "$LAUNCH_PGID" 2>/dev/null || true
}
trap cleanup EXIT
for _ in {1..60}; do
  sleep 1
  if ros2 topic list 2>/dev/null | grep -q '/wamv/sensors/lidars/front_lidar_sensor/points'; then
    break
  fi
done
if ! ros2 topic list 2>/dev/null | grep -q '/wamv/sensors/lidars/front_lidar_sensor/points'; then
  echo "Timed out waiting for official perception sensor topics" >&2
  exit 1
fi
WAMV_X=-710
WAMV_Y=350
if [[ $(grep -c 'WAM-V:' "$RUN_DIR/launch.log") -gt 0 ]]; then
  WAMV_LINE=$(grep -m1 'WAM-V:' "$RUN_DIR/launch.log")
  if [[ $WAMV_LINE =~ \((-?[0-9]+),(-?[0-9]+)\) ]]; then
    WAMV_X=${BASH_REMATCH[1]}
    WAMV_Y=${BASH_REMATCH[2]}
  fi
fi
echo "Controlled-target reference WAM-V: ($WAMV_X,$WAMV_Y)" | tee "$RUN_DIR/target_reference.txt"
if [[ "$WORLD" == "perception_user" ]]; then
  bash "$PROJECT_ROOT/Ubantu/spawn_controlled_calibration_targets.sh" "$RUN_DIR" "$LAYOUT_INDEX" "$WAMV_X" "$WAMV_Y"
else
  echo "Targets are embedded in $WORLD world SDF" > "$RUN_DIR/target_reference.txt"
fi
timeout 3s gz topic -e -n 1 -t "/world/$WORLD/pose/info" > "$RUN_DIR/entity_poses.txt" 2>&1 || true
grep -A8 -B2 'calib_' "$RUN_DIR/entity_poses.txt" > "$RUN_DIR/calibration_target_poses.txt" || true
sleep 3
timeout "${DURATION}s" bash "$PROJECT_ROOT/Ubantu/record_controlled_calibration.sh" "$RUN_DIR/bag" || {
  status=$?
  [[ $status == 124 ]] || exit "$status"
}
ros2 bag info "$RUN_DIR/bag" | tee "$RUN_DIR/bag_info.txt"
python3 "$PROJECT_ROOT/tools/controlled_calibration/create_capture_manifest.py" \
  --run-dir "$RUN_DIR" --layout-index "$LAYOUT_INDEX" --world "$WORLD"
