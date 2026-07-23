#!/usr/bin/env bash
# Project-local setup/check for the official USVSim perception workspace.
# Run this from VS Code opened through Remote-WSL (Ubuntu-24.04).
set -eo pipefail

ROS_SETUP=/opt/ros/jazzy/setup.bash
PROJECT_ROOT=/mnt/f/Challenger\ program
USVSIM_WS=${USVSIM_WS:-"$HOME/USVSim"}
OFFICIAL_VRX_SRC=${OFFICIAL_VRX_SRC:-"$PROJECT_ROOT/2026-challenge-cup-master/src/vrx"}
# ament_python has no Jazzy rosdep key; pycodestyle is only needed for lint
# tests, which this runtime-focused setup deliberately disables below.
ROSDEP_SKIP_KEYS="ament_python ament_cmake_pycodestyle"

if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ROS 2 Jazzy is missing: $ROS_SETUP" >&2
  exit 1
fi
source "$ROS_SETUP"
set -u

if ! command -v colcon >/dev/null; then
  cat >&2 <<'EOF'
colcon is not installed. Run once in Ubuntu:
  sudo apt update && sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-vcstool
EOF
  exit 2
fi

if [[ ! -d "$OFFICIAL_VRX_SRC" ]]; then
  cat >&2 <<EOF
Organizer VRX source was not found at: $OFFICIAL_VRX_SRC
Set OFFICIAL_VRX_SRC to the organizer-provided src/vrx directory, then rerun.
The project PDF requires the organizer's modified USVSim/VRX source; do not substitute upstream VRX as an equivalent test platform.
EOF
  exit 3
fi

mkdir -p "$USVSIM_WS/src"
ln -sfn "$OFFICIAL_VRX_SRC" "$USVSIM_WS/src/vrx"
ln -sfn "$PROJECT_ROOT" "$USVSIM_WS/src/competition_user"

cd "$USVSIM_WS"
if command -v rosdep >/dev/null && [[ -f src/competition_user/package.xml ]]; then
  if ! rosdep db >/dev/null 2>&1; then
    cat >&2 <<'EOF'
rosdep has not been initialized. Run once in an interactive Ubuntu terminal:
  sudo rosdep init
  rosdep update
EOF
    exit 4
  fi
  # rosdep invokes sudo only when it needs to install system packages.  In a
  # non-interactive VS Code task, a password prompt would otherwise appear to
  # hang indefinitely.  Report the exact interactive command instead.
  if sudo -n true 2>/dev/null; then
    rosdep install --from-paths src --ignore-src --skip-keys "$ROSDEP_SKIP_KEYS" -r -y
  elif ! rosdep check --from-paths src --ignore-src --skip-keys "$ROSDEP_SKIP_KEYS" >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Additional ROS/Gazebo system dependencies are required. Run this once in an
interactive Ubuntu terminal (enter your sudo password), then rerun this script:
  cd "$HOME/USVSim"
  sudo rosdep install --from-paths src --ignore-src --skip-keys "$ROSDEP_SKIP_KEYS" -r -y
EOF
    exit 5
  fi
fi

colcon build --merge-install --symlink-install --packages-up-to vrx_gz competition_user \
  --cmake-args -DBUILD_TESTING=OFF
echo
echo "Build succeeded. In this terminal, run:"
echo "  source '$USVSIM_WS/install/setup.bash'"
echo "  ros2 run competition_user perception_user"
echo "  ros2 launch vrx_gz competition.launch.py world:=perception_user headless:=true"
