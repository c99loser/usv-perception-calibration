#!/usr/bin/env bash
set -euo pipefail

# Verify corrected camera MCAP bags with ROS 2 inside WSL/Ubuntu.
# Usage:
#   bash tools/camera_fisheye/verify_ros2_corrected_mcaps.sh
#   ROS_DISTRO=jazzy bash tools/camera_fisheye/verify_ros2_corrected_mcaps.sh
#   bash tools/camera_fisheye/verify_ros2_corrected_mcaps.sh /mnt/e/challenger\ camera

ROOT_DIR="${1:-/mnt/e/challenger camera}"
REPORT_DIR="${REPORT_DIR:-/mnt/f/Challenger program/output/camera_fisheye/ros2_verify}"
REPORT_FILE="${REPORT_DIR}/ros2_corrected_mcap_verify.txt"

CAMERA_TOPICS=(
  "/wamv/sensors/cameras/front_camera_sensor/image_raw"
  "/wamv/sensors/cameras/back_camera_sensor/image_raw"
  "/wamv/sensors/cameras/left_camera_sensor/image_raw"
  "/wamv/sensors/cameras/left2_camera_sensor/image_raw"
  "/wamv/sensors/cameras/right_camera_sensor/image_raw"
  "/wamv/sensors/cameras/right2_camera_sensor/image_raw"
)

source_ros() {
  if [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
    set +u
    # shellcheck source=/dev/null
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    set -u
    return 0
  fi

  for distro in jazzy humble iron rolling; do
    if [[ -f "/opt/ros/${distro}/setup.bash" ]]; then
      set +u
      # shellcheck source=/dev/null
      source "/opt/ros/${distro}/setup.bash"
      set -u
      return 0
    fi
  done

  echo "ERROR: no ROS 2 setup.bash found under /opt/ros" >&2
  return 1
}

write_header() {
  mkdir -p "${REPORT_DIR}"
  {
    echo "# ROS2 corrected MCAP verification"
    echo "timestamp: $(date -Is)"
    echo "wsl_distro: ${WSL_DISTRO_NAME:-unknown}"
    echo "user: $(whoami)"
    echo "pwd: $(pwd)"
    echo "root_dir: ${ROOT_DIR}"
    echo
  } > "${REPORT_FILE}"
}

append_cmd() {
  local title="$1"
  shift
  {
    echo "## ${title}"
    echo "\$ $*"
  } >> "${REPORT_FILE}"
  "$@" >> "${REPORT_FILE}" 2>&1 || {
    local status=$?
    echo "command_failed: ${status}" >> "${REPORT_FILE}"
    return "${status}"
  }
  echo >> "${REPORT_FILE}"
}

main() {
  write_header
  source_ros

  append_cmd "ros2 version" bash -lc 'ros2 --help | head -30'

  shopt -s nullglob
  local bags=("${ROOT_DIR}"/*_fisheye_corrected)
  if (( ${#bags[@]} == 0 )); then
    echo "ERROR: no *_fisheye_corrected bag directories found in ${ROOT_DIR}" | tee -a "${REPORT_FILE}" >&2
    return 2
  fi

  for bag in "${bags[@]}"; do
    {
      echo "## bag: ${bag}"
      echo
    } >> "${REPORT_FILE}"

    append_cmd "ros2 bag info $(basename "${bag}")" ros2 bag info "${bag}"

    for topic in "${CAMERA_TOPICS[@]}"; do
      {
        echo "### topic check: ${topic}"
        ros2 bag info "${bag}" | grep -F "${topic}" || true
        echo
      } >> "${REPORT_FILE}"
    done
  done

  echo "Wrote ${REPORT_FILE}"
}

main "$@"
