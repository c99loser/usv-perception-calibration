#!/usr/bin/env bash
# Record only the permitted production sensor streams; truth/TF are excluded.
set -eo pipefail
OUT_DIR="${1:?Usage: $0 /absolute/output/bag_directory}"
source /opt/ros/jazzy/setup.bash
source "$HOME/USVSim/install/setup.bash"
# ros2 bag owns this path and rejects a pre-existing output directory.
if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite existing bag directory: $OUT_DIR" >&2
  exit 2
fi
exec ros2 bag record --storage mcap --output "$OUT_DIR" \
  /wamv/sensors/lidars/front_lidar_sensor/points \
  /wamv/sensors/lidars/back_lidar_sensor/points \
  /wamv/sensors/lidars/left_lidar_sensor/points \
  /wamv/sensors/lidars/left2_lidar_sensor/points \
  /wamv/sensors/lidars/right_lidar_sensor/points \
  /wamv/sensors/lidars/right2_lidar_sensor/points \
  /wamv/sensors/lidars/forward_lidar_sensor/points \
  /wamv/sensors/cameras/front_camera_sensor/image_raw \
  /wamv/sensors/cameras/back_camera_sensor/image_raw \
  /wamv/sensors/cameras/left_camera_sensor/image_raw \
  /wamv/sensors/cameras/left2_camera_sensor/image_raw \
  /wamv/sensors/cameras/right_camera_sensor/image_raw \
  /wamv/sensors/cameras/right2_camera_sensor/image_raw
