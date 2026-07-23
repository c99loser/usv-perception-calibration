# Ubuntu / VS Code perception environment

Open the project through VS Code Remote-WSL, using distro `Ubuntu-24.04` and
path `/mnt/f/Challenger program`. The project tasks always source ROS 2 Jazzy
explicitly; they do not depend on a user's shell profile.

## Verified local baseline

- Ubuntu 24.04 and ROS 2 Jazzy are installed.
- `rosbag2_storage_mcap` opens the supplied raw MCAP directories.
- VS Code has Remote-WSL installed.

## One-time Ubuntu packages

Run these in an interactive Ubuntu terminal (sudo needs the local Ubuntu user
password):

```bash
sudo apt update
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-vcstool
sudo rosdep init
rosdep update
```

Optional VS Code extensions are `ms-python.python` and `ms-iot.vscode-ros`.

## Official platform prerequisite

The organizer-modified USVSim workspace is not present on this machine. After
obtaining and extracting the official package, its root must contain `src/`.
Set `USVSIM_WS` if it is not located at `$HOME/USVSim`, for example:

```bash
export USVSIM_WS=/path/to/USVSim
bash '/mnt/f/Challenger program/Ubantu/setup_perception_wsl.sh'
```

The script links this project as `src/competition_user` only when that package
is absent, then builds `vrx_gz`, its workspace dependencies, and
`competition_user`. It intentionally does not replace the organizer's USVSim
with upstream VRX: the PDF specifies a modified platform, custom fisheye lens,
and evaluation behaviour that upstream VRX does not prove equivalent.

## VS Code tasks

- `ROS2: verify perception environment` verifies MCAP support and a raw bag.
- `ROS2: replay raw perception bag (09:10)` exposes only recorded topics for
  local development.
- `ROS2: prepare and build official USVSim workspace` runs the guarded build
  script after the platform is supplied.

For deliverable perception code, do not depend on `/world/obstacles/*`, TF,
odom, CameraInfo, or overhead-camera topics. The platform PDF identifies those
as reference/evaluation data; production input is seven LiDAR point streams and
six fisheye image streams.
