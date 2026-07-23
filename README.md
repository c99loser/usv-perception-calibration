# Challenger Program Perception

[简体中文](README.zh-CN.md)

A ROS 2 perception project for an autonomous USV berthing challenge. At this stage, the repository provides a reproducible point-cloud baseline, a runnable single-LiDAR ROS 2 node, raw-fisheye training-data and simulation-label tooling, and a fail-closed calibration gate: unvalidated calibration never enables fusion.

The active direction is to train and validate object detection on six **raw fisheye images**, then advance LiDAR–camera self-calibration only from auditable natural-scene co-observations. See [当前进行路线.md](当前进行路线.md) for the detailed plan.

## Status

| Capability | Status | Scope |
| --- | --- | --- |
| Single-LiDAR detection, tracking, and BEV | Baseline complete | No TF, camera, or extrinsics required; supports offline replay and ROS 2 execution. |
| Offline MCAP perception baseline | Complete | PointCloud2 replay, filtering, clustering, geometric detection, tracking, and a 10 m BEV. |
| ROS 2 execution path | Validated | `competition_user` is the production-facing `ament_python` package. |
| Camera intrinsics and fisheye tools | Development version complete | Used for image-quality audit, auxiliary experiments, and training-data preparation; versions must be recorded. |
| Natural-scene calibration gate | Baseline complete | Audits LiDAR natural clusters and association candidates but does not publish extrinsics. |
| Raw-fisheye training-data pipeline | Complete | Temporal splitting, CVAT/COCO conversion, candidate evaluation, and lineage records. |
| Gazebo semantic training data | Collection tooling complete | RGB/semantic labels are restricted to offline training-data generation. |
| LiDAR–camera extrinsics and pixel-level fusion | Incomplete / disabled | Extrinsic state remains `nominal_unverified` until every gate passes. |

## Operating boundary

Production perception may consume only the seven competition LiDAR point-cloud streams and six fisheye image streams. World truth, `/tf`, odometry, simulator obstacle poses, and semantic labels are offline-only assets for training, evaluation, or debugging; they must never enter a runtime node.

The deployment domain for image detection is the original 1280×720 fisheye image. Undistorted images are only for quality inspection, traceable auxiliary experiments, or ablations; they are not the sole training domain or calibration geometry truth.

## Repository layout

| Path | Purpose |
| --- | --- |
| `competition_user/` | Competition-facing ROS 2 nodes, single-LiDAR pipeline, natural-calibration observations, and association gate. |
| `perception/` | ROS-daemon-free MCAP replay, detection, tracking, and BEV baseline. |
| `preprocessing/` | PointCloud2 parsing, filtering, downsampling, and streaming helpers. |
| `config/` | Default ROS 2 parameters. |
| `tools/lidar/` | MCAP audit, deterministic replay, ROS 2 replay validation, and report summaries. |
| `tools/camera_fisheye/` | Fisheye frame extraction, intrinsics export, correction, and MCAP validation. |
| `tools/annotation/`, `tools/yolo/` | Annotation export, candidate validation, COCO/YOLO conversion, and evaluation. |
| `tools/controlled_calibration/` | Controlled capture, intrinsics, ICP, and quality gates; development validation only. |
| `tools/simulation/` | Gazebo training worlds, semantic cameras, and RGB/label-pair collection. |
| `tests/` | Tests for offline perception, single LiDAR, natural calibration, and annotation tooling. |
| `Ubantu/` | WSL/Ubuntu environment and simulation helper scripts. |
| `step1/` | Historical material. When it conflicts with the active plan, this README and `当前进行路线.md` take precedence. |

The repository excludes raw MCAP files, large training data, model weights, and ROS build products. They remain local under `数据/`, `datasets/`, `output/`, `build/`, `install/`, and `log/`.

## Quick start

### 1. ROS 2 node (Ubuntu 24.04 / ROS 2 Jazzy)

Place the repository under the `src/` directory of a ROS 2 workspace:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/competition_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select competition_user
source install/setup.bash

ros2 run competition_user single_lidar_perception --ros-args \
  --params-file src/competition_user/config/single_lidar_baseline.yaml
```

The node works in the incoming `PointCloud2` `header.frame_id` and publishes cleaned points, detection centres, a local mono8 BEV, and status. See [config/single_lidar_baseline.yaml](config/single_lidar_baseline.yaml) and [README_单路LiDAR基线.md](README_单路LiDAR基线.md) for replay details.

### 2. Offline MCAP baseline

Use a Python environment that provides `numpy`, `scipy`, and `rosbags`:

```powershell
$env:PYTHONPATH = 'F:\Challenger program\.codex_pydeps'
python -m perception.cli inspect --data-root '数据'
python -m perception.cli replay --data-root '数据' --max-frames 20 --output-dir output\perception
```

`inspect` reads metadata only. `replay` writes per-frame BEV, detection/track JSON, and process reports without reading world-obstacle truth.

### 3. Single-LiDAR data audit

```powershell
$env:PYTHONPATH = 'F:\Challenger program\.codex_pydeps'
python tools/lidar/audit_mcap.py
```

`output/lidar_audit/audit.json` records rate, interval, point count, and abnormal frames by MCAP and LiDAR. `replay_manifest.json` defines fixed, reproducible replay frames.

## Validation

Run the offline tests with:

```powershell
$env:PYTHONPATH = 'F:\Challenger program\.codex_pydeps'
python -m unittest tests.test_perception -v
python -m pytest tests -q
```

For ROS 2 validation on WSL, run [Ubantu/setup_perception_wsl.sh](Ubantu/setup_perception_wsl.sh) first, then follow the real-MCAP replay procedure in [README_单路LiDAR基线.md](README_单路LiDAR基线.md).

## Training and calibration workflow

1. Create datasets split by bag and time segment to prevent adjacent-frame leakage.
2. Train or fine-tune raw-fisheye detection using manual annotations and simulation semantic images; report metrics per camera.
3. Feed only validated static image observations and stable LiDAR clusters into the association gate.
4. Allow extrinsics to become `accepted` only after multi-frame, holdout-frame, observability, and stability gates pass.
5. Enable pixel-level LiDAR–camera fusion only for `accepted`; otherwise keep independent-sensor perception.

Controlled checkerboards, ICP, simulator-truth projection, and HSV/contour proposals cannot directly justify extrinsic release. See [自标定前置工作状态.md](自标定前置工作状态.md) for details and known gaps.

## References

- [当前进行路线.md](当前进行路线.md): active execution plan and next-stage work.
- [数据与环境使用说明.md](数据与环境使用说明.md): data boundaries, simulation environment, and runtime constraints.
- [README_单路LiDAR基线.md](README_单路LiDAR基线.md): single-sensor replay, ROS 2 validation, and metric limits.
- [tools/camera_fisheye/README.md](tools/camera_fisheye/README.md): fisheye tooling.
- [自标定汇总/CALIB_TECH.md](自标定汇总/CALIB_TECH.md): calibration algorithms and quality gates.

## Next stage

First complete auditable manual annotations and independent bag-level evaluation for all six raw fisheye cameras, then train a detector that passes the per-camera gates. Only then proceed to natural-object 2D–3D multi-frame optimization. Simulation execution or visible single-sensor results do not substitute for calibration acceptance.
