# Challenger Program 感知项目

[English](README.md)

面向无人船自主靠泊赛事的 ROS 2 感知工程。本阶段已形成可离线复现的点云感知基线、可运行的单路 LiDAR ROS 2 节点、原始鱼眼训练数据与仿真语义数据生成工具，以及“未通过标定绝不融合”的失效关闭门禁。

当前工作主线是：以六路**原始鱼眼图像**训练和验证目标检测器，在可审计的自然场景共同观测基础上推进 LiDAR–相机自标定。详细计划见[当前进行路线.md](当前进行路线.md)。

## 当前状态

| 能力 | 状态 | 范围 |
| --- | --- | --- |
| 单路 LiDAR 检测、跟踪与 BEV | 已完成基础版本 | 不需要 TF、相机或外参；支持离线回放和 ROS 2 运行。 |
| 离线 MCAP 感知基线 | 已完成 | 支持 PointCloud2 回放、过滤、聚类、几何检测、跟踪及 10 m BEV。 |
| ROS 2 运行链路 | 已验证 | `competition_user` 是面向生产的 `ament_python` 包。 |
| 相机内参与鱼眼工具 | 已完成开发版本 | 用于图像质量审计、辅助实验与训练数据准备；必须记录版本。 |
| 自然场景标定门禁 | 已完成基础版本 | 可审计 LiDAR 自然簇和关联候选，但不发布外参。 |
| 原始鱼眼训练数据管线 | 已完成 | 支持时间划分、CVAT/COCO 转换、候选评估与血缘记录。 |
| Gazebo 语义训练数据 | 已完成采集工具 | RGB/语义标签严格限于离线训练数据生成。 |
| LiDAR–相机外参与像素级融合 | 未完成 / 禁用 | 外参状态在全部门禁通过前保持 `nominal_unverified`。 |

## 系统边界

生产感知只允许使用赛事规定的 7 路 LiDAR 点云和 6 路鱼眼图像。世界真值、`/tf`、里程计、仿真内部障碍物位姿与语义标签仅可用于离线训练、评估或调试，绝不能进入运行节点。

图像检测的部署域为原始 1280×720 鱼眼图。去畸变图仅用于质量检查、可追溯的辅助实验或消融，不能成为唯一训练域或标定几何真值。

## 目录导航

| 路径 | 内容 |
| --- | --- |
| `competition_user/` | 竞赛侧 ROS 2 节点、单 LiDAR 管线、自然标定观测与关联门禁。 |
| `perception/` | 不依赖 ROS 守护进程的 MCAP 回放、检测、跟踪和 BEV 基线。 |
| `preprocessing/` | PointCloud2 解析、清洗、降采样与流式回放模块。 |
| `config/` | ROS 2 默认参数。 |
| `tools/lidar/` | MCAP 审计、固定回放、ROS 2 回放验证与报告汇总。 |
| `tools/camera_fisheye/` | 鱼眼帧提取、内参导出、矫正及 MCAP 核验。 |
| `tools/annotation/`、`tools/yolo/` | 标注导出、候选验证、COCO/YOLO 转换与评估。 |
| `tools/controlled_calibration/` | 受控采集、内参、ICP 和质量门禁；仅作开发验证。 |
| `tools/simulation/` | Gazebo 训练世界、语义相机和 RGB/标签对采集。 |
| `tests/` | 离线感知、单 LiDAR、自然标定和标注工具测试。 |
| `Ubantu/` | WSL/Ubuntu 环境与仿真辅助脚本。 |
| `step1/` | 历史资料；若与当前计划冲突，以本 README 和`当前进行路线.md`为准。 |

仓库不提交原始 MCAP、大规模训练数据、模型权重和 ROS 构建产物；它们保留在本地 `数据/`、`datasets/`、`output/`、`build/`、`install/` 与 `log/`。

## 快速开始

### 1. ROS 2 节点（Ubuntu 24.04 / ROS 2 Jazzy）

将仓库置于 ROS 2 工作空间的 `src/` 目录下：

```bash
source /opt/ros/jazzy/setup.bash
cd ~/competition_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select competition_user
source install/setup.bash

ros2 run competition_user single_lidar_perception --ros-args \
  --params-file src/competition_user/config/single_lidar_baseline.yaml
```

节点以输入 `PointCloud2` 的 `header.frame_id` 为坐标系，发布清理点云、检测中心、局部 mono8 BEV 与状态。参数见[config/single_lidar_baseline.yaml](config/single_lidar_baseline.yaml)，回放细节见[README_单路LiDAR基线.md](README_单路LiDAR基线.md)。

### 2. 离线 MCAP 基线

使用包含 `numpy`、`scipy` 和 `rosbags` 的 Python 环境：

```powershell
$env:PYTHONPATH = 'F:\Challenger program\.codex_pydeps'
python -m perception.cli inspect --data-root '数据'
python -m perception.cli replay --data-root '数据' --max-frames 20 --output-dir output\perception
```

`inspect` 只读取元数据；`replay` 写出逐帧 BEV、检测/轨迹 JSON 和过程报告，不读取世界障碍物真值。

### 3. 单路 LiDAR 数据审计

```powershell
$env:PYTHONPATH = 'F:\Challenger program\.codex_pydeps'
python tools/lidar/audit_mcap.py
```

`output/lidar_audit/audit.json` 记录各 MCAP 和 LiDAR 的频率、间隔、点数与异常帧；`replay_manifest.json` 定义固定、可重复的回放帧。

## 验证

运行离线测试：

```powershell
$env:PYTHONPATH = 'F:\Challenger program\.codex_pydeps'
python -m unittest tests.test_perception -v
python -m pytest tests -q
```

在 WSL 上进行 ROS 2 验证时，先运行[Ubantu/setup_perception_wsl.sh](Ubantu/setup_perception_wsl.sh)，再按[README_单路LiDAR基线.md](README_单路LiDAR基线.md)中的真实 MCAP 回放流程执行。

## 训练与标定工作流

1. 按 bag 和时间段创建数据集，防止相邻帧泄漏。
2. 用人工标注和仿真语义图训练或微调原始鱼眼检测器，并逐相机报告指标。
3. 仅将已验证的静态图像观测和稳定 LiDAR 簇输入关联门禁。
4. 只有多帧、留出帧、可观测性和稳定性门禁均通过，外参才可变为 `accepted`。
5. 仅 `accepted` 可启用像素级 LiDAR–相机融合；否则保持独立传感器感知。

受控棋盘格、ICP、仿真真值投影和 HSV/轮廓候选不能直接作为外参发布依据。详见[自标定前置工作状态.md](自标定前置工作状态.md)。

## 参考文档

- [当前进行路线.md](当前进行路线.md)：当前执行计划与下一阶段任务。
- [数据与环境使用说明.md](数据与环境使用说明.md)：数据边界、仿真环境与运行约束。
- [README_单路LiDAR基线.md](README_单路LiDAR基线.md)：单传感器回放、ROS 2 验证与指标边界。
- [tools/camera_fisheye/README.md](tools/camera_fisheye/README.md)：鱼眼工具说明。
- [自标定汇总/CALIB_TECH.md](自标定汇总/CALIB_TECH.md)：标定算法与质量门禁。

## 下一阶段

优先完成六路原始鱼眼图的可审计人工标注和独立 bag 级评估，再训练通过逐相机门槛的检测器。之后才进入自然目标 2D–3D 多帧优化。仿真可运行或单传感器结果可见，均不能替代标定验收。
