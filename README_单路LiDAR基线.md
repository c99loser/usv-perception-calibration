# 单路 LiDAR 感知基线

本基线严格以一条 PointCloud2 的原始 `header.frame_id` 为坐标系。它不查询 TF、
不读取相机、不估计或加载外参，也不合并不同 LiDAR 的目标。

## 离线数据审计与固定回放

在具备 `rosbags` 的环境中运行：

```powershell
$env:PYTHONPATH = 'F:\Challenger program\.codex_pydeps'
& 'C:\Users\23546\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tools/lidar/audit_mcap.py
```

输出 `output/lidar_audit/audit.json`（每个 MCAP、每路 LiDAR 的频率、时间间隔、点数、
异常间隔与可用时长）和 `replay_manifest.json`（每路 12 个均匀、可重复的测试帧）。

## ROS2 节点

```bash
ros2 run competition_user single_lidar_perception --ros-args \
  --params-file config/single_lidar_baseline.yaml
```

发布结果均保留输入的时间戳和 `frame_id`：

- `filtered_points`：清理后的单路点云；
- `detections`：检测中心 PoseArray；
- `bev`：局部 mono8 占据栅格；
- `status`：检测数、轨迹数和各阶段耗时。

`competition_user.offline_evaluation` 只可在离线报告中使用真值；运行节点绝不订阅
`/world/obstacles/*`。中心误差与 BEV IoU 需要先在离线评估器中将真值表示为相应 LiDAR
坐标系，不能把世界坐标真值直接与传感器坐标检测混用。

## 真实 MCAP 在线验证（Ubuntu / ROS 2 Jazzy）

以下命令会启动一个节点，并用原封不动的录制 `PointCloud2` 重放指定的一路数据。通过条件是：
重放帧数完整、节点日志确认处理过输入、`status` 话题已收到单路结果。该过程不发布或消费 TF，
也不涉及外参或多雷达融合。

```bash
source /opt/ros/jazzy/setup.bash
source /mnt/f/Challenger\ program/install/setup.bash
cd /mnt/f/Challenger\ program

python3 tools/lidar/validate_ros2_single_lidar.py \
  '数据/rosbag2_2026_07_02-09_03_17/rosbag2_2026_07_02-09_03_17' \
  --sensor forward \
  --output-dir output/ros2_single_lidar_real_mcap/forward

python3 tools/lidar/summarize_ros2_single_lidar_validation.py \
  output/ros2_single_lidar_real_mcap \
  --output output/ros2_single_lidar_real_mcap/summary.json
```

当前首份 MCAP 的 7 路传感器均已通过该连通性验证（每路重放 8 帧）。它证明的是 ROS2
输入、处理和输出链路可用，不等同于实时性达标或检测精度验收。尤其 `forward` 通道的离线
p95 处理耗时仍偏高，应作为下一项优化对象；优化完成前，建议以低频回放/离线分析作为运行边界。
