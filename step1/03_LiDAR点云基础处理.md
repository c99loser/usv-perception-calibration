# 环节三：LiDAR 点云基础处理

## 1. 环节定位

LiDAR 点云处理是无人船感知系统的核心基础。相比鱼眼相机，LiDAR 可以直接提供目标的空间位置，因此在第一阶段应优先完成 LiDAR 点云读取、滤波、裁剪、聚类和候选目标提取。

本环节不要求一开始完成 7 路 LiDAR 的高精度融合，也不要求立即完成深度学习点云检测。第一阶段的目标是建立一个稳定的单 LiDAR 点云处理 baseline，为后续多 LiDAR 融合、外参标定和目标检测提供基础。

## 2. 输入数据

输入为 MCAP 或 ROS2 实时话题中的 `sensor_msgs/msg/PointCloud2` 数据。感知任务中通常包含以下 LiDAR：

```text
front_lidar
back_lidar
left_lidar
left2_lidar
right_lidar
right2_lidar
forward_lidar
```

其中 Airy LiDAR 主要覆盖半球面近距离区域，forward_lidar 主要覆盖前向较远距离区域。

## 3. 处理流程

推荐处理流程如下：

```text
PointCloud2 读取
→ NaN / Inf 去除
→ 距离裁剪
→ 高度裁剪
→ 船体自身点剔除
→ 下采样
→ 离群点滤波
→ 聚类
→ 目标候选提取
```

## 4. 任务要求

### 4.1 PointCloud2 读取

需要将 ROS2 点云消息转换为 NumPy 数组或 Open3D 点云对象，至少保留：

```text
x, y, z
```

如果点云中包含以下字段，也应保留：

```text
intensity, ring, time
```

这些字段后续可用于去噪、点云投影和深度学习特征输入。

### 4.2 数据清洗

需要去除：

- NaN 点。
- Inf 点。
- 距离过近的异常点。
- 明显超出传感器范围的点。

### 4.3 ROI 裁剪

第一版建议设置两个范围：

```text
检测范围：0.3 m ~ 60 m
BEV 范围：船体周围 ±10 m
```

forward_lidar 可单独保留更远距离数据，但第一阶段重点仍是 10 m 周围 BEV 和近距离障碍物。

### 4.4 高度裁剪

水面场景中可能存在水面反射和天空方向噪声点。可先设置粗略高度范围：

```text
z_min = -2.0 m
z_max = 5.0 m
```

实际范围需根据 LiDAR 坐标系和点云可视化调整。

### 4.5 船体自身点剔除

将点云转换到 base_link 后，剔除船体包围盒内的点。若暂时没有精确外参，可先对单 LiDAR 坐标系中的近距离区域做粗略剔除，待外参初值建立后再切换到 base_link 剔除。

### 4.6 下采样

建议使用体素下采样：

```text
voxel_size = 0.05 ~ 0.15 m
```

目标是减少点数，提高聚类速度，同时尽量保留浮球、立柱等小目标。

### 4.7 离群点滤波

可使用：

```text
StatisticalOutlierRemoval
RadiusOutlierRemoval
```

水面反射点、孤立点、远距离稀疏噪声点应尽量在聚类前剔除。

### 4.8 聚类

第一版推荐使用 DBSCAN 或欧式聚类：

```text
DBSCAN eps = 0.3 ~ 0.8 m
min_samples = 3 ~ 10
```

聚类参数需要按目标尺度调整。浮球较小，eps 不宜过大；船只较大，eps 不宜过小。

## 5. 目标候选特征

每个 cluster 应计算以下特征：

```text
cluster_id
center_x, center_y, center_z
bbox_length, bbox_width, bbox_height
num_points
distance_to_ship
z_min, z_max
aspect_ratio
```

这些特征可用于后续规则分类、深度学习训练标签生成和多传感器匹配。

## 6. 初版目标分类规则

| 目标 | 点云几何特征 |
|---|---|
| 浮球 | 尺寸小、点数少、近似圆形、高度较低 |
| 立柱 | 竖直细长、高度较高、位置稳定 |
| 游动船只 | 尺寸较大、长宽比明显、连续帧运动 |
| 浮块 | 低矮、近似方形，尺寸接近 2 m × 2 m × 0.4 m |

第一版可先输出 unknown obstacle，再逐步分类。

## 7. 建议输出

```text
filtered_points.pcd        # 滤波后点云
clusters.json              # 聚类结果
cluster_markers            # RViz MarkerArray
bev_debug.png              # 点云投影 BEV 图
```

## 8. 验收标准

本环节完成后，应达到：

- 能读取任意一路 LiDAR 点云。
- 能去除 NaN、Inf 和明显噪声点。
- 能裁剪出指定范围内的点云。
- 能剔除船体自身点。
- 能对障碍物形成可视化 cluster。
- 能输出每个 cluster 的中心和包围盒。
- 能在 BEV 图中看到障碍物分布。

## 9. 风险与应对

| 风险 | 应对方法 |
|---|---|
| 水面反射点较多 | 加强高度裁剪和离群点滤波 |
| 小浮球点数太少 | 降低 min_samples，保留较小 cluster |
| 船体自身点残留 | 调整船体遮罩范围 |
| 不同目标粘连 | 减小聚类 eps 或加入高度/距离分层 |
| 点云坐标系不清楚 | 先单传感器可视化，再建立外参转换 |
