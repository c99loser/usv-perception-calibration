# 环节六：LiDAR-Camera 粗对应与投影检查

## 1. 环节定位

LiDAR-Camera 粗对应是后续多传感器融合和外参自标定的前置步骤。当前阶段不要求实现精确标定，但需要建立 LiDAR 目标与相机目标之间的基本对应关系，从而检查传感器方向、相机模型和初始外参是否合理。

任务材料中说明 Airy LiDAR 与对应鱼眼相机上下相差约 0.1 m，且姿态角绑定。这意味着每个 Airy LiDAR 可以与对应鱼眼相机先建立天然配对关系。

## 2. 传感器配对关系

建议初始配对如下：

```text
front_lidar  ↔ front_camera
back_lidar   ↔ back_camera
left_lidar   ↔ left_camera
left2_lidar  ↔ left2_camera
right_lidar  ↔ right_camera
right2_lidar ↔ right2_camera
```

forward_lidar 是前向长距 LiDAR，可暂时作为独立前向障碍检测传感器，不强行与鱼眼相机绑定。

## 3. 任务要求

本环节需要完成：

1. 建立 LiDAR 与 Camera 的名称映射关系。
2. 根据 YAML 或 PDF 初值建立初始外参。
3. 将 LiDAR cluster 中心转换到对应相机坐标系。
4. 将三维点投影到鱼眼图像。
5. 将图像检测框中心转换为方向角。
6. 将 LiDAR 目标方向角与图像目标方向角进行匹配。
7. 输出投影误差和可视化图像。

## 4. 粗对应方法

### 4.1 基于方向角匹配

LiDAR cluster 可以得到目标中心点：

```text
p_lidar = [x, y, z]
```

可计算目标相对于传感器的水平角：

```text
angle_lidar = atan2(y, x)
```

图像检测框中心可通过鱼眼模型转换为视线方向，再得到：

```text
angle_camera
```

若两者角度差小于阈值，则认为可能对应：

```text
|angle_lidar - angle_camera| < threshold
```

第一版阈值可设为 5°~15°，后续根据实际误差调整。

### 4.2 基于投影点与检测框匹配

如果已有初始外参和相机内参，可将 LiDAR 目标中心投影到图像平面。如果投影点落在图像检测框内或附近，则认为匹配成功。

匹配条件可设为：

```text
投影点在 bbox 内
或投影点到 bbox 中心距离 < 设定像素阈值
```

## 5. 可视化要求

建议生成调试图：

```text
原始鱼眼图像
+ 图像检测框
+ LiDAR cluster 投影点
+ 匹配连线或编号
```

该图对于判断外参方向错误、相机模型错误、图像翻转错误非常重要。

## 6. 输出内容

建议输出：

```text
association_results.json
projection_debug_front.png
projection_error.csv
```

每个匹配记录包含：

```text
camera_name
lidar_name
camera_detection_id
lidar_cluster_id
angle_error
pixel_error
matched_or_not
```

## 7. 验收标准

本环节完成后，应满足：

- 能建立 6 对 LiDAR-Camera 初始映射。
- 能将 LiDAR cluster 投影到对应鱼眼图像上。
- 能计算图像目标与 LiDAR 目标的方向角误差。
- 能生成可视化调试图。
- 能初步判断外参或相机模型是否存在明显错误。

## 8. 风险与应对

| 风险 | 应对方法 |
|---|---|
| 没有准确相机内参 | 先用 FOV 估计初值，只做方向角粗匹配 |
| 初始外参误差较大 | 使用较大匹配阈值，后续通过自标定优化 |
| 鱼眼投影模型不准确 | 对比多种模型：等距、等固角、OpenCV fisheye |
| 图像检测误检 | 只用高置信度目标进行对应 |
| LiDAR cluster 分类错误 | 初期只匹配目标中心，不强依赖类别 |
