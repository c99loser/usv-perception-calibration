# 官方 MCAP 直接抽取点云标注 V2 说明

## 判断结论

“只从官方 `F:/Challenger program/数据` 的三份 MCAP 直接抽取点云用于标注”是合理且必要的修正。

旧 raw-only V1 的单帧 PCD 没有再做聚类或降采样，但数据集范围受既有缓存限制：只包含 front/back/left 三路和若干 64 帧窗口，共 704 帧。因此用户观察到的“内容显著小于官方数据”主要是总体时段和传感器覆盖不足，不是 PCD 写入时再次删掉了大量有限点。

## 新包

位置：`F:/Challenger program/output/real_lidar_raw_official_direct_annotation_v2/`

- 数据源：三份官方 MCAP；
- 传感器对：front、back、left、left2、right、right2；
- 训练：090317、090835，共 12 个 episode；
- 留出：091054，共 6 个 episode；
- 输出：5,381 个纯 XYZ PCD、5,381 张同时间 direct `image_raw` 图片；
- 原始槽位：92,983,680；有限 XYZ 点：19,113,380；
- 策略：只删除 NaN/Inf，不做距离过滤、体素化、聚类、语义筛选或几何筛选；
- 同步：最近同名 LiDAR，最大允许时间差 60 ms；31 个超门槛图像帧不强行配对；
- 禁止输入：V4/V4.11 候选与轨迹、TF、里程计、世界真值、名义外参投影；
- overhead camera 与 forward LiDAR 不属于冻结的六路同名标定对，未加入。

与旧 V1 相比，新包帧数为 7.64 倍、有限点数约为 9.99 倍，新增 left2/right/right2，并覆盖三份官方记录的完整可同步时段。旧 V1 不再作为标注输入；截至 2026-08-17，它及原隔离目录均不在活动输出区。

## 标注方法

2026-08-17 工作量修正：完整 V2 目录继续作为不可变补标池，但首轮标注改用 `real_lidar_raw_official_stable_annotation_v3_bundle/` 中的 V3 train ZIP。V3 对每个 bag × 方位选择 64 帧，覆盖六路和完整时段，共 1,152 帧，较 V2 减少 78.59%；选择偏向相邻点云变化较小的帧，但不会完全删除运动/困难场景。V2 的两个全量 Supervisely ZIP 已移至 `F:/Challenger program/Rubbish/2026-08-17_superseded_data/obsolete_annotation_exports/`，不再作为活动入口。详细规则和校验结果见《官方MCAP精简稳定标注V3说明.md》。

方法不变：在 Supervisely Point Cloud Episodes 中跨帧画 3D cuboid，同一物体保持同一 object/track；图片只作场景和类别辅助，没有 2D–3D 投影。静态外参候选只允许 `pillar/buoy/block + static_for_calibration=yes`。自船、水面、多径、粘连或严重稀疏对象使用 `background/ambiguous`；图像中存在但点云中没有可辨回波的对象不得强行画 3D 框。

训练项目可以开始标注；091054 留出项目必须在模型、特征与阈值冻结后再交给独立评估人员。

## 验证

- 18 个 episode；
- 5,381 个 PCD 与 5,381 张图片一一对应；
- 每个 PCD 仅含二进制 `x y z`；
- 逐帧 PCD 载荷与直接解码 XYZ SHA-256 一致；
- 21,620 个包内文件校验和通过；
- train/holdout episode 交集为 0；
- 项目测试集 66 项通过。
