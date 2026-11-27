# 官方 MCAP 精简稳定标注 V3 说明

## 决策

将每个 bag × 方位限制在 100 帧以内是合理的。完整 V2 的 5,381 帧包含大量相邻近重复帧，首轮全量标注的时间成本过高；但如果只截取一个连续短窗口，或彻底删除所有高变化帧，又会损失全时段场景覆盖并导致静态场景偏差。

本次采用折中方案：每个 bag × 方位选择 64 帧，选帧覆盖完整记录时段，并在每个时间分层内优先选择局部变化较小、点数完整、同步误差较低的帧。两份训练 bag 合计每个方位 128 帧，仍满足此前每路约 100–200 帧的真实小样本建议；第三 bag 每方位 64 帧，继续独立留出。

## 位置

- V3 总目录：`F:/Challenger program/output/real_lidar_raw_official_stable_annotation_v3_bundle/`
- 完整精简包：`F:/Challenger program/output/real_lidar_raw_official_stable_annotation_v3_bundle/real_lidar_raw_official_stable_annotation_v3/`
- 首轮人工标注：`F:/Challenger program/output/real_lidar_raw_official_stable_annotation_v3_bundle/real_lidar_raw_official_stable_annotation_v3_supervisely_train.zip`
- 冻结留出：`F:/Challenger program/output/real_lidar_raw_official_stable_annotation_v3_bundle/real_lidar_raw_official_stable_annotation_v3_supervisely_holdout.zip`
- 完整补标池：`F:/Challenger program/output/real_lidar_raw_official_direct_annotation_v2/`

## 规模与校验

- 18 个 episode，front/back/left/left2/right/right2 六路完整覆盖；
- 每个 bag × 方位恰好 64 帧；train 768 帧，holdout 384 帧，共 1,152 帧；
- 相对 V2 的 5,381 帧减少 78.59%，保留 4,012,183 个有限 XYZ 点；
- 每个 episode 的所选时间跨度至少覆盖原始时段的 97.93%；
- 相对普通等距抽样，18 个 episode 的运动分数中位数全部下降，改善中位数为 11.37%，单路约 1.68%–37.05%；
- 4,651 个包内校验和、PCD 点数、原始 XYZ 哈希、图像 sidecar、Supervisely 映射及 train/holdout 隔离全部通过。

ZIP 校验：

- train：118,163,672 字节；SHA-256 `745cd4fa0c24c365cc68c840a1c2ff2ce1a1b0059a3bcac188a6dae084ae4980`
- holdout：57,966,425 字节；SHA-256 `50c854433450feef76bfc568808554cd3cf13154e3b036f5a8d7d8bf82a93a34`

## 选帧原则

运动代理只使用传感器坐标系原始 XYZ：先将每帧构造成方位角 × 高度的最近距离签名，再计算与前后帧的变化。选择分数同时考虑变化、点数完整度和相机–LiDAR 同步差。整个过程没有使用外参、TF、里程计、候选框、轨迹、世界真值或图像投影。

`frame_lineage.csv` 和逐帧 JSON 新增：

- `source_episode_id`、`source_frame_index`：回溯完整 V2；
- `selection_motion_score`：相邻点云变化代理；
- `selection_quality_score`：运动、完整度和同步的综合选帧分数。

逐 episode 的全部选帧索引和相对等距抽样统计位于总目录下 `real_lidar_raw_official_stable_annotation_v3/selection_audit.json`。

## 标注注意事项

1. 先导入 train ZIP；holdout ZIP 必须在模型、规则和阈值冻结后，由独立评估人员打开。
2. 运动分数是无外参的点云变化代理，不是语义运动真值。低分帧仍可能含移动船只、浮标晃动、水面回波或遮挡变化，必须人工判断 `static_for_calibration`。
3. 相机图像只提供同一时刻的场景上下文，仍然不是点云投影；不要按图像像素位置直接画 3D 框。
4. 首轮标注只使用 V3，避免重复劳动。模型初步冻结后，再根据漏检、难例和方位覆盖从完整 V2 定向追加，不恢复全量逐帧标注。
5. 不应追求完全排除运动帧，否则会夸大静态场景效果。当前方案降低高变化帧比例，同时保留完整时段覆盖，用于第一轮高精度候选器更稳妥。
