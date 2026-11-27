# 真实 LiDAR V4.11 标注包使用说明

> 生成日期：2026-08-04。该包用于点云外参评价体系的 L5 真实域语义门禁，不直接产生、接受或发布外参。

## 位置与内容

标注包：`F:/Challenger program/output/real_lidar_manual_annotation_pack_v4_11_l5_v1/`

- `manifest.csv`：不可修改的 572 行谱系清单；
- `annotations/train_labels.csv`：248 行，只来自 090317、090835；
- `annotations/holdout_labels.csv`：324 行，只来自 091054，最终评价专用；
- `clouds/`：572 个传感器坐标系 ASCII PLY 簇；
- `images/`：270 张同时间场景图，只提供上下文，没有做外参投影；
- `leakage_audit.json`：bag 与时间戳交集审计；
- `package_manifest.json`：来源、逐相机数量、阈值和禁止输入；
- `checksums.sha256`：开始标注前的完整性校验；
- `README_标注说明.md`：包内操作规则。

当前数量为 train 248、holdout 324。holdout 中 V4.11 几何合格候选为 front 97、back 35、left 78；这些只是可能进入语义评估的上限，人工标成静态正类后的有效数量仍需复核。

## 具体填写方式

只填写两个 labels CSV 的最后六列，禁止修改前面的谱系与几何列：

1. `human_label`：`pillar`、`buoy`、`block`、`boat`、`background`、`ambiguous`；
2. `static_for_calibration`：`yes`、`no`、`uncertain`；
3. `review_status`：完成初审填 `reviewed`，二审裁决后填 `adjudicated`；
4. `reviewer_confidence`：建议统一用 `high`、`medium`、`low`；
5. `reviewer_id`：填写固定标注者代号；
6. `reviewer_note`：记录遮挡、稀疏、疑似粘连、动态或其他不确定性。

正式静态正类只包括 `pillar/buoy/block + static_for_calibration=yes`。`boat` 必须为 `no`；看不清时填写 `ambiguous/uncertain`，不要猜测。

## 必须注意

- PLY 是**簇级标签**，不是逐点分割标签；单位为米，坐标属于对应 LiDAR 传感器。
- 同时间图像未与 PLY 投影对齐，只能帮助理解当时场景；禁止借助名义外参、TF、里程计、世界真值或固定靶标反推类别。
- 同一 `track_group_id` 的多行是相关帧，不能随机按行划分，也不能把它们当成完全独立的匹配来虚增样本量。
- `geometry_eligible=no` 只能作为审计或训练难例，人工类别不能把它改成几何合格。L5 正式主指标使用 `geometry_eligible=yes` 的冻结候选池。
- holdout 可由独立标注人员提前完成，但开发模型、特征和阈值的人在冻结前不得查看其类别统计或错误案例。
- Excel 容易把 19 位纳秒时间戳转成科学计数法并丢失精度。若必须使用 Excel，应把原始列全部按文本导入；更建议使用不会自动格式化的 UTF-8 CSV 编辑器，只编辑最后六列。
- `checksums.sha256` 只验证未编辑的初始包。标签填写后两个 CSV 的校验和变化是正常的；完成版应另存为新版本并重新生成校验和。

## 标注完成后的质检

在 WSL 项目目录运行：

```bash
python3 tools/natural_scene_calibration/validate_lidar_annotations_v4_11.py \
  --package '/mnt/f/Challenger program/output/real_lidar_manual_annotation_pack_v4_11_l5_v1' \
  --require-complete \
  --output '/mnt/f/Challenger program/output/real_lidar_manual_annotation_pack_v4_11_l5_v1/annotation_validation.json'
```

首次运行前确认 `annotation_validation.json` 不存在；质检器遵循不覆盖已有报告的原则。

质检通过后，训练和阈值选择只能读取 train CSV。模型、特征和阈值全部冻结后，才可读取 holdout CSV，且应同时报告总体、逐相机和逐 `track_group_id` 指标。若任一相机有效静态正匹配不足 20，应从对应允许的 bag 新增不重叠连续窗口，不能降低门槛或把训练样本移入留出集。

## 当前主标注入口：精简稳定 raw-only 全视场 Episode

根据 2026-08-17 对人工工作量和连续运动复杂度的复核，首轮人工真值入口升级为 `F:/Challenger program/output/real_lidar_raw_official_stable_annotation_v3_bundle/`。该总目录同时保存完整 V3 数据目录、train ZIP 和 holdout ZIP。V3 从官方直抽 V2 中按完整时段分层选择局部变化较小、点数完整、同步较好的帧，不使用外参、候选或图像投影。共 18 个 episode、1,152 帧和 4,012,183 个有限 XYZ 点；每个 bag × 方位 64 帧，相对 V2 减少 78.59%。完整 V2 保留为冻结后按失败案例定向补标的来源，不再要求首轮全量标注。

- 立即标注：总目录内 `real_lidar_raw_official_stable_annotation_v3_supervisely_train.zip`，包含 090317/090835 的 12 个六路 episode、768 帧；
- 封存留出：总目录内 `real_lidar_raw_official_stable_annotation_v3_supervisely_holdout.zip`，包含 091054 的 6 个六路 episode、384 帧；
- `cluster_overlay=false`、`projection_valid=false`；不读取 V4/V4.11 候选/轨迹，不使用 TF、里程计、世界真值或名义外参；
- raw-only 真值冻结前，标注人员不得查看下述彩色候选/草稿框版本，避免确认偏差。
- 旧 `real_lidar_raw_full_fov_annotation_v1/` 只有 704 个窗口帧且仅覆盖 front/back/left，已退出活动输出区，不再作为正式标注输入。

## 真值冻结后的候选辅助入口

候选辅助包为：`F:/Challenger program/output/real_lidar_full_fov_episodes_v4_11_v2_patent_assist/`。原 V1 保留为无草稿框的回退版本。两者均只能在 raw-only 真值冻结后用于 TP/FP/FN 评价、误差归因和疑难形状复核。

- 每帧 PCD 保留该 LiDAR 的完整有效 360° 扫描，不再只输出某个物体；
- 灰色为原始有效点，彩色为 V4.11 跟踪候选，同一 episode 内同一 track 保持同色；
- 每个 episode 连续 64 帧，并带同时间鱼眼图、帧映射、逐帧审计元数据和多视角审查图；
- 训练项目为 `supervisely_train_project/`，包含 090317、090835 的 6 个 episode；
- 独立留出项目为 `supervisely_holdout_project/`，只含 091054 的 5 个 episode；
- 全包共 704 帧、1,803,631 个有效点；候选回贴到原始点的总体覆盖率约 98.4%；
- 图像只提供同时间场景上下文；`projection_valid=false`，没有使用名义外参、TF、里程计或世界真值。
- V2 基于 CN202510931459 的可迁移思想增加纯 LiDAR 时序辅助：绿色框为质量合格的已观测提案，黄色虚线框为同一 track 前后重新出现后确认的 1–2 帧缺口补偿；所有框的初始类均为 `ambiguous`，必须人工检查、调整尺寸并重标或删除。
- V2 不向原始 PCD 注入合成点，不外推轨迹末端；补偿结果在元数据中固定标记 `ground_truth=false`。完整包新增 121 个补偿帧提案，其中训练窗口 107 个；另有 149 个训练缺口因端点质量不合格、1 个因双似然过低而被拒绝。

彩色候选只是高精度提议，不是语义真值；灰色区仍可能有漏检对象。留出项目只在模型、特征和阈值冻结后交给独立评估。

原轨迹聚合工作区 `F:/Challenger program/output/real_lidar_track_annotation_v4_11_v1/` 降级为二级形状复核工具。对 Supervisely 中仍难判断的候选，可用其多帧局部聚合和最佳单帧视图辅助判为类别或 `ambiguous`，但不得取代全视场 episode 的场景核对。浏览器导出的轨迹 JSON 仍可用 `apply_track_labels_to_lidar_pack.py` 传播回 572 行逐观测 CSV。

该优化不会让所有轨迹都变得可识别。若多视角轮廓仍呈多层扫描带、疑似粘连或类别不唯一，必须标记 `ambiguous/uncertain`；不得依据相机画面中“看起来最接近”的对象猜标签。
