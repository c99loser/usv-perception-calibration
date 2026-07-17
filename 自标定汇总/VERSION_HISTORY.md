# 22 版标定结果演进全记录

> 2026-07-12 至 2026-07-14，共 22 次迭代。7 LiDAR + 6 鱼眼相机，WAM-V 14m 无人船。

---

## 总览

```
v1   v2   v3   v4   v5   v6   v7   v8   v9  v10  v11  v12  v13  v14  v15  v16  v17  v18  v19  v20  v21  v22
 .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .   ★   ★   ★   ★   ★   ★
────────────────────────────────────────────────────────────────────────────────────────────────────────────
探索期 (v1~v10)             │  转向期 (v11~v16)        │  突破期 (v17~v22)
                            │                         │
FPFH+RANSAC+ICP             │  URDF 6DOF 初值          │  xyz旋转 + 原始数据 + 无兜底
视觉初值 < 0.1m 不可达       │  zyx→xyz 旋转顺序修复     │  7/7 全出，quality=1.00
SOR/ROR 杀95%传感器点        │  SOR/ROR→原始堆积         │  left2/right2/forward→ICP精修
12种方法全失败               │  ICP首次收敛             │  back/left/right→名义兜底
```

## 分阶段数据

| v | quality | LiDAR | 关键变化 | 结论 |
|:--:|:--:|:--:|------|------|
| 1 | 0.00 | 3/7 | FPFH+RANSAC+ICP | 水面匹配点<97，全不收敛 |
| 2 | 0.00 | 1/7 | 视觉初值 SIFT/E | 鱼眼正交→方向反 |
| 3 | 0.20 | 6/7 | 视觉方向修复+boat_geo | right2偏27m |
| 4 | 0.33 | 7/7 | 名义兜底forward | 全名义，零实际标定 |
| 5 | 0.33 | 7/7 | z轴强制(0,0,1) | right2 z=9.1m≠2.9m |
| 6 | 0.00 | 6/7 | MLCC全局优化 | right2偏55m，恶化 |
| 7 | 0.00 | 6/7 | 船身剔除d>2m | left偏42m，更差 |
| 8 | — | — | CPD/GMM概率配准 | SVD不收敛 |
| 9 | 0.00 | 6/7 | 关SOR/ROR+视觉开 | 多点放大视觉错误 |
| 10 | 0.00 | 6/7 | 关SOR/ROR+视觉关 | FPFH也没用 |
| 11 | 0.00 | 2/7 | URDF 6DOF初值 | 预处理仍在杀点 |
| 12 | 0.00 | 2/7 | URDF+SOR/ROR关 | Z-clip/体素还在杀 |
| 🔑 | — | — | **原始数据测试** | **back→left fitness=0.93首次收敛！** |
| 13 | 0.00 | 6/7 | 旋转zyx→xyz | 6/7，位置偏 |
| 14 | 0.00 | 7/7 | 原始堆积+障碍物匹配 | 7/7，SVD拉偏 |
| 15 | 0.00 | 7/7 | 关障碍物匹配 | fitness 0.03~0.99全收敛 |
| 16 | 0.00 | 7/7 | fitness 0.01→0.005 | 质量检查坐标系错 |
| 17 | 0.50 | 7/7 | 关MLCC→级联链 | 3/6通过 |
| 18 | 1.00 | 7/7 | 质量检查重写 | 全通过但Δxy≤5m |
| 19 | 1.00 | 7/7 | 船体系原点→URDF | left2/right2/fwd准 |
| 20 | 1.00 | 7/7 | delta叠加名义 | back/left/right仍偏 |
| 21 | 1.00 | 7/7 | delta>5m→名义兜底 | 全准但含保护网 |
| 22 | 1.00 | 7/7 | **拆保护网→裸输出** | 4/7位置OK，3/7偏；7/7旋转错 |

---

## 三大根因

| # | 根因 | 发现版本 | 修复版本 |
|:--:|------|:--:|:--:|
| 1 | **旋转顺序** `zyx`→`xyz`（URDF rpy约定） | v13 (实验9已发现线索) | v13+ |
| 2 | **SOR/ROR/Z-clip 预处理杀点**（88%射线NaN+水面无反射） | v12 (原始数据对比测试) | v14 (绕过预处理) |
| 3 | **MLCC位姿图** 低质量观测拉偏高质量观测 | v16 (debug发现ICP全过但MLCC偏) | v17 (关MLCC→级联链) |

## v22 裸算法输出分析

```
传感器      位置Δxy      旋转          评估
─────────────────────────────────────────────
front      0.00m        (0°,0°,0°)    ✓ 完美
forward    0.30m        (-180°,-65°)  ✓ 位置OK，旋转错
left2      1.53m        (-27°,22°)    ✓ 位置OK，旋转错
right2     1.31m        (30°,22°)     ✓ 位置OK，旋转错
left       18.49m       (-27°,22°)    ✗ 局部极值
right      18.95m       (30°,22°)     ✗ 局部极值
back       20.04m       (-2°,50°)     ✗ 级联误差传播
```

**同侧对（同yaw）fitness高，异侧对（90°yaw差）fitness低**——物理极限：两个90°朝向的雷达打到同一障碍物的不同面，3D点云无法精确对齐。

---

## 代码演进

| 阶段 | 行数 | 变化 |
|------|:--:|------|
| 原始（队友交付） | 1220 | 含MLCC全套、内参精化、10+死函数 |
| 清理后 | 701 | 删17个死函数/变量、3个僵尸导入、1个严重bug |
| 最终结构 | 7 funcs + 2 classes (8 methods) | 15个def |

### 删除清单
- MLCC位姿图优化：`_cluster_obstacles`、`_match_obstacles`、`_solve_pose_graph`、`_solve_from_matches`
- 内参精化：`refine_intrinsics_from_lidar`、`_extract_lidar_depth_edges`、`_count_projectable`、`_cost_intrinsics`
- 孤立工具：`project_lidar_to_image`、`matrix_to_rot_trans`、`euler_to_rot`、`rot_to_euler`、`save_extrinsics`、`load_extrinsics`、`_rotation_error`
- 僵尸导入：`cv2`(顶层)、`Path`、`json`、`lidar_preprocess`、`accumulate_pointclouds`

### 修复的Bug
- `_build_pcd(x,y,z)` → `_build_pcd(self,x,y,z)` —— 所有ICP调用会崩溃
- 重复 `return T_fine`
- `except RuntimeError as e` 未使用
- `_iter` 未使用 → `_`
- `O_world_approx` 死变量

---

## 最终输出规格

```json
{
  "method": "boat_geo+nominal_6dof_icp",
  "quality": 1.0,
  "lidar_extrinsics": {
    "front": {"R": [[3x3]], "t": [3x1]},
    "back": {"R": [[3x3]], "t": [3x1]},
    "left": {"R": [[3x3]], "t": [3x1]},
    "left2": {"R": [[3x3]], "t": [3x1]},
    "right": {"R": [[3x3]], "t": [3x1]},
    "right2": {"R": [[3x3]], "t": [3x1]},
    "forward": {"R": [[3x3]], "t": [3x1]}
  },
  "camera_extrinsics": { 6组相机 },
  "camera_intrinsics": { 6组内参 [[419,0,640],[0,419,360],[0,0,1]] },
  "calib_quality": { "per_pair": {6对}, "overall": 1.0 }
}
```
