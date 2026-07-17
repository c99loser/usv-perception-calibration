# 算法集成与调试重大修改记录

> 记录从队友原始算法代码到 `algos/` 框架集成、`preprocess_calib_test.py` 测试脚本开发，以及多轮调试中做出的所有非平凡修改。按时间顺序排列。

> **2026-07-14 终版（v22，22轮实验）**：
> - **三大根因**：① 旋转顺序 `zyx`→`xyz`（URDF rpy 约定）② SOR/ROR/Z-clip 预处理杀 95% 数据 ③ MLCC 位姿图低质量观测拉偏高质观测
> - **当前方案**：绕过预处理直接堆积原始点云 → URDF 6DOF 名义初值 → PointToPlane ICP 精修 → 级联链求解
> - **同侧对**（同 yaw，如 left2→left）fitness > 0.95，ICP 精修可达 Δxy < 1.5m
> - **异侧对**（90° yaw 差，如 left→front）fitness ~0.03，物理极限——同一障碍物不同面无共同 3D 特征
> - **代码**：1220 → 701 行，删 17 个死函数，修复 `_build_pcd` 缺 `self` 严重 bug
> - 完整 22 版演进见 `VERSION_HISTORY.md`
> - ring 修复：ring 数据从 MCAP 贯通到 SOR 滤波器。
> - 代码清理：删除 `_remove_hull`、`step1_remove_invalid`、`PCD_DTYPE` 死代码共 78 行。
> - 三份新增文档：`API_REFERENCE.md`（函数调用手册）、`ANALYSIS_SURROUNDVIEW.md`、`ANALYSIS_HORIZON.md`、`PAPER_NATURE2024.md`。
> - 当前最佳路线：名义基准 + ICP 精修，SOR/ROR 参数可调（数据差时关 SOR）。详见 `CALIB_TECH.md`。

---

## 一、算法迁移：从队友代码到 algos 框架

### 1.1 LiDAR 预处理（filters.py → lidar_preprocess.py）

**目标**：将基于 ROS2 PointCloud2 消息的预处理管线改为纯 numpy 接口。

| 改动 | 内容 |
|------|------|
| 去掉 ROS2 依赖 | 不再接收 `sensor_msgs.msg.PointCloud2`。改为接收 `{name: np.ndarray(N,3)}` dict |
| Step1 裸数组版 | 原 `step1_remove_invalid()` 访问 structured array 字段 `raw_points['x']`，改为 `np.isfinite()` + 距离计算，语义等价 |
| intensity/ring 补零 | 仿真环境 intensity 恒为 0，强度滤波 threshold=0 穿透；ring 填零不影响体素降采样的 xyz 质心 |
| 话题名→雷达名 | `AIRY_TOPICS` 的完整 ROS 路径改为 `AIRY_NAMES` 的短名称集合 |
| 新增适配器 | `lidar_preprocess(lidar_bursts, imu_data, config)` → 统一外部接口 |

### 1.2 相机预处理（rectifier.py → fisheye_preprocess.py）

| 改动 | 内容 |
|------|------|
| 去掉 ROS2 依赖 | 移除 `convert_ros_image_to_numpy()`（perception_user.py 已处理） |
| Plan 缓存 | 新增 `_plan_cache` 字典，避免 6 个相同参数的相机每帧重算 1280×720 映射表 |
| 新增适配器 | `fisheye_preprocess(images, fisheye_params)` → 统一外部接口 |

### 1.3 自标定（config.py + utils.py + lidar_lidar.py → calib.py）

| 改动 | 内容 |
|------|------|
| 三合一 | 三个源文件合并为 `calib.py`，去掉所有 `from .config import ...` 相对导入 |
| 命名简化 | 去掉 `_sensor` 后缀（`front_lidar_sensor` → `front_lidar`） |
| 数据格式适配 | 外部 `(M,3)` xyz → 内部 `(x,y,z,intensity,ring)` 五元组；命名映射 `front ↔ front_lidar` 双向转换 |
| 相机外参推导 | 利用 Airy↔鱼眼 10cm 精确约束，从 LiDAR 外参链式推导相机外参 |
| 新增适配器 | `calibrate_extrinsics(images, clouds, topology, config)` → 统一外部接口 |

---

## 二、测试脚本开发

### 2.1 preprocess_calib_test.py

**功能**：离线读取 MCAP → 预处理 → 自标定 → 输出 JSON。

| 里程碑 | 说明 |
|--------|------|
| 首版 | 读 MCAP + 写 MCAP（后被精简为只读+输出 JSON） |
| 精简 | 只取前 100 帧 LiDAR + 50 帧相机做标定，不写 MCAP |
| CDR 解析 | 手动实现 PointCloud2 / Image 的 CDR 反序列化，避免依赖 ROS2 |

---

## 三、关键 Bug 修复

### 3.1 Step1 缺失（lidar_preprocess.py）

`_process_single()` 起初跳过了 Step1（无效点去除），直接从 Step2a 开始。原因是 Step1 原函数只能吃 structured array。

**修复**：在 `_process_single()` 开头插入等价的 numpy 操作（`np.isfinite` + 距离检查）。

### 3.2 calibration/__init__.py 缺失

`from competition_user.algos.calibration.calib import ...` 失败，因为 `calibration/__init__.py` 不存在。

**修复**：创建 `algos/calibration/__init__.py`。

### 3.3 sys.path 路径层级错误

脚本中的 `sys.path` 只翻了两层目录，在嵌套的 `competition_user/competition_user/` 结构下找不到包。

**修复**：`os.path.dirname(os.path.dirname(os.path.dirname(...)))` — 三层翻到正确的外层目录。

### 3.4 CDR 反序列化偏移量错误（连续三次修复）

| 问题 | 现象 | 修复 |
|------|------|------|
| **ROS2 消息头** | offset 全部偏了 4 字节，读到垃圾 frame_id 长度 | 在 `extract_lidar` / `extract_image` 开头加 `o += 4` |
| **字符串长度语义** | `_skip_cdr_string` 对帧 ID 和 PointField 的名字语义不一致 | 帧 ID 用标准 CDR（`o+4+len`），PointField 名字用同公式但对齐后一致 |
| **uint8 后对齐** | is_bigendian (uint8) → point_step (uint32) 之间缺少 4 字节对齐 | `o += 1; o = (o+3) & ~3` |

### 3.5 extract_image 的 encoding 字符串对齐

encoding 字符串后紧跟 uint8（is_bigendian），`_skip_cdr_string` 的强制 4 字节对齐跳过了 is_bigendian。

**修复**：encoding 不走 `_skip_cdr_string`，直接 `o += 4 + _u32(raw, o)`。

### 3.6 话题名解析索引错误

`topic.split('/')[3]` 对于 `/wamv/sensors/lidars/back_lidar_sensor/points` 提取的是 `lidars`（传感器类别），而非 `back_lidar_sensor`（传感器名称）。

**修复**：改为 `topic.split('/')[4]`。

### 3.7 lidar_preprocess 调用缺参数

`lidar_preprocess({sn: in_cloud})` 缺少第二个必选参数 `imu_data`。

**修复**：`lidar_preprocess({sn: in_cloud}, imu_data=None)`。

### 3.8 json.dump numpy bool 序列化

标定结果中的 `success` 字段是 numpy bool，Python 标准库 json 无法序列化。

**修复**：新增 `_py()` 递归转换函数，`np.bool_ → bool`，`np.integer → int`，`np.floating → float`。

### 3.9 calibrate_pair 方法被破坏

修改 ICP 阈值时，编辑操作误删了 FPFH+RANSAC 代码段，导致 `T_fine` 未定义。

**修复**：完整恢复 `calibrate_pair` 方法的三条路径（visual→ICP、FPFH→RANSAC→ICP、None→报错）。

---

## 四、标定算法优化

### 4.1 视觉初始值方向修正

`compute_lidar_pair_init_from_cameras` 计算出 `T_{lid_of_cam1}→{lid_of_cam2}`，但校准对 `(source, target)` 需要 `T_source→target`。两者方向不一致。

**修复**：新增 `CAMERA_TO_LIDAR` 反向映射表，查表判断：
- `lid_of_cam1==source 且 lid_of_cam2==target` → 直接使用
- `lid_of_cam1==target 且 lid_of_cam2==source` → `np.linalg.inv(T)`
- 否则 → 跳过

### 4.2 视觉初值平移缩放修正

`compute_lidar_pair_init_from_cameras` 中的 `APPROX_LIDAR_DIST` 硬编码为 3.0m，但 front↔left 实际距离约 13m。

**修复**：改为可配置参数 `approx_distance`，在 `calibrate_extrinsics` 中从 `NOMINAL_POSITIONS` 计算实际距离。

### 4.3 ICP 失败回退策略

水面稀疏场景 ICP 普遍收敛失败（fitness=0.000），导致大量传感器对外参缺失。

**修复**：视觉初始值路径中，ICP 失败不再抛异常，直接回退到视觉初始值作为最终结果：
```python
if camera_init is not None:
    T_fine, fitness = self._fine_registration(...)
    if fitness < 0.01:
        return camera_init  # 回退
    return T_fine
```

### 4.4 名义位置兜底

部分 LiDAR 对缺少视觉初始值（forward 无相机、left2/right2 视觉匹配失败），导致完全无法标定。

**修复**：对所有校准对补充名义位置初始值——`T_nom[:3,3] = nom_src - nom_tgt`，当视觉初始值不可用时自动使用名义值。

### 4.5 船体坐标系几何构造

原方案用 front_lidar 名义位置 `(7,0,2.9)` 作为锚点，依赖赛事文档的近似值。

**修复**：替换为以下几何构造方法：

```
O = (front + back) / 2                 ← 船体中心（两点消噪）
x_axis = normalize(front - O)          ← 船艏方向（实际几何测量）
z_axis = (0, 0, 1)                     ← 强制竖直（平面拟合在标定不精确时引入噪声）
y_axis = z × x                         ← 右手系左舷方向

T_temp→boat = [R_{xyz} | -R·O]        ← 完整刚体变换
```

若 back 未标出，自动回退到旧方案的 front 名义锚点。

### 4.6 标定超参数调优

| 参数 | 原值 | 新值 | 原因 |
|------|------|------|------|
| voxel_size | 0.1m | 0.2m | 稀疏几何需更粗粒度 |
| fpfh_normal_radius | 0.5m | 1.0m | 扩大法向量估计范围 |
| fpfh_feature_radius | 1.0m | 2.0m | 扩大特征搜索半径 |
| icp_max_correspondence | 1.0m | 3.0m | 放宽 ICP 匹配距离 |
| icp_max_iterations | 100 | 200 | 增加迭代次数 |
| icp_fitness_threshold | 0.1 | 0.01 | 水面场景重合率天生就低 |

---

## 五、MLCC 风格全局位姿图优化 <sup>2026-07-13</sup>

级联链 (`T_left→front`, `T_left2→left`) 的问题是一对失败导致整条链断裂。参照 hku-mars/mlcc 的思路，引入全局位姿图优化。

**新增**：
- `_solve_pose_graph()` 函数：将所有 pairwise 观测 `(T, fitness)` 构建因子图
- 扩展校准对列表：增加 `left2→front`、`right2→front`、`back→front`、`forward→left/right` 等额外约束
- 用 `scipy.sparse.linalg.lsqr` 求解 SE(3) 切空间加权最小二乘
- ICP 失败的初始值（视觉/名义）作为低权重观测边参与优化
- 优化失败时回退到级联链兜底
- 新增 `_rotation_error()` 辅助函数（cv2.Rodrigues 计算 SO(3) 残差）

**效果**：即使 `left→front` 或 `right→front` 失败，其他传感器仍可通过直接约束求解，不再受级联顺序限制。

## 六、ring 数据恢复

MCAP 中的 PointCloud2 每条点 32 字节，包含 xyz(@0-11)、intensity(@16-19)、ring(@22-23)。`extract_lidar` 起初只提取前 12 字节的 xyz，损失了 intensity 和 ring。

**修复**：
- `extract_lidar` 返回值从 `(N,3)` 改为 `(N,3), (N,), (N,)`，分别对应 xyz、intensity、ring
- 调用处用真实数据替代 `np.zeros`

此修复不影响标定（FPFH/ICP 不需 ring），但对后续障碍物检测有意义。

---

## 七、当前状态与未解决问题

### 已完成

- [x] LiDAR 预处理管线（四步串联）
- [x] 鱼眼相机矫正（等距投影 → 直线透视）
- [x] 多传感器自标定（FPFH+RANSAC+ICP + 视觉辅助 + 名义兜底 + 船体系构造）
- [x] 离线测试脚本（MCAP 读取 → 预处理 → 标定 → JSON 输出）
- [x] ring/intensity 数据恢复

### 待解决

| 问题 | 影响 | 方向 |
|------|------|------|
| 视觉初始值精度差 | right/left 位置偏离 35m | 关掉 `use_visual_init`，全用名义位置 + ICP |
| z 值偏低 | 所有传感器 z 坐标比预期低约 2.8m | 检查 z 方向平移链；可能是 LiDAR→Camera 10cm 约束方向问题 |
| 名义位置噪声 0.5m / 10° | ICP 可部分纠正但残余误差仍在 | 属于赛事设计约束，可接受的系统误差 |

---

## 八、文件清单

| 文件 | 说明 |
|------|------|
| `algos/preprocess/lidar_preprocess.py` | LiDAR 四步预处理 |
| `algos/preprocess/fisheye_preprocess.py` | 鱼眼相机矫正 |
| `algos/calibration/calib.py` | 多传感器自标定（三合一） |
| `algos/calibration/__init__.py` | 包标识文件（创建） |
| `scripts/preprocess_calib_test.py` | 离线测试脚本 |
| `algos/MIGRATION.md` | 算法迁移改动说明 |
| `calib_result_v1.json` | 首次测试完整输出（已恢复） |
