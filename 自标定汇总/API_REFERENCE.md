# 预处理算法与自标定算法 API 手册

> 供队友集成使用。`lidar_preprocess.py` + `calib.py`。

---

## 一、lidar_preprocess.py — 激光雷达预处理

### 入口函数

```python
from competition_user.algos.preprocess.lidar_preprocess import lidar_preprocess

result = lidar_preprocess(lidar_bursts, imu_data=None, config=None)
```

### 输入

**`lidar_bursts`** — `dict[str, np.ndarray]`
- key: 雷达名，`'front' | 'back' | 'left' | 'left2' | 'right' | 'right2' | 'forward'`
- value: `(N, 3)` 或 `(N, 5)` float32 数组
  - 3 列：`[x, y, z]`
  - 5 列：`[x, y, z, intensity, ring]`（推荐，ring 用于提高 SOR 质量）

**`imu_data`** — 保留参数，当前忽略，传 `None`。

**`config`** — 可选 `dict`，覆盖默认预处理参数：

| 键 | 类型 | 默认值 | 含义 |
|----|------|:--:|------|
| `intensity_threshold` | float | 0.0 | 强度阈值（0=不过滤） |
| `z_min_airy` | float | -3.0 | Airy Z 下限 (m) |
| `z_max_airy` | float | 5.0 | Airy Z 上限 (m) |
| `z_min_emx` | float | -12.0 | EMX Z 下限 (m) |
| `z_max_emx` | float | 8.0 | EMX Z 上限 (m) |
| `sor_nb` | int | 20 | SOR 邻居数 |
| `sor_std` | float | 2.0 | SOR 标准差倍数 |
| `ror_radius` | float | 0.3 | ROR 半径 (m) |
| `ror_min` | int | 3 | ROR 最小邻居数 |
| `min_range` | float | 0.3 | 最小有效距离 (m) |
| `max_range_airy` | float | 60.0 | Airy 最大距离 (m) |
| `max_range_emx` | float | 200.0 | EMX 最大距离 (m) |
| `voxel_airy` | float | 0.08 | Airy 体素尺寸 (m) |
| `voxel_emx` | float | 0.10 | EMX 体素尺寸 (m) |

### 输出

`dict[str, np.ndarray]` — key 同输入，value 为 `(M, 3)` float32 预处理后点云。未提供数据的雷达不会出现在返回字典中。

### 内部管线（`_process_single` 内部自动执行）

```
Step 1:   NaN/Inf/零距离 剔除
Step 2a:  强度滤波 (threshold, 默认=0 → 不过滤)
Step 2b:  Z 轴裁剪 (传感器专属)
Step 2c:  SOR 统计离群点去除 (EMX 跳过)
Step 2d:  ROR 半径离群点去除 (EMX 跳过)
Step 3:   距离范围裁剪
Step 4:   体素降采样 (质心法, Airy 8cm / EMX 10cm)
```

### 调用示例

```python
import numpy as np
from competition_user.algos.preprocess.lidar_preprocess import lidar_preprocess

# 单帧输入 (仅 xyz)
clouds = {
    'front': np.random.randn(5000, 3).astype('f4'),
    'left':  np.random.randn(5000, 3).astype('f4'),
}

# 带 ring 输入 (5 列)
clouds_5col = {
    'front': np.column_stack([
        np.random.randn(5000).astype('f4'),  # x
        np.random.randn(5000).astype('f4'),  # y
        np.random.randn(5000).astype('f4'),  # z
        np.random.randn(5000).astype('f4'),  # intensity
        np.random.randint(0, 48, 5000).astype('f4'),  # ring
    ]),
}

# 调用 (用默认参数)
result = lidar_preprocess(clouds)

# 调用 (自定义参数)
result = lidar_preprocess(clouds, config={
    'voxel_airy': 0.05,  # 更细的体素
    'max_range_airy': 80.0,
})

for name, pts in result.items():
    print(f'{name}: {len(pts)} points')
```

---

## 二、calib.py — 自标定算法

### 依赖

```bash
pip install scikit-learn  # DBSCAN 聚类
```
已有：`numpy` `scipy` `opencv-python` `open3d`

### 入口函数

```python
from competition_user.algos.calibration.calib import calibrate_extrinsics

result = calibrate_extrinsics(
    rectified_images,          # 矫正后相机图像
    preprocessed_clouds,       # 预处理后 LiDAR 稠密点云
    config=None,               # 可选配置
    camera_config=None,        # 可选相机配置
)
```

### 输入

**`rectified_images`** — `dict[str, np.ndarray]`
- key: 相机名，`'front' | 'back' | 'left' | 'left2' | 'right' | 'right2'`
- value: `(H, W, 3)` uint8 矫正后 RGB 图像
- 必须至少包含对应雷达配对的相机（如调用 `front_lidar` 标定则需要 `front` 图像）

**`preprocessed_clouds`** — `dict[str, tuple]`
- key: 雷达名，`'front' | 'back' | 'left' | 'left2' | 'right' | 'right2' | 'forward'`
- value: `(x, y, z)` 其中 `x, y, z` 均为 float32 一维数组（稠密点云）

**`config`** — 可选 `dict`：

| 键 | 类型 | 默认值 | 含义 |
|----|------|:--:|------|
| `use_visual_init` | bool | True | 是否用相机做视觉初始值 |
| `voxel_size` | float | 0.1 | FPFH 体素尺寸 (m) |
| `fpfh_normal_radius` | float | 1.0 | FPFH 法向量搜索半径 (m) |
| `fpfh_feature_radius` | float | 2.0 | FPFH 特征搜索半径 (m) |
| `method` | str | `'fpfh_ransac_icp'` | 标定方法标签 |

**`camera_config`** — 可选 `dict`：

| 键 | 类型 | 默认值 | 含义 |
|----|------|:--:|------|
| `width` | int | 1280 | 矫正图像宽 (px) |
| `height` | int | 720 | 矫正图像高 (px) |
| `hfov_deg` | float | 175.0 | 鱼眼水平视场角 (°) |

### 输出

`dict` — 结构如下：

```python
{
    "method": "boat_geo+visual_icp",    # 标定方法标识
    "quality": 0.33,                     # 0~1，1=全部成功

    "lidar_extrinsics": {
        "front": {
            "R": [[1,0,0],[0,1,0],[0,0,1]],  # 3×3 旋转矩阵
            "t": [7.0, 0.0, 0.0]              # 3×1 平移向量 (m)
        },
        "left": {"R": [...], "t": [...]},
        ...  # 标定成功的 LiDAR（可能不全）
    },

    "camera_extrinsics": {
        "front": {"R": [...], "t": [...]},
        ...  # 从 LiDAR 外参 + 10cm 约束推导
    },

    "camera_intrinsics": {
        "front": [[419.1, 0, 640], [0, 419.1, 360], [0, 0, 1]],
        ...  # 从文档规格估算的 3×3 内参矩阵
    },

    "calib_quality": {
        "per_pair": {
            "left": {"method": "visual_icp", "success": False},
            "right": {"method": "visual_icp", "success": True},
            ...
        },
        "overall": 0.33
    }
}
```

### 辅助函数

**稠密点云构建**（多帧叠加）：

```python
from competition_user.algos.calibration.calib import accumulate_pointclouds

# lidar_frames: list of (x, y, z, intensity, ring) tuples
# 每帧都是经过 lidar_preprocess 处理后的点云
frames = [(x1, y1, z1, i1, r1), (x2, y2, z2, i2, r2), ...]
x, y, z, intensities = accumulate_pointclouds(
    frames,
    voxel_size=0.05  # 去重体素 (m)
)
# x, y, z: 一维 float32 数组
# intensities: 一维 float32 数组 (若传入则保留均值，否则全零)
```

**单独标定一对雷达**（不通过 `calibrate_extrinsics` 入口）：

```python
from competition_user.algos.calibration.calib import LidarLidarCalibrator

calib = LidarLidarCalibrator(config={'voxel_size': 0.2})
# calibrate_all 标定多对
extrinsics = calib.calibrate_all(
    lidar_data,       # {name: (x,y,z,i,r)}, 含所有待标定雷达
    reference_lidar='front_lidar',
    voxel_size=0.2,
    camera_inits=None,    # dict 或 None
)

# calibrate_pair 标定一对
T = calib.calibrate_pair(
    (src_x, src_y, src_z),   # 源雷达 xyz
    (tgt_x, tgt_y, tgt_z),   # 目标雷达 xyz
    source_name='left_lidar',
    target_name='front_lidar',
    voxel_size=0.1,
    camera_init=None,        # 4×4 初值 或 None
)
# T: 4×4 变换矩阵 T_src→tgt
```

### 完整调用示例

```python
import numpy as np
from competition_user.algos.preprocess.lidar_preprocess import lidar_preprocess
from competition_user.algos.calibration.calib import (
    calibrate_extrinsics, accumulate_pointclouds
)

# ===== 步骤 1: 逐帧预处理 + 多帧叠加 =====
all_frames = {sn: [] for sn in ['front','back','left','left2','right','right2']}

for frame_idx in range(100):
    for sn, raw_pts in read_lidar_frame(frame_idx).items():
        proc = lidar_preprocess({sn: raw_pts})  # raw_pts: (N,3) f4
        if sn in proc:
            pts = proc[sn]
            all_frames[sn].append((
                pts[:,0].astype('f4'), pts[:,1].astype('f4'),
                pts[:,2].astype('f4'),
                np.zeros(len(pts), 'f4'),      # intensity
                np.zeros(len(pts), 'u2'),       # ring
            ))

# 叠加为稠密点云
dense_clouds = {}
for sn, frames in all_frames.items():
    if frames:
        x, y, z, _ = accumulate_pointclouds(frames, voxel_size=0.05)
        dense_clouds[sn] = (x, y, z)

# ===== 步骤 2: 相机图像获取（鱼眼矫正后）=====
rectified = {sn: load_rectified_image(sn) for sn in ['front','back','left','left2','right','right2']}

# ===== 步骤 3: 标定 =====
result = calibrate_extrinsics(
    rectified_images=rectified,
    preprocessed_clouds=dense_clouds,
    config={'use_visual_init': True, 'voxel_size': 0.2},
)

# ===== 步骤 4: 使用结果 =====
print(f'Quality: {result["quality"]}')
for sn, extr in result['lidar_extrinsics'].items():
    print(f'{sn}: t={extr["t"]}')
```

---

## 三、注意事项

1. **forward (EMX) 无配对相机**：通过 FPFH+RANSAC 标定，视觉初值对它不可用
2. **标定输出不保证 7/7 全**：依赖 ICP 收敛质量，缺失的传感器走下游兜底
3. **SOR 可能砍掉稀疏障碍物**：如果标定效果差，优先调大 `sor_std`（默认 2.0 → 3.0）或关 SOR（改 `if not _is_emx(name)` 为 `if False`）
4. **ring 数据**：预处理产出的点云已丢失 ring（SOR+体素合并后无法 1:1 映射），仅预处理内部使用
5. **坐标系**：所有外参输出为船体坐标系（原点=front/back 中点，x=船艏，z=竖直向上）
