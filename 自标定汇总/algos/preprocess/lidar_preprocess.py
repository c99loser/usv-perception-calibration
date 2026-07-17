"""
================================================================================
 lidar_preprocess.py — 激光雷达预处理（算法 + 适配器）
================================================================================

算法核心：从 filters.py 完整迁移，100% 保留原始逻辑不变。
适配器 lidar_preprocess()：将 {name: (N,3)} dict → 适配 → 预处理 → {name: (M,3)} dict

传感器分类:
  Airy (6个): front/back/left/left2/right/right2 — 48线, 360°×90°, P=115°
  EMX  (1个): forward — 96线, 120°×15°, P=0°

============================================================================
第 0 部分：辅助常量与函数（不变）
============================================================================
"""

import numpy as np
from scipy.spatial import cKDTree

# ---- 雷达名称 → 类型识别（适配器用）----
AIRY_NAMES = {'front', 'back', 'left', 'left2', 'right', 'right2'}
EMX_NAME = 'forward'
ALL_LIDAR_NAMES = list(AIRY_NAMES) + [EMX_NAME]


def _is_emx(name):
    """判断是否为 EMX 雷达"""
    return name == EMX_NAME




# ======================================================================
# Step 2a: 强度滤波
# ======================================================================
def step2a_intensity_filter(intensity, threshold=0.0):
    """
    移除低强度水面反射点。

    参考: autoware PassthroughFilter (intensity 字段)

    注意: 仿真环境 intensity 全为 0, threshold=0 自动穿透。
          实测数据建议 threshold=0.01~0.02。
    """
    return intensity >= threshold


# ======================================================================
# Step 2b: Z 轴水面裁剪
# ======================================================================
def step2b_z_clip(z, name, z_min_airy=-3.0, z_min_emx=-12.0,
                   z_max_airy=5.0, z_max_emx=8.0):
    """
    移除水面以下回波和空中高位噪点。

    参考: autoware CropBoxFilter / PassthroughFilter (z 轴)

    Airy: P=115°朝下, z 天然 ≥0, z_min_airy 作安全网; z_max=5m 过滤空中共振
    EMX:  P=0°水平, z_min_emx 极保守; z_max=8m 保留远处立柱/船只等高目标
    """
    if name in AIRY_NAMES:
        return (z >= z_min_airy) & (z <= z_max_airy)
    else:
        return (z >= z_min_emx) & (z <= z_max_emx)


# ======================================================================
# Step 2c: 统计离群点去除 (SOR)
# ======================================================================
def step2c_sor(x, y, z, nb_neighbors=20, std_ratio=2.0):
    """
    基于 k 近邻距离分布的离群点滤除。

    参考: Open3D statistical_outlier_removal

    原理:
      1. 计算每个点到 k 近邻的平均距离 d_i
      2. 全局 μ = mean(d_i), σ = std(d_i)
      3. d_i > μ + std_ratio × σ → 离群点

    EMX 应跳过此步: 200m 远距点间距破坏全局统计假设。
    """
    N = len(x)
    if N < nb_neighbors + 1:
        return np.ones(N, dtype=bool), {'n_outliers': 0}

    xyz = np.column_stack([x, y, z])
    tree = cKDTree(xyz)
    k = min(nb_neighbors + 1, N)
    distances, _ = tree.query(xyz, k=k)

    if distances.shape[1] > 1:
        mean_dists = distances[:, 1:].mean(axis=1)
    else:
        mean_dists = distances[:, 0]

    mu = mean_dists.mean()
    sigma = mean_dists.std()
    threshold = mu + std_ratio * sigma
    keep = mean_dists <= threshold

    return keep, {'n_outliers': int(np.sum(~keep))}


# ======================================================================
# Step 2d: 半径离群点去除 (ROR)
# ======================================================================
def step2d_ror(x, y, z, radius=0.3, min_points=3):
    """
    固定半径内邻居数不足 → 离群点。

    参考: Open3D radius_outlier_removal
          autoware RadiusSearch2DOutlierFilter

    EMX 应跳过此步: 200m 远距点间距 >1m, 固定半径搜索失效。
    """
    N = len(x)
    if N == 0:
        return np.zeros(0, dtype=bool), {'n_outliers': 0}

    xyz = np.column_stack([x, y, z])
    tree = cKDTree(xyz)
    counts = tree.query_ball_point(xyz, r=radius, return_length=True)
    keep = counts >= min_points

    return keep, {'n_outliers': int(np.sum(~keep))}


# ======================================================================
# Step 3: 距离范围裁剪
# ======================================================================
def step3_range_filter(x, y, z, name,
                       min_range=0.3,
                       max_range_airy=60.0,
                       max_range_emx=200.0):
    """
    裁剪有效距离范围。

    参考: autoware PassthroughFilter / CropBoxFilter

    Airy: min=0.5m (去自船反射), max=60m (物理量程)
    EMX:  min=0.5m, max=200m (全量程, 用于障碍物检测+融合)
    """
    dist = np.sqrt(x**2 + y**2 + z**2)
    max_range = max_range_emx if _is_emx(name) else max_range_airy
    return (dist >= min_range) & (dist <= max_range)


# ======================================================================
# Step 4: 体素降采样
# ======================================================================
def step4_voxel_downsample(x, y, z, intensity, ring, name,
                            voxel_airy=0.08, voxel_emx=0.10):
    """
    体素网格降采样 (质心法)。

    参考: Open3D voxel_down_sample (VoxelGrid Centroid)
          autoware DownsampleFilter (VoxelGrid 模式)

    原理:
      1. 将空间划分为 voxel_size 的立方体
      2. 同一体素内 xyz 取质心, intensity 取均值, ring 取众数
    """
    N = len(x)
    if N == 0:
        return (np.array([], dtype='f4'), np.array([], dtype='f4'),
                np.array([], dtype='f4'), np.array([], dtype='f4'),
                np.array([], dtype='u2'))

    voxel_size = voxel_emx if _is_emx(name) else voxel_airy
    xyz = np.column_stack([x, y, z])

    # 体素索引
    vx = np.floor(xyz / voxel_size).astype(np.int64)
    vrec = np.rec.fromarrays([vx[:, 0], vx[:, 1], vx[:, 2]], names='x,y,z')
    _, inv, cnt = np.unique(vrec, return_inverse=True, return_counts=True)
    n_voxels = len(cnt)

    # 质心
    x_ds = np.bincount(inv, weights=xyz[:, 0], minlength=n_voxels) / cnt
    y_ds = np.bincount(inv, weights=xyz[:, 1], minlength=n_voxels) / cnt
    z_ds = np.bincount(inv, weights=xyz[:, 2], minlength=n_voxels) / cnt
    i_ds = np.bincount(inv, weights=intensity, minlength=n_voxels) / cnt

    # ring 众数
    r_ds = np.zeros(n_voxels, dtype=np.uint16)
    for vi in range(n_voxels):
        mask = (inv == vi)
        vals, counts = np.unique(ring[mask], return_counts=True)
        r_ds[vi] = vals[np.argmax(counts)]

    return x_ds, y_ds, z_ds, i_ds, r_ds


# ======================================================================
# 完整单帧流水线（不变）
# ======================================================================
def _process_single(name, x, y, z, intensity, ring,
                    intensity_threshold=0.0,
                    z_min_airy=-3.0, z_min_emx=-12.0,
                    z_max_airy=5.0, z_max_emx=8.0,
                    sor_nb=20, sor_std=2.0,
                    ror_radius=0.3, ror_min=3,
                    min_range=0.3, max_range_airy=60.0, max_range_emx=200.0,
                    voxel_airy=0.08, voxel_emx=0.10):
    """
    对单帧点云执行完整四步预处理。算法逻辑与 filters.py 完全一致。

    参数:
        name: str, 雷达名 (front/back/left/left2/right/right2/forward)
        x, y, z, intensity, ring: 各字段数组

    返回:
        (x, y, z, intensity, ring): 各字段的处理后数组
    """
    if len(x) == 0:
        return x, y, z, intensity, ring

    # Step 1: 无效点去除（NaN / Inf / 零距离）
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    dist = np.sqrt(x**2 + y**2 + z**2)
    valid = valid & (dist > 0.0)
    x, y, z = x[valid], y[valid], z[valid]
    intensity, ring = intensity[valid], ring[valid]
    if len(x) == 0:
        return x, y, z, intensity, ring

    # Step 2a: 强度滤波
    keep = step2a_intensity_filter(intensity, intensity_threshold)
    x, y, z = x[keep], y[keep], z[keep]
    intensity, ring = intensity[keep], ring[keep]
    if len(x) == 0:
        return x, y, z, intensity, ring

    # Step 2b: Z 轴裁剪
    keep = step2b_z_clip(z, name, z_min_airy, z_min_emx,
                         z_max_airy, z_max_emx)
    x, y, z = x[keep], y[keep], z[keep]
    intensity, ring = intensity[keep], ring[keep]
    if len(x) == 0:
        return x, y, z, intensity, ring

    # Step 2c: SOR (EMX 跳过)
    if not _is_emx(name):
        keep, _ = step2c_sor(x, y, z, sor_nb, sor_std)
        x, y, z = x[keep], y[keep], z[keep]
        intensity, ring = intensity[keep], ring[keep]
    if len(x) == 0:
        return x, y, z, intensity, ring

    # Step 2d: ROR (EMX 跳过)
    if not _is_emx(name):
        keep, _ = step2d_ror(x, y, z, ror_radius, ror_min)
        x, y, z = x[keep], y[keep], z[keep]
        intensity, ring = intensity[keep], ring[keep]
    if len(x) == 0:
        return x, y, z, intensity, ring

    # Step 3: 距离范围裁剪
    keep = step3_range_filter(x, y, z, name, min_range,
                              max_range_airy, max_range_emx)
    x, y, z = x[keep], y[keep], z[keep]
    intensity, ring = intensity[keep], ring[keep]
    if len(x) == 0:
        return x, y, z, intensity, ring

    # Step 4: 体素降采样
    x, y, z, intensity, ring = step4_voxel_downsample(
        x, y, z, intensity, ring, name, voxel_airy, voxel_emx)

    return x, y, z, intensity, ring


# ======================================================================
# 适配器：lidar_preprocess() —— 对接感知节点主流水线
# ======================================================================
def lidar_preprocess(lidar_bursts, imu_data, config=None):
    """
    对 7 路激光雷达点云做批量预处理。

    此函数是感知节点流水线中的"步骤一·LiDAR 预处理"。
    算法核心（step1~step4）完全保留 filters.py 原始逻辑。

    参数:
        lidar_bursts: dict
            {lidar_name: np.ndarray (N, 3) float32}
            lidar_name ∈ {front, back, left, left2, right, right2, forward}
            每路数据是点云 xyz 坐标（传感器的 local 坐标系）

        imu_data: 序列/列表 或 None
            预留接口，用于后续运动畸变矫正。
            当前版本未使用。

        config: dict 或 None
            {
                'intensity_threshold': float,    # 默认 0.0
                'z_min_airy':   float,           # 默认 -3.0
                'z_min_emx':    float,           # 默认 -12.0
                'z_max_airy':   float,           # 默认 5.0
                'z_max_emx':    float,           # 默认 8.0
                'sor_nb':       int,             # 默认 20
                'sor_std':      float,           # 默认 2.0
                'ror_radius':   float,           # 默认 0.3
                'ror_min':      int,             # 默认 3
                'min_range':    float,           # 默认 0.3
                'max_range_airy': float,         # 默认 60.0
                'max_range_emx': float,          # 默认 200.0
                'voxel_airy':   float,           # 默认 0.08
                'voxel_emx':    float,           # 默认 0.10
            }

    返回:
        dict: {lidar_name: np.ndarray (M, 3) float32}
            预处理后的点云，每路点的数量 M ≤ N。
            未提供数据的雷达不会出现在返回字典中。
    """
    if config is None:
        config = {}

    # ---- 解析配置（带默认值）----
    intensity_threshold = config.get('intensity_threshold', 0.0)
    z_min_airy   = config.get('z_min_airy',   -3.0)
    z_min_emx    = config.get('z_min_emx',    -12.0)
    z_max_airy   = config.get('z_max_airy',    5.0)
    z_max_emx    = config.get('z_max_emx',     8.0)
    sor_nb       = config.get('sor_nb',       20)
    sor_std      = config.get('sor_std',      2.0)
    ror_radius   = config.get('ror_radius',   0.3)
    ror_min      = config.get('ror_min',      3)
    min_range    = config.get('min_range',    0.3)
    max_range_airy = config.get('max_range_airy', 60.0)
    max_range_emx  = config.get('max_range_emx',  200.0)
    voxel_airy   = config.get('voxel_airy',   0.08)
    voxel_emx    = config.get('voxel_emx',    0.10)

    # ---- 逐路处理 ----
    result = {}
    for name, points in lidar_bursts.items():
        if points is None or len(points) == 0:
            continue

        # 适配①：按列数检测字段 (3列=仅xyz, 5列=含intensity+ring)
        x = points[:, 0].astype(np.float32)
        y = points[:, 1].astype(np.float32)
        z = points[:, 2].astype(np.float32)

        if points.shape[1] >= 5:
            intensity = points[:, 3].astype(np.float32)
            ring = points[:, 4].astype(np.uint16)
        else:
            intensity = np.zeros(len(points), dtype=np.float32)
            ring = np.zeros(len(points), dtype=np.uint16)

        # 调用核心流水线（算法完全不变）
        x, y, z, intensity, ring = _process_single(
            name, x, y, z, intensity, ring,
            intensity_threshold,
            z_min_airy, z_min_emx, z_max_airy, z_max_emx,
            sor_nb, sor_std,
            ror_radius, ror_min,
            min_range, max_range_airy, max_range_emx,
            voxel_airy, voxel_emx,
        )

        # 适配②：输出只保留 (M,3) xyz
        if len(x) > 0:
            result[name] = np.column_stack([x, y, z]).astype(np.float32)
        else:
            result[name] = np.zeros((0, 3), dtype=np.float32)

    return result
