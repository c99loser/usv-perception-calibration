"""
================================================================================
 filters.py — 雷达点云预处理核心算法库
================================================================================
纯 numpy/scipy 函数，无 ROS2/MCAP 依赖。可直接被离线批处理和在线节点复用。

算法来源:
  Step 1: Open3D remove_non_finite_points / PCL removeNaNFromPointCloud
  Step 2: autoware PassthroughFilter + CropBoxFilter + Open3D SOR/ROR
  Step 3: autoware PassthroughFilter / CropBoxFilter
  Step 4: Open3D voxel_down_sample / autoware DownsampleFilter (VoxelGrid)

传感器分类:
  Airy (6个): front/back/left/left2/right/right2 — 48线, 360°×90°, P=115°
  EMX  (1个): forward — 96线, 120°×15°, P=0°
================================================================================
"""

import numpy as np
from scipy.spatial import cKDTree

# ==========================================================================
# PointCloud2 二进制布局 (32 bytes/point)
# ==========================================================================
PCD_DTYPE = np.dtype([
    ('x',         'f4'),
    ('y',         'f4'),
    ('z',         'f4'),
    ('_pad1',     'V4'),
    ('intensity', 'f4'),
    ('_pad2',     'V4'),
    ('ring',      'u2'),
    ('_pad3',     'V6'),
])

# ==========================================================================
# 传感器常量
# ==========================================================================
AIRY_TOPICS = {
    '/wamv/sensors/lidars/front_lidar_sensor/points',
    '/wamv/sensors/lidars/back_lidar_sensor/points',
    '/wamv/sensors/lidars/left_lidar_sensor/points',
    '/wamv/sensors/lidars/left2_lidar_sensor/points',
    '/wamv/sensors/lidars/right_lidar_sensor/points',
    '/wamv/sensors/lidars/right2_lidar_sensor/points',
}
EMX_TOPIC = '/wamv/sensors/lidars/forward_lidar_sensor/points'
ALL_LIDAR_TOPICS = list(AIRY_TOPICS) + [EMX_TOPIC]


def is_emx(topic):
    """判断 topic 是否为 EMX 雷达"""
    return topic == EMX_TOPIC


def validate_pointcloud2_layout(point_step, row_step, width, height,
                                is_bigendian, fields):
    """验证本管线支持的 PointCloud2 二进制布局。

    当前传感器输出是小端、连续的一维 32-byte 点云。显式校验可以避免在
    schema 变化时静默地把字段读错，产生看似可用但坐标错误的结果。
    ``fields`` 可为 ROS2 PointField 对象或具有同名属性的对象。
    """
    expected = {
        'x': (0, 7, 1), 'y': (4, 7, 1), 'z': (8, 7, 1),
        'intensity': (16, 7, 1), 'ring': (24, 4, 1),
    }
    actual = {field.name: (field.offset, field.datatype, field.count)
              for field in fields}
    errors = []
    if is_bigendian:
        errors.append('big-endian point clouds are not supported')
    if point_step != PCD_DTYPE.itemsize:
        errors.append(f'point_step={point_step}, expected {PCD_DTYPE.itemsize}')
    if row_step != width * point_step:
        errors.append(f'row_step={row_step}, expected width * point_step={width * point_step}')
    if actual != expected:
        errors.append(f'fields={actual}, expected {expected}')
    if errors:
        raise ValueError('Unsupported PointCloud2 layout: ' + '; '.join(errors))


# ======================================================================
# Step 1: 无效点去除
# ======================================================================
def step1_remove_invalid(raw_points):
    """
    移除 NaN / Inf / 零距离点。

    参考: Open3D remove_non_finite_points, PCL removeNaNFromPointCloud

    参数:
        raw_points: numpy structured array (dtype=PCD_DTYPE)

    返回:
        (x, y, z, intensity, ring): 各字段的有效值数组
    """
    x, y, z = raw_points['x'], raw_points['y'], raw_points['z']
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    # hypot 避免 float32 平方在极端输入上溢出。
    dist = np.hypot(np.hypot(x, y), z)
    valid = valid & (dist > 0.0)

    return (
        x[valid], y[valid], z[valid],
        raw_points['intensity'][valid],
        raw_points['ring'][valid],
    )


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
def step2b_z_clip(z, topic, z_min_airy=-3.0, z_min_emx=-12.0):
    """
    移除水面以下回波。

    参考: autoware CropBoxFilter / PassthroughFilter (z 轴)

    Airy: P=115°朝下, z 天然 ≥0, z_min_airy 作为安全网
    EMX:  P=0°水平, z_min_emx 极保守, 避免误删远距合法回波
    """
    if topic in AIRY_TOPICS:
        return z >= z_min_airy
    else:
        return z >= z_min_emx


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
def step3_range_filter(x, y, z, topic,
                       min_range=0.5,
                       max_range_airy=60.0,
                       max_range_emx=200.0):
    """
    裁剪有效距离范围。

    参考: autoware PassthroughFilter / CropBoxFilter

    Airy: min=0.5m (去自船反射), max=60m (物理量程)
    EMX:  min=0.5m, max=200m (全量程, 用于障碍物检测+融合)
    """
    dist = np.hypot(np.hypot(x, y), z)
    max_range = max_range_emx if is_emx(topic) else max_range_airy
    return (dist >= min_range) & (dist <= max_range)


# ======================================================================
# Step 4: 体素降采样
# ======================================================================
def step4_voxel_downsample(x, y, z, intensity, ring, topic,
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

    voxel_size = voxel_emx if is_emx(topic) else voxel_airy
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

    # ring 众数。按 (voxel, ring) 排序并统计连续段，避免为每个体素
    # 全量扫描 inv（旧实现复杂度近似 O(体素数 × 点数)）。并列时选择较小 ring，
    # 与 np.unique(...); np.argmax(...) 的原有行为一致。
    order = np.lexsort((ring, inv))
    sorted_voxel = inv[order]
    sorted_ring = ring[order]
    run_start = np.empty(N, dtype=bool)
    run_start[0] = True
    run_start[1:] = ((sorted_voxel[1:] != sorted_voxel[:-1]) |
                     (sorted_ring[1:] != sorted_ring[:-1]))
    start_idx = np.flatnonzero(run_start)
    run_voxel = sorted_voxel[start_idx]
    run_ring = sorted_ring[start_idx]
    run_counts = np.diff(np.append(start_idx, N))
    max_counts = np.zeros(n_voxels, dtype=run_counts.dtype)
    np.maximum.at(max_counts, run_voxel, run_counts)
    candidates = np.flatnonzero(run_counts == max_counts[run_voxel])
    chosen_run = np.full(n_voxels, len(run_voxel), dtype=np.int64)
    np.minimum.at(chosen_run, run_voxel[candidates], candidates)
    r_ds = run_ring[chosen_run].astype(np.uint16, copy=False)

    return x_ds, y_ds, z_ds, i_ds, r_ds


# ======================================================================
# 完整流水线 (四步串联, 单帧处理)
# ======================================================================
class PreprocessingPipeline:
    """
    单帧点云预处理流水线。

    用法:
        pipe = PreprocessingPipeline()
        x, y, z, intensity, ring = pipe.process(topic, raw_points)
    """

    def __init__(self,
                 intensity_threshold=0.0,      # 2a
                 z_min_airy=-3.0,              # 2b
                 z_min_emx=-12.0,              # 2b
                 sor_nb=20, sor_std=2.0,       # 2c
                 ror_radius=0.3, ror_min=3,    # 2d
                 min_range=0.5,                # 3
                 max_range_airy=60.0,          # 3
                 max_range_emx=200.0,          # 3
                 voxel_airy=0.08,              # 4
                 voxel_emx=0.10):              # 4
        self.intensity_threshold = intensity_threshold
        self.z_min_airy = z_min_airy
        self.z_min_emx = z_min_emx
        self.sor_nb = sor_nb
        self.sor_std = sor_std
        self.ror_radius = ror_radius
        self.ror_min = ror_min
        self.min_range = min_range
        self.max_range_airy = max_range_airy
        self.max_range_emx = max_range_emx
        self.voxel_airy = voxel_airy
        self.voxel_emx = voxel_emx

    def process(self, topic, raw_points):
        """
        对单帧点云执行完整四步预处理。

        参数:
            topic:       str,  雷达 ROS2 topic
            raw_points:  numpy structured array (dtype=PCD_DTYPE)

        返回:
            (x, y, z, intensity, ring): 各字段的处理后数组
        """
        # Step 1
        x, y, z, intensity, ring = step1_remove_invalid(raw_points)
        if len(x) == 0:
            return x, y, z, intensity, ring

        # Step 2a
        keep = step2a_intensity_filter(intensity, self.intensity_threshold)
        x, y, z = x[keep], y[keep], z[keep]
        intensity, ring = intensity[keep], ring[keep]
        if len(x) == 0:
            return x, y, z, intensity, ring

        # Step 2b
        keep = step2b_z_clip(z, topic, self.z_min_airy, self.z_min_emx)
        x, y, z = x[keep], y[keep], z[keep]
        intensity, ring = intensity[keep], ring[keep]
        if len(x) == 0:
            return x, y, z, intensity, ring

        # Step 2c (EMX 跳过)
        if not is_emx(topic):
            keep, _ = step2c_sor(x, y, z, self.sor_nb, self.sor_std)
            x, y, z = x[keep], y[keep], z[keep]
            intensity, ring = intensity[keep], ring[keep]
        if len(x) == 0:
            return x, y, z, intensity, ring

        # Step 2d (EMX 跳过)
        if not is_emx(topic):
            keep, _ = step2d_ror(x, y, z, self.ror_radius, self.ror_min)
            x, y, z = x[keep], y[keep], z[keep]
            intensity, ring = intensity[keep], ring[keep]
        if len(x) == 0:
            return x, y, z, intensity, ring

        # Step 3
        keep = step3_range_filter(x, y, z, topic,
                                  self.min_range,
                                  self.max_range_airy,
                                  self.max_range_emx)
        x, y, z = x[keep], y[keep], z[keep]
        intensity, ring = intensity[keep], ring[keep]
        if len(x) == 0:
            return x, y, z, intensity, ring

        # Step 4
        x, y, z, intensity, ring = step4_voxel_downsample(
            x, y, z, intensity, ring, topic,
            self.voxel_airy, self.voxel_emx)

        return x, y, z, intensity, ring
