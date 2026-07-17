"""
================================================================================
 Step 2: 水面噪声滤除 (Water Surface Noise Filtering)
================================================================================
用途: 对 Step1 清洗后的点云执行四级级联噪声滤除:
      2a. 强度滤波 —— 移除低强度水面反射点
      2b. Z轴水面裁剪 —— 移除水面以下的虚假回波 (Airy已天然过滤, 主要针对EMX)
      2c. 统计离群点去除(SOR) —— 移除空间孤立的飞溅/反射散斑
      2d. 半径离群点去除(ROR) —— 二次确认，移除更大尺度的孤立噪声簇

参考开源方案:
      2a: autoware PassthroughFilter (intensity 字段)
          来源: autowarefoundation/autoware_universe → passthrough_filter
      2b: autoware CropBoxFilter / PassthroughFilter (z 轴)
          来源: autowarefoundation/autoware_universe → crop_box_filter
      2c: Open3D statistical_outlier_removal
          来源: isl-org/Open3D
      2d: Open3D radius_outlier_removal / autoware RadiusSearch2DOutlierFilter
          来源: isl-org/Open3D, autowarefoundation/autoware_universe → outlier_filter

注意:
      - 仿真环境 (Gazebo) 中 intensity 全为 0，Step 2a 在此环境下自动穿透
      - 强度滤波代码保留完整实现，切换到真实 LiDAR 数据后自动生效
      - 基于 scipy.spatial.cKDTree 实现 SOR/ROR，替代不可用的 Open3D (py3.13 无预编译包)

输入: Step1 输出 MCAP 中的 _clean topic 点云
输出: 含有 _step2 后缀 topic 的 MCAP 文件
================================================================================
"""

import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
try:
    from mcap.reader import make_reader
    from mcap_ros2.reader import read_ros2_messages
    from mcap_ros2._dynamic import serialize_dynamic
    from mcap.writer import Writer as McapWriter
except ImportError:
    make_reader = read_ros2_messages = serialize_dynamic = McapWriter = None
try:
    from .streaming_mcap import process_mcap_streaming
except ImportError:
    from streaming_mcap import process_mcap_streaming

# ==============================================================================
# PointCloud2 二进制布局 (与 Step1 一致)
# ==============================================================================
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

LIDAR_TOPICS = [
    '/wamv/sensors/lidars/front_lidar_sensor/points',
    '/wamv/sensors/lidars/back_lidar_sensor/points',
    '/wamv/sensors/lidars/left_lidar_sensor/points',
    '/wamv/sensors/lidars/left2_lidar_sensor/points',
    '/wamv/sensors/lidars/right_lidar_sensor/points',
    '/wamv/sensors/lidars/right2_lidar_sensor/points',
    '/wamv/sensors/lidars/forward_lidar_sensor/points',
]

# 处理后 topic 后缀
STEP1_SUFFIX = '_clean'     # Step1 输出: 去除 NaN 后的点云
STEP2_SUFFIX = '_step2'     # Step2 输出: 水面噪声滤除后的点云

# ------------------------------------------------------------------
# 传感器分类: Airy (6个半球面) vs EMX (1个前向)
# ------------------------------------------------------------------
AIRY_TOPICS = {t for t in LIDAR_TOPICS if 'forward' not in t}
EMX_TOPIC = '/wamv/sensors/lidars/forward_lidar_sensor/points'


# ==============================================================================
# 2a. 强度滤波
# ==============================================================================
def filter_by_intensity(intensity, threshold=0.01):
    """
    移除低强度点。
    
    原理: 水面反射点强度远低于障碍物(柱/浮球/船体)反射。
          Gazebo 仿真中 intensity 全为 0，此滤波自动穿透(不删除任何点)。
          切换到真实 LiDAR 后阈值需根据实测数据调参。
    
    参考: autoware PassthroughFilter 应用于 intensity 字段
          来源: autowarefoundation/autoware_universe → passthrough_filter
          实现方式: filter_field_name='intensity', filter_limit_min=threshold
    
    参数:
        intensity: (N,) float32, 强度值数组
        threshold: float, 强度下界 (低于此值视为水面反射)
    
    返回:
        keep: (N,) bool, True=保留, False=滤除
    """
    return intensity >= threshold


# ==============================================================================
# 2b. Z 轴水面裁剪
# ==============================================================================
def filter_below_water(z, topic, z_min_airy=-3.0, z_min_emx=-12.0):
    """
    移除水面以下的虚假回波。
    
    原理: 雷达在 base_link 坐标系中的点云，z 轴为真实世界高度(z≈0=水面)。
          Airy 雷达(P=115°朝下)的 z 值天然 ≥0 (Gazebo 物理不产生水下回波)。
          EMX 雷达(P=0°水平, 200m远距)远距离波束天然低于水面，z 裁剪须极保守。
    
    参考: autoware CropBoxFilter / PassthroughFilter (z 轴)
          来源: autowarefoundation/autoware_universe → crop_box_filter
          实现方式: filter_field_name='z', filter_limit_min=z_min
    
    参数:
        z:      (N,) float32, 点云 z 坐标 (base_link 系，z≈0=水面)
        topic:  str, 雷达 topic 名 (用于区分 Airy/EMX)
        z_min_airy: float, Airy 雷达 z 轴下界 (留波浪余量)
        z_min_emx:  float, EMX 雷达 z 轴下界 (极保守，仅裁剪极端值)
    
    返回:
        keep: (N,) bool
    """
    if topic in AIRY_TOPICS:
        # Airy: z 天然 ≥0, 此处作为安全网保留 z≥-3m
        return z >= z_min_airy
    else:
        # EMX: 水平安装(P=0°), 96线远距(200m), 远距离波束自然低于水面。
        #      z_min_emx 需保守设置，避免误删合法远距回波。
        return z >= z_min_emx


# ==============================================================================
# 2c. 统计离群点去除 (SOR)
# ==============================================================================
def statistical_outlier_removal(xyz, nb_neighbors=30, std_ratio=0.8):
    """
    基于 k 近邻距离分布的离群点滤除。
    
    原理 (等效于 Open3D statistical_outlier_removal):
        1. 对每个点，计算其到 k 个最近邻的平均距离 d_i
        2. 对所有点的 d_i 计算全局均值 μ 和标准差 σ
        3. 若 d_i > μ + std_ratio × σ，则该点为离群点(空间孤点)
    
    参考: Open3D statistical_outlier_removal
          isl-org/Open3D (cpp/open3d/geometry/PointCloud.cpp)
          算法: StatisticalOutlierRemoval 类
    
    参数:
        xyz:          (N, 3) float32, 点云坐标
        nb_neighbors: int, k 近邻数 (激进配置=30)
        std_ratio:    float, 标准差倍数 (激进配置=0.8, 越小移除越多)
    
    返回:
        keep: (N,) bool, True=保留, False=离群点
        stats: dict, 统计信息
    """
    N = len(xyz)
    if N < nb_neighbors + 1:
        return np.ones(N, dtype=bool), {'mean_dist': 0, 'std_dist': 0, 'threshold': 0, 'n_outliers': 0}

    # 构建 KD-Tree (scipy 替代 Open3D)
    tree = cKDTree(xyz)
    
    # 查询每个点的 k+1 近邻 (k+1 因为包含自身, 即距离=0)
    k = min(nb_neighbors + 1, N)
    distances, _ = tree.query(xyz, k=k)
    
    # 排除自身 (距离=0 总是排在第一位)
    if distances.shape[1] > 1:
        mean_dists = distances[:, 1:].mean(axis=1)  # 到 k 个近邻的平均距离
    else:
        mean_dists = distances[:, 0]  # 只有自身的情况 (极端)
    
    # 全局统计
    mu = mean_dists.mean()
    sigma = mean_dists.std()
    threshold = mu + std_ratio * sigma
    
    keep = mean_dists <= threshold
    
    stats = {
        'mean_dist': mu,
        'std_dist': sigma,
        'threshold': threshold,
        'n_outliers': int(np.sum(~keep)),
    }
    return keep, stats


# ==============================================================================
# 2d. 半径离群点去除 (ROR)
# ==============================================================================
def radius_outlier_removal(xyz, radius=1.0, min_points=10):
    """
    基于固定半径内邻居数量的离群点滤除。
    
    原理 (等效于 Open3D radius_outlier_removal):
        对每个点，统计半径 radius 内的邻居数。
        若邻居数 < min_points，则该点为离群点(孤立的噪声簇)。
    
    与 SOR 互补: SOR 关注局部密度异常，ROR 关注绝对空间孤立度。
    SOR 之后执行 ROR 作为二次确认。
    
    参考: Open3D radius_outlier_removal
          isl-org/Open3D
          autoware RadiusSearch2DOutlierFilter (2D 平面版本)
          autowarefoundation/autoware_universe → outlier_filter
    
    参数:
        xyz:        (N, 3) float32, 点云坐标
        radius:     float, 搜索半径 (m)
        min_points: int, 最少邻居数 (≤此值视为离群点)
    
    返回:
        keep: (N,) bool
        stats: dict
    """
    N = len(xyz)
    if N == 0:
        return np.zeros(0, dtype=bool), {'n_outliers': 0}

    tree = cKDTree(xyz)
    
    # 对每个点，查询半径内的邻居数 (包含自身)
    counts = tree.query_ball_point(xyz, r=radius, return_length=True)
    keep = counts >= min_points
    
    stats = {
        'n_outliers': int(np.sum(~keep)),
        'radius': radius,
        'min_points': min_points,
    }
    return keep, stats


# ==============================================================================
# PointCloud2 序列化 (与 Step1 共用)
# ==============================================================================
def build_pointcloud2_dict(xyz, intensity, ring, frame_id, stamp_sec, stamp_nsec):
    """从 numpy 数组构建 PointCloud2 dict"""
    N = len(xyz)
    buf = np.zeros(N, dtype=PCD_DTYPE)
    buf['x'] = xyz[:, 0]
    buf['y'] = xyz[:, 1]
    buf['z'] = xyz[:, 2]
    buf['intensity'] = intensity
    buf['ring'] = ring

    return {
        'header': {'stamp': {'sec': stamp_sec, 'nanosec': stamp_nsec},
                   'frame_id': frame_id},
        'height': 1, 'width': N,
        'fields': [
            {'name': 'x',         'offset': 0,  'datatype': 7, 'count': 1},
            {'name': 'y',         'offset': 4,  'datatype': 7, 'count': 1},
            {'name': 'z',         'offset': 8,  'datatype': 7, 'count': 1},
            {'name': 'intensity', 'offset': 16, 'datatype': 7, 'count': 1},
            {'name': 'ring',      'offset': 24, 'datatype': 4, 'count': 1},
        ],
        'is_bigendian': False,
        'point_step': 32,
        'row_step': 32 * N,
        'data': buf.tobytes(),
        'is_dense': True,
    }


def process_step2_full(input_mcap, output_mcap,
                       # 2a 参数
                       intensity_threshold=0.01,
                       # 2b 参数
                       z_min_airy=-3.0, z_min_emx=-12.0,
                       # 2c 参数
                       sor_nb_neighbors=20, sor_std_ratio=2.0,
                       # 2d 参数
                       ror_radius=0.3, ror_min_points=3):
    return process_mcap_streaming(
        input_mcap, output_mcap, stage=2,
        intensity_threshold=intensity_threshold, z_min_airy=z_min_airy,
        z_min_emx=z_min_emx, sor_nb=sor_nb_neighbors, sor_std=sor_std_ratio,
        ror_radius=ror_radius, ror_min=ror_min_points,
    )

    """
    主处理流程: 读取原始 MCAP → Step1(无效点去除) → Step2(四级噪声滤除)
    
    直接读取原始 MCAP (包含 NaN 网格)，完整执行 Step1+Step2，
    输出到 _step2 后缀 topic。
    """
    input_mcap = Path(input_mcap)
    output_mcap = Path(output_mcap)
    lidar_set = set(LIDAR_TOPICS)

    print("=" * 70)
    print("  Step 2: 水面噪声滤除")
    print("    2a: 强度滤波        (autoware PassthroughFilter)")
    print("    2b: Z轴水面裁剪      (autoware CropBoxFilter)")
    print("    2c: 统计离群点 SOR   (Open3D)")
    print("    2d: 半径离群点 ROR   (Open3D + autoware RadiusSearch2D)")
    print("=" * 70)
    print(f"  输入: {input_mcap}")
    print(f"  输出: {output_mcap}")
    print()

    # ==================================================================
    # 阶段 1: 读取 + Step1/Step2 处理
    # ==================================================================
    print("[1/3] Step1 + Step2 流水线处理 ...")

    # 获取 PointCloud2 schema
    with open(str(input_mcap), 'rb') as f:
        summary_in = make_reader(f).get_summary()
        for sid, sc in summary_in.schemas.items():
            if sc.name == 'sensor_msgs/msg/PointCloud2':
                pcd_encoder = serialize_dynamic(sc.name, sc.data.decode())[
                    'sensor_msgs/msg/PointCloud2'
                ]
                break
        schemas = summary_in.schemas
        channels = summary_in.channels

    processed_data = {}      # topic + '_step2' → {log_time_ns: serialized_bytes}
    step1_stats = {}          # topic → stats for Step1
    step2_stats = {}          # topic → {substep: stats}

    for msg in read_ros2_messages(str(input_mcap), topics=sorted(lidar_set)):
        topic = msg.channel.topic
        pc = msg.ros_msg
        log_time = msg.log_time_ns

        # ---- Step 1: 无效点去除 (与 step1_remove_invalid 完全一致的逻辑) ----
        raw = np.frombuffer(bytes(pc.data), dtype=PCD_DTYPE)
        x = raw['x']; y = raw['y']; z = raw['z']
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        dist = np.sqrt(x**2 + y**2 + z**2)
        valid = valid & (dist > 0.0)

        if topic not in step1_stats:
            step1_stats[topic] = {'total': 0, 'valid': 0}
        step1_stats[topic]['total'] += len(raw)
        step1_stats[topic]['valid'] += np.sum(valid)

        x, y, z = x[valid], y[valid], z[valid]
        intensity = raw['intensity'][valid]
        ring = raw['ring'][valid]
        dist = dist[valid]
        
        if len(x) == 0:
            continue

        n_before_step2 = len(x)

        # ---- Step 2a: 强度滤波 ----
        # 参考: autoware PassthroughFilter (intensity 字段)
        keep_2a = filter_by_intensity(intensity, intensity_threshold)
        x, y, z = x[keep_2a], y[keep_2a], z[keep_2a]
        intensity = intensity[keep_2a]
        ring = ring[keep_2a]

        # ---- Step 2b: Z轴水面裁剪 ----
        # 参考: autoware CropBoxFilter / PassthroughFilter (z 轴)
        keep_2b = filter_below_water(z, topic, z_min_airy, z_min_emx)
        x, y, z = x[keep_2b], y[keep_2b], z[keep_2b]
        intensity = intensity[keep_2b]
        ring = ring[keep_2b]

        # ---- Step 2c: SOR ----
        # 参考: Open3D statistical_outlier_removal
        # EMX (200m远距) 跳过 SOR: 远场点密度差异大，全局统计(Mu/Sigma) 失效
        is_emx = (topic == EMX_TOPIC)
        if not is_emx and len(x) >= sor_nb_neighbors + 1:
            xyz = np.column_stack([x, y, z])
            keep_2c, sor_stats = statistical_outlier_removal(xyz, sor_nb_neighbors, sor_std_ratio)
            x, y, z = x[keep_2c], y[keep_2c], z[keep_2c]
            intensity = intensity[keep_2c]
            ring = ring[keep_2c]
        else:
            sor_stats = {'n_outliers': 0}

        # ---- Step 2d: ROR ----
        # 参考: Open3D radius_outlier_removal
        # EMX 跳过 ROR: 200m 远距点间距 >1m，固定半径搜索失效
        if (not is_emx) and len(x) > 0:
            xyz = np.column_stack([x, y, z])
            keep_2d, ror_stats = radius_outlier_removal(xyz, ror_radius, ror_min_points)
            x, y, z = x[keep_2d], y[keep_2d], z[keep_2d]
            intensity = intensity[keep_2d]
            ring = ring[keep_2d]
        else:
            ror_stats = {'n_outliers': 0}

        n_after_step2 = len(x)
        if n_after_step2 == 0:
            continue

        # 累计统计
        if topic not in step2_stats:
            step2_stats[topic] = {
                'before': 0, 'after': 0,
                '2a_removed': 0, '2b_removed': 0,
                '2c_removed': 0, '2d_removed': 0,
            }
        step2_stats[topic]['before'] += n_before_step2
        step2_stats[topic]['after'] += n_after_step2
        step2_stats[topic]['2a_removed'] += int(n_before_step2 - np.sum(keep_2a))
        # 需要追踪中间量...

        # ---- 序列化输出 ----
        xyz = np.column_stack([x, y, z])
        stamp_sec = log_time // 1_000_000_000
        stamp_nsec = log_time % 1_000_000_000
        pcd_dict = build_pointcloud2_dict(xyz, intensity, ring, pc.header.frame_id, stamp_sec, stamp_nsec)
        serialized = pcd_encoder(pcd_dict)

        step2_topic = topic + STEP2_SUFFIX
        if step2_topic not in processed_data:
            processed_data[step2_topic] = {}
        processed_data[step2_topic][log_time] = serialized

    # ---- 打印统计 ----
    print()
    for topic in sorted(step1_stats.keys()):
        name = topic.replace('/wamv/sensors/lidars/', '').replace('_lidar_sensor/points', '')
        s1 = step1_stats[topic]
        s2 = step2_stats.get(topic, {'before': 0, 'after': 0})
        print(f"  {name}:")
        print(f"    Step1: {s1['total']:,} grid → {s1['valid']:,} valid ({s1['valid']/max(s1['total'],1)*100:.1f}%)")
        if s2['before'] > 0:
            removed = s2['before'] - s2['after']
            print(f"    Step2: {s2['before']:,} → {s2['after']:,} ({removed:,} removed, {removed/s2['before']*100:.1f}%)")
        print()

    # ==================================================================
    # 阶段 2: 写入输出 MCAP
    # ==================================================================
    print("[2/3] 写入输出 MCAP ...")

    with open(str(output_mcap), 'wb') as f:
        writer = McapWriter(f)
        writer.start(profile='ros2', library='step2-noise-filter')

        schema_map = {}
        for sid, sc in schemas.items():
            schema_map[sid] = writer.register_schema(sc.name, sc.encoding, sc.data)

        channel_map = {}
        for cid, ch in channels.items():
            meta = dict(ch.metadata) if ch.metadata else {}
            channel_map[cid] = writer.register_channel(
                ch.topic, ch.message_encoding, schema_map[ch.schema_id], meta
            )

        step2_channel_map = {}
        for topic in lidar_set:
            for cid, ch in channels.items():
                if ch.topic == topic:
                    meta = dict(ch.metadata) if ch.metadata else {}
                    step2_topic = topic + STEP2_SUFFIX
                    step2_channel_map[topic] = writer.register_channel(
                        step2_topic, ch.message_encoding, schema_map[ch.schema_id], meta
                    )
                    break

        li_raw = li_step2 = oth = 0
        with open(str(input_mcap), 'rb') as f2:
            for schema, channel, message in make_reader(f2).iter_messages():
                topic = channel.topic
                log_time = message.log_time
                pub_time = message.publish_time
                raw_data = message.data
                cid = channel_map[channel.id]

                writer.add_message(cid, log_time, raw_data, pub_time, 0)

                if topic in lidar_set:
                    li_raw += 1
                    step2_topic = topic + STEP2_SUFFIX
                    step2_data = processed_data.get(step2_topic, {}).get(log_time)
                    if step2_data is not None:
                        writer.add_message(step2_channel_map[topic], log_time, step2_data, pub_time, 0)
                        li_step2 += 1
                else:
                    oth += 1

                if (li_raw + oth) % 10000 == 0:
                    print(f"    已处理 {li_raw + oth:,} 条消息 ...")

        writer.finish()

    # ==================================================================
    # 阶段 3: 结果汇报
    # ==================================================================
    size_mb = output_mcap.stat().st_size / (1024 * 1024)
    total_s1_grid = sum(s['total'] for s in step1_stats.values())
    total_s1_valid = sum(s['valid'] for s in step1_stats.values())
    total_s2_in = sum(s['before'] for s in step2_stats.values())
    total_s2_out = sum(s['after'] for s in step2_stats.values())

    print(f"\n[3/3] 完成!")
    print("=" * 70)
    print(f"  输出大小:      {size_mb:>8.1f} MB")
    print(f"  LiDAR 原始帧:  {li_raw:>6,d}")
    print(f"  LiDAR 输出帧:  {li_step2:>6,d}  (topic + '{STEP2_SUFFIX}')")
    print(f"  非LiDAR:       {oth:>6,d} 条")
    print()
    print(f"  Step1 无效→有效:  {total_s1_grid:,} → {total_s1_valid:,}")
    print(f"  Step2 级联滤除:   {total_s2_in:,} → {total_s2_out:,}  "
          f"({(total_s2_in - total_s2_out) / max(total_s2_in, 1) * 100:.1f}% removed)")
    print()
    print(f"  参数:")
    print(f"    2a intensity_threshold = {intensity_threshold}  "
          f"({'穿透(仿真intensity=0)' if intensity_threshold == 0 else '激活'})")
    print(f"    2b z_min_airy = {z_min_airy}m  |  z_min_emx = {z_min_emx}m")
    print(f"    2c sor_nb_neighbors = {sor_nb_neighbors}  |  sor_std_ratio = {sor_std_ratio}")
    print(f"    2d ror_radius = {ror_radius}m  |  ror_min_points = {ror_min_points}")
    print("=" * 70)
    print()
    print("  Foxglove 对比方法:")
    print("    同一帧同时可视化:")
    print(f"      - .../points       (原始, 含 NaN 网格)")
    print(f"      - .../points{STEP2_SUFFIX}  (Step1+2 清洗后)")
    print("    不同颜色对比：原始大量虚点 → 清洗后仅真实障碍物回波")
    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        input_mcap = sys.argv[1]
    else:
        input_mcap = (
            Path(__file__).parent.parent
            / 'rosbag2_2026_07_02-09_03_17'
            / 'rosbag2_2026_07_02-09_03_17'
            / 'rosbag2_2026_07_02-09_03_17_0.mcap'
        )

    if len(sys.argv) >= 3:
        output_mcap = sys.argv[2]
    else:
        output_mcap = (
            Path(__file__).parent.parent
            / 'output'
            / 'step2_noise_filter.mcap'
        )

    output_mcap = Path(output_mcap)
    output_mcap.parent.mkdir(parents=True, exist_ok=True)

    process_step2_full(
        input_mcap, output_mcap,
        intensity_threshold=0.0,  # 仿真环境穿透(0.01 用于实测)
        z_min_airy=-3.0,
        z_min_emx=-12.0,
        sor_nb_neighbors=20,
        sor_std_ratio=2.0,
        ror_radius=0.3,
        ror_min_points=3,
    )
