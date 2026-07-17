"""
================================================================================
 Step 4: 体素降采样 (Voxel Downsampling)
================================================================================
用途: 在 Step1+Step2+Step3 基础上，对点云进行体素网格降采样:
      - 将空间划分为等大立方体(voxel)
      - 每个体素内所有点取质心(centroid)
      - 大幅减少点数，同时保留几何结构

参考方案:
      Open3D: voxel_down_sample (VoxelGrid Centroid 模式)
      来源: isl-org/Open3D
      autoware: DownsampleFilter (VoxelGrid 模式)
      来源: autowarefoundation/autoware_universe → downsample_filter

参数:
      Airy (48线, 近距离密集): voxel_size = 0.10m
      EMX  (96线, 超密扫描):   voxel_size = 0.15m

输入: 原始 MCAP (自动内部执行完整四步流水线)
输出: 含有 _step4 后缀 topic 的 MCAP (最终预处理结果)
================================================================================
"""

import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
# 旧的批量实现依赖这些包；默认入口已切换到 rosbags 流式实现，故允许缺失。
try:
    from mcap.reader import make_reader
    from mcap_ros2.reader import read_ros2_messages
    from mcap_ros2._dynamic import serialize_dynamic
    from mcap.writer import Writer as McapWriter
except ImportError:
    make_reader = read_ros2_messages = serialize_dynamic = McapWriter = None

try:
    from .streaming_mcap import process_mcap_streaming
except ImportError:  # 支持 `python preprocessing/step4_downsample.py`
    from streaming_mcap import process_mcap_streaming

PCD_DTYPE = np.dtype([
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('_pad1', 'V4'), ('intensity', 'f4'),
    ('_pad2', 'V4'), ('ring', 'u2'), ('_pad3', 'V6'),
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

AIRY_TOPICS = {t for t in LIDAR_TOPICS if 'forward' not in t}
EMX_TOPIC = '/wamv/sensors/lidars/forward_lidar_sensor/points'
STEP4_SUFFIX = '_step4'


def build_pointcloud2_dict(xyz, intensity, ring, frame_id, stamp_sec, stamp_nsec):
    N = len(xyz)
    buf = np.zeros(N, dtype=PCD_DTYPE)
    buf['x'] = xyz[:, 0]; buf['y'] = xyz[:, 1]; buf['z'] = xyz[:, 2]
    buf['intensity'] = intensity; buf['ring'] = ring
    return {
        'header': {'stamp': {'sec': stamp_sec, 'nanosec': stamp_nsec}, 'frame_id': frame_id},
        'height': 1, 'width': N,
        'fields': [
            {'name': 'x', 'offset': 0, 'datatype': 7, 'count': 1},
            {'name': 'y', 'offset': 4, 'datatype': 7, 'count': 1},
            {'name': 'z', 'offset': 8, 'datatype': 7, 'count': 1},
            {'name': 'intensity', 'offset': 16, 'datatype': 7, 'count': 1},
            {'name': 'ring', 'offset': 24, 'datatype': 4, 'count': 1},
        ],
        'is_bigendian': False, 'point_step': 32, 'row_step': 32 * N,
        'data': buf.tobytes(), 'is_dense': True,
    }


def voxel_downsample(xyz, intensity, ring, voxel_size):
    """
    体素降采样: 每个边长为 voxel_size 的立方体中取质心。

    参考:
      Open3D voxel_down_sample — 体素内所有点取平均值(质心)
      来源: isl-org/Open3D (cpp/open3d/geometry/PointCloud.cpp)
      autoware DownsampleFilter (VoxelGrid Centroid 模式)
      来源: autowarefoundation/autoware_universe → downsample_filter

    原理:
      (1) 计算每个点的体素索引: idx = floor(xyz / voxel_size)
      (2) 同一体素的点合并: 对 xyz/intensity 取均值, ring 取众数
      (3) 不同于 step1_voxel_downsample.py 中的"取第一个点"策略,
          质心法更好地保留障碍物的空间几何信息

    参数:
        xyz:       (N, 3) float32
        intensity: (N,)  float32
        ring:      (N,)  uint16
        voxel_size: float (m)

    返回:
        xyz_ds, intensity_ds, ring_ds
    """
    N = len(xyz)
    if N == 0:
        return xyz, intensity, ring

    # 计算体素索引
    vx = np.floor(xyz / voxel_size).astype(np.int64)

    # 用 structured array 去重，获取唯一的体素索引及其逆映射
    vrec = np.rec.fromarrays([vx[:, 0], vx[:, 1], vx[:, 2]], names='x,y,z')
    _, inv, cnt = np.unique(vrec, return_inverse=True, return_counts=True)
    n_voxels = len(cnt)

    # 质心法: 对每个体素内的点取均值
    # 使用 np.bincount + weights 做加权求和，再除以计数
    x_ds = np.bincount(inv, weights=xyz[:, 0], minlength=n_voxels) / cnt
    y_ds = np.bincount(inv, weights=xyz[:, 1], minlength=n_voxels) / cnt
    z_ds = np.bincount(inv, weights=xyz[:, 2], minlength=n_voxels) / cnt
    i_ds = np.bincount(inv, weights=intensity, minlength=n_voxels) / cnt

    # 按 (voxel, ring) 统计众数，避免每个体素重复扫描全部点。
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
    ring_ds = run_ring[chosen_run].astype(np.uint16, copy=False)

    xyz_ds = np.column_stack([x_ds, y_ds, z_ds])
    return xyz_ds, i_ds, ring_ds


def process_mcap(input_mcap, output_mcap,
                 voxel_airy=0.08, voxel_emx=0.10,
                 min_range=0.5, max_range_airy=60.0, max_range_emx=200.0):
    """运行完整四步流式流水线，避免缓存整段 MCAP。"""
    if (voxel_airy, voxel_emx, min_range, max_range_airy, max_range_emx) != (0.08, 0.10, 0.5, 60.0, 200.0):
        raise ValueError('自定义参数请通过 PreprocessingPipeline 配置；离线入口当前使用文档默认值。')
    return process_mcap_streaming(input_mcap, output_mcap, stage=4)

    # 以下旧实现保留仅供历史对照；入口已在上方返回。
    input_mcap = Path(input_mcap)
    output_mcap = Path(output_mcap)
    lidar_set = set(LIDAR_TOPICS)

    print("=" * 70)
    print("  Step 4: 体素降采样 (完整四步流水线)")
    print("    参考: Open3D voxel_down_sample / autoware DownsampleFilter")
    print(f"    Airy voxel={voxel_airy}m  |  EMX voxel={voxel_emx}m")
    print("=" * 70)

    with open(str(input_mcap), 'rb') as f:
        summary_in = make_reader(f).get_summary()
        for sid, sc in summary_in.schemas.items():
            if sc.name == 'sensor_msgs/msg/PointCloud2':
                pcd_encoder = serialize_dynamic(sc.name, sc.data.decode())['sensor_msgs/msg/PointCloud2']
                break
        schemas = summary_in.schemas
        channels = summary_in.channels

    processed_data = {}
    stats = {}

    print("[1/2] Step1 → Step2 → Step3 → Step4 流水线 ...\n")

    for msg in read_ros2_messages(str(input_mcap), topics=sorted(lidar_set)):
        topic = msg.channel.topic
        pc = msg.ros_msg
        log_time = msg.log_time_ns
        is_emx = (topic == EMX_TOPIC)

        # -------- Step 1: 无效点去除 --------
        raw = np.frombuffer(bytes(pc.data), dtype=PCD_DTYPE)
        x, y, z = raw['x'], raw['y'], raw['z']
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        dist = np.sqrt(x**2 + y**2 + z**2)
        valid = valid & (dist > 0.0)
        x, y, z = x[valid], y[valid], z[valid]
        intensity = raw['intensity'][valid]; ring = raw['ring'][valid]
        if len(x) == 0: continue

        # -------- Step 2: 水面噪声 --------
        # 2b z-clip
        keep = z >= (-12.0 if is_emx else -3.0)
        x, y, z = x[keep], y[keep], z[keep]
        intensity = intensity[keep]; ring = ring[keep]
        if len(x) == 0: continue

        # 2c SOR (skip EMX)
        if not is_emx and len(x) >= 21:
            xyz_arr = np.column_stack([x, y, z])
            tree = cKDTree(xyz_arr)
            k = min(21, len(x))
            distances, _ = tree.query(xyz_arr, k=k)
            mean_dists = distances[:, 1:].mean(axis=1)
            mu, sigma = mean_dists.mean(), mean_dists.std()
            keep = mean_dists <= mu + 2.0 * sigma
            x, y, z = x[keep], y[keep], z[keep]
            intensity = intensity[keep]; ring = ring[keep]
        if len(x) == 0: continue

        # 2d ROR (skip EMX)
        if not is_emx and len(x) > 0:
            xyz_arr = np.column_stack([x, y, z])
            tree = cKDTree(xyz_arr)
            counts = tree.query_ball_point(xyz_arr, r=0.3, return_length=True)
            keep = counts >= 3
            x, y, z = x[keep], y[keep], z[keep]
            intensity = intensity[keep]; ring = ring[keep]
        if len(x) == 0: continue

        # -------- Step 3: 距离裁剪 --------
        dist = np.sqrt(x**2 + y**2 + z**2)
        max_range = max_range_emx if is_emx else max_range_airy
        keep_range = (dist >= min_range) & (dist <= max_range)
        x, y, z = x[keep_range], y[keep_range], z[keep_range]
        intensity = intensity[keep_range]; ring = ring[keep_range]
        if len(x) == 0: continue

        # -------- Step 4: 体素降采样 (新增) --------
        # 参考: Open3D voxel_down_sample (质心法)
        vs = voxel_emx if is_emx else voxel_airy
        xyz_arr = np.column_stack([x, y, z])
        xyz_ds, i_ds, r_ds = voxel_downsample(xyz_arr, intensity, ring, vs)

        # 累计统计
        if topic not in stats:
            stats[topic] = {'before': 0, 'after': 0}
        stats[topic]['before'] += len(x)
        stats[topic]['after'] += len(xyz_ds)

        # 序列化
        ss = log_time // 1_000_000_000; sns = log_time % 1_000_000_000
        pcd_dict = build_pointcloud2_dict(xyz_ds, i_ds, r_ds, pc.header.frame_id, ss, sns)
        serialized = pcd_encoder(pcd_dict)

        step4_topic = topic + STEP4_SUFFIX
        if step4_topic not in processed_data:
            processed_data[step4_topic] = {}
        processed_data[step4_topic][log_time] = serialized

    for t in sorted(stats):
        name = t.replace('/wamv/sensors/lidars/', '').replace('_lidar_sensor/points', '')
        s = stats[t]; r = (1 - s['after']/max(s['before'],1)) * 100
        print(f"  {name}: {s['before']:,} → {s['after']:,} ({r:.1f}% reduced)")

    print("\n[2/2] 写入 MCAP ...")
    with open(str(output_mcap), 'wb') as f:
        writer = McapWriter(f)
        writer.start(profile='ros2', library='step4-downsample')

        smap = {sid: writer.register_schema(sc.name, sc.encoding, sc.data)
                for sid, sc in schemas.items()}
        cmap = {}
        for cid, ch in channels.items():
            meta = dict(ch.metadata) if ch.metadata else {}
            cmap[cid] = writer.register_channel(ch.topic, ch.message_encoding, smap[ch.schema_id], meta)

        s4map = {}
        for topic in lidar_set:
            for cid, ch in channels.items():
                if ch.topic == topic:
                    meta = dict(ch.metadata) if ch.metadata else {}
                    s4map[topic] = writer.register_channel(
                        topic + STEP4_SUFFIX, ch.message_encoding, smap[ch.schema_id], meta)
                    break

        li_raw = li_s4 = oth = 0
        with open(str(input_mcap), 'rb') as f2:
            for schema, channel, message in make_reader(f2).iter_messages():
                t = channel.topic; lt = message.log_time; pt = message.publish_time
                writer.add_message(cmap[channel.id], lt, message.data, pt, 0)
                if t in lidar_set:
                    li_raw += 1
                    sd = processed_data.get(t + STEP4_SUFFIX, {}).get(lt)
                    if sd is not None:
                        writer.add_message(s4map[t], lt, sd, pt, 0)
                        li_s4 += 1
                else:
                    oth += 1
                if (li_raw + oth) % 10000 == 0:
                    print(f"    {li_raw + oth:,} msgs ...")

        writer.finish()

    mb = output_mcap.stat().st_size / (1024**2)
    total_before = sum(s['before'] for s in stats.values())
    total_after = sum(s['after'] for s in stats.values())
    print(f"\n{'='*70}")
    print(f"  完成! {mb:.1f} MB")
    print(f"  LiDAR: {li_raw} raw → {li_s4} step4  非LiDAR: {oth}")
    print(f"  降采样: {total_before:,} → {total_after:,} "
          f"({(1-total_after/max(total_before,1))*100:.1f}% reduced)")
    print(f"{'='*70}")
    print(f"\n  Foxglove 对比:")
    print(f"    .../points        原始 (含 NaN 网格)")
    print(f"    .../points_step4  完整预处理 (Step1-4)")
    print()


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) >= 2 else (
        Path(__file__).parent.parent / 'rosbag2_2026_07_02-09_03_17'
        / 'rosbag2_2026_07_02-09_03_17' / 'rosbag2_2026_07_02-09_03_17_0.mcap')
    out = sys.argv[2] if len(sys.argv) >= 3 else (
        Path(__file__).parent.parent / 'output' / 'step4_downsample.mcap')
    process_mcap(inp, out)
