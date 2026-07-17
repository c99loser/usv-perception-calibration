"""
================================================================================
 Step 3: 距离范围裁剪 (Distance Range Clipping)
================================================================================
用途: 在 Step1+Step2 基础上，裁剪有效距离范围:
      - 去除过近的点 (min_range): 过滤船体自身反射
      - 去除过远的点 (max_range): 过滤量程外/超远距无效噪声

参考方案:
      autoware PassthroughFilter / CropBoxFilter
      来源: autowarefoundation/autoware_universe → passthrough_filter

参数:
      Airy (48线, 60m量程):  min=0.5m, max=60m  (物理上限)
      EMX  (96线, 200m量程): min=0.5m, max=200m (保留全量程用于障碍物检测+点云融合)

输入: 原始 MCAP (自动内部执行 Step1 + Step2 + Step3)
输出: 含有 _step3 后缀 topic 的 MCAP
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
STEP1_SUFFIX = '_clean'
STEP2_SUFFIX = '_step2'
STEP3_SUFFIX = '_step3'


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


def process_mcap(input_mcap, output_mcap,
                 min_range=0.5, max_range_airy=60.0, max_range_emx=200.0):
    if (min_range, max_range_airy, max_range_emx) != (0.5, 60.0, 200.0):
        raise ValueError('自定义参数请通过 PreprocessingPipeline 配置；离线入口使用文档默认值。')
    return process_mcap_streaming(input_mcap, output_mcap, stage=3)

    input_mcap = Path(input_mcap)
    output_mcap = Path(output_mcap)
    lidar_set = set(LIDAR_TOPICS)

    print("=" * 70)
    print("  Step 3: 距离范围裁剪")
    print("    参考: autoware PassthroughFilter / CropBoxFilter")
    print(f"    Airy: {min_range}-{max_range_airy}m  |  EMX: {min_range}-{max_range_emx}m (全量程)")
    print(f"    注意: EMX保留200m用于障碍物检测+融合; 自标定可内部选取近场数据")
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

    print("[1/2] Step1 → Step2 → Step3 流水线 ...\n")

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
        if len(x) == 0:
            continue

        # -------- Step 2: 水面噪声 --------
        # 2b z-clip
        if topic in AIRY_TOPICS:
            keep = z >= -3.0
        else:
            keep = z >= -12.0
        x, y, z = x[keep], y[keep], z[keep]
        intensity = intensity[keep]; ring = ring[keep]
        if len(x) == 0:
            continue

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
        if len(x) == 0:
            continue

        # 2d ROR (skip EMX)
        if not is_emx and len(x) > 0:
            xyz_arr = np.column_stack([x, y, z])
            tree = cKDTree(xyz_arr)
            counts = tree.query_ball_point(xyz_arr, r=0.3, return_length=True)
            keep = counts >= 3
            x, y, z = x[keep], y[keep], z[keep]
            intensity = intensity[keep]; ring = ring[keep]
        if len(x) == 0:
            continue

        # -------- Step 3: 距离裁剪 (新增) --------
        # 参考: autoware PassthroughFilter 应用于距离字段
        dist = np.sqrt(x**2 + y**2 + z**2)
        max_range = max_range_emx if is_emx else max_range_airy
        keep_range = (dist >= min_range) & (dist <= max_range)
        x, y, z = x[keep_range], y[keep_range], z[keep_range]
        intensity = intensity[keep_range]; ring = ring[keep_range]
        if len(x) == 0:
            continue

        # 累计统计
        if topic not in stats:
            stats[topic] = {'total': 0, 'after': 0}
        stats[topic]['after'] += len(x)

        # 序列化
        xyz_out = np.column_stack([x, y, z])
        ss = log_time // 1_000_000_000; sns = log_time % 1_000_000_000
        pcd_dict = build_pointcloud2_dict(xyz_out, intensity, ring, pc.header.frame_id, ss, sns)
        serialized = pcd_encoder(pcd_dict)

        step3_topic = topic + STEP3_SUFFIX
        if step3_topic not in processed_data:
            processed_data[step3_topic] = {}
        processed_data[step3_topic][log_time] = serialized

    for t in sorted(stats):
        name = t.replace('/wamv/sensors/lidars/', '').replace('_lidar_sensor/points', '')
        print(f"  {name}: {stats[t]['after']:,} pts after Step3")

    print("\n[2/2] 写入 MCAP ...")
    with open(str(output_mcap), 'wb') as f:
        writer = McapWriter(f)
        writer.start(profile='ros2', library='step3-range-filter')

        smap = {sid: writer.register_schema(sc.name, sc.encoding, sc.data)
                for sid, sc in schemas.items()}
        cmap = {}
        for cid, ch in channels.items():
            meta = dict(ch.metadata) if ch.metadata else {}
            cmap[cid] = writer.register_channel(ch.topic, ch.message_encoding, smap[ch.schema_id], meta)

        s3map = {}
        for topic in lidar_set:
            for cid, ch in channels.items():
                if ch.topic == topic:
                    meta = dict(ch.metadata) if ch.metadata else {}
                    s3map[topic] = writer.register_channel(
                        topic + STEP3_SUFFIX, ch.message_encoding, smap[ch.schema_id], meta)
                    break

        li_raw = li_s3 = oth = 0
        with open(str(input_mcap), 'rb') as f2:
            for schema, channel, message in make_reader(f2).iter_messages():
                t = channel.topic; lt = message.log_time; pt = message.publish_time
                writer.add_message(cmap[channel.id], lt, message.data, pt, 0)
                if t in lidar_set:
                    li_raw += 1
                    sd = processed_data.get(t + STEP3_SUFFIX, {}).get(lt)
                    if sd is not None:
                        writer.add_message(s3map[t], lt, sd, pt, 0)
                        li_s3 += 1
                else:
                    oth += 1
                if (li_raw + oth) % 10000 == 0:
                    print(f"    {li_raw + oth:,} msgs ...")

        writer.finish()

    mb = output_mcap.stat().st_size / (1024**2)
    print(f"\n完成! {output_mcap} ({mb:.1f} MB)")
    print(f"  LiDAR: {li_raw} raw → {li_s3} step3  非LiDAR: {oth}\n")


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) >= 2 else (
        Path(__file__).parent.parent / 'rosbag2_2026_07_02-09_03_17'
        / 'rosbag2_2026_07_02-09_03_17' / 'rosbag2_2026_07_02-09_03_17_0.mcap')
    out = sys.argv[2] if len(sys.argv) >= 3 else (
        Path(__file__).parent.parent / 'output' / 'step3_range_filter.mcap')
    process_mcap(inp, out)
