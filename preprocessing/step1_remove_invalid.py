"""
================================================================================
 Step 1: 无效点去除 (Invalid Point Removal)
================================================================================
用途: 移除 LiDAR 原始点云中的无效数据点，包括：
      - NaN / Inf 坐标值（传感器未获得有效回波的"空槽位"）
      - 距离为零的点（传感器原点噪声）

原理: 机械旋转式 LiDAR（Airy/EMX）以固定网格输出点云（48线×360方位=17280点/帧
      或 96线×360方位=34560点/帧），但绝大多数网格位置因无回波而被填充为 NaN。
      Step 1 仅保留有实际回波的有限值点。

参考开源方案:
      - Open3D: remove_non_finite_points()
      - PCL (via perception_pcl): pcl::removeNaNFromPointCloud()
      - 开源地址: isl-org/Open3D, ros-perception/perception_pcl

输入: 原始 ROS2 MCAP 文件中的 7 个 LiDAR PointCloud2 topic
输出: 仅含有效点的 MCAP 文件 (每个 LiDAR 保持原始 topic 名)
================================================================================
"""

import numpy as np
from pathlib import Path
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
# PointCloud2 二进制布局 (由 Airy/EMX 仿真传感器确定，不可随意修改)
# ==============================================================================
# 字段布局 (已验证: point_step=32 bytes):
#   offset  0: x (float32, 4 bytes)       datatype=7 (FLOAT32)
#   offset  4: y (float32, 4 bytes)       datatype=7
#   offset  8: z (float32, 4 bytes)       datatype=7
#   offset 12: padding (4 bytes)          -- 结构对齐填充
#   offset 16: intensity (float32, 4 bytes) datatype=7
#   offset 20: padding (4 bytes)          -- 结构对齐填充
#   offset 24: ring (uint16, 2 bytes)     datatype=4 (UINT16)
#   offset 26: padding (6 bytes)          -- 尾部填充至 32 字节对齐
#
# 参考: sensor_msgs/PointCloud2 的 ROS2 消息规范
#       autoware point_types (PointXYZIRCAEDT) 类似布局
PCD_DTYPE = np.dtype([
    ('x',         'f4'),   # offset 0,  float32
    ('y',         'f4'),   # offset 4,  float32
    ('z',         'f4'),   # offset 8,  float32
    ('_pad1',     'V4'),   # offset 12, 4 bytes padding
    ('intensity', 'f4'),   # offset 16, float32
    ('_pad2',     'V4'),   # offset 20, 4 bytes padding
    ('ring',      'u2'),   # offset 24, uint16
    ('_pad3',     'V6'),   # offset 26, 6 bytes padding → total 32 bytes
])

# ==============================================================================
# 7 个 LiDAR 传感器的 ROS2 topic 列表
# ==============================================================================
LIDAR_TOPICS = [
    '/wamv/sensors/lidars/front_lidar_sensor/points',     # L1 Airy, 船头
    '/wamv/sensors/lidars/back_lidar_sensor/points',      # L2 Airy, 船尾
    '/wamv/sensors/lidars/left_lidar_sensor/points',      # L3 Airy, 左舷前
    '/wamv/sensors/lidars/left2_lidar_sensor/points',     # L4 Airy, 左舷后
    '/wamv/sensors/lidars/right_lidar_sensor/points',     # L5 Airy, 右舷前
    '/wamv/sensors/lidars/right2_lidar_sensor/points',    # L6 Airy, 右舷后
    '/wamv/sensors/lidars/forward_lidar_sensor/points',   # L7 EMX, 船头前向
]

# 处理后点云的 topic 后缀 —— 在 Foxglove 中与原始 topic 并列显示，便于对比
CLEAN_SUFFIX = '_clean'


def parse_pointcloud2(pc_msg):
    """
    将 ROS2 PointCloud2 消息解析为 numpy 结构化数组。

    参考: Open3D 的 read_point_cloud 内部使用了类似的字节缓冲区解析逻辑
          PCL 的 fromROSMsg() 同样从 data 字段按字段偏移量读取

    参数:
        pc_msg: mcap_ros2 反序列化后的 sensor_msgs/PointCloud2 对象

    返回:
        points: numpy structured array, dtype=PCD_DTYPE
            - 每个元素包含 x, y, z, intensity, ring 字段
            - 形状 (N,), 其中 N = height × width
    """
    # PointCloud2.data 是 sequence<uint8>，在 mcap_ros2 中解码后为 list[int] 或 bytes
    raw_bytes = bytes(pc_msg.data)

    # 按 PCD_DTYPE 解析二进制数据为结构化数组
    # 每个点占 32 字节 (point_step=32)
    points = np.frombuffer(raw_bytes, dtype=PCD_DTYPE)
    return points


def remove_invalid_points(points):
    """
    移除无效点：NaN、Inf、距离为零的点。

    参考方案:
        Open3D: o3d.geometry.PointCloud.remove_non_finite_points()
            函数签名: remove_non_finite_points(remove_nan=True, remove_infinite=True)
            原理: 遍历所有点，检查 xyz 是否为有限值
            来源: isl-org/Open3D (cpp/open3d/geometry/PointCloud.cpp)

        PCL: pcl::removeNaNFromPointCloud()
            函数签名: removeNaNFromPointCloud(cloud_in, cloud_out, index)
            原理: 检查 pcl::isFinite(p)，内部判断 xyz 是否既非 NaN 也非 Inf
            来源: ros-perception/perception_pcl (filters/remove_nan.cpp)

    参数:
        points: numpy structured array (dtype=PCD_DTYPE)

    返回:
        valid_points: 仅包含有效点的 numpy structured array
        stats: dict, 包含过滤统计信息
    """
    x = points['x']
    y = points['y']
    z = points['z']

    # ---------------------------------------------------------------
    # 条件 1: 坐标分量必须为有限值 (非 NaN, 非 ±Inf)
    # ---------------------------------------------------------------
    # 参考: Open3D 的 remove_non_finite_points
    #       isfinite() 等价于 ~isnan() & ~isinf()
    is_finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    # ---------------------------------------------------------------
    # 条件 2: 距离必须大于零 (排除传感器原点附近的噪声点)
    # ---------------------------------------------------------------
    # 参考: autoware CropBoxFilter 的 min_range 概念
    #       距离为零的点通常是传感器自身噪声或仿真中未初始化的"空"回波
    #       注意: 对 NaN 值做 sqrt 仍得到 NaN，需先用 is_finite 过滤
    #             但我们只在有限值上计算距离以避免警告
    distance = np.sqrt(x**2 + y**2 + z**2)
    is_positive = distance > 0.0

    # ---------------------------------------------------------------
    # 组合条件
    # ---------------------------------------------------------------
    valid_mask = is_finite & is_positive

    n_total = len(points)
    n_nan_inf = np.sum(~is_finite)
    n_zero = np.sum(is_finite & ~is_positive)
    n_valid = np.sum(valid_mask)

    stats = {
        'total': n_total,
        'nan_inf': n_nan_inf,
        'zero_dist': n_zero,
        'valid': n_valid,
        'removed': n_total - n_valid,
    }

    return points[valid_mask], stats


def build_pointcloud2_dict(xyz, intensity, ring, frame_id, stamp_sec, stamp_nsec):
    """
    从 numpy 数组构建可被 serialize_dynamic 序列化为 PointCloud2 消息的 dict。

    参考: 仿照 sensor_msgs/PointCloud2 的字段声明布局
          PCL 的 toROSMsg() 以类似方式将 PCL 点云序列化为 ROS 消息
          来源: ros-perception/perception_pcl

    参数:
        xyz:       (N, 3) float32, xyz 坐标
        intensity: (N,)  float32, 强度值
        ring:      (N,)  uint16,  扫描环编号
        frame_id:  str, 坐标系名称
        stamp_sec: int, 时间戳秒部分
        stamp_nsec: int, 时间戳纳秒部分

    返回:
        dict, 可直接传给 serialize_dynamic 的 PointCloud2 格式
    """
    N = len(xyz)

    # 构建与原始布局一致的结构化数组
    buf = np.zeros(N, dtype=PCD_DTYPE)
    buf['x'] = xyz[:, 0]
    buf['y'] = xyz[:, 1]
    buf['z'] = xyz[:, 2]
    buf['intensity'] = intensity
    buf['ring'] = ring

    return {
        'header': {
            'stamp': {'sec': stamp_sec, 'nanosec': stamp_nsec},
            'frame_id': frame_id,
        },
        'height': 1,          # 无序点云格式: height=1, width=N
        'width': N,           # 有效点数
        'fields': [
            # 字段声明必须与 PCD_DTYPE 布局一致
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
        'is_dense': True,       # 标记为稠密点云（无 NaN/Inf）
    }


def process_mcap(input_mcap, output_mcap):
    return process_mcap_streaming(input_mcap, output_mcap, stage=1)

    """
    主处理流程：读取 MCAP → 逐帧去除无效点 → 写入新 MCAP

    处理逻辑:
        - 仅处理 LIDAR_TOPICS 中列出的 7 个 LiDAR topic
        - 非 LiDAR topic 直接透传（保留原始数据）
        - 每个 LiDAR 独立处理，互不依赖

    参考: autoware_pointcloud_preprocessor 的点云管道设计模式
          每个 LiDAR 数据流经过独立的过滤节点后输出
          来源: autowarefoundation/autoware_universe → sensing/autoware_pointcloud_preprocessor

    参数:
        input_mcap:  Path, 输入 MCAP 文件路径
        output_mcap: Path, 输出 MCAP 文件路径
    """
    input_mcap = Path(input_mcap)
    output_mcap = Path(output_mcap)
    lidar_set = set(LIDAR_TOPICS)

    print("=" * 70)
    print("  Step 1: 无效点去除 (NaN / Inf / Zero-Distance Removal)")
    print("=" * 70)
    print(f"  输入: {input_mcap}")
    print(f"  输出: {output_mcap}")
    print(f"  参考: Open3D remove_non_finite_points / PCL removeNaNFromPointCloud")
    print()

    # ==================================================================
    # 阶段 1: 读取 LiDAR 数据，逐帧执行无效点去除
    # ==================================================================
    print("[1/3] 读取并处理 LiDAR 点云 ...")

    # 从输入 MCAP 获取 PointCloud2 的 schema，用于后续序列化
    with open(str(input_mcap), 'rb') as f:
        summary_in = make_reader(f).get_summary()
        # 找到 sensor_msgs/PointCloud2 的 schema
        for sid, sc in summary_in.schemas.items():
            if sc.name == 'sensor_msgs/msg/PointCloud2':
                pcd_encoder = serialize_dynamic(sc.name, sc.data.decode())[
                    'sensor_msgs/msg/PointCloud2'
                ]
                break
        schemas = summary_in.schemas
        channels = summary_in.channels

    # 使用 read_ros2_messages 正确反序列化 PointCloud2 消息
    # (read_ros2_messages 内部处理 CDR 序列化，比手动字节解析更可靠)
    processed_data = {}  # topic → {log_time_ns: serialized_bytes}
    stats_per_topic = {}  # topic → {total, valid, ...}

    for msg in read_ros2_messages(str(input_mcap), topics=sorted(lidar_set)):
        topic = msg.channel.topic
        pc = msg.ros_msg
        log_time = msg.log_time_ns

        # ---- 解析 PointCloud2 → numpy 结构化数组 ----
        points_raw = parse_pointcloud2(pc)

        # ---- 执行无效点去除 (核心步骤) ----
        # 参考: Open3D remove_non_finite_points 的逻辑等价实现
        points_valid, stats = remove_invalid_points(points_raw)

        # 记录统计信息
        if topic not in stats_per_topic:
            stats_per_topic[topic] = {'total': 0, 'nan_inf': 0, 'zero_dist': 0, 'valid': 0, 'removed': 0}
        for k in stats:
            stats_per_topic[topic][k] += stats[k]

        if len(points_valid) == 0:
            # 整帧均为无效点（极端情况，几乎不会发生），跳过
            continue

        # ---- 重建 PointCloud2 消息 ----
        # 从结构化数组中提取各字段
        xyz = np.column_stack([
            points_valid['x'], points_valid['y'], points_valid['z']
        ])
        intensity = points_valid['intensity']
        ring = points_valid['ring']

        stamp_sec = log_time // 1_000_000_000
        stamp_nsec = log_time % 1_000_000_000
        frame_id = pc.header.frame_id

        pcd_dict = build_pointcloud2_dict(
            xyz, intensity, ring,
            frame_id, stamp_sec, stamp_nsec
        )

        # 序列化为 CDR 字节流
        serialized = pcd_encoder(pcd_dict)

        # 使用 _clean 后缀的 topic 名存储处理后数据
        clean_topic = topic + CLEAN_SUFFIX
        if clean_topic not in processed_data:
            processed_data[clean_topic] = {}
        processed_data[clean_topic][log_time] = serialized

    # ---- 打印各 LiDAR 统计 ----
    print()
    for topic in sorted(stats_per_topic.keys()):
        s = stats_per_topic[topic]
        name = topic.replace('/wamv/sensors/lidars/', '').replace('_lidar_sensor/points', '')
        print(f"  {name}:")
        print(f"    总网格点: {s['total']:>10,d}")
        print(f"    NaN/Inf:  {s['nan_inf']:>10,d}  ({s['nan_inf']/max(s['total'],1)*100:.1f}%)")
        print(f"    零距离:   {s['zero_dist']:>10,d}")
        print(f"    有效点:   {s['valid']:>10,d}  ({s['valid']/max(s['total'],1)*100:.1f}%)")
        print()

    # ==================================================================
    # 阶段 2: 写入输出 MCAP
    # ==================================================================
    print("[2/3] 写入输出 MCAP ...")

    with open(str(output_mcap), 'wb') as f:
        writer = McapWriter(f)
        writer.start(profile='ros2', library='step1-remove-invalid')

        # 注册所有 schema（与输入 MCAP 保持一致）
        schema_map = {}
        for sid, sc in schemas.items():
            schema_map[sid] = writer.register_schema(sc.name, sc.encoding, sc.data)

        # 注册所有 channel（原始 topic）

        channel_map = {}  # (cid_from_input) → new_cid
        for cid, ch in channels.items():
            meta = dict(ch.metadata) if ch.metadata else {}
            channel_map[cid] = writer.register_channel(
                ch.topic, ch.message_encoding, schema_map[ch.schema_id], meta
            )

        # 为 LiDAR topic 额外注册 _clean 后缀的 channel
        # 用于存放处理后的点云，与原始 topic 并存于同一 MCAP
        clean_channel_map = {}  # original_topic → new_cid
        for topic in lidar_set:
            # 找到该 topic 对应的原始 channel，复用其 schema/meta
            orig_cid = None
            for cid, ch in channels.items():
                if ch.topic == topic:
                    orig_cid = cid
                    break
            if orig_cid is not None:
                ch = channels[orig_cid]
                clean_topic = topic + CLEAN_SUFFIX
                meta = dict(ch.metadata) if ch.metadata else {}
                clean_channel_map[topic] = writer.register_channel(
                    clean_topic, ch.message_encoding, schema_map[ch.schema_id], meta
                )

        print(f"  原始 channel: {len(channel_map)}  |  clean channel: {len(clean_channel_map)}")
        print()

        # 按原始时间顺序遍历所有消息，逐条写入
        li_msg_count = 0
        clean_msg_count = 0
        oth_msg_count = 0

        with open(str(input_mcap), 'rb') as f2:
            for schema, channel, message in make_reader(f2).iter_messages():
                topic = channel.topic
                log_time = message.log_time
                pub_time = message.publish_time
                raw_data = message.data
                cid = channel_map[channel.id]

                if topic in lidar_set:
                    li_msg_count += 1
                    # ---- 写入原始点云（透传，不做任何修改） ----
                    writer.add_message(cid, log_time, raw_data, pub_time, 0)

                    # ---- 写入处理后的点云到 _clean topic ----
                    clean_topic = topic + CLEAN_SUFFIX
                    new_data = processed_data.get(clean_topic, {}).get(log_time)
                    if new_data is not None:
                        clean_cid = clean_channel_map[topic]
                        writer.add_message(clean_cid, log_time, new_data, pub_time, 0)
                        clean_msg_count += 1
                else:
                    oth_msg_count += 1
                    writer.add_message(cid, log_time, raw_data, pub_time, 0)

                if (li_msg_count + oth_msg_count) % 10000 == 0:
                    print(f"    已处理 {li_msg_count + oth_msg_count:,} 条消息 ...")

        writer.finish()

    # ==================================================================
    # 阶段 3: 结果汇报
    # ==================================================================

    # 汇总统计
    total_grid = sum(s['total'] for s in stats_per_topic.values())
    total_valid = sum(s['valid'] for s in stats_per_topic.values())
    total_rm = total_grid - total_valid

    size_mb = output_mcap.stat().st_size / (1024 * 1024)
    print(f"\n[3/3] 完成!")
    print("=" * 70)
    print(f"  输出大小:    {size_mb:>8.1f} MB")
    print(f"  LiDAR 原始:  {li_msg_count:>6,d} 帧 (原样透传)")
    print(f"  LiDAR 清洗:  {clean_msg_count:>6,d} 帧 (topic + '{CLEAN_SUFFIX}')")
    print(f"  非LiDAR:     {oth_msg_count:>6,d} 条 (透传)")
    print()
    print(f"  总网格点:    {total_grid:>8,d}")
    print(f"  有效点:      {total_valid:>8,d}  ({total_valid/total_grid*100:.1f}%)")
    print(f"  移除无效:    {total_rm:>8,d}  ({total_rm/total_grid*100:.1f}%)")
    print("=" * 70)
    print()
    print("  Foxglove 对比方法:")
    print("    1. 打开本 MCAP 文件")
    print("    2. 3D 面板中添加两个 topic，例如:")
    print("       - /wamv/sensors/lidars/front_lidar_sensor/points       (原始)")
    print("       - /wamv/sensors/lidars/front_lidar_sensor/points_clean (清洗)")
    print("    3. 给两个 topic 设置不同颜色")
    print("    4. 播放同一帧，直观对比无效点去除效果")
    print()
    print("  下一步: Step 2 — 水面噪声滤除（强度滤波 + Z轴裁剪 + SOR + ROR）")
    print()


# ==============================================================================
# 主入口
# ==============================================================================
if __name__ == "__main__":
    import sys

    # 默认使用第一个 rosbag 数据包
    # 用户可通过命令行参数指定输入/输出路径
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
            / 'step1_remove_invalid.mcap'
        )

    # 确保输出目录存在
    output_mcap = Path(output_mcap)
    output_mcap.parent.mkdir(parents=True, exist_ok=True)

    process_mcap(input_mcap, output_mcap)
