"""低内存 MCAP 离线预处理。

逐条读取、处理和写入消息；不会把整段录包的处理结果缓存在内存中。输出为
rosbags 约定的目录，其中包含同名 ``.mcap`` 文件。
"""

from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import Writer
from rosbags.rosbag2.writer import StoragePlugin

from .filters import (
    ALL_LIDAR_TOPICS,
    PCD_DTYPE,
    is_emx,
    step1_remove_invalid,
    step2a_intensity_filter,
    step2b_z_clip,
    step2c_sor,
    step2d_ror,
    step3_range_filter,
    step4_voxel_downsample,
    validate_pointcloud2_layout,
)


STAGE_SUFFIX = {1: '_clean', 2: '_step2', 3: '_step3', 4: '_step4'}


def _process_to_stage(topic, raw_points, stage, config):
    x, y, z, intensity, ring = step1_remove_invalid(raw_points)
    if stage == 1 or len(x) == 0:
        return x, y, z, intensity, ring

    keep = step2a_intensity_filter(intensity, config['intensity_threshold'])
    x, y, z, intensity, ring = x[keep], y[keep], z[keep], intensity[keep], ring[keep]
    if len(x) == 0:
        return x, y, z, intensity, ring
    keep = step2b_z_clip(z, topic, config['z_min_airy'], config['z_min_emx'])
    x, y, z, intensity, ring = x[keep], y[keep], z[keep], intensity[keep], ring[keep]
    if len(x) == 0:
        return x, y, z, intensity, ring
    if not is_emx(topic):
        keep, _ = step2c_sor(x, y, z, config['sor_nb'], config['sor_std'])
        x, y, z, intensity, ring = x[keep], y[keep], z[keep], intensity[keep], ring[keep]
        if len(x) == 0:
            return x, y, z, intensity, ring
        keep, _ = step2d_ror(x, y, z, config['ror_radius'], config['ror_min'])
        x, y, z, intensity, ring = x[keep], y[keep], z[keep], intensity[keep], ring[keep]
    if stage == 2 or len(x) == 0:
        return x, y, z, intensity, ring

    keep = step3_range_filter(x, y, z, topic, config['min_range'], config['max_range_airy'], config['max_range_emx'])
    x, y, z, intensity, ring = x[keep], y[keep], z[keep], intensity[keep], ring[keep]
    if stage == 3 or len(x) == 0:
        return x, y, z, intensity, ring
    return step4_voxel_downsample(x, y, z, intensity, ring, topic, config['voxel_airy'], config['voxel_emx'])


def process_mcap_streaming(input_mcap, output_path, stage=4, max_lidar_messages=None, **overrides):
    """流式保留原始消息，并以附加 topic 写入指定处理阶段的结果。"""
    if stage not in STAGE_SUFFIX:
        raise ValueError(f'stage must be 1..4, got {stage}')
    config = {'intensity_threshold': 0.0, 'z_min_airy': -3.0, 'z_min_emx': -12.0,
              'sor_nb': 20, 'sor_std': 2.0, 'ror_radius': 0.3, 'ror_min': 3,
              'min_range': 0.5, 'max_range_airy': 60.0, 'max_range_emx': 200.0,
              'voxel_airy': 0.08, 'voxel_emx': 0.10}
    unknown = set(overrides) - set(config)
    if unknown:
        raise ValueError(f'Unknown preprocessing options: {sorted(unknown)}')
    config.update(overrides)
    source = Path(input_mcap)
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir = Path(output_path).with_suffix('')
    if output_dir.exists():
        raise FileExistsError(f'Output directory already exists: {output_dir}')

    lidar_topics = set(ALL_LIDAR_TOPICS)
    suffix = STAGE_SUFFIX[stage]
    counts = {'raw_messages': 0, 'processed_messages': 0, 'raw_points': 0, 'output_points': 0}
    reader = AnyReader([source])
    reader.open()
    try:
        writer = Writer(output_dir, version=Writer.VERSION_LATEST, storage_plugin=StoragePlugin.MCAP)
        writer.open()
        try:
            original = {
                connection.id: writer.add_connection(
                    connection.topic, connection.msgtype, typestore=reader.typestore,
                    msgdef=connection.msgdef.data,
                    serialization_format=connection.ext.serialization_format,
                    offered_qos_profiles=connection.ext.offered_qos_profiles,
                )
                for connection in reader.connections
            }
            processed = {
                connection.id: writer.add_connection(
                    connection.topic + suffix, connection.msgtype, typestore=reader.typestore,
                    msgdef=connection.msgdef.data,
                    serialization_format=connection.ext.serialization_format,
                    offered_qos_profiles=connection.ext.offered_qos_profiles,
                )
                for connection in reader.connections if connection.topic in lidar_topics
            }
            for connection, timestamp, data in reader.messages():
                writer.write(original[connection.id], timestamp, data)
                counts['raw_messages'] += 1
                if connection.topic not in lidar_topics:
                    continue
                message = reader.deserialize(data, connection.msgtype)
                validate_pointcloud2_layout(
                    message.point_step, message.row_step, message.width, message.height,
                    message.is_bigendian, message.fields,
                )
                raw = np.frombuffer(bytes(message.data), dtype=PCD_DTYPE)
                x, y, z, intensity, ring = _process_to_stage(connection.topic, raw, stage, config)
                counts['raw_points'] += len(raw)
                if len(x) == 0:
                    continue
                buffer = np.zeros(len(x), dtype=PCD_DTYPE)
                buffer['x'], buffer['y'], buffer['z'] = x, y, z
                buffer['intensity'], buffer['ring'] = intensity, ring
                message.height = 1
                message.width = len(buffer)
                message.row_step = message.width * PCD_DTYPE.itemsize
                # rosbags 的 CDR serializer 要求 uint8 ndarray，而非 bytes。
                message.data = np.frombuffer(buffer.tobytes(), dtype=np.uint8)
                message.is_dense = True
                writer.write(processed[connection.id], timestamp,
                             reader.typestore.serialize_cdr(message, connection.msgtype))
                counts['processed_messages'] += 1
                counts['output_points'] += len(buffer)
                if (max_lidar_messages is not None and
                        counts['processed_messages'] >= max_lidar_messages):
                    break
        finally:
            writer.close()
    finally:
        reader.close()
    result = output_dir / f'{output_dir.name}.mcap'
    print(f'Wrote {result}')
    print(counts)
    return result, counts
