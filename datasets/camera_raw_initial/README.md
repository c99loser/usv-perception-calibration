# 六路相机初始训练候选集

本目录由三段原始 ROS2 MCAP 的六路 `image_raw` 话题抽取生成。图像保持原始鱼眼域，不进行几何矫正；JPEG 质量为 95。

## 数量与划分

| 相机 | train | val | test | 合计 |
|---|---:|---:|---:|---:|
| front | 160 | 40 | 40 | 240 |
| back | 160 | 40 | 40 | 240 |
| left | 160 | 40 | 40 | 240 |
| left2 | 160 | 40 | 40 | 240 |
| right | 160 | 40 | 40 | 240 |
| right2 | 160 | 40 | 40 | 240 |
| 总计 | 960 | 240 | 240 | 1440 |

训练、验证和测试来自不同 MCAP。每个片段包含每路相机 40 个连续原始帧，避免将相邻帧拆到不同集合造成时间泄漏。

## 文件结构

```text
images/<camera>/<split>/<clip_id>/frame_<source_index>_<timestamp_ns>.jpg
manifest.csv
summary.json
```

`manifest.csv` 是追溯主清单，关键字段包括：

- `source_bag`、`source_topic`
- `source_frame_index_0based`、`source_frame_number_1based`
- `timestamp_ns`、`timestamp_sec`、`timestamp_nanosec`
- `clip_id`、`clip_position_0based`、`split`
- 原始像素和输出 JPEG 的 SHA-256
- 相对文件路径、图像宽高及相邻帧像素差

## 去重与核验结果

- 按相机对解码后的原始像素 SHA-256 去除完全重复帧。
- 对片段内相邻候选帧使用极保守的平均绝对像素差阈值 `0.25`，只排除近乎完全静止的副本，避免破坏连续性。
- 本批数据未发现完全重复、近零变化或解码失败帧。
- 已核验清单 1440 行、图像 1440 张、缺失文件 0、片段内帧序号断点 0、时间戳逆序 0。

## 使用建议

这批数量足够启动当前目标检测或分割方案的初步训练和数据管线验证，但不代表最终训练集规模。建议先完成标注规范、小批量标注一致性复核和首轮 baseline；根据各相机与目标类别的召回结果，再定向追加困难场景、远距离小目标、遮挡和极端视角帧。

复现工具：`tools/camera_fisheye/extract_training_clips.py`。
