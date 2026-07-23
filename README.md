# Challenger Program 感知项目

当前唯一执行路线见：[当前进行路线.md](当前进行路线.md)。

`step1/` 中的文档保留为阶段性技术参考；当其内容与当前路线冲突时，以
`当前进行路线.md` 为准。鱼眼处理工具说明见
[tools/camera_fisheye/README.md](tools/camera_fisheye/README.md)。

## WAM-V 离线感知基线

This branch now contains an offline, ROS-daemon-free chain for raw ROS2 MCAP:

`MCAP PointCloud2 replay -> identity/bootstrap sensor transform -> filtering + clustering -> geometric detection -> multi-object tracking -> 10 m occupancy BEV`.

The initial transform is deliberately identity per LiDAR. It is a bootstrap for single-sensor replay, **not** a claim of multi-LiDAR calibration. Replace `sensor_to_base` in a copied JSON configuration after calibration is available.

## Quick start

```powershell
$py = 'C:\Users\23546\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
# Only if dependencies use a shared cache rather than .codex_pydeps:
$env:PERCEPTION_PYDEPS = 'F:\Challenger program\.codex_pydeps'
& $py -m perception.cli inspect --data-root 'F:\Challenger program\数据'
& $py -m perception.cli replay --data-root 'F:\Challenger program\数据' --max-frames 20 --output-dir output\perception
```

`inspect` only reads metadata. `replay` opens the original MCAP through the project-local `rosbags` dependency, writes `.npy` and directly-viewable `.pgm` BEV files plus a JSON track list for every processed LiDAR frame, and never reads `/world/obstacles/*` ground truth. Camera-model detection and ROS2 publishing are explicit integration points, not hidden dependencies of this bootstrap.
