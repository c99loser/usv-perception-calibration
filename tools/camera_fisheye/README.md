# Fisheye Camera Baseline

Offline baseline for the six fisheye cameras in the WAM-V docking challenge.

## Inputs

The tools expect ROS2 MCAP bag directories under `数据/`, each containing a
`metadata.yaml` and a `.mcap` file. The six challenge cameras are:

- `front`
- `back`
- `left`
- `left2`
- `right`
- `right2`

`overhead_camera` is intentionally ignored because this stage targets the six
fisheye cameras required by the challenge perception task.

## Commands

Metadata-only inspection works in plain Python:

PowerShell example from the repository root:

```powershell
python tools/camera_fisheye/camera_fisheye_pipeline.py inspect `
  --data-root "F:\Challenger program\数据" `
  --output-root output/camera_fisheye
```

Full extraction must run in a sourced ROS2 Jazzy environment with `rosbag2_py`,
`cv_bridge`, OpenCV, and NumPy available:

```bash
python tools/camera_fisheye/camera_fisheye_pipeline.py run \
  --data-root "/mnt/f/Challenger program/数据" \
  --output-root output/camera_fisheye \
  --samples-per-camera 3
```

To produce corrected MCAP bags for acceptance:

```bash
source /opt/ros/jazzy/setup.bash
cd "/mnt/f/Challenger program"
python3 tools/camera_fisheye/camera_fisheye_pipeline.py correct-all \
  --data-root "/mnt/f/Challenger program/数据" \
  --output-root "/mnt/f/Challenger program/output/camera_fisheye"
```

The corrected bags are written under:

```text
output/camera_fisheye/corrected_mcaps/
```

The full run writes:

- `camera_stage_report.md`
- `camera_intrinsics.yaml`
- `raw_frames/`
- `undistorted_frames/`
- `debug_detections/`
- `extraction_index.json`

The correction run writes:

- one `<bag_name>_fisheye_corrected/` ROS2 MCAP bag directory per input bag
- `<bag_name>_fisheye_corrected_stats.json`
- `manifest.json`

## Notes

Camera intrinsics prefer MCAP `CameraInfo`. If that is unavailable, the tools
fall back to a 1280x720, 175 degree horizontal FOV estimate. The fallback is
only a bootstrap for visual checking and should be replaced by calibrated
parameters later.
