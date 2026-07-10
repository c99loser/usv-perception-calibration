# ROS2 verification handoff

Current purpose: verify the corrected fisheye MCAP bags through ROS 2 in WSL/Ubuntu.

Corrected bags expected under Windows:

- `E:\challenger camera\rosbag2_2026_07_02-09_08_35_fisheye_corrected`
- `E:\challenger camera\rosbag2_2026_07_02-09_10_54_fisheye_corrected`

Equivalent WSL path:

- `/mnt/e/challenger camera`

Run from Windows cmd:

```bat
cd /d "F:\Challenger program"
tools\camera_fisheye\run_ros2_verify_from_windows.cmd
```

Run from WSL/Ubuntu:

```bash
cd "/mnt/f/Challenger program"
bash tools/camera_fisheye/verify_ros2_corrected_mcaps.sh
```

Expected report:

```text
F:\Challenger program\output\camera_fisheye\ros2_verify\ros2_corrected_mcap_verify.txt
```

The report should prove:

- ROS 2 command line is available.
- Each corrected MCAP can be read by `ros2 bag info`.
- The six fisheye camera `image_raw` topics are present in each corrected bag.
