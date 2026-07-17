"""
preprocessing — 雷达点云预处理模块

核心文件:
  filters.py             — 纯 numpy 算法库 (Step1-4, 无 ROS2 依赖)
  mcap_publisher.py      — 读 MCAP → 预处理 → 发布 ROS2 话题 (比赛入口)
  step1_remove_invalid.py  — 离线分析 Step1 (读 MCAP→处理→写 MCAP)
  step2_noise_filter.py    — 离线分析 Step2
  step3_range_filter.py    — 离线分析 Step3
  step4_downsample.py       — 离线分析 Step4 (完整流水线)
"""

from .filters import (
    PreprocessingPipeline,
    step1_remove_invalid,
    step2a_intensity_filter,
    step2b_z_clip,
    step2c_sor,
    step2d_ror,
    step3_range_filter,
    step4_voxel_downsample,
    PCD_DTYPE,
    ALL_LIDAR_TOPICS,
    is_emx,
)
