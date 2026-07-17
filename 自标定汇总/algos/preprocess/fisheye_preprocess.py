"""
================================================================================
 fisheye_preprocess.py — 鱼眼相机矫正（算法 + 适配器）
================================================================================

算法核心：从 rectifier.py 完整迁移，100% 保留原始逻辑不变。
适配器 fisheye_preprocess()：将 {name: (H,W,3)} dict → 去畸变 → {name: (H,W,3)} dict

矫正模型: 等距鱼眼 → 直线透视 (equidistant_fov_approx)
默认参数: 1280×720, 175° HFOV → 120° rectified HFOV

============================================================================
第 0 部分：数据类与默认内参（不变）
============================================================================
"""

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class RectificationPlan:
    """矫正计划 (预计算映射表, 可复用于同一相机的所有帧)"""
    map1: np.ndarray          # x 映射表 (float32)
    map2: np.ndarray          # y 映射表 (float32)
    width: int                # 图像宽
    height: int               # 图像高
    model: str                # 矫正模型


# ---- 默认内参: 1280×720, 175° HFOV 等距鱼眼 ----
CAMERA_PARAMS = {
    "front":  {"width": 1280, "height": 720, "hfov": 175.0, "down_tilt": 25.0},
    "back":   {"width": 1280, "height": 720, "hfov": 175.0, "down_tilt": 25.0},
    "left":   {"width": 1280, "height": 720, "hfov": 175.0, "down_tilt": 25.0},
    "left2":  {"width": 1280, "height": 720, "hfov": 175.0, "down_tilt": 25.0},
    "right":  {"width": 1280, "height": 720, "hfov": 175.0, "down_tilt": 25.0},
    "right2": {"width": 1280, "height": 720, "hfov": 175.0, "down_tilt": 25.0},
}


# ======================================================================
# build_plan() —— 预计算矫正映射表（不变）
# ======================================================================
def build_plan(
    camera_name: str,
    width: int = None,
    height: int = None,
    hfov_degrees: float = None,
    rectified_hfov: float = 120.0,
) -> RectificationPlan:
    """
    为指定相机预计算矫正映射表。

    调用一次即可，后续每帧调用 apply_rectification(image, plan)。

    参数:
        camera_name:  相机名 (front/back/left/left2/right/right2)
        width, height: 图像尺寸 (默认 1280×720)
        hfov_degrees:  原始水平 FOV (默认 175°)
        rectified_hfov: 矫正后水平 FOV (默认 120°)

    返回:
        RectificationPlan, 可复用
    """
    params = CAMERA_PARAMS.get(camera_name, CAMERA_PARAMS["front"])
    w = width or params["width"]
    h = height or params["height"]
    fov = hfov_degrees or params["hfov"]

    # 等距鱼眼 → 直线投影
    source_cx = w / 2.0
    source_cy = h / 2.0
    source_f = w / math.radians(fov)

    target_f = w / (2.0 * math.tan(math.radians(rectified_hfov) / 2.0))
    target_cx = w / 2.0
    target_cy = h / 2.0

    yy, xx = np.indices((h, w), dtype=np.float32)
    x = (xx - target_cx) / target_f
    y = (yy - target_cy) / target_f
    radius = np.sqrt(x * x + y * y)
    theta = np.arctan(radius)
    scale = np.divide(
        source_f * theta, radius,
        out=np.zeros_like(radius, dtype=np.float32),
        where=radius > 1e-8,
    )
    mapx = (source_cx + x * scale).astype(np.float32)
    mapy = (source_cy + y * scale).astype(np.float32)

    return RectificationPlan(
        map1=mapx,
        map2=mapy,
        width=w,
        height=h,
        model="equidistant_fov_approx",
    )


# ======================================================================
# apply_rectification() —— 单帧矫正（不变）
# ======================================================================
def apply_rectification(image: np.ndarray, plan: RectificationPlan) -> np.ndarray:
    """
    对单帧鱼眼图像做矫正。

    参数:
        image: (H, W, 3) uint8, BGR 格式
        plan:  RectificationPlan (由 build_plan 预计算)

    返回:
        (H, W, 3) uint8, BGR 矫正后图像
    """
    import cv2
    return cv2.remap(image, plan.map1, plan.map2, interpolation=cv2.INTER_LINEAR)


# ======================================================================
# 内部辅助：plan 缓存（避免每帧重新计算映射表）
# ======================================================================
_plan_cache = {}


def _get_or_build_plan(camera_name, width, height, hfov, rectified_hfov):
    """按 (camera_name, width, height, hfov, rectified_hfov) 缓存 plan"""
    key = (camera_name, width, height, hfov, rectified_hfov)
    if key not in _plan_cache:
        _plan_cache[key] = build_plan(camera_name, width, height,
                                       hfov, rectified_hfov)
    return _plan_cache[key]


# ======================================================================
# 适配器：fisheye_preprocess() —— 对接感知节点主流水线
# ======================================================================
def fisheye_preprocess(images, fisheye_params=None):
    """
    对 6 路鱼眼相机图像做批量去畸变矫正。

    此函数是感知节点流水线中的"步骤一·鱼眼预处理"。
    算法核心（build_plan + apply_rectification）完全保留 rectifier.py 原始逻辑。

    参数:
        images: dict
            {camera_name: np.ndarray (H, W, 3) uint8}
            camera_name ∈ {front, back, left, left2, right, right2}
            图像格式: BGR (OpenCV 默认)，与 ROS Image 消息转换后一致

        fisheye_params: dict 或 None
            {
                'width':           int,     # 图像宽, 默认 1280
                'height':          int,     # 图像高, 默认 720
                'hfov_degrees':    float,   # 原始鱼眼 HFOV, 默认 175.0
                'rectified_hfov':  float,   # 矫正后 HFOV, 默认 120.0
            }
            不传则使用内置默认值（1280×720, 175°, 120°）。

    返回:
        dict: {camera_name: np.ndarray (H, W, 3) uint8}
            去畸变后的透视图像，与输入 image 同尺寸。
            未提供数据的相机不会出现在返回字典中。
    """
    if fisheye_params is None:
        fisheye_params = {}

    width  = fisheye_params.get('width', 1280)
    height = fisheye_params.get('height', 720)
    hfov   = fisheye_params.get('hfov_degrees', 175.0)
    rect_hfov = fisheye_params.get('rectified_hfov', 120.0)

    result = {}
    for camera_name, image in images.items():
        if image is None:
            continue

        # 获取或构建矫正映射表（首次调用构建，后续命中缓存）
        plan = _get_or_build_plan(camera_name, width, height, hfov, rect_hfov)

        # 矫正
        rectified = apply_rectification(image, plan)
        result[camera_name] = rectified

    return result
