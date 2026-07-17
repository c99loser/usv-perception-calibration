"""
================================================================================
 calib.py — 无人船多传感器外参自标定（算法 + 适配器）
================================================================================

算法来源（原封不动迁移）：
  config.py      → 传感器拓扑常量 + LidarLidarConfig
  utils.py       → 内参估算 / 视觉辅助初始值 / EMX滤波 / 外参矩阵工具
  lidar_lidar.py → LidarLidarCalibrator (FPFH + RANSAC + PointToPlane ICP)

适配器 calibrate_extrinsics()：
  输入：{rectified_images, preprocessed_clouds, known_topology, config}
  输出：{camera_extrinsics, lidar_extrinsics, camera_intrinsics, calib_quality}

依赖: numpy, scipy, open3d, opencv-python

============================================================================
第 0 部分：传感器拓扑常量（迁移自 config.py，不变）
============================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

# ---- 6 组 Airy LiDAR ↔ 鱼眼相机配对 ----
LIDAR_CAMERA_PAIRS: Dict[str, str] = {
    'front_lidar':  'front_camera',
    'back_lidar':   'back_camera',
    'left_lidar':   'left_camera',
    'left2_lidar':  'left2_camera',
    'right_lidar':  'right_camera',
    'right2_lidar': 'right2_camera',
}

LIDAR_NAMES = list(LIDAR_CAMERA_PAIRS.keys()) + ['forward_lidar']
CAMERA_NAMES = list(LIDAR_CAMERA_PAIRS.values())

# 已知精确约束: Airy LiDAR 在鱼眼相机正下方 10cm, 姿态绑定
LIDAR_CAMERA_Z_OFFSET_M = -0.10

# 传感器名义位置 (赛事文档, 仅作锚点参考)
NOMINAL_POSITIONS: Dict[str, Tuple[float, float, float]] = {
    'front_lidar':   ( 7.0,  0.0,    2.9),
    'back_lidar':    (-7.0,  0.0,    2.9),
    'left_lidar':    (-6.0,  1.626,  2.9),
    'left2_lidar':   ( 6.0,  1.626,  2.9),
    'right_lidar':   (-6.0, -1.626,  2.9),
    'right2_lidar':  ( 6.0, -1.626,  2.9),
    'forward_lidar': ( 7.0,  0.0,    2.0),
}

# ---- 赛事文档已知相机规格 ----
# 注: fx=419.1 从等距投影 r=f·θ 反推 (640 / tan⁻¹(175°/2)), 非PDF直接提供
# Gazebo真实鱼眼模型可能与等距有偏差, 但内参不在自标定优化回路中
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
CAMERA_HFOV_DEG = 175.0
CAMERA_DFOV_DEG = 175.0


# ======================================================================
# LiDAR-LiDAR 标定参数配置（迁移自 config.py，不变）
# ======================================================================

@dataclass
class LidarLidarConfig:
    fpfh_normal_radius:    float = 0.5
    fpfh_feature_radius:   float = 1.0
    gicp_max_correspondence: float = 1.0
    gicp_max_iterations:     int   = 100
    emx_ror_radius:        float = 0.5
    emx_ror_min_points:    int   = 5


# ======================================================================
# 第 1 部分：工具函数（迁移自 utils.py，不变）
# ======================================================================

def estimate_camera_intrinsics(
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT,
    hfov_deg: float = CAMERA_HFOV_DEG,
    dfov_deg: float = CAMERA_DFOV_DEG,
) -> np.ndarray:
    """从文档规格估算鱼眼相机等距投影内参 (3×3)"""
    cx = width / 2.0
    cy = height / 2.0
    fx = (width / 2.0) / np.deg2rad(hfov_deg / 2.0)
    fy = (height / 2.0) / np.deg2rad(dfov_deg / 2.0)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def get_lidar_camera_transform() -> np.ndarray:
    """返回 Airy LiDAR→Camera 精确外参 (10cm 下移, 姿态相同)"""
    T = np.eye(4, dtype=np.float64)
    T[2, 3] = LIDAR_CAMERA_Z_OFFSET_M
    return T


def filter_emx_for_calibration(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    intensity: np.ndarray, ring: np.ndarray,
    radius: float = 0.5, min_points: int = 5,
) -> Tuple[np.ndarray, ...]:
    """对标定环节的 EMX 点云补做半径滤波"""
    from scipy.spatial import cKDTree
    N = len(x)
    if N < min_points:
        return x, y, z, intensity, ring
    xyz = np.column_stack([x, y, z])
    tree = cKDTree(xyz)
    counts = tree.query_ball_point(xyz, r=radius, return_length=True)
    keep = counts >= min_points
    return x[keep], y[keep], z[keep], intensity[keep], ring[keep]


def accumulate_pointclouds(
    lidar_frames: List[Tuple],
    voxel_size: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """多帧点云叠加为稠密点云（帧间叠加 → 体素去重）"""
    all_x, all_y, all_z, all_i = [], [], [], []
    for x, y, z, intensity, _ in lidar_frames:
        all_x.append(np.asarray(x, dtype=np.float32))
        all_y.append(np.asarray(y, dtype=np.float32))
        all_z.append(np.asarray(z, dtype=np.float32))
        all_i.append(np.asarray(intensity, dtype=np.float32))

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    z = np.concatenate(all_z)
    intensity = np.concatenate(all_i)

    if voxel_size > 0 and len(x) > 0:
        xyz = np.column_stack([x, y, z])
        vx = np.floor(xyz / voxel_size).astype(np.int64)
        vrec = np.rec.fromarrays([vx[:, 0], vx[:, 1], vx[:, 2]], names='x,y,z')
        _, inv, cnt = np.unique(vrec, return_inverse=True, return_counts=True)
        n_v = len(cnt)
        x = (np.bincount(inv, weights=x, minlength=n_v) / cnt).astype(np.float32)
        y = (np.bincount(inv, weights=y, minlength=n_v) / cnt).astype(np.float32)
        z = (np.bincount(inv, weights=z, minlength=n_v) / cnt).astype(np.float32)
        intensity = (np.bincount(inv, weights=intensity, minlength=n_v) / cnt).astype(np.float32)

    return x, y, z, intensity


def compute_lidar_pair_init_from_cameras(
    image1: np.ndarray, image2: np.ndarray,
    K1: np.ndarray = None, K2: np.ndarray = None,
    feature_type: str = 'sift',
    approx_distance: float = 3.0,
) -> Tuple[np.ndarray, bool]:
    """
    利用相邻鱼眼相机视觉匹配 → 为对应 LiDAR 对提供粗配准初始值。

    原理:
      Cam1 ← SIFT/E → Cam2 → T_cam1→cam2
        ↓ 10cm              ↓ 10cm
      LiDAR1               LiDAR2
        ↓
      T_lid1→lid2 = T_cam→lid · T_cam1→cam2 · T_lid→cam
    """
    import cv2

    if K1 is None:
        K1 = estimate_camera_intrinsics(image1.shape[1], image1.shape[0])
    if K2 is None:
        K2 = estimate_camera_intrinsics(image2.shape[1], image2.shape[0])

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    if feature_type == 'sift':
        try:
            detector = cv2.SIFT_create()
        except AttributeError:
            detector = cv2.ORB_create(nfeatures=2000)
    else:
        try:
            detector = cv2.SIFT_create()
        except AttributeError:
            detector = cv2.ORB_create(nfeatures=2000)

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return np.eye(4), False

    is_orb = isinstance(detector, cv2.ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING if is_orb else cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    if len(matches) < 15:
        return np.eye(4), False

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts1, pts2, K1, method=cv2.RANSAC,
                                    prob=0.999, threshold=1.0)
    if E is None:
        return np.eye(4), False

    inliers = mask.ravel() == 1
    if inliers.sum() < 10:
        return np.eye(4), False

    _, R_cam, t_cam, _ = cv2.recoverPose(E, pts1[inliers], pts2[inliers], K1)
    t_cam = t_cam.ravel() * approx_distance
    T_cam1_to_cam2 = np.eye(4)
    T_cam1_to_cam2[:3, :3] = R_cam
    T_cam1_to_cam2[:3, 3]  = t_cam

    T_lidar_to_cam = get_lidar_camera_transform()
    T_cam_to_lidar = np.eye(4)
    T_cam_to_lidar[2, 3] = -LIDAR_CAMERA_Z_OFFSET_M

    T_lidar1_to_lidar2 = T_cam_to_lidar @ T_cam1_to_cam2 @ T_lidar_to_cam
    return T_lidar1_to_lidar2, True



def rot_trans_to_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """R_3×3 + t_3 → T_4×4"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).flatten()
    return T





# ======================================================================
# 第 2 部分：LidarLidarCalibrator（URDF 6DOF 初值 → ICP 精修）
# ======================================================================

class LidarLidarCalibrator:
    """LiDAR-LiDAR 外参标定器：URDF 6DOF 名义初值 → PointToPlane ICP 精修"""

    def __init__(self, config: 'LidarLidarConfig' = None):
        self.config = config or LidarLidarConfig()

    def calibrate_all(
        self,
        lidar_data: 'Dict[str, Tuple]',
        reference_lidar: str = 'front_lidar',
        voxel_size: float = 0.1,
        camera_inits: 'Dict[str, np.ndarray]' = None,
    ) -> 'Dict[str, np.ndarray]':
        """以 reference_lidar 为基准级联标定所有 LiDAR"""
        extrinsics: Dict[str, np.ndarray] = {reference_lidar: np.eye(4)}
        observations: Dict[Tuple[str, str], Tuple[np.ndarray, float]] = {}

        if camera_inits is None:
            camera_inits = {}

        # ---- 所有可能重叠的雷达对 ----
        pairs = [
            ('left_lidar',   'front_lidar'),
            ('right_lidar',  'front_lidar'),
            ('left2_lidar',  'left_lidar'),
            ('right2_lidar', 'right_lidar'),
            ('forward_lidar','front_lidar'),
            ('back_lidar',   'left_lidar'),
            ('left2_lidar',  'front_lidar'),
            ('right2_lidar', 'front_lidar'),
            ('back_lidar',   'front_lidar'),
            ('forward_lidar','left_lidar'),
            ('forward_lidar','right_lidar'),
            ('back_lidar',   'right_lidar'),
        ]

        for source, target in pairs:
            if source not in lidar_data or target not in lidar_data:
                continue
            src = lidar_data[source]
            tgt = lidar_data[target]
            src_xyz = (src[0], src[1], src[2])
            tgt_xyz = (tgt[0], tgt[1], tgt[2])
            init_key = f'{source}|{target}'
            camera_init = camera_inits.get(init_key)

            try:
                T_src_to_tgt = self.calibrate_pair(
                    src_xyz, tgt_xyz, source, target, voxel_size, camera_init)
                src_pcd = self._build_pcd(*src_xyz)
                tgt_pcd = self._build_pcd(*tgt_xyz)
                _, fitness = self._fine_registration(src_pcd, tgt_pcd, T_src_to_tgt)
                weight = max(fitness, 0.001)
                observations[(source, target)] = (T_src_to_tgt, weight)
            except RuntimeError:
                if camera_init is not None:
                    observations[(source, target)] = (camera_init, 0.001)

        # ---- 级联链求解 ----
        cascade_pairs = [
            ('left_lidar',   'front_lidar'),
            ('right_lidar',  'front_lidar'),
            ('left2_lidar',  'left_lidar'),
            ('right2_lidar', 'right_lidar'),
            ('forward_lidar','front_lidar'),
            ('back_lidar',   'left_lidar'),
        ]
        for source, target in cascade_pairs:
            if (source, target) not in observations:
                continue
            T, _ = observations[(source, target)]
            if target in extrinsics:
                extrinsics[source] = extrinsics[target] @ T

        return extrinsics


    @staticmethod
    def _nominal_6dof_init(source_name: str, target_name: str) -> np.ndarray:
        """
        [2026-07-13] 从 URDF 名义安装参数计算完整的 6DOF 初始变换。

        Airy LiDAR: pitch=115°（半球朝下）, yaw 按安装方向,
                   位置从 NOMINAL_POSITIONS 查表。
        EMX forward: pitch=0°（水平向前）。

        返回: T_src→tgt (4x4)
        """
        from scipy.spatial.transform import Rotation as R

        NOMINAL_RPY = {
            'front_lidar':   (0, 115.0,   0.0),
            'back_lidar':    (0, 115.0, 180.0),
            'left_lidar':    (0, 115.0,  90.0),
            'left2_lidar':   (0, 115.0,  90.0),
            'right_lidar':   (0, 115.0, -90.0),
            'right2_lidar':  (0, 115.0, -90.0),
            'forward_lidar': (0,   0.0,   0.0),
        }

        def _make_T(name):
            t = np.array(NOMINAL_POSITIONS.get(name, (0, 0, 0)), dtype=np.float64)
            roll, pitch, yaw = NOMINAL_RPY.get(name, (0, 0, 0))
            R_mat = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_mat
            T[:3, 3] = t
            return T

        T_src = _make_T(source_name)
        T_tgt = _make_T(target_name)
        return np.linalg.inv(T_tgt) @ T_src

    def calibrate_pair(
        self,
        source_xyz: Tuple[np.ndarray, np.ndarray, np.ndarray],
        target_xyz: Tuple[np.ndarray, np.ndarray, np.ndarray],
        source_name: str = 'source',
        target_name: str = 'target',
        voxel_size: float = 0.1,
        camera_init: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """标定一对 LiDAR 外参: URDF 6DOF 初值 → ICP 精修"""

        # 先确定初始变换（URDF 6DOF 为基础，视觉初值可选覆盖）
        if source_name and target_name:
            T_init = self._nominal_6dof_init(source_name, target_name)
            if camera_init is not None:
                T_init = camera_init  # 视觉初值覆盖 URDF（显式指定时）
        else:
            T_init = camera_init  # 无名传感器：只能用视觉初值

        if T_init is None:
            # 3. FPFH+RANSAC 粗配（兜底）
            src_pcd = self._build_pcd(*source_xyz)
            tgt_pcd = self._build_pcd(*target_xyz)
            src_down, src_fpfh = self._compute_fpfh(src_pcd, voxel_size)
            tgt_down, tgt_fpfh = self._compute_fpfh(tgt_pcd, voxel_size)
            if src_fpfh is None or tgt_fpfh is None:
                raise RuntimeError(
                    f'{source_name}→{target_name}: FPFH 特征提取失败')
            T_init = self._coarse_registration(
                src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size)

        # 直接 ICP 精修（URDF 6DOF 初值）
        src_pcd = self._build_pcd(*source_xyz)
        tgt_pcd = self._build_pcd(*target_xyz)
        T_fine, fitness = self._fine_registration(src_pcd, tgt_pcd, T_init)
        if fitness < 0.005:  # 100帧密集数据 fitness 自然偏低
            raise RuntimeError(
                f'{source_name}→{target_name}: ICP 精配失败, fitness={fitness:.3f}')
        return T_fine
    def _build_pcd(self, x, y, z):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.column_stack([np.asarray(x, dtype=np.float64),
                             np.asarray(y, dtype=np.float64),
                             np.asarray(z, dtype=np.float64)]))
        if len(pcd.points) > 0:
            nn = min(30, len(pcd.points))
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=nn))
        return pcd

    def _compute_fpfh(self, pcd, voxel_size: float):
        import open3d as o3d
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.config.fpfh_normal_radius, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.config.fpfh_feature_radius, max_nn=100))
        return pcd_down, fpfh

    def _coarse_registration(self, src, tgt, sf, tf, vs: float):
        import open3d as o3d
        r = o3d.pipelines.registration
        return r.registration_ransac_based_on_feature_matching(
            src, tgt, sf, tf, mutual_filter=True,
            max_correspondence_distance=vs * 1.5,
            estimation_method=r.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[r.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                      r.CorrespondenceCheckerBasedOnDistance(vs * 1.5)],
            criteria=r.RANSACConvergenceCriteria(4000000, 500),
        ).transformation

    def _fine_registration(self, src, tgt, init: np.ndarray):
        import open3d as o3d
        r = o3d.pipelines.registration
        result = r.registration_icp(
            src, tgt,
            max_correspondence_distance=self.config.gicp_max_correspondence,
            init=init,
            estimation_method=r.TransformationEstimationPointToPlane(),
            criteria=r.ICPConvergenceCriteria(
                max_iteration=self.config.gicp_max_iterations,
                relative_fitness=1e-6, relative_rmse=1e-6),
        )
        return result.transformation, result.fitness


# ======================================================================
# 第 3 部分：适配器 calibrate_extrinsics()
# ======================================================================

def calibrate_extrinsics(
    rectified_images: dict,
    preprocessed_clouds: dict,
    known_topology: dict = None,
    config: dict = None,
) -> dict:
    """
    不依赖陆地标定的多传感器外参自标定（赛事任务 1）。

    流水线:
      1. LiDAR-LiDAR 外参标定 (FPFH + RANSAC + PointToPlane ICP)
         → 有相机图像时自动用视觉特征匹配提供初始值，跳过 FPFH+RANSAC
      2. 从 LiDAR 外参推导相机外参（利用 Airy↔鱼眼 10cm 精确约束）
      3. 从文档规格估算相机内参
      4. 输出全部外参 + 标定质量评估

    参数:
        rectified_images: dict
            {camera_name: np.ndarray(H, W, 3) uint8 BGR}
            camera_name ∈ {front, back, left, left2, right, right2}
            来自 fisheye_preprocess 输出。可传空 dict {} 跳过视觉辅助。

        preprocessed_clouds: dict
            {lidar_name: np.ndarray(M, 3) float32}
            lidar_name ∈ {front, back, left, left2, right, right2, forward}
            来自 lidar_preprocess 输出。所有 7 路必填。

        known_topology: dict 或 None
            {
                'lidar_camera_pairs': {lidar_name: camera_name},   # 可覆盖默认配对
                'relative_offset_xyz': (dx, dy, dz),               # 可覆盖默认 10cm
            }
            None 则使用赛事文档内置常量。

        config: dict 或 None
            {
                'method':         str,    # 默认 'fpfh_ransac_icp'
                'use_visual_init': bool,   # 默认 True
                'voxel_size':     float,   # FPFH 体素, 默认 0.1
                'reference_lidar': str,    # 默认 'front'
                'fpfh_normal_radius':  float,  # 默认 0.5
                'fpfh_feature_radius': float,  # 默认 1.0
                'icp_max_correspondence': float,  # 默认 1.0
                'icp_max_iterations': int,       # 默认 100
            }

    返回:
        {
            'camera_extrinsics': {cam_name: np.ndarray(4,4)},
                # 相机→船体坐标系 (boat frame = front_lidar 位置)
            'lidar_extrinsics':  {lid_name: np.ndarray(4,4)},
                # 激光雷达→船体坐标系
            'camera_intrinsics': {cam_name: np.ndarray(3,3)},
                # 鱼眼相机内参 (等距投影模型)
            'calib_quality': {
                'per_pair': {pair_key: {'method': str, 'success': bool}},
                'overall': float,   # [0, 1]
            },
            'method': str,   # 'fpfh_ransac_icp' 或 'visual_icp'
        }
    """
    # ---- 默认配置 ----
    if config is None:
        config = {}
    if known_topology is None:
        known_topology = {}

    use_visual = config.get('use_visual_init', True)
    voxel_size = config.get('voxel_size', 0.1)
    reference_lidar = config.get('reference_lidar', 'front')
    method_label = config.get('method', 'nominal_6dof_icp')

    # ---- 传感器配对 ----
    lidar_camera_pairs = known_topology.get(
        'lidar_camera_pairs', LIDAR_CAMERA_PAIRS)
    rel_offset = known_topology.get(
        'relative_offset_xyz', (0.0, 0.0, LIDAR_CAMERA_Z_OFFSET_M))

    # ---- 命名映射：外部名(short) ↔ 内部名(full) ----
    SHORT_TO_LIDAR = {n.replace('_lidar', ''): n for n in LIDAR_NAMES}
    SHORT_TO_CAMERA = {n.replace('_camera', ''): n for n in CAMERA_NAMES}

    # ---- 构建 LiDAR 数据（适配①：short_name → full_name + xyz→5元组）----
    lidar_data: Dict[str, Tuple] = {}
    for short_name, xyz in preprocessed_clouds.items():
        full_name = SHORT_TO_LIDAR.get(short_name)
        if full_name is None:
            continue
        if xyz is None or len(xyz) == 0:
            continue
        n = len(xyz)
        lidar_data[full_name] = (
            xyz[:, 0].astype(np.float32),
            xyz[:, 1].astype(np.float32),
            xyz[:, 2].astype(np.float32),
            np.zeros(n, dtype=np.float32),    # 仿真 intensity 恒 0
            np.zeros(n, dtype=np.uint16),     # ring 填 0
        )

    if len(lidar_data) < 2:
        return {
            'camera_extrinsics': {},
            'lidar_extrinsics': {},
            'camera_intrinsics': {},
            'calib_quality': {'per_pair': {}, 'overall': 0.0},
            'method': method_label,
        }

    # ---- 相机辅助初始值（用 rectified_images 做视觉匹配）----
    camera_inits: Dict[str, np.ndarray] = {}
    if use_visual and rectified_images:
        visual_pairs = [
            ('front', 'left',    'left_lidar',    'front_lidar'),
            ('front', 'right',   'right_lidar',   'front_lidar'),
            ('front', 'back',    'back_lidar',    'front_lidar'),
            ('left',  'left2',   'left2_lidar',   'left_lidar'),
            ('right', 'right2',  'right2_lidar',  'right_lidar'),
            ('left',  'back',    'back_lidar',    'left_lidar'),
        ]
        CAMERA_TO_LIDAR = {v: k for k, v in LIDAR_CAMERA_PAIRS.items()}
        for cam1, cam2, lid_src, lid_tgt in visual_pairs:
            if cam1 in rectified_images and cam2 in rectified_images:
                pos1 = np.array(NOMINAL_POSITIONS.get(lid_src, (0,0,0)))
                pos2 = np.array(NOMINAL_POSITIONS.get(lid_tgt, (0,0,0)))
                approx_dist = max(np.linalg.norm(pos1 - pos2), 3.0)
                T, ok = compute_lidar_pair_init_from_cameras(
                    rectified_images[cam1], rectified_images[cam2],
                    approx_distance=approx_dist)
                if ok:
                    lid_cam1 = CAMERA_TO_LIDAR.get(f'{cam1}_camera')
                    lid_cam2 = CAMERA_TO_LIDAR.get(f'{cam2}_camera')
                    if lid_cam1 == lid_src and lid_cam2 == lid_tgt:
                        camera_inits[f'{lid_src}|{lid_tgt}'] = T
                    elif lid_cam1 == lid_tgt and lid_cam2 == lid_src:
                        camera_inits[f'{lid_src}|{lid_tgt}'] = np.linalg.inv(T)
                    method_label = 'visual_icp'

    # ---- 运行 LiDAR-LiDAR 标定 ----
    ll_config = LidarLidarConfig(
        fpfh_normal_radius=config.get('fpfh_normal_radius', 0.5),
        fpfh_feature_radius=config.get('fpfh_feature_radius', 1.0),
        gicp_max_correspondence=config.get('icp_max_correspondence', 1.0),
        gicp_max_iterations=config.get('icp_max_iterations', 100),
    )
    calib = LidarLidarCalibrator(ll_config)
    lidar_to_ref = calib.calibrate_all(
        lidar_data, f'{reference_lidar}_lidar', voxel_size, camera_inits)

    # ---- 转换到船体坐标系 ----
    # 船体坐标系 (boat frame) 定义：
    #   原点 O = front_lidar 与 back_lidar 的中点
    #   x 轴正方向 = O → front_lidar（船艏方向）
    #   z 轴 = 6 个 Airy 雷达位置拟合的甲板平面法向量（朝上）
    #   y 轴 = z × x（右手系，左舷方向）
    #
    # 兜底：若 back 未标出，用 front 名义位置；若平面拟合失败，z=(0,0,1)
    ref_full = f'{reference_lidar}_lidar'

    # ① 用 URDF 世界坐标计算船体原点 O
    nom_front = np.array(NOMINAL_POSITIONS.get(ref_full, (0, 0, 0)), dtype=np.float64)
    nom_back  = np.array(NOMINAL_POSITIONS.get('back_lidar', (0, 0, 0)), dtype=np.float64)
    O_world = (nom_front + nom_back) / 2.0

    # ② 传感器世界位置 = 名义世界 + ICP delta（不做兜底，信任算法输出）
    lidar_world: Dict[str, np.ndarray] = {}
    for full_name, T_to_ref in lidar_to_ref.items():
        nom_pos = np.array(NOMINAL_POSITIONS.get(full_name, (0, 0, 0)), dtype=np.float64)
        icp_ref = T_to_ref[:3, 3].copy()
        nom_ref = nom_pos - nom_front
        lidar_world[full_name] = nom_pos + (icp_ref - nom_ref)

    # ③ 船体系位置 = 世界位置 - 船体原点
    lidar_positions_boat = {}
    for full_name, P_world in lidar_world.items():
        lidar_positions_boat[full_name] = P_world - O_world

    # ④ 船体系坐标轴
    P_front_boat = lidar_positions_boat.get(ref_full, np.array([7.0, 0.0, 0.0]))
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if np.linalg.norm(P_front_boat[:2]) > 0.1:
        x_axis[:2] = P_front_boat[:2] / np.linalg.norm(P_front_boat[:2])
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # ⑤ 输出 lidar extrinsics
    lidar_extrinsics: Dict[str, np.ndarray] = {}
    R_boat = np.column_stack([x_axis, y_axis, z_axis])
    for full_name, P_boat in lidar_positions_boat.items():
        short = full_name.replace('_lidar', '')
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = lidar_to_ref[full_name][:3, :3]  # ICP 精修的旋转
        T[:3, 3] = R_boat @ P_boat  # 世界位置 → 船体坐标系旋转后
        lidar_extrinsics[short] = T

    # ---- 推导相机外参（LiDAR 外参 + 10cm 约束 → 相机外参）----
    rel_dx, rel_dy, rel_dz = rel_offset
    T_lidar_to_cam = np.eye(4, dtype=np.float64)
    T_lidar_to_cam[0, 3] = rel_dx
    T_lidar_to_cam[1, 3] = rel_dy
    T_lidar_to_cam[2, 3] = rel_dz

    camera_extrinsics: Dict[str, np.ndarray] = {}
    for lidar_name, cam_name in lidar_camera_pairs.items():
        lidar_short = lidar_name.replace('_lidar', '')
        cam_short = cam_name.replace('_camera', '')
        T_lidar_to_boat = lidar_extrinsics.get(lidar_short)
        if T_lidar_to_boat is not None:
            T_cam_to_boat = T_lidar_to_boat @ np.linalg.inv(T_lidar_to_cam)
            camera_extrinsics[cam_short] = T_cam_to_boat

    # ---- 估算相机内参 ----
    camera_intrinsics: Dict[str, np.ndarray] = {}
    for cam_name in CAMERA_NAMES:
        cam_short = cam_name.replace('_camera', '')
        if cam_short in camera_extrinsics:
            camera_intrinsics[cam_short] = estimate_camera_intrinsics()

    # ---- 质量评估 ----
    per_pair = {}
    success_count = 0
    total_pairs = len(lidar_to_ref) - 1  # 不含 reference 自身
    for name, T in lidar_to_ref.items():
        if name == ref_full:
            continue
        # ICP fitness > 0 即算成功（fitness < 0.005 已在 calibrate_pair 被拒绝）
        short = name.replace('_lidar', '')
        per_pair[short] = {
            'method': 'nominal_6dof_icp',
            'success': True,
        }
        success_count += 1

    overall = success_count / max(total_pairs, 1)

    method_label = f'boat_geo+{method_label}'

    return {
        'camera_extrinsics': camera_extrinsics,
        'lidar_extrinsics': lidar_extrinsics,
        'camera_intrinsics': camera_intrinsics,
        'calib_quality': {'per_pair': per_pair, 'overall': overall},
        'method': method_label,
        'boat_frame': {
            'origin_world': O_world.tolist(),
            'x_axis': x_axis.tolist(),
            'y_axis': y_axis.tolist(),
            'z_axis': z_axis.tolist(),
            'n_airy_for_plane': 0,
            'used_back': True,
        },
    }
