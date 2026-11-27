import numpy as np

from tools.natural_scene_calibration.evaluate_pointcloud_calibration_readiness import (
    DEFAULT_THRESHOLDS,
    observability,
    window_metrics,
)


def _payload(*, lagged: bool = False) -> dict:
    frames = []
    centers = [
        np.asarray((4.0, -3.0, -0.5)),
        np.asarray((6.0, 0.0, 0.2)),
        np.asarray((9.0, 4.0, 1.2)),
        np.asarray((13.0, -1.0, 2.0)),
    ]
    for frame_index in range(48):
        clusters = []
        for track_id, base in enumerate(centers, 1):
            raw = base + np.asarray((0.002 * frame_index, 0.0, 0.0))
            calibration = raw + (np.asarray((0.1, 0.0, 0.0)) if lagged else 0.0)
            clusters.append(
                {
                    "points": 20,
                    "center_m": raw.tolist(),
                    "raw_center_m": raw.tolist(),
                    "calibration_center_m": calibration.tolist(),
                    "extent_m": [0.4, 0.3, 0.8],
                    "raw_extent_m": [0.4, 0.3, 0.8],
                    "range_m": float(np.linalg.norm(raw)),
                    "pca_eigenvalues_m2": [0.02, 0.01, 0.005],
                    "calibration_quality": {
                        "eligible": True,
                        "center_acceleration_m_per_frame2": 0.01,
                    },
                    "temporal_track": {
                        "track_id": track_id,
                        "calibration_center_source": "causal_current_timestamp_alpha_beta",
                    },
                }
            )
        frames.append(
            {
                "image_stamp_ns": frame_index,
                "lidar_stamp_ns": frame_index,
                "sync_delta_ms": 5.0,
                "clusters": clusters,
            }
        )
    return {"prohibited_inputs": ["/tf", "/tf_static", "odometry", "/world/*"], "frames": frames}


def test_window_metrics_accepts_unlagged_stable_geometry() -> None:
    metrics, representatives = window_metrics(_payload(), DEFAULT_THRESHOLDS)
    assert metrics["passed"]
    assert metrics["persistent_tracks_ge_6"] == 4
    assert len(representatives) == 4


def test_window_metrics_rejects_lagged_calibration_center() -> None:
    payload = _payload(lagged=True)
    for frame in payload["frames"]:
        for cluster in frame["clusters"]:
            cluster["temporal_track"]["calibration_center_source"] = "ema_center"
    metrics, _ = window_metrics(payload, DEFAULT_THRESHOLDS)
    assert not metrics["gates"]["unlagged_calibration_center"]


def test_se3_observability_reports_full_rank_for_diverse_points() -> None:
    report = observability(
        np.asarray(((4, -3, -0.5), (6, 0, 0.2), (9, 4, 1.2), (13, -1, 2.0)))
    )
    assert report["se3_information_rank"] == 6
    assert report["bearing_span_deg"] > 20.0
