"""Sensor-only temporal confirmation and stable geometry for point-cloud clusters.

The instantaneous axis-aligned extent is retained for audit, while confirmed
tracks expose a robust, frozen reference extent. Static calibration objects
should not change physical size because a few beams become occluded or an
axis-aligned box changes with viewpoint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


TRACKER_VERSION = "pointcloud_tracker_v1.5.0_calibration_kinematic_quality"


@dataclass(frozen=True)
class PointCloudTrackConfig:
    max_center_jump_m: float = 1.5
    max_extent_delta_m: float = 1.0
    min_confirmed_hits: int = 3
    max_frame_gap: int = 2
    ema_alpha: float = 0.35
    center_alpha: float = 0.85
    center_beta: float = 0.10
    reference_extent_hits: int = 3
    min_calibration_points: int = 8
    min_calibration_range_m: float = 2.0
    max_calibration_extent_m: float = 2.0
    max_calibration_center_se_m: float = 0.15
    max_calibration_residual_m: float = 0.35
    max_calibration_acceleration_m_per_frame2: float = 0.50

    def to_dict(self) -> dict[str, object]:
        return {"version": TRACKER_VERSION, **asdict(self)}


@dataclass
class _State:
    track_id: int
    last_frame: int
    hits: int
    center: np.ndarray
    velocity: np.ndarray
    center_step: np.ndarray
    center_acceleration: float
    association_extent: np.ndarray
    extent_samples: list[np.ndarray]
    reference_extent: np.ndarray | None = None


class PointCloudTrackFilter:
    """Confirm adjacent-frame tracks and emit stable, auditable geometry."""

    def __init__(self, config: PointCloudTrackConfig = PointCloudTrackConfig()) -> None:
        if not 0.0 < config.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0,1]")
        if not 0.0 < config.center_alpha <= 1.0:
            raise ValueError("center_alpha must be in (0,1]")
        if not 0.0 <= config.center_beta <= 1.0:
            raise ValueError("center_beta must be in [0,1]")
        if config.min_confirmed_hits < 1:
            raise ValueError("min_confirmed_hits must be positive")
        if config.reference_extent_hits < 1:
            raise ValueError("reference_extent_hits must be positive")
        self.config = config
        self.states: dict[int, _State] = {}
        self.next_id = 1

    def update(self, frame_index: int, clusters: list[dict]) -> list[dict]:
        expired = [
            track_id
            for track_id, state in self.states.items()
            if frame_index - state.last_frame > self.config.max_frame_gap
        ]
        for track_id in expired:
            del self.states[track_id]

        centers = [np.asarray(cluster["center_m"], dtype=float) for cluster in clusters]
        extents = [np.asarray(cluster["extent_m"], dtype=float) for cluster in clusters]
        options = []
        for local, (center, extent) in enumerate(zip(centers, extents)):
            for track_id, state in self.states.items():
                frame_gap = frame_index - state.last_frame
                if frame_gap < 1 or frame_gap > self.config.max_frame_gap:
                    continue
                predicted = state.center + state.velocity * frame_gap
                center_delta = float(np.linalg.norm(center - predicted))
                extent_delta = float(np.linalg.norm(extent - state.association_extent))
                if (
                    center_delta <= self.config.max_center_jump_m
                    and extent_delta <= self.config.max_extent_delta_m
                ):
                    gap_penalty = 0.05 * (frame_gap - 1)
                    options.append(
                        (center_delta + 0.25 * extent_delta + gap_penalty, local, track_id)
                    )
        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, local, track_id in sorted(options):
            if local not in assignments and track_id not in used_tracks:
                assignments[local] = track_id
                used_tracks.add(track_id)

        output = []
        alpha = self.config.ema_alpha
        for local, cluster in enumerate(clusters):
            track_id = assignments.get(local)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                state = _State(
                    track_id,
                    frame_index,
                    1,
                    centers[local],
                    np.zeros(3, dtype=float),
                    np.zeros(3, dtype=float),
                    0.0,
                    extents[local],
                    [extents[local].copy()],
                )
                self.states[track_id] = state
            else:
                state = self.states[track_id]
                frame_gap = frame_index - state.last_frame
                previous_center = state.center.copy()
                previous_step = state.center_step.copy()
                predicted = state.center + state.velocity * frame_gap
                innovation = centers[local] - predicted
                state.center = predicted + self.config.center_alpha * innovation
                state.velocity = (
                    state.velocity
                    + self.config.center_beta * innovation / max(frame_gap, 1)
                )
                state.center_step = (state.center - previous_center) / max(frame_gap, 1)
                state.center_acceleration = float(
                    np.linalg.norm(state.center_step - previous_step)
                )
                state.association_extent = (
                    alpha * extents[local] + (1.0 - alpha) * state.association_extent
                )
                state.last_frame = frame_index
                state.hits += 1
                if state.reference_extent is None:
                    state.extent_samples.append(extents[local].copy())

            if (
                state.reference_extent is None
                and len(state.extent_samples) >= self.config.reference_extent_hits
            ):
                state.reference_extent = np.median(
                    np.asarray(state.extent_samples, dtype=float), axis=0
                )
                state.extent_samples.clear()
            if state.hits < self.config.min_confirmed_hits:
                continue
            if state.reference_extent is None:
                stable_extent = np.median(
                    np.asarray(state.extent_samples, dtype=float), axis=0
                )
                extent_state = "provisional_median"
            else:
                stable_extent = state.reference_extent
                extent_state = "frozen_median"

            item = dict(cluster)
            item["raw_center_m"] = centers[local].round(4).tolist()
            item["raw_extent_m"] = extents[local].round(4).tolist()
            # A causal alpha-beta estimate represents the current LiDAR timestamp
            # without the systematic lag of an EMA under constant motion.
            item["calibration_center_m"] = state.center.round(4).tolist()
            item["association_center_m"] = state.center.round(4).tolist()
            item["center_m"] = state.center.round(4).tolist()
            item["extent_m"] = stable_extent.round(4).tolist()
            item["reference_extent_m"] = stable_extent.round(4).tolist()
            item["range_m"] = round(float(np.linalg.norm(state.center)), 4)
            center_residual = float(np.linalg.norm(centers[local] - state.center))
            eigenvalues = np.asarray(cluster.get("pca_eigenvalues_m2", []), dtype=float)
            point_count = int(cluster.get("points", 0))
            explicit_se = cluster.get("center_standard_error_m")
            if isinstance(explicit_se, (int, float)) and np.isfinite(explicit_se):
                center_se = float(explicit_se)
            elif eigenvalues.shape == (3,) and point_count > 0:
                center_se = float(
                    np.sqrt(np.sum(np.maximum(eigenvalues, 0.0)) / point_count)
                )
            else:
                center_se = float("inf")
            quality_reasons = []
            if point_count < self.config.min_calibration_points:
                quality_reasons.append("too_few_points")
            if float(np.linalg.norm(state.center)) < self.config.min_calibration_range_m:
                quality_reasons.append("near_field")
            if float(np.max(stable_extent)) > self.config.max_calibration_extent_m:
                quality_reasons.append("oversized_extent")
            if center_se > self.config.max_calibration_center_se_m:
                quality_reasons.append("center_uncertainty")
            if center_residual > self.config.max_calibration_residual_m:
                quality_reasons.append("center_innovation")
            if (
                state.center_acceleration
                > self.config.max_calibration_acceleration_m_per_frame2
            ):
                quality_reasons.append("center_acceleration")
            item["calibration_quality"] = {
                "eligible": not quality_reasons,
                "reasons": quality_reasons,
                "center_standard_error_m": (
                    round(center_se, 6) if np.isfinite(center_se) else None
                ),
                "center_residual_m": round(center_residual, 6),
                "center_acceleration_m_per_frame2": round(
                    state.center_acceleration, 6
                ),
                "inverse_variance_weight": (
                    round(1.0 / max(center_se * center_se, 1e-4), 6)
                    if np.isfinite(center_se)
                    else 0.0
                ),
            }
            item["temporal_track"] = {
                "track_id": state.track_id,
                "confirmed_hits": state.hits,
                "geometry": "causal_alpha_beta_center_frozen_extent_quality_gate",
                "calibration_center_source": "causal_current_timestamp_alpha_beta",
                "extent_state": extent_state,
                "reference_extent_hits": self.config.reference_extent_hits,
            }
            item["source_cluster_index"] = local
            output.append(item)
        return output
