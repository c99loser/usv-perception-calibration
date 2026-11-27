"""Sensor-only temporal proposals for point-cloud episode annotation.

This module is deliberately an annotation aid, not a source of ground truth and
not a replacement for the validated V4.11 cluster geometry.  It borrows three
ideas from CN202510931459/CN120871086A:

* a five-degree-of-freedom planar state (x, y, heading, speed, yaw rate),
* dual likelihoods that penalise both observation residual and motion-history
  deviation, and
* bounded observer correction across short missing-observation gaps.

The implementation is retrospective because an offline annotation episode can
verify that the same producer track reappears after a gap.  Only such bounded
interior gaps are compensated; it never extrapolates beyond the last observed
frame and never consumes camera projection, TF, odometry, or world state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import atan2, cos, exp, pi, sin

import numpy as np


ASSIST_VERSION = "pointcloud_annotation_assist_v1.0.0_patent_inspired"


@dataclass(frozen=True)
class PointCloudAnnotationAssistConfig:
    max_bridge_gap: int = 2
    min_track_hits: int = 3
    min_eligible_hits: int = 2
    observation_sigma_m: float = 0.45
    history_sigma_m: float = 0.65
    observer_correction_limit_m: float = 0.60
    min_proposal_confidence: float = 0.08

    def to_dict(self) -> dict[str, object]:
        return {"version": ASSIST_VERSION, **asdict(self)}


def _wrap_angle(value: float) -> float:
    return (value + pi) % (2.0 * pi) - pi


def _center(cluster: dict) -> np.ndarray:
    for key in ("raw_center_m", "calibration_center_m", "center_m"):
        value = np.asarray(cluster.get(key, ()), dtype=float)
        if value.shape == (3,) and np.isfinite(value).all():
            return value
    raise ValueError("tracked cluster has no finite 3D center")


def _extent(cluster: dict) -> np.ndarray:
    for key in ("reference_extent_m", "extent_m", "raw_extent_m"):
        value = np.asarray(cluster.get(key, ()), dtype=float)
        if value.shape == (3,) and np.isfinite(value).all() and np.all(value > 0):
            return value
    return np.full(3, 0.25, dtype=float)


def _track_id(cluster: dict) -> int | None:
    temporal = cluster.get("temporal_track")
    if not isinstance(temporal, dict) or "track_id" not in temporal:
        return None
    return int(temporal["track_id"])


def _motion_state(observations: list[dict], position: int) -> dict[str, float | np.ndarray]:
    current = observations[position]
    center = current["center"]
    if position < 1:
        return {"center": center, "heading": 0.0, "speed": 0.0, "yaw_rate": 0.0, "vz": 0.0}
    previous = observations[position - 1]
    dt = max(1, int(current["frame_index"] - previous["frame_index"]))
    step = (center - previous["center"]) / dt
    planar_speed = float(np.linalg.norm(step[:2]))
    heading = atan2(float(step[1]), float(step[0])) if planar_speed > 1e-8 else 0.0
    yaw_rate = 0.0
    if position >= 2:
        older = observations[position - 2]
        older_dt = max(1, int(previous["frame_index"] - older["frame_index"]))
        older_step = (previous["center"] - older["center"]) / older_dt
        older_speed = float(np.linalg.norm(older_step[:2]))
        if planar_speed > 1e-8 and older_speed > 1e-8:
            older_heading = atan2(float(older_step[1]), float(older_step[0]))
            yaw_rate = _wrap_angle(heading - older_heading) / dt
    return {
        "center": center,
        "heading": heading,
        "speed": planar_speed,
        "yaw_rate": yaw_rate,
        "vz": float(step[2]),
    }


def _predict_5dof(state: dict[str, float | np.ndarray], dt: float) -> np.ndarray:
    center = np.asarray(state["center"], dtype=float)
    heading = float(state["heading"])
    speed = float(state["speed"])
    yaw_rate = float(state["yaw_rate"])
    result = center.copy()
    if abs(yaw_rate) < 1e-6:
        result[0] += speed * cos(heading) * dt
        result[1] += speed * sin(heading) * dt
    else:
        result[0] += speed / yaw_rate * (sin(heading + yaw_rate * dt) - sin(heading))
        result[1] += speed / yaw_rate * (-cos(heading + yaw_rate * dt) + cos(heading))
    result[2] += float(state["vz"]) * dt
    return result


def _normalised_dual_weights(
    endpoint_predictions: list[np.ndarray],
    observed_endpoint: np.ndarray,
    history_deviations: list[float],
    config: PointCloudAnnotationAssistConfig,
) -> tuple[np.ndarray, list[float], float, float]:
    observation_residuals = [
        float(np.linalg.norm(prediction - observed_endpoint))
        for prediction in endpoint_predictions
    ]
    log_weights = np.asarray(
        [
            -0.5 * (residual / config.observation_sigma_m) ** 2
            -0.5 * (history / config.history_sigma_m) ** 2
            for residual, history in zip(observation_residuals, history_deviations)
        ],
        dtype=float,
    )
    shifted = np.exp(log_weights - float(np.max(log_weights)))
    weights = shifted / max(float(np.sum(shifted)), 1e-12)
    effective_hypotheses = 1.0 / float(np.sum(weights * weights))
    confidence = exp(float(np.max(log_weights)))
    return weights, observation_residuals, effective_hypotheses, confidence


def build_annotation_assists(
    frames: list[dict],
    config: PointCloudAnnotationAssistConfig = PointCloudAnnotationAssistConfig(),
) -> tuple[list[list[dict]], dict[str, object]]:
    """Return per-frame observed/compensated annotation proposals and audit data."""
    if config.max_bridge_gap < 0:
        raise ValueError("max_bridge_gap must be non-negative")
    if config.min_track_hits < 2:
        raise ValueError("min_track_hits must be at least two")
    if config.min_eligible_hits < 1:
        raise ValueError("min_eligible_hits must be positive")
    if config.observation_sigma_m <= 0 or config.history_sigma_m <= 0:
        raise ValueError("likelihood sigmas must be positive")
    if config.observer_correction_limit_m <= 0:
        raise ValueError("observer_correction_limit_m must be positive")

    by_track: dict[int, list[dict]] = {}
    for frame_index, frame in enumerate(frames):
        seen: set[int] = set()
        for cluster in frame.get("clusters", []):
            track_id = _track_id(cluster)
            if track_id is None:
                continue
            if track_id in seen:
                raise ValueError(f"duplicate temporal track_id {track_id} in frame {frame_index}")
            seen.add(track_id)
            by_track.setdefault(track_id, []).append(
                {
                    "frame_index": frame_index,
                    "center": _center(cluster),
                    "extent": _extent(cluster),
                    "eligible": cluster.get("calibration_quality", {}).get("eligible") is True,
                    "source_cluster_index": int(cluster.get("source_cluster_index", -1)),
                }
            )

    per_frame: list[list[dict]] = [[] for _ in frames]
    compensated = rejected = rejected_ineligible = 0
    eligible_tracks = 0
    for track_id, observations in sorted(by_track.items()):
        observations.sort(key=lambda item: item["frame_index"])
        if (
            len(observations) < config.min_track_hits
            or sum(bool(item["eligible"]) for item in observations) < config.min_eligible_hits
        ):
            continue
        eligible_tracks += 1
        for item in observations:
            if not item["eligible"]:
                continue
            per_frame[item["frame_index"]].append(
                {
                    "track_id": track_id,
                    "status": "observed",
                    "center_m": item["center"].round(5).tolist(),
                    "extent_m": item["extent"].round(5).tolist(),
                    "heading_rad": 0.0,
                    "confidence": 1.0,
                    "effective_hypotheses": None,
                    "source_cluster_index": item["source_cluster_index"],
                    "ground_truth": False,
                }
            )

        for right_position in range(1, len(observations)):
            left = observations[right_position - 1]
            right = observations[right_position]
            frame_span = int(right["frame_index"] - left["frame_index"])
            missing = frame_span - 1
            if missing <= 0 or missing > config.max_bridge_gap:
                continue
            if not left["eligible"] or not right["eligible"]:
                rejected_ineligible += missing
                continue
            state = _motion_state(observations, right_position - 1)
            cv_state = dict(state)
            cv_state["yaw_rate"] = 0.0
            stationary_state = dict(state)
            stationary_state["speed"] = 0.0
            stationary_state["yaw_rate"] = 0.0
            stationary_state["vz"] = 0.0
            endpoint_predictions = [
                _predict_5dof(state, frame_span),
                _predict_5dof(cv_state, frame_span),
                _predict_5dof(stationary_state, frame_span),
            ]
            prior_step = np.zeros(3, dtype=float)
            if right_position >= 2:
                prior = observations[right_position - 2]
                prior_dt = max(1, int(left["frame_index"] - prior["frame_index"]))
                prior_step = (left["center"] - prior["center"]) / prior_dt
            current_step = (right["center"] - left["center"]) / frame_span
            history_change = float(np.linalg.norm(current_step - prior_step))
            histories = [history_change, 0.8 * history_change, float(np.linalg.norm(current_step))]
            weights, residuals, effective, confidence = _normalised_dual_weights(
                endpoint_predictions, right["center"], histories, config
            )
            if confidence < config.min_proposal_confidence:
                rejected += missing
                continue
            endpoint_fused = np.sum(
                np.asarray(endpoint_predictions) * weights[:, None], axis=0
            )
            endpoint_error = right["center"] - endpoint_fused
            correction_length = float(np.linalg.norm(endpoint_error))
            if correction_length > config.observer_correction_limit_m:
                endpoint_error *= config.observer_correction_limit_m / correction_length
            for offset in range(1, frame_span):
                fraction = offset / frame_span
                hypotheses = np.asarray(
                    [
                        _predict_5dof(state, offset),
                        _predict_5dof(cv_state, offset),
                        _predict_5dof(stationary_state, offset),
                    ]
                )
                center = np.sum(hypotheses * weights[:, None], axis=0)
                center += fraction * endpoint_error
                extent = (1.0 - fraction) * left["extent"] + fraction * right["extent"]
                heading = float(state["heading"] + state["yaw_rate"] * offset)
                per_frame[left["frame_index"] + offset].append(
                    {
                        "track_id": track_id,
                        "status": "compensated",
                        "center_m": center.round(5).tolist(),
                        "extent_m": extent.round(5).tolist(),
                        "heading_rad": round(_wrap_angle(heading), 6),
                        "confidence": round(float(confidence), 6),
                        "effective_hypotheses": round(float(effective), 6),
                        "dual_likelihood_weights": [round(float(value), 6) for value in weights],
                        "endpoint_residuals_m": [round(value, 6) for value in residuals],
                        "source_cluster_index": None,
                        "ground_truth": False,
                    }
                )
                compensated += 1

    for items in per_frame:
        items.sort(key=lambda item: (int(item["track_id"]), item["status"]))
    summary = {
        "config": config.to_dict(),
        "sensor_only": True,
        "retrospective_verified_gaps_only": True,
        "ground_truth": False,
        "eligible_tracks": eligible_tracks,
        "observed_proposals": sum(
            item["status"] == "observed" for items in per_frame for item in items
        ),
        "compensated_proposals": compensated,
        "rejected_low_confidence_gaps": rejected,
        "rejected_ineligible_endpoint_gaps": rejected_ineligible,
    }
    return per_frame, summary
