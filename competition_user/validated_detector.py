"""Convert frozen, validated raw-fisheye detections into candidate observations.

This module deliberately contains no model loading or geometry.  The caller
supplies detections from a frozen model artifact and a camera timestamp; only
the validated static-candidate classes may pass on to the association gate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from competition_user.association_gate import ImageNaturalObservation


VALIDATED_STATIC_CANDIDATE_CLASSES = frozenset(("buoy", "pillar", "block"))


@dataclass(frozen=True)
class DetectorBox:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def as_validated_observations(
    *, camera: str, frame_stamp_ns: int, detections: Iterable[DetectorBox], model_version: str,
    min_confidence: float = 0.50,
) -> tuple[ImageNaturalObservation, ...]:
    """Return only validated, static-candidate detections for candidate matching.

    ``boat`` is intentionally excluded: a well-detected dynamic object is not
    evidence for a static LiDAR--camera calibration constraint.  This function
    does not assert an external calibration and every result remains subject to
    the LiDAR stability and time-synchronisation gates.
    """
    if not camera or not model_version:
        raise ValueError("camera and frozen model_version are required")
    if frame_stamp_ns <= 0 or not 0.0 <= min_confidence <= 1.0:
        raise ValueError("invalid timestamp or confidence threshold")
    result: list[ImageNaturalObservation] = []
    for index, box in enumerate(detections):
        values = (box.confidence, box.x1, box.y1, box.x2, box.y2)
        if box.label not in VALIDATED_STATIC_CANDIDATE_CLASSES or not all(math.isfinite(value) for value in values):
            continue
        if box.confidence < min_confidence or box.x2 <= box.x1 or box.y2 <= box.y1:
            continue
        result.append(ImageNaturalObservation(
            camera=camera, frame_stamp_ns=frame_stamp_ns,
            feature_id=f"{camera}:{frame_stamp_ns}:{index}:{box.label}",
            pixel_u=(box.x1 + box.x2) / 2, pixel_v=(box.y1 + box.y2) / 2,
            source=f"validated_raw_fisheye_detector:{model_version}:{box.label}",
            source_validated=True, static_confidence=box.confidence,
        ))
    return tuple(result)
