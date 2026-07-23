"""Safe LiDAR--camera correspondence gate for natural-scene self-calibration.

This module consumes *measurements*, not simulator truth, TF, odometry, a
calibration fixture, or a nominal transform.  It intentionally stops at the
``candidate`` state: optimization and acceptance are separate steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from competition_user.natural_calibration import Track


@dataclass(frozen=True)
class ImageNaturalObservation:
    """A camera feature/object supplied by a separately validated extractor.

    ``source_validated`` is deliberately explicit.  Raw image pixels, a
    hand-tuned colour threshold, or an unverified detector may be logged for
    development, but cannot become a calibration correspondence.
    """
    camera: str
    frame_stamp_ns: int
    feature_id: str
    pixel_u: float
    pixel_v: float
    source: str
    source_validated: bool
    static_confidence: float


@dataclass(frozen=True)
class AssociationCandidate:
    lidar: str
    camera: str
    track_id: int
    image_feature_id: str
    time_delta_ms: float
    provenance: str


@dataclass(frozen=True)
class AssociationReport:
    candidate_ready: bool
    reason: str
    distinct_tracks: int
    distinct_cameras: int
    candidates: tuple[AssociationCandidate, ...]


def make_candidates(
    *, lidar: str, tracks: dict[int, Track], image_observations: Iterable[ImageNaturalObservation],
    max_time_delta_ms: float = 50.0, min_static_confidence: float = 0.80,
) -> AssociationReport:
    """Make auditable, temporal candidates from persistent natural features.

    Geometry is not used to silently select a match: doing so would bake an
    unverified nominal external parameter into the evidence.  The caller must
    provide feature identities from a validated image extractor or later add a
    robust multi-hypothesis optimizer.  This gate only says whether it is safe
    to advance to that *candidate optimization* stage.
    """
    stable = {track_id: track for track_id, track in tracks.items() if track.stable}
    usable_images = [
        image for image in image_observations
        if image.source_validated and image.static_confidence >= min_static_confidence
    ]
    if len(stable) < 3:
        return AssociationReport(False, "fewer_than_three_persistent_lidar_tracks", len(stable), 0, ())
    if not usable_images:
        return AssociationReport(False, "no_validated_static_image_observations", len(stable), 0, ())

    candidates: list[AssociationCandidate] = []
    for track_id, track in stable.items():
        for image in usable_images:
            delta_ms = abs(track.latest.frame_stamp_ns - image.frame_stamp_ns) / 1e6
            if delta_ms <= max_time_delta_ms:
                candidates.append(AssociationCandidate(
                    lidar=lidar, camera=image.camera, track_id=track_id,
                    image_feature_id=image.feature_id, time_delta_ms=delta_ms,
                    provenance=image.source,
                ))
    cameras = {candidate.camera for candidate in candidates}
    if len(candidates) < 3:
        return AssociationReport(False, "insufficient_time_synchronised_cross_sensor_candidates", len(stable), len(cameras), tuple(candidates))
    return AssociationReport(
        True, "ready_for_prior_bounded_robust_candidate_optimization",
        len({candidate.track_id for candidate in candidates}), len(cameras), tuple(candidates),
    )
