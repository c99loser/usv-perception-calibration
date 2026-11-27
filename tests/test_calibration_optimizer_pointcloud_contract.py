import numpy as np

from tools.natural_scene_calibration.optimize_box_alignment import prepare_frames
from tools.natural_scene_calibration.optimize_pillar_track_alignment import (
    LidarObservation,
    lidar_tracks,
)


def _cluster(
    center,
    *,
    track_id: int,
    eligible: bool,
    weight: float = 1.0,
) -> dict:
    return {
        "center_m": [99.0, 99.0, 99.0],
        "calibration_center_m": list(center),
        "extent_m": [0.4, 0.4, 1.0],
        "calibration_quality": {
            "eligible": eligible,
            "inverse_variance_weight": weight,
        },
        "temporal_track": {"track_id": track_id},
    }


def test_box_optimizer_uses_only_eligible_calibration_centers() -> None:
    payload = {
        "frames": [
            {
                "clusters": [
                    _cluster((4.0, 0.0, 0.0), track_id=7, eligible=True, weight=3.5),
                    _cluster((8.0, 0.0, 0.0), track_id=8, eligible=False),
                ],
                "detections": [{"label": "pillar", "xyxy": [1, 2, 3, 4]}],
            }
        ]
    }

    frames = prepare_frames(payload)

    assert len(frames) == 1
    points, boxes, weights = frames[0]
    np.testing.assert_allclose(points, [[4.0, 0.0, 0.0]])
    np.testing.assert_allclose(weights, [3.5])
    assert boxes == [[1, 2, 3, 4]]


def test_track_optimizer_preserves_production_ids_and_rejects_ineligible() -> None:
    frames = []
    for frame_index in range(3):
        frames.append(
            {
                "pillar_candidate_indices": [0, 1],
                "clusters": [
                    _cluster(
                        (4.0 + frame_index, 0.0, 0.0),
                        track_id=41,
                        eligible=True,
                        weight=2.0,
                    ),
                    _cluster(
                        (7.0, 0.0, 0.0),
                        track_id=42,
                        eligible=False,
                    ),
                ],
            }
        )

    tracks = lidar_tracks({"frames": frames}, min_length=3)

    assert [track.track_id for track in tracks] == [41]
    assert [index for index, _ in tracks[0].entries] == [0, 1, 2]
    observations = [item for _, item in tracks[0].entries]
    assert all(isinstance(item, LidarObservation) for item in observations)
    np.testing.assert_allclose([item.point for item in observations], [[4, 0, 0], [5, 0, 0], [6, 0, 0]])
    np.testing.assert_allclose([item.weight for item in observations], [2, 2, 2])
