import pytest

from tools.natural_scene_calibration.compare_pointcloud_pipelines import metrics, tracks


def item(track_id: int, x: float) -> dict:
    return {
        "center_m": [x, 0.0, 0.0],
        "extent_m": [1.0, 1.0, 1.0],
        "temporal_track": {"track_id": track_id},
    }


def test_tracks_respects_embedded_track_identity() -> None:
    frames = [
        {"clusters": [item(10, 0.0), item(20, 0.1)]},
        {"clusters": [item(20, 0.0), item(10, 0.1)]},
    ]
    result = tracks(frames, min_length=2)
    assert len(result) == 2
    assert all(
        len({cluster["temporal_track"]["track_id"] for cluster in track}) == 1
        for track in result
    )


def test_duplicate_embedded_track_identity_is_rejected() -> None:
    frames = [{"clusters": [item(10, 0.0), item(10, 0.1)]}]
    with pytest.raises(ValueError, match="duplicate temporal track_id"):
        tracks(frames, min_length=1)


def test_metrics_counts_persistent_evidence_without_rewarding_fragments() -> None:
    frames = []
    for frame_index in range(8):
        frames.append(
            {
                "clusters": [
                    {
                        **item(10, float(frame_index)),
                        "range_m": 4.0,
                    }
                ]
            }
        )
    result = metrics({"frames": frames})
    assert result["persistent_tracks_ge_6"] == 1
    assert result["persistent_track_observations_ge_6"] == 8
    assert result["persistent_track_observation_fraction"] == 1.0
    assert result["median_persistent_track_length"] == 8.0
