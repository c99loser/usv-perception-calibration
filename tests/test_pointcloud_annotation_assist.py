from competition_user.pointcloud_annotation_assist import (
    PointCloudAnnotationAssistConfig,
    build_annotation_assists,
)


def tracked(track_id: int, x: float, *, source: int = 0) -> dict:
    return {
        "center_m": [x, 1.0, 0.5],
        "extent_m": [1.0, 0.6, 0.8],
        "source_cluster_index": source,
        "temporal_track": {"track_id": track_id},
        "calibration_quality": {"eligible": True},
    }


def test_short_verified_gap_gets_sensor_only_compensation() -> None:
    frames = [
        {"clusters": [tracked(7, 0.0)]},
        {"clusters": [tracked(7, 0.2)]},
        {"clusters": []},
        {"clusters": [tracked(7, 0.6)]},
        {"clusters": [tracked(7, 0.8)]},
    ]
    assists, summary = build_annotation_assists(frames)
    assert len(assists[2]) == 1
    proposal = assists[2][0]
    assert proposal["status"] == "compensated"
    assert proposal["track_id"] == 7
    assert abs(proposal["center_m"][0] - 0.4) < 0.1
    assert 1.0 <= proposal["effective_hypotheses"] <= 3.0
    assert proposal["ground_truth"] is False
    assert summary["compensated_proposals"] == 1
    assert summary["sensor_only"] is True


def test_assist_does_not_extrapolate_past_last_observation() -> None:
    frames = [
        {"clusters": [tracked(2, 0.0)]},
        {"clusters": [tracked(2, 0.1)]},
        {"clusters": [tracked(2, 0.2)]},
        {"clusters": []},
        {"clusters": []},
    ]
    assists, summary = build_annotation_assists(frames)
    assert assists[3] == []
    assert assists[4] == []
    assert summary["compensated_proposals"] == 0


def test_low_likelihood_identity_jump_is_not_compensated() -> None:
    frames = [
        {"clusters": [tracked(3, 0.0)]},
        {"clusters": [tracked(3, 0.1)]},
        {"clusters": []},
        {"clusters": [tracked(3, 6.0)]},
    ]
    config = PointCloudAnnotationAssistConfig(min_proposal_confidence=0.2)
    assists, summary = build_annotation_assists(frames, config)
    assert assists[2] == []
    assert summary["rejected_low_confidence_gaps"] == 1
