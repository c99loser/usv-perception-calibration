from competition_user.pointcloud_tracking import PointCloudTrackConfig, PointCloudTrackFilter


def cluster(center, extent=(1.0, 1.0, 1.0)):
    return {"center_m": list(center), "extent_m": list(extent), "range_m": 1.0, "points": 10}


def test_track_is_emitted_only_after_adjacent_confirmation() -> None:
    tracker = PointCloudTrackFilter(PointCloudTrackConfig(min_confirmed_hits=3))
    assert tracker.update(0, [cluster((4.0, 0.0, 0.0))]) == []
    assert tracker.update(1, [cluster((4.1, 0.0, 0.0))]) == []
    output = tracker.update(2, [cluster((3.9, 0.0, 0.0))])
    assert len(output) == 1
    assert output[0]["temporal_track"]["confirmed_hits"] == 3


def test_alpha_beta_center_and_reference_extent_freeze() -> None:
    tracker = PointCloudTrackFilter(
        PointCloudTrackConfig(min_confirmed_hits=1, ema_alpha=0.25, reference_extent_hits=2)
    )
    first = tracker.update(0, [cluster((4.0, 0.0, 0.0), (1.0, 1.0, 1.0))])[0]
    second = tracker.update(1, [cluster((4.4, 0.0, 0.0), (1.4, 1.0, 1.0))])[0]
    third = tracker.update(2, [cluster((4.2, 0.0, 0.0), (1.8, 1.0, 1.0))])[0]
    assert first["center_m"][0] == 4.0
    assert second["center_m"][0] == 4.34
    assert second["extent_m"][0] == 1.2
    assert third["extent_m"][0] == 1.2
    assert third["raw_extent_m"][0] == 1.8
    assert abs(third["calibration_center_m"][0] - 4.2) < 0.04
    assert third["association_center_m"][0] == third["center_m"][0]
    assert third["calibration_center_m"][0] != third["raw_center_m"][0]
    assert third["temporal_track"]["extent_state"] == "frozen_median"


def test_causal_alpha_beta_center_tracks_constant_motion_without_ema_lag() -> None:
    tracker = PointCloudTrackFilter(PointCloudTrackConfig(min_confirmed_hits=1))
    output = None
    for frame_index in range(4):
        output = tracker.update(
            frame_index, [cluster((float(frame_index), 0.0, 0.0))]
        )[0]
    assert output is not None
    assert abs(output["calibration_center_m"][0] - 3.0) < 0.2
    assert output["temporal_track"]["calibration_center_source"] == (
        "causal_current_timestamp_alpha_beta"
    )


def test_default_confirmation_emits_only_frozen_reference_extent() -> None:
    tracker = PointCloudTrackFilter(PointCloudTrackConfig())
    tracker.update(0, [cluster((4.0, 0.0, 0.0), (0.8, 1.0, 1.0))])
    tracker.update(1, [cluster((4.0, 0.0, 0.0), (1.4, 1.0, 1.0))])
    third = tracker.update(2, [cluster((4.0, 0.0, 0.0), (1.0, 1.0, 1.0))])[0]
    fourth = tracker.update(3, [cluster((4.0, 0.0, 0.0), (1.7, 1.0, 1.0))])[0]
    assert third["extent_m"][0] == 1.0
    assert fourth["extent_m"][0] == 1.0
    assert fourth["raw_extent_m"][0] == 1.7


def test_track_is_not_reconnected_after_a_missing_frame() -> None:
    tracker = PointCloudTrackFilter(
        PointCloudTrackConfig(min_confirmed_hits=1, max_frame_gap=1)
    )
    first = tracker.update(0, [cluster((4.0, 0.0, 0.0))])[0]
    tracker.update(1, [])
    second = tracker.update(2, [cluster((4.0, 0.0, 0.0))])[0]
    assert first["temporal_track"]["track_id"] != second["temporal_track"]["track_id"]


def test_track_bridges_one_missing_frame_when_configured() -> None:
    tracker = PointCloudTrackFilter(
        PointCloudTrackConfig(min_confirmed_hits=1, max_frame_gap=2)
    )
    first = tracker.update(0, [cluster((4.0, 0.0, 0.0))])[0]
    tracker.update(1, [])
    second = tracker.update(2, [cluster((4.1, 0.0, 0.0))])[0]
    assert first["temporal_track"]["track_id"] == second["temporal_track"]["track_id"]
