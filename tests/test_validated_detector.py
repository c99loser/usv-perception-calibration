from competition_user.validated_detector import DetectorBox, as_validated_observations


def test_only_validated_static_classes_and_valid_boxes_pass():
    observations = as_validated_observations(
        camera="front", frame_stamp_ns=42, model_version="three_bag_v2",
        detections=(
            DetectorBox("buoy", 0.8, 10, 20, 30, 40),
            DetectorBox("boat", 0.99, 10, 20, 30, 40),
            DetectorBox("pillar", 0.49, 10, 20, 30, 40),
            DetectorBox("block", 0.9, 30, 20, 10, 40),
        ),
    )
    assert len(observations) == 1
    assert observations[0].pixel_u == 20
    assert observations[0].pixel_v == 30
    assert observations[0].source_validated


def test_rejects_invalid_runtime_metadata():
    try:
        as_validated_observations(camera="", frame_stamp_ns=1, model_version="v", detections=())
    except ValueError:
        pass
    else:
        raise AssertionError("missing camera must be rejected")
