import numpy as np

from competition_user.association_gate import ImageNaturalObservation, make_candidates
from competition_user.natural_calibration import NaturalTrackManager, observations_from_cloud


def _tracks():
    rng = np.random.default_rng(3)
    cloud = np.vstack((
        rng.normal((7.0, -3.0, 1.0), 0.08, size=(12, 3)),
        rng.normal((12.0, 1.0, 1.0), 0.08, size=(12, 3)),
        rng.normal((18.0, 5.0, 1.0), 0.08, size=(12, 3)),
    ))
    manager = NaturalTrackManager()
    for stamp in (0, 100_000_000, 200_000_000):
        manager.update(observations_from_cloud(cloud, stamp))
    return manager.tracks


def test_only_validated_synchronised_image_observations_make_candidates():
    image = ImageNaturalObservation("front", 200_000_000, "edge-1", 640, 360,
                                   "validated_static_edge_geometry", True, 0.9)
    report = make_candidates(lidar="front", tracks=_tracks(), image_observations=[image])
    assert report.candidate_ready
    assert report.distinct_tracks == 3
    assert len(report.candidates) == 3


def test_unvalidated_image_heuristic_cannot_advance_the_gate():
    image = ImageNaturalObservation("front", 200_000_000, "colour-1", 640, 360,
                                   "hand_tuned_colour", False, 0.99)
    report = make_candidates(lidar="front", tracks=_tracks(), image_observations=[image])
    assert not report.candidate_ready
    assert report.reason == "no_validated_static_image_observations"
