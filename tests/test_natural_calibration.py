import numpy as np

from competition_user.natural_calibration import NaturalTrackManager, observations_from_cloud


def _cloud():
    rng = np.random.default_rng(7)
    return np.vstack((
        rng.normal((8.0, -3.0, 1.0), 0.08, size=(20, 3)),
        rng.normal((12.0, 1.0, 1.5), 0.08, size=(20, 3)),
        rng.normal((18.0, 5.0, 2.0), 0.08, size=(20, 3)),
        np.array([[np.nan, 0, 0], [0, 0, 0]]),
    ))


def test_persistent_diverse_clusters_are_observable():
    tracker = NaturalTrackManager()
    for frame in range(3):
        tracker.update(observations_from_cloud(_cloud(), frame * 100_000_000))
    report = tracker.observability()
    assert report.observable
    assert report.stable_track_count == 3


def test_one_cluster_is_not_observable():
    tracker = NaturalTrackManager()
    point = np.repeat([[10.0, 0.0, 1.0]], 10, axis=0)
    for frame in range(3):
        tracker.update(observations_from_cloud(point, frame))
    assert not tracker.observability().observable
