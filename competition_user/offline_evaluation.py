"""Offline-only metrics for a single-LiDAR run.

This module never participates in the ROS node.  Truth coordinates supplied
to it must already be expressed in the evaluated LiDAR frame; converting
recorded world truth with recorded TF is an offline audit operation only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from competition_user.lidar_baseline import Detection, Track


@dataclass(frozen=True)
class TruthObject:
    center: np.ndarray
    label: str


def greedy_match(detections: Iterable[Detection], truth: Iterable[TruthObject], gate_m: float = 3.0):
    detections, truth = list(detections), list(truth)
    candidates = []
    for di, detection in enumerate(detections):
        for ti, target in enumerate(truth):
            distance = float(np.linalg.norm(detection.center[:2] - target.center[:2]))
            if distance <= gate_m:
                candidates.append((distance, di, ti))
    matched_d, matched_t, pairs = set(), set(), []
    for distance, di, ti in sorted(candidates):
        if di not in matched_d and ti not in matched_t:
            matched_d.add(di); matched_t.add(ti); pairs.append((distance, di, ti))
    return pairs, len(detections) - len(matched_d), len(truth) - len(matched_t)


class SingleLidarEvaluator:
    def __init__(self, distance_bins: tuple[float, ...] = (0.0, 10.0, 20.0, 40.0, float("inf"))) -> None:
        self.distance_bins = distance_bins
        self.rows: list[dict] = []
        self._previous_truth_track: dict[int, int] = {}
        self.id_switches = 0
        self.track_lifetimes: dict[int, int] = {}

    def update(self, detections: Iterable[Detection], tracks: Iterable[Track], truth: Iterable[TruthObject],
               bev: np.ndarray, truth_bev: np.ndarray | None, timings_ms: dict[str, float]) -> None:
        detections, tracks, truth = list(detections), list(tracks), list(truth)
        pairs, fp, fn = greedy_match(detections, truth)
        row = {"false_positive": fp, "false_negative": fn, "matches": len(pairs), "timings_ms": timings_ms}
        row["center_errors_m"] = [distance for distance, _, _ in pairs]
        row["class_matches"] = sum(detections[di].label == truth[ti].label for _, di, ti in pairs)
        row["match_details"] = [{"distance_m": distance, "truth_label": truth[ti].label,
                                 "prediction_label": detections[di].label,
                                 "truth_range_m": float(np.linalg.norm(truth[ti].center[:2]))}
                                for distance, di, ti in pairs]
        row["false_negative_details"] = [{"label": item.label,
                                           "range_m": float(np.linalg.norm(item.center[:2]))}
                                          for index, item in enumerate(truth)
                                          if index not in {ti for _, _, ti in pairs}]
        row["false_positive_details"] = [{"label": item.label,
                                           "range_m": float(np.linalg.norm(item.center[:2]))}
                                          for index, item in enumerate(detections)
                                          if index not in {di for _, di, _ in pairs}]
        row["bev_iou"] = None
        if truth_bev is not None:
            predicted = bev > 0; expected = truth_bev > 0
            union = np.logical_or(predicted, expected).sum()
            row["bev_iou"] = float(np.logical_and(predicted, expected).sum() / union) if union else 1.0
        # Map truth to nearest active track only for temporal ID stability.
        for truth_index, target in enumerate(truth):
            if not tracks:
                continue
            nearest = min(tracks, key=lambda track: float(np.linalg.norm(track.center[:2] - target.center[:2])))
            if np.linalg.norm(nearest.center[:2] - target.center[:2]) <= 3.0:
                old = self._previous_truth_track.get(truth_index)
                if old is not None and old != nearest.track_id:
                    self.id_switches += 1
                self._previous_truth_track[truth_index] = nearest.track_id
                self.track_lifetimes[nearest.track_id] = self.track_lifetimes.get(nearest.track_id, 0) + 1
        self.rows.append(row)

    def summary(self) -> dict:
        matches = sum(row["matches"] for row in self.rows)
        fp, fn = sum(row["false_positive"] for row in self.rows), sum(row["false_negative"] for row in self.rows)
        errors = [error for row in self.rows for error in row["center_errors_m"]]
        timing_keys = sorted({key for row in self.rows for key in row["timings_ms"]})
        slices = {}
        for label in sorted({item["truth_label"] for row in self.rows for item in row["match_details"]} |
                            {item["label"] for row in self.rows for item in row["false_negative_details"]}):
            matched = sum(item["truth_label"] == label for row in self.rows for item in row["match_details"])
            missed = sum(item["label"] == label for row in self.rows for item in row["false_negative_details"])
            slices[f"class:{label}"] = {"matches": matched, "false_negative": missed,
                                        "recall": matched / max(1, matched + missed)}
        for low, high in zip(self.distance_bins, self.distance_bins[1:]):
            matched = sum(low <= item["truth_range_m"] < high for row in self.rows for item in row["match_details"])
            missed = sum(low <= item["range_m"] < high for row in self.rows for item in row["false_negative_details"])
            slices[f"distance_m:{low:g}-{high:g}"] = {"matches": matched, "false_negative": missed,
                                                       "recall": matched / max(1, matched + missed)}
        return {
            "offline_evaluation_only": True,
            "frames": len(self.rows), "matches": matches, "false_positive": fp, "false_negative": fn,
            "precision": matches / max(1, matches + fp), "recall": matches / max(1, matches + fn),
            "median_center_error_m": float(np.median(errors)) if errors else None,
            "mean_bev_iou": float(np.mean([row["bev_iou"] for row in self.rows if row["bev_iou"] is not None])) if any(row["bev_iou"] is not None for row in self.rows) else None,
            "id_switches": self.id_switches,
            "track_lifetime_frames": {str(key): value for key, value in self.track_lifetimes.items()},
            "timings_ms": {key: {"p50": float(np.percentile([row["timings_ms"][key] for row in self.rows], 50)), "p95": float(np.percentile([row["timings_ms"][key] for row in self.rows], 95))} for key in timing_keys},
            "slices": slices,
        }
