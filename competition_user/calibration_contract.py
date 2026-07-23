"""Runtime contract for self-calibration outputs.

No truth, TF, odometry, fixture coordinates or nominal transform is a
publishable calibration result.  Downstream pixel fusion must check this
state, so an unobservable scene safely degrades to independent sensors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class ExtrinsicState(str, Enum):
    NOMINAL_UNVERIFIED = "nominal_unverified"
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ExtrinsicStatus:
    state: ExtrinsicState = ExtrinsicState.NOMINAL_UNVERIFIED
    version: str = "none"
    source: str = "no_natural_scene_solution"
    quality: Mapping[str, float] = field(default_factory=dict)

    @property
    def allows_pixel_fusion(self) -> bool:
        return self.state is ExtrinsicState.ACCEPTED
