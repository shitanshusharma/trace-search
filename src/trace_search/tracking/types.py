"""Typed tracking contract.

Per-camera trackers share these records. This module does not load a model
or weights. A track id is unique inside one camera only.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from trace_search.attributes.attributes import Box
from trace_search.detection.types import DetClass, Detection


def _require_finite(label: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}")


def _require_non_negative_int(label: str, value: object) -> None:
    """Reject a bool or any other value that is not an int >= 0."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an int >= 0, got {value!r}")
    if value < 0:
        raise ValueError(f"{label} must be an int >= 0, got {value!r}")


@dataclass(frozen=True)
class TrackObservation:
    """One tracked detection in one camera frame.

    ``track_id`` is unique per camera only. ``box`` is ``(x1, y1, x2, y2)``
    in pixels, the same shape as ``Detection.box``.
    """

    camera_id: str
    track_id: int
    det_class: DetClass
    frame_number: int
    timestamp_seconds: float
    box: Box
    score: float

    def __post_init__(self) -> None:
        """Reject a track observation that fails a guard.

        Checks: track_id and frame_number as non-negative ints (not bool),
        box ordering (x2 > x1 and y2 > y1), score in [0, 1], non-finite box
        coordinates, timestamp, or score, and camera_id as a non-empty str.
        """
        if not isinstance(self.camera_id, str) or self.camera_id == "":
            raise ValueError(f"camera_id must be a non-empty string, got {self.camera_id!r}")
        _require_non_negative_int("track_id", self.track_id)
        _require_non_negative_int("frame_number", self.frame_number)
        x1, y1, x2, y2 = self.box
        _require_finite("x1", x1)
        _require_finite("y1", y1)
        _require_finite("x2", x2)
        _require_finite("y2", y2)
        _require_finite("timestamp_seconds", self.timestamp_seconds)
        _require_finite("score", self.score)
        if x2 <= x1:
            raise ValueError(f"box x2 must be greater than x1, got x1={x1} x2={x2}")
        if y2 <= y1:
            raise ValueError(f"box y2 must be greater than y1, got y1={y1} y2={y2}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")


class Tracker(Protocol):
    """Structural type for every per-camera tracker."""

    def update(self, detections: Sequence[Detection]) -> list[TrackObservation]:
        """Return this camera's tracks for one frame of detections."""

    def reset(self) -> None:
        """Drop this camera's tracks."""
