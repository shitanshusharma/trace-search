"""Typed detection contract.

Detectors share these records. This module does not load a model or weights.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from trace_search.attributes.attributes import Box
from trace_search.ingest.video_source import Frame


class DetClass(Enum):
    """Classes a detector may emit."""

    PERSON = "person"
    BACKPACK = "backpack"
    HANDBAG = "handbag"


def _require_finite(label: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}")


@dataclass(frozen=True)
class Detection:
    """One detection in one camera frame.

    ``box`` is ``(x1, y1, x2, y2)`` in pixels, the same shape as
    ``trace_search.attributes.attributes.Box``.
    """

    camera_id: str
    frame_number: int
    timestamp_seconds: float
    det_class: DetClass
    box: Box
    score: float

    def __post_init__(self) -> None:
        """Reject a bad box, a score outside [0, 1], or any non-finite value."""
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


class Detector(Protocol):
    """Structural type for every detector."""

    def detect(self, frame: Frame) -> list[Detection]:
        """Return the detections in one frame."""
        ...
