"""The vector is an anonymous whole-body appearance descriptor for linking tracks.

It carries no identity, face, gender, age or attribute meaning, and
placeholder vectors have no accuracy meaning (PRD §0).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

# A vector is L2-normalised when its norm is within this distance of 1.
_NORM_TOLERANCE = 1e-5


def _require_unit_vector(vector: tuple[float, ...]) -> tuple[float, ...]:
    """Return a copy of ``vector`` or raise ``ValueError``.

    The copy is finite, non-zero, and L2-normalised within 1e-5.
    """
    values = tuple(float(item) for item in vector)
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"vector values must be finite, got {value!r}")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ValueError("vector must be non-zero")
    if abs(norm - 1.0) > _NORM_TOLERANCE:
        raise ValueError(f"vector must be L2-normalised within {_NORM_TOLERANCE}, got norm {norm}")
    return values


@dataclass(frozen=True)
class AppearanceEmbedding:
    """One anonymous appearance vector for one track in one camera frame.

    ``track_id`` is unique per camera only, as on ``TrackObservation``.
    ``vector`` is stored as a tuple so the frozen record cannot be edited
    in place.
    """

    camera_id: str
    track_id: int
    frame_number: int
    vector: tuple[float, ...]
    embedder_name: str
    placeholder: bool

    def __post_init__(self) -> None:
        """Raise ``ValueError`` for an empty camera or a vector that is not a unit vector.

        A unit vector is finite, non-zero, and L2-normalised within 1e-5.
        """
        if self.camera_id == "":
            raise ValueError(f"camera_id must be a non-empty string, got {self.camera_id!r}")
        object.__setattr__(self, "vector", _require_unit_vector(self.vector))


class AppearanceEmbedder(Protocol):
    """Structural type for every appearance embedder.

    ``name`` is this embedder's short name. It is read-only and is not a
    person name. ``embed`` takes uint8 crops and returns float32 of shape
    ``(len(crops), dim)`` with each row L2-normalised.
    """

    dim: int
    placeholder: bool

    @property
    def name(self) -> str:
        raise NotImplementedError

    def embed(self, crops: Sequence[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        raise NotImplementedError
