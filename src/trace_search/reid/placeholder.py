"""Untrained placeholder embedder.

Each crop is resized to 64 pixels high and 32 pixels wide with
``cv2.resize`` (default interpolation), flattened, and multiplied by a
Gaussian matrix drawn from ``np.random.default_rng(seed)``. Each output
row is L2-normalised. The constructor takes ``dim`` and ``seed`` only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from trace_search.reid.types import AppearanceEmbedder

# Spatial size after resize. OpenCV takes (width, height).
_HEIGHT = 64
_WIDTH = 32


class UntrainedPlaceholderEmbedder:
    """Untrained placeholder.

    Similarity scores from this embedder have no accuracy meaning and may
    not be reported as one (PRD §0).
    """

    placeholder: bool = True

    def __init__(self, dim: int = 256, seed: int = 0) -> None:
        if dim < 1:
            raise ValueError(f"dim must be a positive int, got {dim}")
        self.dim = dim
        self._seed = seed

    @property
    def name(self) -> str:
        """Short embedder label. It is read-only and is not a person name."""
        return "untrained-placeholder-v1"

    def embed(self, crops: Sequence[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        """Project uint8 crops to ``dim`` L2-normalised float32 rows."""
        if len(crops) == 0:
            return _empty(self.dim)
        flat = _flat_batch(crops)
        matrix = _gaussian(self._seed, int(flat.shape[1]), self.dim)
        return _unit_rows(flat @ matrix)


def _as_appearance_embedder(
    embedder: UntrainedPlaceholderEmbedder,
) -> AppearanceEmbedder:
    """Return ``embedder`` as an ``AppearanceEmbedder``.

    ``mypy src/ --strict`` checks this assignment. CI does not type-check tests.
    """
    checked: AppearanceEmbedder = embedder
    return checked


def _empty(dim: int) -> npt.NDArray[np.float32]:
    empty = np.zeros((0, dim), dtype=np.float32)
    return cast(npt.NDArray[np.float32], empty)


def _flat_batch(crops: Sequence[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float64]:
    rows = [_flat_crop(crop) for crop in crops]
    width = int(rows[0].shape[0])
    _require_same_width(rows, width)
    stacked = np.ascontiguousarray(np.stack(rows), dtype=np.float64)
    return cast(npt.NDArray[np.float64], stacked)


def _require_same_width(rows: Sequence[npt.NDArray[np.float64]], width: int) -> None:
    for row in rows:
        if int(row.shape[0]) != width:
            raise ValueError("crops in one call must flatten to the same length")


def _flat_crop(crop: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    _require_crop(crop)
    contiguous = np.ascontiguousarray(crop)
    resized = cv2.resize(contiguous, (_WIDTH, _HEIGHT))
    image = np.asarray(resized, dtype=np.float64)
    return cast(npt.NDArray[np.float64], image.reshape(-1))


def _require_crop(crop: npt.NDArray[np.uint8]) -> None:
    if not isinstance(crop, np.ndarray):
        raise ValueError(f"crop must be a numpy array, got {type(crop).__name__}")
    if crop.dtype != np.uint8:
        raise ValueError(f"crop must be uint8, got {crop.dtype}")
    if crop.ndim not in (2, 3):
        raise ValueError(f"crop must be HxW or HxWxC, got ndim {crop.ndim}")
    if crop.size == 0:
        raise ValueError("crop must have a non-zero height and width")


def _gaussian(seed: int, in_dim: int, out_dim: int) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    drawn = rng.standard_normal((in_dim, out_dim))
    matrix = np.ascontiguousarray(drawn, dtype=np.float64)
    return cast(npt.NDArray[np.float64], matrix)


def _unit_rows(projected: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    if _any_zero(norms):
        raise ValueError("projected crop is zero and cannot be normalised")
    unit = np.asarray(projected / norms, dtype=np.float32)
    return cast(npt.NDArray[np.float32], np.ascontiguousarray(unit))


def _any_zero(norms: npt.NDArray[np.float64]) -> bool:
    flat = np.asarray(norms, dtype=np.float64).reshape(-1)
    for value in flat:
        if float(value) == 0.0:
            return True
    return False
