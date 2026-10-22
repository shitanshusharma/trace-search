"""Contract tests for AppearanceEmbedding and AppearanceEmbedder."""

import math
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, fields
from typing import get_type_hints

import numpy as np
import numpy.typing as npt
import pytest

from trace_search.reid import AppearanceEmbedder, AppearanceEmbedding
from trace_search.reid import types as reid_types

_EXACT_FIELDS = {
    "camera_id",
    "track_id",
    "frame_number",
    "vector",
    "embedder_name",
    "placeholder",
}


def _seeded_unit(seed: int, dim: int = 4) -> tuple[float, ...]:
    """Build one L2-normalised vector from a seeded numpy generator."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(dim)
    unit = raw / float(np.linalg.norm(raw))
    return tuple(float(value) for value in unit)


def _embedding(
    *,
    camera_id: str = "cam-1",
    track_id: int = 3,
    frame_number: int = 7,
    vector: tuple[float, ...] | None = None,
    embedder_name: str = "test-embedder",
    placeholder: bool = True,
) -> AppearanceEmbedding:
    if vector is None:
        vector = _seeded_unit(1)
    return AppearanceEmbedding(
        camera_id=camera_id,
        track_id=track_id,
        frame_number=frame_number,
        vector=vector,
        embedder_name=embedder_name,
        placeholder=placeholder,
    )


def test_embedding_fields_exact() -> None:
    """A planted extra field changes this set and fails the test."""
    assert {field.name for field in fields(AppearanceEmbedding)} == _EXACT_FIELDS


def test_embedding_field_types() -> None:
    hints = get_type_hints(AppearanceEmbedding)
    assert hints["camera_id"] is str
    assert hints["track_id"] is int
    assert hints["frame_number"] is int
    assert hints["vector"] == tuple[float, ...]
    assert hints["embedder_name"] is str
    assert hints["placeholder"] is bool


def test_module_docstring_states_the_vector_has_no_accuracy_meaning() -> None:
    doc = reid_types.__doc__ or ""
    assert "anonymous whole-body appearance descriptor for linking tracks" in doc
    assert "placeholder vectors have no accuracy meaning" in doc


def test_valid_embedding_keeps_each_field() -> None:
    vector = _seeded_unit(4)
    found = _embedding(vector=vector, placeholder=False, embedder_name="test-embedder")
    assert found.camera_id == "cam-1"
    assert found.track_id == 3
    assert found.frame_number == 7
    assert found.vector == vector
    assert found.embedder_name == "test-embedder"
    assert found.placeholder is False


def test_same_track_id_is_allowed_on_two_cameras() -> None:
    """A track id is unique per camera only."""
    vector = _seeded_unit(5)
    left = _embedding(camera_id="cam-a", track_id=4, vector=vector)
    right = _embedding(camera_id="cam-b", track_id=4, vector=vector)
    assert left.track_id == right.track_id == 4
    assert left.camera_id != right.camera_id


def test_embedding_is_frozen() -> None:
    found = _embedding()
    with pytest.raises(FrozenInstanceError):
        setattr(found, "track_id", 9)


def test_vector_is_stored_as_its_own_tuple() -> None:
    source = _seeded_unit(6)
    found = _embedding(vector=source)
    assert isinstance(found.vector, tuple)
    assert found.vector == source
    assert found.vector is not source
    norm = math.sqrt(sum(value * value for value in found.vector))
    assert abs(norm - 1.0) <= 1e-5


def test_norm_within_tolerance_is_accepted() -> None:
    """A norm inside 1e-5 of 1 is kept as given."""
    raw = np.array([1.0 + 1e-6], dtype=np.float64)
    vector = (float(raw[0]),)
    norm = math.sqrt(vector[0] * vector[0])
    assert abs(norm - 1.0) <= 1e-5
    found = _embedding(vector=vector)
    assert found.vector == vector


def test_non_finite_vector_raises() -> None:
    values = np.array(_seeded_unit(3), dtype=np.float64)
    values[0] = np.nan
    vector = tuple(float(value) for value in values)
    with pytest.raises(ValueError, match="finite"):
        _embedding(vector=vector)


def test_zero_vector_raises() -> None:
    vector = tuple(float(value) for value in np.zeros(4, dtype=np.float64))
    with pytest.raises(ValueError, match="non-zero"):
        _embedding(vector=vector)


def test_vector_not_l2_normalised_raises() -> None:
    unit = np.array(_seeded_unit(2), dtype=np.float64)
    raw = unit * np.float64(2.0)
    assert abs(float(np.linalg.norm(raw)) - 1.0) > 1e-5
    vector = tuple(float(value) for value in raw)
    with pytest.raises(ValueError, match="L2-normalised"):
        _embedding(vector=vector)


def test_empty_camera_id_raises() -> None:
    with pytest.raises(ValueError, match="camera_id"):
        _embedding(camera_id="")


def test_embedder_returns_one_l2_normalised_row_per_crop() -> None:
    """A conforming embedder returns shape (len(crops), dim), rows L2-normalised."""

    class _StubEmbedder:
        dim = 4
        placeholder = True

        @property
        def name(self) -> str:
            return "test-embedder"

        def embed(self, crops: Sequence[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
            count = len(crops)
            if count == 0:
                return np.zeros((0, self.dim), dtype=np.float32)
            rng = np.random.default_rng(0)
            rows = rng.standard_normal((count, self.dim)).astype(np.float32)
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            unit = np.asarray(rows / norms, dtype=np.float32)
            return unit

    embedder: AppearanceEmbedder = _StubEmbedder()
    assert embedder.name == "test-embedder"
    assert embedder.dim == 4
    assert embedder.placeholder is True

    rng = np.random.default_rng(1)
    crops = [rng.integers(0, 256, size=(8, 6, 3), dtype=np.uint8) for _ in range(3)]
    found = embedder.embed(crops)
    assert found.shape == (len(crops), embedder.dim)
    assert found.dtype == np.float32
    row_norms = np.linalg.norm(found, axis=1)
    assert np.all(np.abs(row_norms - np.float32(1.0)) <= np.float32(1e-5))

    empty = embedder.embed([])
    assert empty.shape == (0, embedder.dim)
    assert empty.dtype == np.float32


def test_embedder_protocol_types() -> None:
    hints = get_type_hints(AppearanceEmbedder)
    assert hints["dim"] is int
    assert hints["placeholder"] is bool
    embed_hints = get_type_hints(AppearanceEmbedder.embed)
    assert embed_hints["crops"] == Sequence[npt.NDArray[np.uint8]]
    assert embed_hints["return"] == npt.NDArray[np.float32]
    name_attr = AppearanceEmbedder.__dict__["name"]
    assert isinstance(name_attr, property)
    getter = name_attr.fget
    assert getter is not None
    assert get_type_hints(getter)["return"] is str
