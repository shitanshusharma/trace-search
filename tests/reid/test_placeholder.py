"""Tests for the untrained placeholder embedder."""

import inspect
import socket
from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from trace_search.reid import AppearanceEmbedder
from trace_search.reid.placeholder import (
    UntrainedPlaceholderEmbedder,
    _as_appearance_embedder,
)

_SPEC_DOC = (
    "Untrained placeholder. Similarity scores from this embedder have no "
    "accuracy meaning and may not be reported as one (PRD §0)."
)
_OUTPUT_HEIGHT = 64
_OUTPUT_WIDTH = 32


def _crops(
    seed: int,
    count: int = 2,
    shape: tuple[int, ...] = (40, 20, 3),
) -> list[npt.NDArray[np.uint8]]:
    """Build crops in the test from a seeded numpy generator."""
    rng = np.random.default_rng(seed)
    crops: list[npt.NDArray[np.uint8]] = []
    for _ in range(count):
        crop = rng.integers(0, 256, size=shape, dtype=np.uint8)
        crops.append(np.asarray(crop, dtype=np.uint8))
    return crops


def _reference_rows(
    crops: Sequence[npt.NDArray[np.uint8]],
    *,
    dim: int,
    seed: int,
) -> npt.NDArray[np.float32]:
    """Independent copy of the spec projection, used only as a test oracle."""
    flats: list[npt.NDArray[np.float64]] = []
    for crop in crops:
        resized = cv2.resize(crop, (_OUTPUT_WIDTH, _OUTPUT_HEIGHT))
        assert resized.shape[:2] == (_OUTPUT_HEIGHT, _OUTPUT_WIDTH)
        flat = np.asarray(resized, dtype=np.float64).reshape(-1)
        flats.append(flat)
    stacked = np.stack(flats)
    matrix = np.random.default_rng(seed).standard_normal((stacked.shape[1], dim))
    projected = stacked @ matrix
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    unit = np.asarray(projected / norms, dtype=np.float32)
    return np.asarray(unit, dtype=np.float32)


def test_class_docstring_quotes_the_spec() -> None:
    doc = UntrainedPlaceholderEmbedder.__doc__ or ""
    assert " ".join(doc.split()) == _SPEC_DOC


def test_package_exports_the_embedder() -> None:
    from trace_search import reid

    assert reid.UntrainedPlaceholderEmbedder is UntrainedPlaceholderEmbedder
    assert "UntrainedPlaceholderEmbedder" in reid.__all__


def test_defaults_and_placeholder_flag() -> None:
    embedder = UntrainedPlaceholderEmbedder()
    assert embedder.dim == 256
    assert embedder.placeholder is True
    assert embedder.name == "untrained-placeholder-v1"


def test_name_is_a_read_only_property() -> None:
    embedder = UntrainedPlaceholderEmbedder()
    name_attr = UntrainedPlaceholderEmbedder.__dict__["name"]
    assert isinstance(name_attr, property)
    assert name_attr.fset is None
    with pytest.raises(AttributeError):
        setattr(embedder, "name", "other")
    assert embedder.name == "untrained-placeholder-v1"


def test_assignable_to_appearance_embedder() -> None:
    """Same assignment ``mypy src/ --strict`` checks in ``_as_appearance_embedder``."""
    concrete = UntrainedPlaceholderEmbedder(dim=8, seed=0)
    embedder: AppearanceEmbedder = concrete
    assert _as_appearance_embedder(concrete) is embedder
    crops = _crops(seed=1, count=2)
    found = embedder.embed(crops)
    assert embedder.placeholder is True
    assert embedder.name == "untrained-placeholder-v1"
    assert embedder.dim == 8
    assert found.shape == (len(crops), embedder.dim)
    assert found.dtype == np.float32


def test_same_seed_and_crop_match_across_instances_and_calls() -> None:
    crops = _crops(seed=7, count=3)
    first = UntrainedPlaceholderEmbedder(dim=32, seed=0)
    second = UntrainedPlaceholderEmbedder(dim=32, seed=0)
    once = first.embed(crops)
    twice = first.embed(crops)
    other = second.embed(crops)
    assert np.array_equal(once, twice)
    assert np.array_equal(once, other)


def test_seeds_zero_and_one_differ() -> None:
    crops = _crops(seed=7, count=2)
    seed_zero = UntrainedPlaceholderEmbedder(dim=32, seed=0).embed(crops)
    seed_one = UntrainedPlaceholderEmbedder(dim=32, seed=1).embed(crops)
    assert not np.array_equal(seed_zero, seed_one)


def test_different_crops_differ_for_one_seed() -> None:
    embedder = UntrainedPlaceholderEmbedder(dim=32, seed=0)
    left = embedder.embed(_crops(seed=1, count=1))
    right = embedder.embed(_crops(seed=2, count=1))
    assert not np.array_equal(left, right)


def test_output_shape_dtype_and_unit_norm() -> None:
    crops = _crops(seed=3, count=4)
    dim = 16
    found = UntrainedPlaceholderEmbedder(dim=dim, seed=2).embed(crops)
    assert found.shape == (len(crops), dim)
    assert found.dtype == np.float32
    norms = np.linalg.norm(found, axis=1)
    assert np.all(np.abs(norms - np.float32(1.0)) <= np.float32(1e-5))


def test_default_dim_output_shape() -> None:
    crops = _crops(seed=4, count=1)
    found = UntrainedPlaceholderEmbedder().embed(crops)
    assert found.shape == (1, 256)
    assert found.dtype == np.float32
    norm = float(np.linalg.norm(found[0]))
    assert abs(norm - 1.0) <= 1e-5


def test_matches_fixed_gaussian_projection() -> None:
    crops = _crops(seed=11, count=2, shape=(15, 18, 3))
    dim = 8
    seed = 4
    found = UntrainedPlaceholderEmbedder(dim=dim, seed=seed).embed(crops)
    expected = _reference_rows(crops, dim=dim, seed=seed)
    assert found.shape == expected.shape
    assert np.allclose(found, expected, atol=1e-5, rtol=1e-5)


def test_same_instance_embed_calls_match() -> None:
    crops = _crops(seed=9, count=2)
    embedder = UntrainedPlaceholderEmbedder(dim=16, seed=3)
    assert np.array_equal(embedder.embed(crops), embedder.embed(crops))


def test_projection_matrix_is_drawn_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """One draw per instance. Same crops match across calls and instances."""
    draws: list[int] = []
    real_rng = np.random.default_rng

    def counting(seed: int) -> np.random.Generator:
        draws.append(int(seed))
        return real_rng(seed)

    crops = _crops(seed=1, count=2)
    monkeypatch.setattr(np.random, "default_rng", counting)
    embedder = UntrainedPlaceholderEmbedder(dim=4, seed=5)
    other = UntrainedPlaceholderEmbedder(dim=4, seed=5)
    different = UntrainedPlaceholderEmbedder(dim=4, seed=6)
    first = embedder.embed(crops)
    second = embedder.embed(crops)
    assert np.array_equal(first, second)
    assert np.array_equal(first, other.embed(crops))
    assert not np.array_equal(first, different.embed(crops))
    assert draws == [5, 5, 6]


def test_two_dimensional_crop_raises() -> None:
    crop = _crops(seed=6, count=1, shape=(18, 9))[0]
    assert crop.ndim == 2
    assert crop.size > 0
    embedder = UntrainedPlaceholderEmbedder(dim=4, seed=0)
    with pytest.raises(ValueError, match="HxWx3"):
        embedder.embed([crop])


def test_one_channel_crop_raises() -> None:
    crop = _crops(seed=6, count=1, shape=(12, 8, 1))[0]
    assert crop.shape[2] == 1
    embedder = UntrainedPlaceholderEmbedder(dim=4, seed=0)
    with pytest.raises(ValueError, match="3 channels"):
        embedder.embed([crop])


def test_four_channel_crop_raises() -> None:
    crop = _crops(seed=6, count=1, shape=(12, 8, 4))[0]
    assert crop.shape[2] == 4
    embedder = UntrainedPlaceholderEmbedder(dim=4, seed=0)
    with pytest.raises(ValueError, match="3 channels"):
        embedder.embed([crop])


def test_different_spatial_shapes_share_one_output_shape() -> None:
    rng = np.random.default_rng(8)
    crops = [
        np.asarray(rng.integers(0, 256, size=(50, 10, 3), dtype=np.uint8)),
        np.asarray(rng.integers(0, 256, size=(9, 40, 3), dtype=np.uint8)),
    ]
    found = UntrainedPlaceholderEmbedder(dim=8, seed=0).embed(crops)
    assert found.shape == (2, 8)
    norms = np.linalg.norm(found, axis=1)
    assert np.all(np.abs(norms - np.float32(1.0)) <= np.float32(1e-5))


def test_noncontiguous_crop_embeds() -> None:
    rng = np.random.default_rng(12)
    full = np.asarray(rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8))
    crop = full[::2, ::2, :]
    assert crop.flags.c_contiguous is False
    found = UntrainedPlaceholderEmbedder(dim=8, seed=0).embed([crop])
    assert found.shape == (1, 8)
    norm = float(np.linalg.norm(found[0]))
    assert abs(norm - 1.0) <= 1e-5


def test_empty_crop_list_has_shape_zero_by_dim() -> None:
    found = UntrainedPlaceholderEmbedder(dim=8, seed=0).embed([])
    assert found.shape == (0, 8)
    assert found.dtype == np.float32


@pytest.mark.parametrize("dim", [0, -3])
def test_dim_below_one_raises(dim: int) -> None:
    with pytest.raises(ValueError, match="dim"):
        UntrainedPlaceholderEmbedder(dim=dim)


@pytest.mark.parametrize(
    "shape",
    [(0, 8, 3), (8, 0, 3), (0, 0, 3), (0, 0), (4, 4, 0)],
)
def test_zero_sized_crop_raises(shape: tuple[int, ...]) -> None:
    crop = np.zeros(shape, dtype=np.uint8)
    assert crop.size == 0
    embedder = UntrainedPlaceholderEmbedder(dim=4, seed=0)
    with pytest.raises(ValueError, match="non-zero"):
        embedder.embed([crop])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.uint16])
def test_non_uint8_crop_raises(dtype: type) -> None:
    crop: npt.NDArray[np.generic] = np.zeros((8, 8, 3), dtype=dtype)
    assert crop.dtype != np.uint8
    embedder = UntrainedPlaceholderEmbedder(dim=4, seed=0)
    with pytest.raises(ValueError, match="uint8"):
        embedder.embed([crop])


def test_crop_rank_other_than_image_raises() -> None:
    embedder = UntrainedPlaceholderEmbedder(dim=4, seed=0)
    flat = np.zeros((8,), dtype=np.uint8)
    with pytest.raises(ValueError, match="HxW"):
        embedder.embed([flat])
    video = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="HxW"):
        embedder.embed([video])


def test_non_array_crop_raises() -> None:
    embedder = UntrainedPlaceholderEmbedder(dim=4, seed=0)
    raw = cast(npt.NDArray[np.uint8], [[0, 1], [2, 3]])
    with pytest.raises(ValueError, match="numpy"):
        embedder.embed([raw])


def test_mixed_channel_counts_raise() -> None:
    rng = np.random.default_rng(3)
    color = np.asarray(rng.integers(0, 256, size=(10, 12, 3), dtype=np.uint8))
    gray = np.asarray(rng.integers(0, 256, size=(10, 12), dtype=np.uint8))
    embedder = UntrainedPlaceholderEmbedder(dim=4, seed=0)
    with pytest.raises(ValueError, match="HxWx3"):
        embedder.embed([color, gray])


def test_all_zero_crop_cannot_be_normalised() -> None:
    crop = np.zeros((12, 9, 3), dtype=np.uint8)
    assert crop.size > 0
    embedder = UntrainedPlaceholderEmbedder(dim=8, seed=0)
    with pytest.raises(ValueError, match="projected"):
        embedder.embed([crop])


def test_signature_is_only_dim_and_seed() -> None:
    params = inspect.signature(UntrainedPlaceholderEmbedder).parameters
    assert list(params) == ["dim", "seed"]
    assert all(item.kind is not inspect.Parameter.VAR_KEYWORD for item in params.values())


def _construct_rejected(keyword: str, value: object) -> None:
    """Call the constructor with one keyword it does not accept."""
    rejected: dict[str, object] = {}
    rejected[keyword] = value
    UntrainedPlaceholderEmbedder(**rejected)


def test_weights_keyword_raises_type_error() -> None:
    with pytest.raises(TypeError, match="'weights'"):
        _construct_rejected("weights", "x")


def test_weights_path_keyword_raises_type_error() -> None:
    with pytest.raises(TypeError, match="'weights_path'"):
        _construct_rejected("weights_path", "x")


def test_checkpoint_keyword_raises_type_error() -> None:
    with pytest.raises(TypeError, match="'checkpoint'"):
        _construct_rejected("checkpoint", "x")


def test_pretrained_keyword_raises_type_error() -> None:
    with pytest.raises(TypeError, match="'pretrained'"):
        _construct_rejected("pretrained", True)


def test_no_network_guard_records_zero_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def refuse_create_connection(*_args: object, **_kwargs: object) -> None:
        calls.append("create_connection")
        raise OSError("refused")

    monkeypatch.setattr(socket, "create_connection", refuse_create_connection)
    crops = _crops(seed=5, count=2)
    embedder = UntrainedPlaceholderEmbedder(dim=8, seed=1)
    found = embedder.embed(crops)
    assert found.shape == (len(crops), 8)
    assert calls == []
