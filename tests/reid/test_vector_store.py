"""Tests for the in-memory appearance vector store."""

import math
import re
import socket

import numpy as np
import pytest

from trace_search.reid.types import AppearanceEmbedding
from trace_search.reid.vector_store import (
    ALLOWED_DERIVED_FROM,
    InMemoryVectorStore,
    _conformance_checked_store,
)
from trace_search.tracking import store as track_store

_DIM = 3
_QUERY_DIM = r"query dim must be \d+, got"


def _unit(*values: float) -> tuple[float, ...]:
    """L2-normalise a hand-built vector with numpy inside the test."""
    raw = np.asarray(values, dtype=np.float64)
    unit = raw / float(np.linalg.norm(raw))
    return tuple(float(item) for item in unit)


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    one = np.asarray(left, dtype=np.float64)
    other = np.asarray(right, dtype=np.float64)
    return float(np.dot(one, other) / (np.linalg.norm(one) * np.linalg.norm(other)))


def _embedding(
    camera_id: str,
    track_id: int,
    frame_number: int,
    vector: tuple[float, ...],
) -> AppearanceEmbedding:
    return AppearanceEmbedding(
        camera_id=camera_id,
        track_id=track_id,
        frame_number=frame_number,
        vector=vector,
        embedder_name="hand-built",
        placeholder=True,
    )


def _axis_store() -> tuple[InMemoryVectorStore, AppearanceEmbedding, tuple[float, ...]]:
    query = _unit(1.0, 0.0, 0.0)
    row = _embedding("cam-a", 1, 1, query)
    store = InMemoryVectorStore(_DIM)
    store.add("run-1", "CODE_TEST", [row])
    return store, row, query


def test_top_two_order_and_cosine_values() -> None:
    """Three hand-built unit vectors: top-2 order and cosine within 1e-6."""
    query = _unit(1.0, 0.0, 0.0)
    opposite = _embedding("cam-a", 0, 0, _unit(-1.0, 0.0, 0.0))
    near = _embedding("cam-m", 4, 2, _unit(1.0, 1.0, 0.0))
    same = _embedding("cam-z", 9, 9, query)
    store = InMemoryVectorStore(_DIM)
    store.add("run-1", "CODE_TEST", [opposite, near, same])

    found = store.search(query, k=2)
    assert [(item.camera_id, item.track_id, item.frame_number) for item, _ in found] == [
        ("cam-z", 9, 9),
        ("cam-m", 4, 2),
    ]
    assert found[0][0] is same
    assert found[1][0] is near
    assert type(found[0][1]) is float
    assert abs(found[0][1] - 1.0) <= 1e-6
    assert abs(found[1][1] - _cosine(query, near.vector)) <= 1e-6


def test_camera_filter_is_applied_before_top_k() -> None:
    """A better row outside the filter must not consume a top-k slot."""
    query = _unit(1.0, 0.0, 0.0)
    outsider = _embedding("cam-out", 1, 1, query)
    second = _embedding("cam-in", 2, 1, _unit(1.0, 1.0, 0.0))
    third = _embedding("cam-in", 3, 1, _unit(0.0, 1.0, 0.0))
    store = InMemoryVectorStore(_DIM)
    store.add("run-1", "CODE_TEST", [outsider, third, second])

    found = store.search(query, k=2, camera_ids=["cam-in"])
    assert [(item.track_id, item.camera_id) for item, _ in found] == [
        (2, "cam-in"),
        (3, "cam-in"),
    ]
    assert abs(found[0][1] - _cosine(query, second.vector)) <= 1e-6
    assert abs(found[1][1] - 0.0) <= 1e-6


def test_camera_filter_matching_nothing_returns_empty() -> None:
    store, _row, query = _axis_store()
    assert store.search(query, k=2, camera_ids=["cam-missing"]) == []
    assert store.search(query, k=2, camera_ids={"cam-missing"}) == []


def test_empty_camera_ids_returns_empty() -> None:
    store, _row, query = _axis_store()
    assert store.search(query, k=1, camera_ids=[]) == []
    assert store.search(query, k=1, camera_ids=set()) == []


def test_empty_store_search_returns_empty() -> None:
    store = InMemoryVectorStore(_DIM)
    query = _unit(1.0, 0.0, 0.0)
    assert store.search(query, k=3) == []
    assert store.search(query, k=3, camera_ids=["cam-a"]) == []


def test_ties_break_by_camera_track_and_frame() -> None:
    """Equal cosine yields to (camera_id, track_id, frame_number) ascending."""
    query = _unit(1.0, 0.0, 0.0)
    same = query
    rows = [
        _embedding("cam-b", 1, 1, same),
        _embedding("cam-a", 5, 1, same),
        _embedding("cam-a", 2, 9, same),
        _embedding("cam-a", 2, 3, same),
        _embedding("cam-a", 0, 0, _unit(0.0, 1.0, 0.0)),
    ]
    store = InMemoryVectorStore(_DIM)
    store.add("run-1", "CODE_TEST", rows)

    found = store.search(query, k=10)
    assert [(item.camera_id, item.track_id, item.frame_number) for item, _ in found] == [
        ("cam-a", 2, 3),
        ("cam-a", 2, 9),
        ("cam-a", 5, 1),
        ("cam-b", 1, 1),
        ("cam-a", 0, 0),
    ]
    assert abs(found[0][1] - 1.0) <= 1e-6
    assert abs(found[-1][1] - 0.0) <= 1e-6


def test_derived_from_train_raises() -> None:
    store, row, _query = _axis_store()
    with pytest.raises(ValueError, match="derived_from"):
        store.add("run-2", "TRAIN", [row])
    assert store.count() == 1
    assert store.count(derived_from="CODE_TEST") == 1
    assert store.count(run_id="run-2") == 0


def test_delete_where_smoke_keeps_every_code_test_row() -> None:
    smoke = [
        _embedding("cam-s", 1, 1, _unit(1.0, 0.0, 0.0)),
        _embedding("cam-s", 2, 1, _unit(0.0, 1.0, 0.0)),
    ]
    code = [
        _embedding("cam-c", 1, 1, _unit(1.0, 0.0, 0.0)),
        _embedding("cam-c", 2, 1, _unit(0.0, 1.0, 0.0)),
        _embedding("cam-c", 3, 1, _unit(0.0, 0.0, 1.0)),
    ]
    site = [_embedding("cam-t", 1, 1, _unit(0.0, 0.0, 1.0))]
    store = InMemoryVectorStore(_DIM)
    store.add("run-smoke", "SMOKE", smoke)
    store.add("run-code", "CODE_TEST", code)
    store.add("run-site", "SITE", site)

    deleted = store.delete_where(derived_from="SMOKE")
    assert deleted == 2
    assert store.count(derived_from="SMOKE") == 0
    assert store.count(derived_from="CODE_TEST") == 3
    assert store.count(derived_from="SITE") == 1
    assert store.count() == 4


def test_wrong_dimension_embedding_raises() -> None:
    store = InMemoryVectorStore(_DIM)
    bad = _embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="dim"):
        store.add("run-1", "CODE_TEST", [bad])
    assert store.count() == 0


def test_wrong_dimension_query_raises() -> None:
    store, _row, _query = _axis_store()
    with pytest.raises(ValueError, match=_QUERY_DIM):
        store.search(_unit(1.0, 0.0), k=1)
    with pytest.raises(ValueError, match=_QUERY_DIM):
        store.search(_unit(1.0, 0.0, 0.0, 0.0), k=1)
    with pytest.raises(ValueError, match="query"):
        store.search(np.zeros((1, _DIM), dtype=np.float64), k=1)


def test_wrong_dimension_query_on_an_empty_store_raises() -> None:
    store = InMemoryVectorStore(_DIM)
    with pytest.raises(ValueError, match=_QUERY_DIM):
        store.search(_unit(1.0, 0.0), k=1)


def test_wrong_dimension_query_with_an_empty_camera_filter_raises() -> None:
    store, _row, _query = _axis_store()
    with pytest.raises(ValueError, match=_QUERY_DIM):
        store.search(_unit(1.0, 0.0), k=1, camera_ids=["cam-missing"])


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_query_raises(bad: float) -> None:
    store, _row, _query = _axis_store()
    with pytest.raises(ValueError, match="finite"):
        store.search((bad, 0.0, 0.0), k=1)


@pytest.mark.parametrize("k", [0, -1, False, True, 2.0])
def test_non_positive_k_raises(k: object) -> None:
    store = InMemoryVectorStore(_DIM)
    query = _unit(1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="k"):
        store.search(query, k)  # type: ignore[arg-type]


def test_bad_batch_does_not_change_the_store() -> None:
    store, kept, query = _axis_store()
    incoming = _embedding("cam-new", 8, 4, _unit(0.0, 1.0, 0.0))
    wrong = _embedding("cam-bad", 9, 4, _unit(1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="dim"):
        store.add("run-b", "SITE", [incoming, wrong])
    assert store.count() == 1
    assert store.count(run_id="run-b") == 0
    assert store.count(derived_from="SITE") == 0
    found = store.search(query, k=5)
    assert [item.camera_id for item, _score in found] == [kept.camera_id]


def test_delete_where_camera_keeps_the_other_camera_in_the_run() -> None:
    store = InMemoryVectorStore(_DIM)
    first = _embedding("cam-1", 1, 1, _unit(1.0, 0.0, 0.0))
    second = _embedding("cam-2", 1, 1, _unit(0.0, 1.0, 0.0))
    other_run = _embedding("cam-1", 1, 1, _unit(0.0, 0.0, 1.0))
    store.add("run-1", "CODE_TEST", [first, second])
    store.add("run-2", "CODE_TEST", [other_run])
    with pytest.raises(ValueError, match="derived_from or run_id"):
        store.delete_where(camera_id="cam-1")
    with pytest.raises(ValueError, match="non-empty"):
        store.delete_where(run_id="run-1", camera_id="")
    assert store.delete_where(run_id="run-1", camera_id="cam-1") == 1
    assert store.count(run_id="run-1") == 1
    assert store.count(run_id="run-2") == 1
    found = store.search(_unit(0.0, 1.0, 0.0), k=1)
    assert found[0][0].camera_id == "cam-2"
    assert found[0][0] is second


def test_delete_where_with_no_filter_raises() -> None:
    store, _row, _query = _axis_store()
    before = store.count()
    with pytest.raises(ValueError, match="derived_from or run_id"):
        store.delete_where()
    with pytest.raises(ValueError, match="derived_from or run_id"):
        store.delete_where(None, None)
    assert store.count() == before


def test_count_filters_by_derived_from_and_run_id() -> None:
    store = InMemoryVectorStore(_DIM)
    smoke = _embedding("cam-s", 1, 1, _unit(1.0, 0.0, 0.0))
    code_a = _embedding("cam-a", 1, 1, _unit(0.0, 1.0, 0.0))
    code_b = _embedding("cam-b", 1, 1, _unit(0.0, 0.0, 1.0))
    store.add("run-a", "SMOKE", [smoke, smoke])
    store.add("run-a", "CODE_TEST", [code_a])
    store.add("run-b", "CODE_TEST", [code_b])
    assert store.count() == 4
    assert store.count(derived_from=None, run_id=None) == 4
    assert store.count(derived_from="SMOKE") == 2
    assert store.count(run_id="run-a") == 3
    assert store.count(derived_from="CODE_TEST", run_id="run-b") == 1
    assert store.count(derived_from="SITE", run_id="run-a") == 0


def test_allowed_derived_from_matches_track_store_check() -> None:
    """The set is defined once and must equal the track-store CHECK."""
    match = re.search(r"CHECK \(derived_from IN \(([^)]*)\)\)", track_store._CREATE_SQL)
    assert match is not None
    labels = frozenset(re.findall(r"'([^']*)'", match.group(1)))
    assert labels
    assert ALLOWED_DERIVED_FROM == labels


def test_delete_where_matches_run_id_and_both_filters() -> None:
    store = InMemoryVectorStore(_DIM)
    one = _embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))
    two = _embedding("cam-b", 2, 1, _unit(0.0, 1.0, 0.0))
    three = _embedding("cam-c", 3, 1, _unit(0.0, 0.0, 1.0))
    store.add("run-1", "SMOKE", [one])
    store.add("run-1", "CODE_TEST", [two])
    store.add("run-2", "SMOKE", [three])

    assert store.count() == 3
    assert store.delete_where(derived_from="SITE") == 0
    assert store.count() == 3
    removed = store.delete_where(derived_from="SMOKE", run_id="run-1")
    assert removed == 1
    assert store.count(run_id="run-1", derived_from="CODE_TEST") == 1
    assert store.count(run_id="run-2", derived_from="SMOKE") == 1
    assert store.delete_where(run_id="run-2") == 1
    assert store.count() == 1
    assert store.count(derived_from="CODE_TEST") == 1


def test_add_returns_inserted_count() -> None:
    store = InMemoryVectorStore(_DIM)
    row = _embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))
    assert store.add("run-1", "SITE", []) == 0
    assert store.count() == 0
    assert store.add("run-1", "SITE", [row, row]) == 2
    assert store.count() == 2


def test_search_score_is_cosine_for_a_non_unit_query() -> None:
    """A scaled query stays on the unit-vector cosine, not the raw dot."""
    store, row, _query = _axis_store()
    found = store.search((2.0, 0.0, 0.0), k=1)
    assert found[0][0] is row
    assert abs(found[0][1] - 1.0) <= 1e-6


@pytest.mark.parametrize("label", ["TRAIN", "smoke"])
def test_delete_where_rejects_derived_from_outside_the_set(label: str) -> None:
    store, _row, _query = _axis_store()
    before = store.count()
    with pytest.raises(ValueError, match="derived_from"):
        store.delete_where(derived_from=label)
    assert store.count() == before


@pytest.mark.parametrize("label", ["TRAIN", "smoke"])
def test_count_rejects_derived_from_outside_the_set(label: str) -> None:
    store, _row, _query = _axis_store()
    with pytest.raises(ValueError, match="derived_from"):
        store.count(derived_from=label)
    assert store.count() == 1


def test_search_sees_rows_from_every_add() -> None:
    store = InMemoryVectorStore(_DIM)
    first = _embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))
    second = _embedding("cam-b", 2, 1, _unit(0.0, 1.0, 0.0))
    store.add("run-1", "CODE_TEST", [first])
    store.add("run-2", "SMOKE", [second])
    found = store.search(_unit(0.0, 1.0, 0.0), k=2)
    assert [item.camera_id for item, _score in found] == ["cam-b", "cam-a"]


def test_zero_query_raises() -> None:
    store, _row, _query = _axis_store()
    with pytest.raises(ValueError, match="non-zero"):
        store.search((0.0, 0.0, 0.0), k=1)


def test_string_camera_ids_raise() -> None:
    store, _row, query = _axis_store()
    with pytest.raises(ValueError, match="camera_ids"):
        store.search(query, k=1, camera_ids="cam-a")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="camera_ids"):
        store.search(query, k=1, camera_ids=b"cam-a")  # type: ignore[arg-type]


def test_camera_ids_must_be_a_collection() -> None:
    store, _row, query = _axis_store()
    with pytest.raises(ValueError, match="camera_ids"):
        store.search(query, k=1, camera_ids=1)  # type: ignore[arg-type]


def test_blank_camera_id_in_filter_raises() -> None:
    store, _row, query = _axis_store()
    with pytest.raises(ValueError, match="camera_id"):
        store.search(query, k=1, camera_ids=[""])


def test_blank_run_id_and_derived_from_raise() -> None:
    store = InMemoryVectorStore(_DIM)
    with pytest.raises(ValueError, match="run_id"):
        store.add("", "CODE_TEST", [])
    with pytest.raises(ValueError, match="derived_from"):
        store.add("run-1", "", [])
    with pytest.raises(ValueError, match="run_id"):
        store.add(1, "CODE_TEST", [])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="derived_from"):
        store.count(derived_from="")
    with pytest.raises(ValueError, match="run_id"):
        store.delete_where(run_id="")
    assert store.count() == 0


def test_embeddings_must_be_appearance_embeddings() -> None:
    store = InMemoryVectorStore(_DIM)
    with pytest.raises(TypeError, match="AppearanceEmbedding"):
        store.add("run-1", "CODE_TEST", "nope")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AppearanceEmbedding"):
        store.add("run-1", "CODE_TEST", [object()])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="AppearanceEmbedding"):
        store.add("run-1", "CODE_TEST", 3)  # type: ignore[arg-type]
    assert store.count() == 0


@pytest.mark.parametrize("dim", [0, -1, True, 1.5])
def test_invalid_dim_raises(dim: object) -> None:
    with pytest.raises(ValueError, match="dim"):
        InMemoryVectorStore(dim)  # type: ignore[arg-type]


def test_search_accepts_numpy_query() -> None:
    store, row, query = _axis_store()
    as_float32 = np.asarray(query, dtype=np.float32)
    as_float64 = np.asarray(query, dtype=np.float64)
    found32 = store.search(as_float32, k=1)
    found64 = store.search(as_float64, k=1)
    assert found32[0][0] is row
    assert found64[0][0] is row
    assert abs(found32[0][1] - 1.0) <= 1e-6
    assert abs(found64[0][1] - 1.0) <= 1e-6


def test_query_must_be_a_vector_of_floats() -> None:
    store, _row, _query = _axis_store()
    with pytest.raises(ValueError, match="query"):
        store.search("1,0,0", k=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="query"):
        store.search(None, k=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="query"):
        store.search(["a", "b", "c"], k=1)  # type: ignore[arg-type]


def test_small_batches_match_one_batch_search() -> None:
    """Several small adds match one add, including the exact scores."""
    vectors = [
        _unit(1.0, 0.0, 0.0),
        _unit(0.0, 1.0, 0.0),
        _unit(0.0, 0.0, 1.0),
        _unit(1.0, 1.0, 0.0),
        _unit(1.0, 0.0, 1.0),
        _unit(0.0, 1.0, 1.0),
        _unit(1.0, 1.0, 1.0),
        _unit(-1.0, 0.5, 0.0),
    ]
    rows = [
        _embedding(f"cam-{index}", index, index + 1, vector) for index, vector in enumerate(vectors)
    ]
    one = InMemoryVectorStore(_DIM)
    one.add("run-1", "CODE_TEST", rows)
    many = InMemoryVectorStore(_DIM)
    many.add("run-1", "CODE_TEST", rows[:1])
    many.add("run-1", "CODE_TEST", rows[1:3])
    assert many.count() == 3
    assert many._matrix.dtype == np.float32
    assert many._matrix.shape[0] > many.count()
    assert many._matrix.shape[1] == _DIM
    many.add("run-1", "CODE_TEST", rows[3:6])
    many.add("run-1", "CODE_TEST", rows[6:])
    query = _unit(1.0, 0.2, -0.3)
    left = one.search(query, k=len(rows))
    right = many.search(query, k=len(rows))
    assert len(left) == len(rows)
    assert left == right


def test_store_is_a_float32_matrix_with_parallel_keys() -> None:
    first = _embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))
    second = _embedding("cam-b", 2, 2, _unit(0.0, 1.0, 0.0))
    store = InMemoryVectorStore(_DIM)
    store.add("run-1", "CODE_TEST", [first])
    store.add("run-2", "SMOKE", [second])
    assert store.dim == _DIM
    assert store._matrix.dtype == np.float32
    assert store._count == len(store._rows) == store.count() == 2
    assert store._matrix.shape[0] >= store._count
    assert store._matrix.shape[1] == _DIM
    assert store._rows[0].embedding is first
    assert store._rows[0].run_id == "run-1"
    assert store._rows[0].derived_from == "CODE_TEST"
    assert store._rows[1].embedding is second
    assert store.delete_where(run_id="run-1") == 1
    assert store._matrix.dtype == np.float32
    assert store._count == len(store._rows) == store.count() == 1
    assert store._matrix.shape[0] >= store._count
    assert store._matrix.shape[1] == _DIM
    assert store._rows[0].embedding is second
    found = store.search(second.vector, k=1)
    assert found[0][0] is second
    assert abs(found[0][1] - 1.0) <= 1e-6


def test_store_satisfies_vector_store() -> None:
    row = _embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))
    store = _conformance_checked_store(InMemoryVectorStore(_DIM))
    assert store.count() == 0
    assert store.add("run-1", "CODE_TEST", [row]) == 1
    found = store.search(row.vector, 1)
    assert found[0][0] is row
    assert store.delete_where(derived_from="CODE_TEST") == 1
    assert store.count() == 0


def test_search_limits_run_and_derived_from() -> None:
    north = _unit(0.0, 1.0, 0.0)
    east = _unit(1.0, 0.0, 0.0)
    store = InMemoryVectorStore(_DIM)
    store.add("run-1", "SMOKE", [_embedding("cam-a", 1, 0, east)])
    store.add("run-2", "CODE_TEST", [_embedding("cam-a", 1, 1, east)])
    store.add("run-1", "CODE_TEST", [_embedding("cam-a", 1, 2, north)])
    found = store.search(north, k=10, run_id="run-1", derived_from="CODE_TEST")
    assert [(item.frame_number, item.vector) for item, _score in found] == [(2, north)]
    assert len(store.search(north, k=10)) == 3


def test_no_network_on_add_search_delete_and_count(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def refuse_create_connection(*_args: object, **_kwargs: object) -> None:
        calls.append("create_connection")
        raise OSError("refused")

    monkeypatch.setattr(socket, "create_connection", refuse_create_connection)
    store, _row, query = _axis_store()
    assert store.search(query, k=1)
    assert store.count(derived_from="CODE_TEST") == 1
    assert store.delete_where(run_id="run-1") == 1
    assert calls == []
