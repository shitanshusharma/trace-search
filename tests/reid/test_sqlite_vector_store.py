"""Persisted appearance vectors in a stdlib SQLite file outside the repo.

Rows are seeded in the test. Nothing is read from ``smoke/`` and nothing is
written into the repository tree. ``tmp_path`` is the database directory.
"""

import math
import socket
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from trace_search.reid.sqlite_vector_store import _CREATE_SQL as _STORE_SQL
from trace_search.reid.sqlite_vector_store import (
    DEFAULT_MAX_AGE_DAYS,
    SqliteVectorStore,
    _conformance_checked_store,
    _placeholder_int,
    _vector_from_blob,
)
from trace_search.reid.types import AppearanceEmbedding
from trace_search.reid.vector_store import ALLOWED_DERIVED_FROM, InMemoryVectorStore
from trace_search.retention.config import FOOTAGE_DERIVED_DAYS

_DIM = 3
_NOW = 1_800_000_000.0
_DAY = 86400
_QUERY_DIM = r"query dim must be \d+, got"


class _Clock:
    """Mutable UTC epoch. Tests move it. Nothing sleeps."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _unit(*values: float) -> tuple[float, ...]:
    raw = np.asarray(values, dtype=np.float64)
    unit = raw / float(np.linalg.norm(raw))
    return tuple(float(item) for item in np.asarray(unit, dtype="<f4"))


def _embedding(
    camera_id: str,
    track_id: int,
    frame_number: int,
    vector: tuple[float, ...],
    *,
    embedder_name: str = "hand-built",
    placeholder: bool = True,
) -> AppearanceEmbedding:
    return AppearanceEmbedding(
        camera_id=camera_id,
        track_id=track_id,
        frame_number=frame_number,
        vector=vector,
        embedder_name=embedder_name,
        placeholder=placeholder,
    )


def _store(tmp_path: Path, clock: _Clock | None = None, *, dim: int = _DIM) -> SqliteVectorStore:
    ticking = _Clock(_NOW) if clock is None else clock
    return SqliteVectorStore(tmp_path / "embeddings.sqlite", dim, clock=ticking)


def _raw_count(db_path: Path) -> int:
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute("SELECT COUNT(*) FROM appearance_embeddings").fetchone()
    finally:
        connection.close()
    assert row is not None
    return int(row[0])


def _hits(
    found: Sequence[tuple[AppearanceEmbedding, float]],
) -> list[tuple[object, ...]]:
    return [
        (
            item.camera_id,
            item.track_id,
            item.frame_number,
            item.embedder_name,
            item.placeholder,
            item.vector,
            score,
        )
        for item, score in found
    ]


def _seeded_batch(
    count: int,
    dim: int,
    seed: int,
) -> tuple[list[AppearanceEmbedding], tuple[float, ...]]:
    """Seeded float32 unit vectors. Three rows share one vector, so scores tie."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((count + 1, dim))
    raw[1] = raw[0]
    raw[2] = raw[0]
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    values = [tuple(float(item) for item in row) for row in np.asarray(unit, dtype="<f4")]
    values[count // 2] = values[count]
    rows = [
        _embedding(f"cam-{index % 7}", index, index % 5, values[index], embedder_name="seeded")
        for index in range(count)
    ]
    return rows, values[count]


def _store_source() -> str:
    path = Path(__file__).resolve().parents[2] / "src" / "trace_search" / "reid"
    return (path / "sqlite_vector_store.py").read_text(encoding="utf-8")


def test_create_table_ddl_is_a_source_literal() -> None:
    """The scanner reads this string. It is not built with format or concatenation."""
    text = _store_source()
    start = text.index("CREATE TABLE IF NOT EXISTS appearance_embeddings (")
    end = text.index("created_at REAL NOT NULL", start)
    body = text[start:end]
    assert "created_at" not in body
    assert "{" not in body
    assert "format(" not in body
    assert "+" not in body
    closing = text.index('"""', start)
    assert "created_at REAL NOT NULL" in text[start:closing]
    assert "CHECK (derived_from IN ('SMOKE','CODE_TEST','SITE'))" in text[start:closing]
    assert _STORE_SQL.strip().startswith("CREATE TABLE IF NOT EXISTS appearance_embeddings (")
    assert "created_at REAL NOT NULL" in _STORE_SQL


def test_scoring_reuses_the_in_memory_helpers() -> None:
    text = _store_source()
    assert "def _cosine_scores" not in text
    assert "def _rank" not in text
    assert "_cosine_scores(" in text
    assert "_rank(" in text


def test_retention_uses_the_shared_cutoff() -> None:
    text = _store_source()
    assert "retention_cutoff" in text
    assert "86400" not in text
    assert "_retention_seconds" not in text
    assert "_SECONDS_PER_DAY" not in text
    assert DEFAULT_MAX_AGE_DAYS == FOOTAGE_DERIVED_DAYS == 7


def test_pragma_table_info_exact_columns(tmp_path: Path) -> None:
    store = _store(tmp_path)
    connection = sqlite3.connect(store.db_path)
    try:
        info = connection.execute("PRAGMA table_info(appearance_embeddings)").fetchall()
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        ddl_row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'appearance_embeddings'"
        ).fetchone()
    finally:
        connection.close()
    assert ddl_row is not None
    names_and_types = [(str(row[1]), str(row[2])) for row in info]
    assert names_and_types == [
        ("camera_id", "TEXT"),
        ("track_id", "INTEGER"),
        ("frame_number", "INTEGER"),
        ("embedder_name", "TEXT"),
        ("placeholder", "INTEGER"),
        ("vector", "BLOB"),
        ("run_id", "TEXT"),
        ("derived_from", "TEXT"),
        ("created_at", "REAL"),
    ]
    assert [str(row[0]) for row in tables] == ["appearance_embeddings"]
    assert "CHECK (derived_from IN ('SMOKE','CODE_TEST','SITE'))" in str(ddl_row[0])


def test_vector_blob_is_float32_little_endian(tmp_path: Path) -> None:
    store = _store(tmp_path)
    row = _embedding("cam-a", 1, 1, (1.0, 0.0, 0.0))
    assert store.add("run-1", "CODE_TEST", [row]) == 1
    connection = sqlite3.connect(store.db_path)
    try:
        fetched = connection.execute("SELECT vector FROM appearance_embeddings").fetchone()
    finally:
        connection.close()
    assert fetched is not None
    blob = fetched[0]
    assert isinstance(blob, bytes)
    assert len(blob) == 4 * _DIM
    little = [float(item) for item in np.frombuffer(blob, dtype="<f4")]
    big = [float(item) for item in np.frombuffer(blob, dtype=">f4")]
    assert little == [1.0, 0.0, 0.0]
    assert big != little


def test_hundred_row_search_matches_in_memory_including_exact_ties(tmp_path: Path) -> None:
    """Same seeded rows, same order, same scores, including equal-score ties."""
    rows, query = _seeded_batch(100, 8, seed=7)
    clock = _Clock(_NOW)
    disk = SqliteVectorStore(tmp_path / "embeddings.sqlite", 8, clock=clock)
    memory = InMemoryVectorStore(8)
    for start in range(0, 100, 25):
        batch = rows[start : start + 25]
        assert disk.add("run-1", "CODE_TEST", batch) == len(batch)
        assert memory.add("run-1", "CODE_TEST", batch) == len(batch)
    assert disk.count() == memory.count() == 100
    full_disk = disk.search(query, k=100)
    full_memory = memory.search(query, k=100)
    assert _hits(full_disk) == _hits(full_memory)
    scores = [score for _item, score in full_disk]
    assert any(scores[index] == scores[index + 1] for index in range(len(scores) - 1))
    assert _hits(disk.search(query, k=2)) == _hits(memory.search(query, k=2))
    cameras = {rows[0].camera_id}
    assert _hits(disk.search(query, k=20, camera_ids=cameras)) == _hits(
        memory.search(query, k=20, camera_ids=cameras)
    )


def test_hand_built_ties_match_the_in_memory_store(tmp_path: Path) -> None:
    query = _unit(1.0, 0.0, 0.0)
    rows = [
        _embedding("cam-b", 1, 1, query),
        _embedding("cam-a", 5, 1, query),
        _embedding("cam-a", 2, 9, query),
        _embedding("cam-a", 2, 3, query),
        _embedding("cam-a", 0, 0, _unit(0.0, 1.0, 0.0)),
    ]
    disk = _store(tmp_path)
    memory = InMemoryVectorStore(_DIM)
    disk.add("run-1", "CODE_TEST", rows)
    memory.add("run-1", "CODE_TEST", rows)
    assert _hits(disk.search(query, k=10)) == _hits(memory.search(query, k=10))
    assert abs(disk.search(query, k=1)[0][1] - 1.0) <= 1e-6


def test_restart_keeps_the_rows(tmp_path: Path) -> None:
    rows, query = _seeded_batch(100, 8, seed=11)
    clock = _Clock(_NOW)
    path = tmp_path / "embeddings.sqlite"
    disk = SqliteVectorStore(path, 8, clock=clock)
    disk.add("run-1", "SITE", rows)
    before = _hits(disk.search(query, k=100))
    db_path = disk.db_path
    del disk
    assert _raw_count(db_path) == 100
    reopened = SqliteVectorStore(db_path, 8, clock=clock)
    assert reopened.count() == 100
    assert _hits(reopened.search(query, k=100)) == before
    assert reopened.db_path == path.resolve()


def test_placeholder_must_be_a_bool() -> None:
    with pytest.raises(TypeError, match="placeholder"):
        _placeholder_int(1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="placeholder"):
        _placeholder_int(0)  # type: ignore[arg-type]


def test_placeholder_and_embedder_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    real = _embedding(
        "cam-a", 1, 2, _unit(1.0, 0.0, 0.0), embedder_name="seeded", placeholder=False
    )
    marked = _embedding(
        "cam-b", 3, 4, _unit(0.0, 1.0, 0.0), embedder_name="other", placeholder=True
    )
    store.add("run-1", "SMOKE", [real, marked])
    found = store.search(_unit(1.0, 0.0, 0.0), k=2)
    assert [(item.embedder_name, item.placeholder, item.vector) for item, _score in found] == [
        ("seeded", False, real.vector),
        ("other", True, marked.vector),
    ]


def test_camera_filter_is_applied_before_top_k(tmp_path: Path) -> None:
    """A better row outside the filter must not consume a top-k slot."""
    query = _unit(1.0, 0.0, 0.0)
    outsider = _embedding("cam-out", 1, 1, query)
    second = _embedding("cam-in", 2, 1, _unit(1.0, 1.0, 0.0))
    third = _embedding("cam-in", 3, 1, _unit(0.0, 1.0, 0.0))
    store = _store(tmp_path)
    memory = InMemoryVectorStore(_DIM)
    batch = [outsider, third, second]
    store.add("run-1", "CODE_TEST", batch)
    memory.add("run-1", "CODE_TEST", batch)
    found = store.search(query, k=2, camera_ids=["cam-in"])
    assert _hits(found) == _hits(memory.search(query, k=2, camera_ids=["cam-in"]))
    assert [(item.track_id, item.camera_id) for item, _score in found] == [
        (2, "cam-in"),
        (3, "cam-in"),
    ]


def test_camera_filter_matching_nothing_returns_empty(tmp_path: Path) -> None:
    store = _store(tmp_path)
    row = _embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))
    store.add("run-1", "CODE_TEST", [row])
    query = row.vector
    assert store.search(query, k=2, camera_ids=["cam-missing"]) == []
    assert store.search(query, k=1, camera_ids=[]) == []
    assert store.search(query, k=1, camera_ids=set()) == []


def test_empty_store_search_returns_empty(tmp_path: Path) -> None:
    store = _store(tmp_path)
    query = _unit(1.0, 0.0, 0.0)
    assert store.search(query, k=3) == []
    assert store.search(query, k=3, camera_ids=["cam-a"]) == []


def test_default_max_age_days_is_7() -> None:
    assert DEFAULT_MAX_AGE_DAYS == 7
    purge_default = SqliteVectorStore.purge_expired.__defaults__
    count_default = SqliteVectorStore.count_older_than.__defaults__
    assert purge_default is not None
    assert count_default is not None
    assert purge_default[-1] == 7
    assert count_default[-1] == 7


def test_retention_purges_an_8_day_row_and_keeps_a_6_day_row(tmp_path: Path) -> None:
    """The default window is used. An explicit day count would hide a planted 30."""
    clock = _Clock(_NOW)
    store = _store(tmp_path, clock)
    old = _embedding("cam-old", 1, 1, _unit(1.0, 0.0, 0.0))
    fresh = _embedding("cam-fresh", 2, 1, _unit(0.0, 1.0, 0.0))
    clock.now = _NOW - 8 * _DAY
    store.add("run-old", "CODE_TEST", [old])
    clock.now = _NOW - 6 * _DAY
    store.add("run-fresh", "CODE_TEST", [fresh])
    assert _raw_count(store.db_path) == 2
    clock.now = _NOW
    removed = store.purge_expired()
    assert removed == 1
    assert store.count() == 1
    found = store.search(fresh.vector, k=5)
    assert [item.camera_id for item, _score in found] == ["cam-fresh"]


def test_retention_keeps_a_row_one_second_older_than_6_days(tmp_path: Path) -> None:
    """A planted default of 6 deletes this row. A default of 7 keeps it."""
    created_at = _NOW - 6 * _DAY - 1.0
    clock = _Clock(created_at)
    store = _store(tmp_path, clock)
    row = _embedding("cam-six", 6, 1, _unit(0.0, 1.0, 0.0))
    store.add("run-six", "CODE_TEST", [row])
    clock.now = _NOW
    assert store.purge_expired() == 0
    assert store.count() == 1
    assert store.search(row.vector, k=1)[0][0].camera_id == "cam-six"


def test_retention_keeps_a_row_aged_exactly_7_days(tmp_path: Path) -> None:
    clock = _Clock(_NOW - 7 * _DAY)
    store = _store(tmp_path, clock)
    row = _embedding("cam-exact", 3, 1, _unit(0.0, 0.0, 1.0))
    store.add("run-exact", "CODE_TEST", [row])
    clock.now = _NOW
    assert store.purge_expired() == 0
    assert [item.camera_id for item, _score in store.search(row.vector, k=1)] == ["cam-exact"]


def test_purge_runs_on_open(tmp_path: Path) -> None:
    clock = _Clock(_NOW - 8 * _DAY)
    store = _store(tmp_path, clock)
    store.add("run-old", "CODE_TEST", [_embedding("cam-old", 1, 1, _unit(1.0, 0.0, 0.0))])
    db_path = store.db_path
    del store
    assert _raw_count(db_path) == 1
    clock.now = _NOW
    SqliteVectorStore(db_path, _DIM, clock=clock)
    assert _raw_count(db_path) == 0


def test_purge_runs_before_add(tmp_path: Path) -> None:
    clock = _Clock(_NOW - 8 * _DAY)
    store = _store(tmp_path, clock)
    store.add("run-old", "CODE_TEST", [_embedding("cam-old", 8, 1, _unit(1.0, 0.0, 0.0))])
    clock.now = _NOW
    fresh = _embedding("cam-fresh", 9, 1, _unit(0.0, 1.0, 0.0))
    assert store.add("run-fresh", "CODE_TEST", [fresh]) == 1
    assert _raw_count(store.db_path) == 1
    assert [item.camera_id for item, _score in store.search(fresh.vector, k=5)] == ["cam-fresh"]


def test_empty_add_still_purges(tmp_path: Path) -> None:
    clock = _Clock(_NOW - 8 * _DAY)
    store = _store(tmp_path, clock)
    store.add("run-old", "CODE_TEST", [_embedding("cam-old", 1, 1, _unit(1.0, 0.0, 0.0))])
    clock.now = _NOW
    assert store.add("run-empty", "SITE", []) == 0
    assert _raw_count(store.db_path) == 0


def test_rejected_add_leaves_expired_rows(tmp_path: Path) -> None:
    clock = _Clock(_NOW - 8 * _DAY)
    store = _store(tmp_path, clock)
    row = _embedding("cam-old", 1, 1, _unit(1.0, 0.0, 0.0))
    store.add("run-old", "CODE_TEST", [row])
    clock.now = _NOW
    with pytest.raises(ValueError, match="derived_from"):
        store.add("run-bad", "TRAIN", [row])
    assert _raw_count(store.db_path) == 1
    assert store.count() == 1


def test_purge_on_open_false_keeps_the_count(tmp_path: Path) -> None:
    clock = _Clock(_NOW - 8 * _DAY)
    store = _store(tmp_path, clock)
    store.add("run-old", "CODE_TEST", [_embedding("cam-old", 1, 1, _unit(1.0, 0.0, 0.0))])
    db_path = store.db_path
    del store
    clock.now = _NOW
    reopened = SqliteVectorStore(db_path, _DIM, clock=clock, purge_on_open=False)
    assert _raw_count(db_path) == 1
    assert reopened.count_older_than() == 1
    assert _raw_count(db_path) == 1
    assert reopened.purge_expired() == 1
    assert reopened.count_older_than() == 0
    assert _raw_count(db_path) == 0


def test_count_older_than_matches_purge_and_does_not_delete(tmp_path: Path) -> None:
    clock = _Clock(_NOW - 8 * _DAY)
    store = SqliteVectorStore(
        tmp_path / "embeddings.sqlite", _DIM, clock=clock, purge_on_open=False
    )
    store.add("run-old", "CODE_TEST", [_embedding("cam-old", 1, 1, _unit(1.0, 0.0, 0.0))])
    clock.now = _NOW - 6 * _DAY
    store.add("run-fresh", "CODE_TEST", [_embedding("cam-fresh", 2, 1, _unit(0.0, 1.0, 0.0))])
    clock.now = _NOW
    assert store.count_older_than() == 1
    assert _raw_count(store.db_path) == 2
    assert store.purge_expired() == 1
    assert store.count_older_than(now=_NOW) == 0


def test_explicit_now_does_not_move_the_clock(tmp_path: Path) -> None:
    clock = _Clock(_NOW - 8 * _DAY)
    store = _store(tmp_path, clock)
    store.add("run-old", "CODE_TEST", [_embedding("cam-old", 1, 1, _unit(1.0, 0.0, 0.0))])
    assert store.count_older_than(now=_NOW) == 1
    assert store.purge_expired(now=_NOW) == 1
    assert clock.now == _NOW - 8 * _DAY


def test_db_path_inside_the_repo_raises() -> None:
    repo = Path(__file__).resolve().parents[2]
    target = repo / "t-r-7-should-not-exist.sqlite"
    try:
        with pytest.raises(ValueError, match="outside the repo"):
            SqliteVectorStore(target, _DIM, clock=lambda: 0.0)
        assert not target.exists()
    finally:
        if target.is_file():
            target.unlink()


def test_missing_parent_is_created(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "embeddings.sqlite"
    store = SqliteVectorStore(path, _DIM, clock=lambda: 1.0)
    assert store.db_path == path.resolve()
    assert store.db_path.is_file()
    assert store.label == "embeddings"
    assert store.covered_tables == ("appearance_embeddings",)
    assert store.dim == _DIM


@pytest.mark.parametrize("dim", [0, -1, True, 1.5])
def test_invalid_dim_raises(tmp_path: Path, dim: object) -> None:
    with pytest.raises(ValueError, match="dim"):
        SqliteVectorStore(tmp_path / "embeddings.sqlite", dim, clock=lambda: 0.0)  # type: ignore[arg-type]


def test_wrong_dimension_embedding_raises(tmp_path: Path) -> None:
    store = _store(tmp_path)
    bad = _embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="dim"):
        store.add("run-1", "CODE_TEST", [bad])
    assert store.count() == 0
    assert _raw_count(store.db_path) == 0


def test_bad_batch_does_not_insert(tmp_path: Path) -> None:
    store = _store(tmp_path)
    kept = _embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))
    store.add("run-1", "CODE_TEST", [kept])
    incoming = _embedding("cam-new", 8, 4, _unit(0.0, 1.0, 0.0))
    wrong = _embedding("cam-bad", 9, 4, _unit(1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="dim"):
        store.add("run-b", "SITE", [incoming, wrong])
    assert store.count() == 1
    assert store.count(run_id="run-b") == 0
    assert _raw_count(store.db_path) == 1


def test_wrong_blob_length_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)
    row = _embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))
    store.add("run-1", "CODE_TEST", [row])
    connection = sqlite3.connect(store.db_path)
    try:
        connection.execute(
            "UPDATE appearance_embeddings SET vector = ?",
            (b"\x00" * (4 * _DIM + 4),),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ValueError, match="vector blob length"):
        store.search(row.vector, k=1)


def test_blob_must_be_bytes_and_known_lengths_round_trip() -> None:
    raw = np.asarray([1.0, 0.0, 0.0], dtype="<f4").tobytes()
    assert _vector_from_blob(memoryview(raw), _DIM) == (1.0, 0.0, 0.0)
    assert _vector_from_blob(bytearray(raw), _DIM) == (1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="vector blob must be bytes"):
        _vector_from_blob(1, _DIM)
    with pytest.raises(ValueError, match="vector blob length"):
        _vector_from_blob(b"\x00\x00", _DIM)


def test_stored_placeholder_other_than_zero_or_one_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)
    row = _embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))
    store.add("run-1", "CODE_TEST", [row])
    connection = sqlite3.connect(store.db_path)
    try:
        connection.execute("UPDATE appearance_embeddings SET placeholder = 2")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ValueError, match="placeholder"):
        store.search(row.vector, k=1)


def test_check_constraint_rejects_an_unknown_label(tmp_path: Path) -> None:
    store = _store(tmp_path)
    connection = sqlite3.connect(store.db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO appearance_embeddings (
                    camera_id, track_id, frame_number, embedder_name, placeholder,
                    vector, run_id, derived_from, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("cam-a", 1, 1, "hand-built", 1, b"\x00" * 12, "run-1", "TRAIN", 1.0),
            )
    finally:
        connection.close()
    assert store.count() == 0


@pytest.mark.parametrize("label", ["TRAIN", "smoke", ""])
def test_add_rejects_derived_from_outside_the_set(tmp_path: Path, label: str) -> None:
    store = _store(tmp_path)
    row = _embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))
    store.add("run-1", "CODE_TEST", [row])
    with pytest.raises(ValueError, match="derived_from"):
        store.add("run-2", label, [row])
    assert store.count() == 1
    assert ALLOWED_DERIVED_FROM == frozenset({"SMOKE", "CODE_TEST", "SITE"})


@pytest.mark.parametrize("label", ["TRAIN", "smoke"])
def test_delete_where_rejects_derived_from_outside_the_set(tmp_path: Path, label: str) -> None:
    store = _store(tmp_path)
    store.add("run-1", "CODE_TEST", [_embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))])
    with pytest.raises(ValueError, match="derived_from"):
        store.delete_where(derived_from=label)
    assert store.count() == 1


@pytest.mark.parametrize("label", ["TRAIN", "smoke"])
def test_count_rejects_derived_from_outside_the_set(tmp_path: Path, label: str) -> None:
    store = _store(tmp_path)
    store.add("run-1", "CODE_TEST", [_embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))])
    with pytest.raises(ValueError, match="derived_from"):
        store.count(derived_from=label)
    assert store.count() == 1


def test_delete_where_with_no_filter_raises(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("run-1", "CODE_TEST", [_embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))])
    with pytest.raises(ValueError, match="derived_from or run_id"):
        store.delete_where()
    with pytest.raises(ValueError, match="derived_from or run_id"):
        store.delete_where(None, None)
    assert store.count() == 1


def test_count_and_delete_filters(tmp_path: Path) -> None:
    store = _store(tmp_path)
    one = _embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))
    two = _embedding("cam-b", 2, 1, _unit(0.0, 1.0, 0.0))
    three = _embedding("cam-c", 3, 1, _unit(0.0, 0.0, 1.0))
    store.add("run-1", "SMOKE", [one])
    store.add("run-1", "CODE_TEST", [two])
    store.add("run-2", "SMOKE", [three])
    assert store.count() == 3
    assert store.count(derived_from=None, run_id=None) == 3
    assert store.count(derived_from="SMOKE") == 2
    assert store.count(run_id="run-1") == 2
    assert store.count(derived_from="CODE_TEST", run_id="run-1") == 1
    assert store.delete_where(derived_from="SITE") == 0
    assert store.delete_where(derived_from="SMOKE", run_id="run-1") == 1
    assert store.count(run_id="run-2", derived_from="SMOKE") == 1
    assert store.delete_where(run_id="run-2") == 1
    assert store.count() == 1
    assert store.count(derived_from="CODE_TEST") == 1


def test_add_returns_inserted_count(tmp_path: Path) -> None:
    store = _store(tmp_path)
    row = _embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))
    assert store.add("run-1", "SITE", []) == 0
    assert store.count() == 0
    assert store.add("run-1", "SITE", [row, row]) == 2
    assert store.count() == 2


def test_blank_run_id_and_bad_embeddings_raise(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="run_id"):
        store.add("", "CODE_TEST", [])
    with pytest.raises(ValueError, match="run_id"):
        store.add(1, "CODE_TEST", [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AppearanceEmbedding"):
        store.add("run-1", "CODE_TEST", "nope")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AppearanceEmbedding"):
        store.add("run-1", "CODE_TEST", [object()])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="run_id"):
        store.delete_where(run_id="")
    with pytest.raises(ValueError, match="derived_from"):
        store.count(derived_from="")
    assert store.count() == 0


def test_wrong_dimension_query_raises(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("run-1", "CODE_TEST", [_embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))])
    with pytest.raises(ValueError, match=_QUERY_DIM):
        store.search(_unit(1.0, 0.0), k=1)
    with pytest.raises(ValueError, match=_QUERY_DIM):
        store.search(_unit(1.0, 0.0), k=1, camera_ids=["cam-missing"])
    empty = SqliteVectorStore(tmp_path / "empty.sqlite", _DIM, clock=lambda: 1.0)
    with pytest.raises(ValueError, match=_QUERY_DIM):
        empty.search(_unit(1.0, 0.0), k=1)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_query_raises(tmp_path: Path, bad: float) -> None:
    store = _store(tmp_path)
    store.add("run-1", "CODE_TEST", [_embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))])
    with pytest.raises(ValueError, match="finite"):
        store.search((bad, 0.0, 0.0), k=1)


def test_zero_query_raises(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("run-1", "CODE_TEST", [_embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))])
    with pytest.raises(ValueError, match="non-zero"):
        store.search((0.0, 0.0, 0.0), k=1)


@pytest.mark.parametrize("k", [0, -1, False, True, 2.0])
def test_non_positive_k_raises(tmp_path: Path, k: object) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="k"):
        store.search(_unit(1.0, 0.0, 0.0), k)  # type: ignore[arg-type]


def test_string_camera_ids_raise(tmp_path: Path) -> None:
    store = _store(tmp_path)
    query = _unit(1.0, 0.0, 0.0)
    store.add("run-1", "CODE_TEST", [_embedding("cam-a", 1, 1, query)])
    with pytest.raises(ValueError, match="camera_ids"):
        store.search(query, k=1, camera_ids="cam-a")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="camera_ids"):
        store.search(query, k=1, camera_ids=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="camera_id"):
        store.search(query, k=1, camera_ids=[""])


def test_query_must_be_a_vector_of_floats(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="query"):
        store.search("1,0,0", k=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="query"):
        store.search(None, k=1)  # type: ignore[arg-type]
    as_float32 = np.asarray(_unit(1.0, 0.0, 0.0), dtype=np.float32)
    store.add("run-1", "CODE_TEST", [_embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))])
    found = store.search(as_float32, k=1)
    assert found[0][0].camera_id == "cam-a"


def test_default_clock_stores_utc_epoch(tmp_path: Path) -> None:
    import time

    before = time.time()
    store = SqliteVectorStore(tmp_path / "embeddings.sqlite", _DIM)
    store.add("run-1", "CODE_TEST", [_embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))])
    after = time.time()
    connection = sqlite3.connect(store.db_path)
    try:
        row = connection.execute("SELECT created_at FROM appearance_embeddings").fetchone()
    finally:
        connection.close()
    assert row is not None
    assert before <= float(row[0]) <= after


def test_integer_clock_is_accepted(tmp_path: Path) -> None:
    store = SqliteVectorStore(tmp_path / "embeddings.sqlite", _DIM, clock=lambda: 15)
    store.add("run-1", "CODE_TEST", [_embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))])
    connection = sqlite3.connect(store.db_path)
    try:
        row = connection.execute("SELECT created_at FROM appearance_embeddings").fetchone()
    finally:
        connection.close()
    assert row is not None
    assert float(row[0]) == 15.0


def test_bad_clock_and_bad_window_raise(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="timestamp"):
        SqliteVectorStore(tmp_path / "bool.sqlite", _DIM, clock=lambda: True)  # type: ignore[arg-type,return-value]
    with pytest.raises(ValueError, match="finite"):
        SqliteVectorStore(tmp_path / "inf.sqlite", _DIM, clock=lambda: math.inf)
    store = _store(tmp_path)
    with pytest.raises(TypeError, match="max_age_days"):
        store.purge_expired(max_age_days=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_age_days"):
        store.count_older_than(max_age_days=-1)
    with pytest.raises(TypeError, match="timestamp"):
        store.purge_expired(now=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        store.purge_expired(now=math.nan)


def test_store_satisfies_vector_store(tmp_path: Path) -> None:
    row = _embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))
    store = _conformance_checked_store(_store(tmp_path))
    assert store.count() == 0
    assert store.add("run-1", "CODE_TEST", [row]) == 1
    found = store.search(row.vector, 1)
    assert found[0][0] == row
    assert store.delete_where(derived_from="CODE_TEST") == 1
    assert store.count() == 0


def test_no_network_on_add_search_delete_count_and_purge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def refuse_connect(*_args: object, **_kwargs: object) -> None:
        calls.append("connect")
        raise OSError("refused")

    def refuse_create_connection(*_args: object, **_kwargs: object) -> None:
        calls.append("create_connection")
        raise OSError("refused")

    monkeypatch.setattr(socket.socket, "connect", refuse_connect)
    monkeypatch.setattr(socket, "create_connection", refuse_create_connection)
    store = _store(tmp_path)
    row = _embedding("cam-a", 1, 1, _unit(1.0, 0.0, 0.0))
    assert store.add("run-1", "CODE_TEST", [row]) == 1
    assert store.search(row.vector, k=1)
    assert store.count(derived_from="CODE_TEST") == 1
    assert store.count_older_than() == 0
    assert store.purge_expired() == 0
    assert store.delete_where(run_id="run-1") == 1
    assert calls == []
