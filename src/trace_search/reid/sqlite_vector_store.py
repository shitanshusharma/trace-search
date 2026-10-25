"""SQLite store of anonymous appearance vectors.

The file is separate from the track database. The only table is
``appearance_embeddings``. Each vector is float32 little-endian bytes of
length ``4 * dim``. The path must sit outside the repository tree. The
guard is ``tracking.store._db_path_outside_repo``. This module calls that
helper and does not copy it.

``created_at`` is a UTC epoch from the injected clock, or from
``time.time`` when no clock is injected. Rows older than the shipped
footage window (7 days) are deleted on open and before every ``add``. A
row whose age is exactly that window is kept. ``purge_on_open=False``
skips the open purge so a registry can still count expired rows.
``count_older_than`` uses the same cutoff and does not delete.

Search loads the rows that pass the camera filter into a float32 matrix
and ranks them with ``vector_store._cosine_scores`` and
``vector_store._rank``. ``derived_from`` must be a member of
``ALLOWED_DERIVED_FROM`` on ``add``, ``delete_where``, and ``count``.

The table does not store crops or pixels. ``add`` inserts only the new
batch. It does not read the rows already stored.
"""

from __future__ import annotations

import math
import sqlite3
import time
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt

from trace_search.reid.types import AppearanceEmbedding
from trace_search.reid.vector_store import (
    VectorStore,
    _cosine_scores,
    _optional_derived_from,
    _optional_text,
    _rank,
    _require_camera_ids,
    _require_delete_target,
    _require_derived_from,
    _require_dim,
    _require_embeddings,
    _require_k,
    _require_query,
    _require_text,
    _StoredRow,
)
from trace_search.retention._time import retention_cutoff
from trace_search.retention.config import FOOTAGE_DERIVED_DAYS
from trace_search.tracking.store import _db_path_outside_repo

# Shipped footage-derived window. T-P-7 is the later override.
DEFAULT_MAX_AGE_DAYS = FOOTAGE_DERIVED_DAYS

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS appearance_embeddings (
    camera_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    frame_number INTEGER NOT NULL,
    embedder_name TEXT NOT NULL,
    placeholder INTEGER NOT NULL,
    vector BLOB NOT NULL,
    run_id TEXT NOT NULL,
    derived_from TEXT NOT NULL CHECK (derived_from IN ('SMOKE','CODE_TEST','SITE')),
    created_at REAL NOT NULL
)
"""

_INDEX_CREATED_SQL = """
CREATE INDEX IF NOT EXISTS idx_appearance_embeddings_created_at
ON appearance_embeddings (created_at)
"""

_INSERT_SQL = """
INSERT INTO appearance_embeddings (
    camera_id,
    track_id,
    frame_number,
    embedder_name,
    placeholder,
    vector,
    run_id,
    derived_from,
    created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_LOAD_SQL = """
SELECT
    camera_id,
    track_id,
    frame_number,
    embedder_name,
    placeholder,
    vector,
    run_id,
    derived_from
FROM appearance_embeddings
ORDER BY rowid ASC
"""

_COUNT_SQL = """
SELECT COUNT(*)
FROM appearance_embeddings
WHERE (? IS NULL OR derived_from = ?)
  AND (? IS NULL OR run_id = ?)
"""

_DELETE_SQL = """
DELETE FROM appearance_embeddings
WHERE (? IS NULL OR derived_from = ?)
  AND (? IS NULL OR run_id = ?)
"""

_PURGE_SQL = "DELETE FROM appearance_embeddings WHERE created_at < ?"
_COUNT_OLDER_SQL = "SELECT COUNT(*) FROM appearance_embeddings WHERE created_at < ?"

_InsertRow = tuple[str, int, int, str, int, bytes, str, str, float]
_Query = Sequence[float] | npt.NDArray[np.float64] | npt.NDArray[np.float32]
_T = TypeVar("_T")
_Clock = Callable[[], float]


def _require_timestamp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"timestamp must be a real UTC epoch value, got {value!r}")
    timestamp = float(value)
    if not math.isfinite(timestamp):
        raise ValueError(f"timestamp must be a finite UTC epoch value, got {value!r}")
    return timestamp


def _expired_cutoff(clock: _Clock, now: float | None, max_age_days: int) -> float:
    current = _require_timestamp(clock() if now is None else now)
    return retention_cutoff(current, max_age_days)


def _placeholder_int(value: bool) -> int:
    """Store a bool as 0 or 1. Any other value, including 0 and 1, is refused."""
    if value is True:
        return 1
    if value is False:
        return 0
    raise TypeError(f"placeholder must be a bool, got {value!r}")


def _placeholder_bool(value: object) -> bool:
    """Map a stored 0 or 1 back to a bool. Any other integer is refused."""
    if value == 0:
        return False
    if value == 1:
        return True
    raise ValueError(f"placeholder must be stored as 0 or 1, got {value!r}")


def _vector_blob(vector: tuple[float, ...]) -> bytes:
    """Encode one vector as little-endian float32 bytes."""
    array = np.ascontiguousarray(np.asarray(vector, dtype="<f4"))
    return array.tobytes()


def _blob_bytes(value: object) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    kind = type(value).__name__
    raise ValueError(f"vector blob must be bytes, got {kind}")


def _vector_from_blob(value: object, dim: int) -> tuple[float, ...]:
    """Decode one blob. A length other than ``4 * dim`` is refused."""
    raw = _blob_bytes(value)
    expected = 4 * dim
    if len(raw) != expected:
        raise ValueError(f"vector blob length must be {expected}, got {len(raw)}")
    viewed = np.frombuffer(raw, dtype="<f4")
    return tuple(float(item) for item in viewed)


def _insert_row(
    item: AppearanceEmbedding,
    run_id: str,
    derived_from: str,
    created_at: float,
) -> _InsertRow:
    return (
        item.camera_id,
        item.track_id,
        item.frame_number,
        item.embedder_name,
        _placeholder_int(item.placeholder),
        _vector_blob(item.vector),
        run_id,
        derived_from,
        created_at,
    )


def _stored_from_row(row: sqlite3.Row, dim: int) -> _StoredRow:
    vector = _vector_from_blob(row["vector"], dim)
    embedding = AppearanceEmbedding(
        camera_id=str(row["camera_id"]),
        track_id=int(row["track_id"]),
        frame_number=int(row["frame_number"]),
        vector=vector,
        embedder_name=str(row["embedder_name"]),
        placeholder=_placeholder_bool(row["placeholder"]),
    )
    return _StoredRow(str(row["run_id"]), str(row["derived_from"]), embedding)


def _matching_rows(
    records: Sequence[sqlite3.Row],
    camera_ids: frozenset[str] | None,
    dim: int,
) -> tuple[_StoredRow, ...]:
    """Keep rows whose camera is allowed, then decode those vectors.

    ``None`` keeps every camera. The filter runs before the matrix is
    built, so a row from another camera is not ranked.
    """
    chosen: list[_StoredRow] = []
    for record in records:
        if camera_ids is not None and str(record["camera_id"]) not in camera_ids:
            continue
        chosen.append(_stored_from_row(record, dim))
    return tuple(chosen)


def _matrix_for_rows(rows: Sequence[_StoredRow]) -> npt.NDArray[np.float32]:
    stacked = np.asarray([row.embedding.vector for row in rows], dtype=np.float32)
    return cast(npt.NDArray[np.float32], np.ascontiguousarray(stacked))


def _one_count(cursor: sqlite3.Cursor) -> int:
    # COUNT(*) returns one row. The cast records that invariant.
    fetched = cast(sqlite3.Row, cursor.fetchone())
    return int(fetched[0])


class SqliteVectorStore:
    """SQLite store of ``AppearanceEmbedding`` rows in one local file.

    Construction opens the file and creates ``appearance_embeddings`` when
    missing. It purges rows older than the shipped footage window (7 days)
    unless ``purge_on_open`` is False. The default stays True so every
    other caller still purges on open. The registry passes False: that
    open-time purge throws the count away, and a fresh registry must still
    be able to count the expired rows. ``add`` purges again before it
    inserts. ``count_older_than`` uses the same cutoff and does not delete.
    The clock is injectable so retention tests do not sleep.
    """

    def __init__(
        self,
        db_path: Path,
        dim: int,
        *,
        clock: _Clock | None = None,
        purge_on_open: bool = True,
    ) -> None:
        self._dim = _require_dim(dim)
        resolved = _db_path_outside_repo(db_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.db_path: Path = resolved
        self._clock: _Clock = time.time if clock is None else clock
        self._ensure_schema()
        if purge_on_open:
            self.purge_expired()

    @property
    def dim(self) -> int:
        """Length of every stored vector."""
        return self._dim

    @property
    def label(self) -> str:
        """Registry name for this store."""
        return "embeddings"

    @property
    def covered_tables(self) -> tuple[str, ...]:
        """Tables whose ``created_at`` column this store purges."""
        return ("appearance_embeddings",)

    def add(
        self,
        run_id: str,
        derived_from: str,
        embeddings: Sequence[AppearanceEmbedding],
    ) -> int:
        """Store one batch and return how many rows were stored.

        The batch is checked before any row is stored and before the purge.
        A failure leaves the table unchanged. A valid call, including an
        empty batch, purges expired rows and then inserts only the new rows.
        """
        run = _require_text("run_id", run_id)
        origin = _require_derived_from(derived_from)
        items = _require_embeddings(embeddings, self._dim)
        self.purge_expired()
        if not items:
            return 0
        created_at = _require_timestamp(self._clock())
        rows = tuple(_insert_row(item, run, origin, created_at) for item in items)

        def _insert(connection: sqlite3.Connection) -> None:
            connection.executemany(_INSERT_SQL, rows)

        self._using(_insert)
        return len(rows)

    def search(
        self,
        vector: _Query,
        k: int,
        camera_ids: Collection[str] | None = None,
    ) -> list[tuple[AppearanceEmbedding, float]]:
        """Return up to ``k`` cosine hits after the camera filter."""
        query = _require_query(vector, self._dim)
        limit = _require_k(k)
        cameras = _require_camera_ids(camera_ids)
        rows = self._load_matching(cameras)
        if not rows:
            return []
        matrix = _matrix_for_rows(rows)
        indexes = tuple(range(len(rows)))
        scores = _cosine_scores(matrix, indexes, query)
        return _rank(rows, indexes, scores)[:limit]

    def delete_where(
        self,
        derived_from: str | None = None,
        run_id: str | None = None,
    ) -> int:
        """Delete matching rows. Both arguments omitted raises ``ValueError``."""
        origin = _optional_derived_from(derived_from)
        run = _optional_text("run_id", run_id)
        _require_delete_target(origin, run)

        def _query(connection: sqlite3.Connection) -> int:
            cursor = connection.execute(_DELETE_SQL, (origin, origin, run, run))
            return int(cursor.rowcount)

        return self._using(_query)

    def count(
        self,
        derived_from: str | None = None,
        run_id: str | None = None,
    ) -> int:
        """Count rows. Both arguments omitted counts every row."""
        origin = _optional_derived_from(derived_from)
        run = _optional_text("run_id", run_id)

        def _query(connection: sqlite3.Connection) -> int:
            cursor = connection.execute(_COUNT_SQL, (origin, origin, run, run))
            return _one_count(cursor)

        return self._using(_query)

    def count_older_than(
        self,
        now: float | None = None,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    ) -> int:
        """Return how many rows ``purge_expired`` would delete.

        The cutoff matches ``purge_expired``. This does not delete, and it
        does not call ``purge_expired``. ``add`` purges first, so calling
        ``add`` before this method drops the rows the count would have
        reported.
        """
        cutoff = _expired_cutoff(self._clock, now, max_age_days)

        def _query(connection: sqlite3.Connection) -> int:
            cursor = connection.execute(_COUNT_OLDER_SQL, (cutoff,))
            return _one_count(cursor)

        return self._using(_query)

    def purge_expired(
        self,
        now: float | None = None,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    ) -> int:
        """Delete rows whose ``created_at`` is older than ``max_age_days``.

        The default window is the shipped footage window (7 days). ``now``
        is a UTC epoch. When it is omitted, the injected clock supplies it.
        A row is kept when its age is exactly the window. The return value
        is the number of rows deleted.
        """
        cutoff = _expired_cutoff(self._clock, now, max_age_days)

        def _query(connection: sqlite3.Connection) -> int:
            cursor = connection.execute(_PURGE_SQL, (cutoff,))
            return int(cursor.rowcount)

        return self._using(_query)

    def _load_matching(self, camera_ids: frozenset[str] | None) -> tuple[_StoredRow, ...]:
        dim = self._dim

        def _query(connection: sqlite3.Connection) -> tuple[_StoredRow, ...]:
            cursor = connection.execute(_LOAD_SQL)
            records = cast(Sequence[sqlite3.Row], cursor.fetchall())
            return _matching_rows(records, camera_ids, dim)

        return self._using(_query)

    def _ensure_schema(self) -> None:
        def _create(connection: sqlite3.Connection) -> None:
            connection.execute(_CREATE_SQL)
            connection.execute(_INDEX_CREATED_SQL)

        self._using(_create)

    def _using(self, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        connection: sqlite3.Connection | None = None
        try:
            with sqlite3.connect(self.db_path) as connection:
                connection.row_factory = sqlite3.Row
                return fn(connection)
        finally:
            if connection is not None:
                connection.close()


def _conformance_checked_store(store: SqliteVectorStore) -> VectorStore:
    """Return ``store`` typed as ``VectorStore``.

    The assignment is what ``mypy src/ --strict`` checks. Tests are not
    type-checked, so the helper stays in this module.
    """
    checked: VectorStore = store
    return checked
