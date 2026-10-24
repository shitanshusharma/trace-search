"""Protocol and in-memory store for anonymous appearance vectors.

Rows stay in the process. Nothing is written to a file.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from trace_search.reid.types import AppearanceEmbedding

# The track observation table stores these three labels in its CHECK.
# That module has no Python constant for them, so the set is defined once here.
ALLOWED_DERIVED_FROM: frozenset[str] = frozenset({"SMOKE", "CODE_TEST", "SITE"})

_Query = Sequence[float] | npt.NDArray[np.float64] | npt.NDArray[np.float32]


@dataclass(frozen=True)
class _StoredRow:
    """One matrix row: run metadata and the original embedding."""

    run_id: str
    derived_from: str
    embedding: AppearanceEmbedding


class VectorStore(Protocol):
    """Add, search, and purge anonymous appearance embeddings."""

    def add(
        self,
        run_id: str,
        derived_from: str,
        embeddings: Sequence[AppearanceEmbedding],
    ) -> int:
        """Store one batch and return how many rows were stored.

        ``derived_from`` must be a member of ``ALLOWED_DERIVED_FROM``.
        Every vector length must equal the store dimension. The batch is
        checked before any row is stored. A failure leaves the store unchanged.
        """

    def search(
        self,
        vector: _Query,
        k: int,
        camera_ids: Collection[str] | None = None,
    ) -> list[tuple[AppearanceEmbedding, float]]:
        """Return up to ``k`` rows, cosine highest first.

        Equal scores are ordered by ``(camera_id, track_id, frame_number)``
        ascending. ``camera_ids`` is applied before that ranking. ``None``
        keeps every camera. A collection that matches no row returns ``[]``.
        ``k`` must be a positive int. A non-finite query, a zero query, or a
        query of the wrong length raises ``ValueError``.
        """

    def delete_where(
        self,
        derived_from: str | None = None,
        run_id: str | None = None,
    ) -> int:
        """Delete matching rows and return how many were deleted.

        At least one of ``derived_from`` and ``run_id`` is required. When
        both are set, a row must match both. Calling this with neither
        raises ``ValueError`` and deletes nothing.
        """

    def count(
        self,
        derived_from: str | None = None,
        run_id: str | None = None,
    ) -> int:
        """Count rows. An omitted filter matches every value in that column."""


class InMemoryVectorStore:
    """Float32 matrix with one parallel key per row.

    ``search`` keeps rows whose camera is allowed, ranks those rows by
    cosine, and then returns the top ``k``.
    """

    def __init__(self, dim: int) -> None:
        self._dim = _require_dim(dim)
        self._rows: list[_StoredRow] = []
        self._matrix = _empty_matrix(self._dim)

    @property
    def dim(self) -> int:
        """Length of every stored vector."""
        return self._dim

    def add(
        self,
        run_id: str,
        derived_from: str,
        embeddings: Sequence[AppearanceEmbedding],
    ) -> int:
        """Store one batch and return how many rows were stored."""
        run = _require_text("run_id", run_id)
        origin = _require_derived_from(derived_from)
        items = _require_embeddings(embeddings, self._dim)
        if not items:
            return 0
        added = tuple(_StoredRow(run, origin, item) for item in items)
        updated = [*self._rows, *added]
        matrix = _matrix_for(updated, self._dim)
        self._matrix = matrix
        self._rows = updated
        return len(items)

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
        indexes = _candidate_indexes(self._rows, cameras)
        if not indexes:
            return []
        scores = _cosine_scores(self._matrix, indexes, query)
        return _rank(self._rows, indexes, scores)[:limit]

    def delete_where(
        self,
        derived_from: str | None = None,
        run_id: str | None = None,
    ) -> int:
        """Delete matching rows. Both arguments omitted raises ``ValueError``."""
        origin = _optional_text("derived_from", derived_from)
        run = _optional_text("run_id", run_id)
        _require_delete_target(origin, run)
        kept = [row for row in self._rows if not _row_matches(row, origin, run)]
        removed = len(self._rows) - len(kept)
        if removed == 0:
            return 0
        self._matrix = _matrix_for(kept, self._dim)
        self._rows = kept
        return removed

    def count(
        self,
        derived_from: str | None = None,
        run_id: str | None = None,
    ) -> int:
        """Count rows. Both arguments omitted counts every row."""
        origin = _optional_text("derived_from", derived_from)
        run = _optional_text("run_id", run_id)
        if origin is None and run is None:
            return len(self._rows)
        return sum(1 for row in self._rows if _row_matches(row, origin, run))


def _as_vector_store(store: InMemoryVectorStore) -> VectorStore:
    """Return ``store`` as a ``VectorStore``.

    ``mypy src/ --strict`` checks this assignment.
    """
    checked: VectorStore = store
    return checked


def _require_dim(dim: object) -> int:
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        raise ValueError(f"dim must be a positive int, got {dim!r}")
    return dim


def _require_text(label: str, value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")
    return value


def _optional_text(label: str, value: object) -> str | None:
    if value is None:
        return None
    return _require_text(label, value)


def _require_derived_from(value: object) -> str:
    text = _require_text("derived_from", value)
    if text not in ALLOWED_DERIVED_FROM:
        allowed = ", ".join(sorted(ALLOWED_DERIVED_FROM))
        raise ValueError(f"derived_from must be one of {allowed}, got {text!r}")
    return text


def _require_k(k: object) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive int, got {k!r}")
    return k


def _require_embeddings(value: object, dim: int) -> tuple[AppearanceEmbedding, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("embeddings must be a sequence of AppearanceEmbedding")
    return tuple(_one_embedding(item, dim) for item in value)


def _one_embedding(value: object, dim: int) -> AppearanceEmbedding:
    if not isinstance(value, AppearanceEmbedding):
        raise TypeError("embeddings must be a sequence of AppearanceEmbedding")
    width = len(value.vector)
    if width != dim:
        raise ValueError(f"embedding dim must be {dim}, got {width}")
    return value


def _require_delete_target(derived_from: str | None, run_id: str | None) -> None:
    if derived_from is None and run_id is None:
        raise ValueError("delete_where requires derived_from or run_id")


def _row_matches(row: _StoredRow, derived_from: str | None, run_id: str | None) -> bool:
    if derived_from is not None and row.derived_from != derived_from:
        return False
    if run_id is not None and row.run_id != run_id:
        return False
    return True


def _empty_matrix(dim: int) -> npt.NDArray[np.float32]:
    empty = np.zeros((0, dim), dtype=np.float32)
    return cast(npt.NDArray[np.float32], empty)


def _matrix_for(rows: Sequence[_StoredRow], dim: int) -> npt.NDArray[np.float32]:
    if not rows:
        return _empty_matrix(dim)
    stacked = np.asarray([row.embedding.vector for row in rows], dtype=np.float32)
    return cast(npt.NDArray[np.float32], np.ascontiguousarray(stacked))


def _as_query(vector: object) -> npt.NDArray[np.float64]:
    if isinstance(vector, (str, bytes)) or vector is None:
        raise ValueError("query must be a vector of finite floats")
    try:
        array = np.asarray(vector, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("query must be a vector of finite floats") from exc
    if array.ndim != 1:
        shape = "x".join(str(int(size)) for size in array.shape)
        raise ValueError(f"query must be a 1-d vector, got shape {shape}")
    return cast(npt.NDArray[np.float64], array)


def _require_query(vector: object, dim: int) -> npt.NDArray[np.float64]:
    array = _as_query(vector)
    width = int(array.shape[0])
    if width != dim:
        raise ValueError(f"query dim must be {dim}, got {width}")
    _reject_non_finite(array)
    if float(np.linalg.norm(array)) == 0.0:
        raise ValueError("query must be non-zero")
    return array


def _reject_non_finite(array: npt.NDArray[np.float64]) -> None:
    if not bool(np.all(np.isfinite(array))):
        raise ValueError("query values must be finite")


def _require_camera_ids(value: object) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        kind = type(value).__name__
        raise ValueError(
            f"camera_ids must be a collection of camera ids, not a {kind}, got {value!r}"
        )
    if not isinstance(value, Collection):
        raise ValueError(f"camera_ids must be a collection of camera ids, got {value!r}")
    return frozenset(_require_camera_id(item) for item in value)


def _require_camera_id(value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"camera_id must be a non-empty string, got {value!r}")
    return value


def _candidate_indexes(
    rows: Sequence[_StoredRow],
    camera_ids: frozenset[str] | None,
) -> tuple[int, ...]:
    if camera_ids is None:
        return tuple(range(len(rows)))
    return tuple(index for index, row in enumerate(rows) if row.embedding.camera_id in camera_ids)


def _cosine_scores(
    matrix: npt.NDArray[np.float32],
    indexes: Sequence[int],
    query: npt.NDArray[np.float64],
) -> tuple[float, ...]:
    chosen = np.asarray(matrix[list(indexes)], dtype=np.float64)
    dots = chosen @ query
    row_norms = np.linalg.norm(chosen, axis=1)
    query_norm = float(np.linalg.norm(query))
    scaled = np.asarray(dots / (row_norms * query_norm), dtype=np.float64).reshape(-1)
    return tuple(float(value) for value in scaled)


def _rank(
    rows: Sequence[_StoredRow],
    indexes: Sequence[int],
    scores: Sequence[float],
) -> list[tuple[AppearanceEmbedding, float]]:
    order = sorted(range(len(indexes)), key=lambda pos: _sort_key(rows, indexes, scores, pos))
    return [(rows[indexes[pos]].embedding, scores[pos]) for pos in order]


def _sort_key(
    rows: Sequence[_StoredRow],
    indexes: Sequence[int],
    scores: Sequence[float],
    pos: int,
) -> tuple[float, str, int, int]:
    embedding = rows[indexes[pos]].embedding
    return (-scores[pos], embedding.camera_id, embedding.track_id, embedding.frame_number)
