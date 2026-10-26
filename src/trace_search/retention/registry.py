"""Registry of the footage-derived stores a purge walks.

``purgeables`` builds each store from one resolved data root. It does not
search the repository and it does not glob. The files under that directory
are:

- ``tracks.sqlite`` — ``TrackStore``, table ``track_observations``. This is
  the filename ``tracking.store`` already uses.
- ``attributes.sqlite`` — ``TrackAttributeStore``, table
  ``track_attributes``. The attribute store has no default filename of its
  own. The registry places the file at ``attributes.sqlite``.
- ``embeddings.sqlite`` — ``SqliteVectorStore``, table
  ``appearance_embeddings``. The embedding store has no default filename of
  its own. The registry places the file at ``embeddings.sqlite`` and opens
  it at a fixed vector width. That width is only the constructor argument.
  ``purge_expired`` and ``count_older_than`` do not read the vectors, so
  they still run when the file holds a different width.

The export store (T-W-8) is not on main, so it is not registered.

Paths are the resolved data root plus one of those filenames. A filename
that leaves the directory is refused. The track store's outside-repo guard
still runs on the resulting path, so a data root inside the repository is
refused and no file is created there.

``count_older_than`` counts with the same cutoff as ``purge_expired`` and
does not delete. On the attribute store, ``read``, ``query``, ``count``,
and ``write`` still purge first, so a count taken through those methods
would drop the expired rows and then return zero. On the embedding store,
``add``, ``search``, and ``count`` purge first. ``count_older_than`` does
not. That is why ``purgeables`` passes ``purge_on_open=False``: a fresh
registry can still count expired rows, and a later ``purge_expired``
deletes them.

Each store still purges on open by default, and that call throws the count
away. ``purgeables`` passes ``purge_on_open=False`` so a fresh registry
still sees the expired rows and a later purge can report them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from trace_search.attributes.store import TrackAttributeStore
from trace_search.reid.sqlite_vector_store import SqliteVectorStore
from trace_search.retention.config import FOOTAGE_DERIVED_DAYS
from trace_search.tracking.store import TrackStore, _db_path_outside_repo

TRACK_DB_FILENAME = "tracks.sqlite"
ATTRIBUTE_DB_FILENAME = "attributes.sqlite"
EMBEDDING_DB_FILENAME = "embeddings.sqlite"
# Constructor width for the registry embedding file. Purge and
# count_older_than do not read vectors, so this is not required to match
# the blobs already stored. ``add`` on this instance refuses a mismatch.
_EMBEDDING_DIM = 4

_Clock = Callable[[], float]


@runtime_checkable
class Purgeable(Protocol):
    """One footage-derived store the purge can count and delete."""

    @property
    def label(self) -> str:
        """Short name of this store."""

    @property
    def covered_tables(self) -> tuple[str, ...]:
        """SQLite tables whose ``created_at`` rows this store purges."""

    def purge_expired(
        self,
        now: float | None = None,
        max_age_days: int = FOOTAGE_DERIVED_DAYS,
    ) -> int:
        """Delete expired rows and return how many were deleted."""

    def count_older_than(
        self,
        now: float | None = None,
        max_age_days: int = FOOTAGE_DERIVED_DAYS,
    ) -> int:
        """Return how many rows ``purge_expired`` would delete. Do not delete them."""


def _resolved_data_root(data_root: Path) -> Path:
    if not isinstance(data_root, Path):
        raise TypeError(f"data_root must be a Path, got {data_root!r}")
    root = data_root.expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"data_root must be a directory, got {root}")
    return root


def _path_under_data_root(root: Path, filename: str) -> Path:
    """Join ``filename`` onto ``root`` and refuse a path that leaves it."""
    if filename == "" or filename != Path(filename).name:
        raise ValueError(f"store filename must not escape the data root, got {filename!r}")
    path = (root / filename).resolve()
    if path.parent != root:
        raise ValueError(f"store path must stay inside the data root {root}, got {path}")
    return _db_path_outside_repo(path)


def purgeables(data_root: Path, *, clock: _Clock | None = None) -> list[Purgeable]:
    """Return the footage-derived stores under ``data_root``.

    ``data_root`` is expanded and resolved. Each database path is that
    directory plus a fixed filename. Nothing outside that directory is
    opened or created. ``clock`` is passed through to each store. When it
    is omitted, the store uses ``time.time``.

    ``purge_on_open`` is False. The store constructors otherwise delete
    expired rows while they open and discard that count, so
    ``count_older_than`` on a fresh registry would return zero.

    The annotated assignments are the mypy check that each real store
    satisfies ``Purgeable``.
    """
    root = _resolved_data_root(data_root)
    track_path = _path_under_data_root(root, TRACK_DB_FILENAME)
    attribute_path = _path_under_data_root(root, ATTRIBUTE_DB_FILENAME)
    embedding_path = _path_under_data_root(root, EMBEDDING_DB_FILENAME)
    track: Purgeable = TrackStore(track_path, clock=clock, purge_on_open=False)
    attributes: Purgeable = TrackAttributeStore(
        attribute_path,
        clock=clock,
        purge_on_open=False,
    )
    embeddings: Purgeable = SqliteVectorStore(
        embedding_path,
        _EMBEDDING_DIM,
        clock=clock,
        purge_on_open=False,
    )
    return [track, attributes, embeddings]
