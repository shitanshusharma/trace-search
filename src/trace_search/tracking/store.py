"""Local store of per-camera track boxes.

Rows live in one stdlib SQLite table so a later reader can load them and a
retention purge can delete them. The table holds boxes and track fields
only. It does not store image bytes, crops, or embeddings.

Pass ``db_path`` to ``TrackStore``, or build one from config.
``TRACE_SEARCH_TRACK_DB`` is the config value. When it is unset, the file
is ``tracks.sqlite`` under ``$XDG_DATA_HOME/trace-search`` or
``~/.local/share/trace-search``. That directory sits outside the repository
tree. The gitignored ``data/`` directory inside the checkout is not used.
"""

from __future__ import annotations

import math
import os
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

from trace_search.detection.types import DetClass
from trace_search.tracking.types import TrackObservation

TRACK_DB_ENV = "TRACE_SEARCH_TRACK_DB"
DEFAULT_MAX_AGE_DAYS = 7
_SECONDS_PER_DAY = 86400
_DATA_DIR_NAME = "trace-search"
_DB_FILENAME = "tracks.sqlite"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS track_observations (
    camera_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    det_class TEXT NOT NULL,
    frame_number INTEGER NOT NULL,
    timestamp_seconds REAL NOT NULL,
    box_x1 REAL NOT NULL,
    box_y1 REAL NOT NULL,
    box_x2 REAL NOT NULL,
    box_y2 REAL NOT NULL,
    score REAL NOT NULL,
    run_id TEXT NOT NULL,
    source_fixture_id TEXT NOT NULL,
    derived_from TEXT NOT NULL CHECK (derived_from IN ('SMOKE','CODE_TEST','SITE')),
    created_at REAL NOT NULL
)
"""

_INDEX_CREATED_SQL = """
CREATE INDEX IF NOT EXISTS idx_track_observations_created_at
ON track_observations (created_at)
"""

_INDEX_CAMERA_SQL = """
CREATE INDEX IF NOT EXISTS idx_track_observations_camera_run
ON track_observations (camera_id, run_id)
"""

_INDEX_DERIVED_SQL = """
CREATE INDEX IF NOT EXISTS idx_track_observations_derived_from
ON track_observations (derived_from)
"""

_INSERT_SQL = """
INSERT INTO track_observations (
    camera_id,
    track_id,
    det_class,
    frame_number,
    timestamp_seconds,
    box_x1,
    box_y1,
    box_x2,
    box_y2,
    score,
    run_id,
    source_fixture_id,
    derived_from,
    created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_READ_SQL = """
SELECT
    camera_id,
    track_id,
    det_class,
    frame_number,
    timestamp_seconds,
    box_x1,
    box_y1,
    box_x2,
    box_y2,
    score
FROM track_observations
WHERE camera_id = ?
  AND (? IS NULL OR run_id = ?)
ORDER BY rowid ASC
"""

_COUNT_SQL = """
SELECT COUNT(*)
FROM track_observations
WHERE (? IS NULL OR camera_id = ?)
  AND (? IS NULL OR run_id = ?)
  AND (? IS NULL OR derived_from = ?)
"""

_PURGE_SQL = "DELETE FROM track_observations WHERE created_at < ?"

_InsertRow = tuple[
    str,
    int,
    str,
    int,
    float,
    float,
    float,
    float,
    float,
    float,
    str,
    str,
    str,
    float,
]

_T = TypeVar("_T")
_Clock = Callable[[], float]


@dataclass(frozen=True)
class TrackStoreConfig:
    """Track database location from config or the default data directory."""

    db_path: Path


def default_db_path() -> Path:
    """Return the configured database path, else one outside the repo tree."""
    configured = os.environ.get(TRACK_DB_ENV)
    if configured is not None and configured.strip() != "":
        return Path(configured).expanduser()
    return _default_data_dir() / _DB_FILENAME


def load_config() -> TrackStoreConfig:
    """Load the track database path from config, else the default data dir."""
    return TrackStoreConfig(db_path=default_db_path())


def _default_data_dir() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home is not None and data_home.strip() != "":
        return Path(data_home).expanduser() / _DATA_DIR_NAME
    return Path.home() / ".local" / "share" / _DATA_DIR_NAME


def _repo_root() -> Path | None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        package = parent / "src" / "trace_search" / "tracking" / "types.py"
        if (parent / "pyproject.toml").is_file() and package.is_file():
            return parent
    return None


def _db_path_outside_repo(db_path: Path) -> Path:
    resolved = Path(db_path).expanduser().resolve()
    root = _repo_root()
    if root is not None and (resolved == root or root in resolved.parents):
        raise ValueError(f"track db must sit outside the repo tree: {resolved}")
    return resolved


def _require_text(label: str, value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")
    return value


def _require_timestamp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"timestamp must be a real UTC epoch value, got {value!r}")
    timestamp = float(value)
    if not math.isfinite(timestamp):
        raise ValueError(f"timestamp must be a finite UTC epoch value, got {value!r}")
    return timestamp


def _retention_seconds(max_age_days: int) -> int:
    if isinstance(max_age_days, bool) or not isinstance(max_age_days, int):
        raise TypeError(f"max_age_days must be an int >= 0, got {max_age_days!r}")
    if max_age_days < 0:
        raise ValueError(f"max_age_days must be an int >= 0, got {max_age_days!r}")
    return max_age_days * _SECONDS_PER_DAY


def _require_observations(observations: object) -> Sequence[TrackObservation]:
    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        raise TypeError("observations must be a sequence of TrackObservation")
    items: list[TrackObservation] = []
    for item in observations:
        if not isinstance(item, TrackObservation):
            raise TypeError("observations must be a sequence of TrackObservation")
        items.append(item)
    return items


def _insert_row(
    observation: TrackObservation,
    run_id: str,
    fixture_id: str,
    derived_from: str,
    created_at: float,
) -> _InsertRow:
    x1, y1, x2, y2 = observation.box
    return (
        observation.camera_id,
        observation.track_id,
        observation.det_class.value,
        observation.frame_number,
        observation.timestamp_seconds,
        float(x1),
        float(y1),
        float(x2),
        float(y2),
        observation.score,
        run_id,
        fixture_id,
        derived_from,
        created_at,
    )


def _observation_from_row(row: sqlite3.Row) -> TrackObservation:
    return TrackObservation(
        camera_id=str(row["camera_id"]),
        track_id=int(row["track_id"]),
        det_class=DetClass(str(row["det_class"])),
        frame_number=int(row["frame_number"]),
        timestamp_seconds=float(row["timestamp_seconds"]),
        box=(
            float(row["box_x1"]),
            float(row["box_y1"]),
            float(row["box_x2"]),
            float(row["box_y2"]),
        ),
        score=float(row["score"]),
    )


def _delete_matching(
    connection: sqlite3.Connection,
    derived_from: str | None,
    run_id: str | None,
) -> int:
    if derived_from is not None and run_id is not None:
        cursor = connection.execute(
            "DELETE FROM track_observations WHERE derived_from = ? AND run_id = ?",
            (derived_from, run_id),
        )
        return int(cursor.rowcount)
    if derived_from is not None:
        cursor = connection.execute(
            "DELETE FROM track_observations WHERE derived_from = ?",
            (derived_from,),
        )
        return int(cursor.rowcount)
    if run_id is not None:
        cursor = connection.execute(
            "DELETE FROM track_observations WHERE run_id = ?",
            (run_id,),
        )
        return int(cursor.rowcount)
    raise ValueError("delete_where requires derived_from or run_id")


class TrackStore:
    """SQLite store of ``TrackObservation`` rows for one local data directory.

    Construction opens the file, creates ``track_observations`` when missing,
    and purges rows older than 7 days. ``write`` purges again before it
    inserts. The clock is injectable so retention tests do not sleep.
    """

    def __init__(self, db_path: Path, *, clock: _Clock | None = None) -> None:
        resolved = _db_path_outside_repo(db_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.db_path: Path = resolved
        self._clock: _Clock = time.time if clock is None else clock
        self._ensure_schema()
        self.purge_expired()

    def write(
        self,
        run_id: str,
        fixture_id: str,
        derived_from: str,
        observations: Sequence[TrackObservation],
    ) -> None:
        """Insert observations for one run after purging expired rows.

        ``fixture_id`` is stored as ``source_fixture_id``. ``derived_from``
        must be ``SMOKE``, ``CODE_TEST``, or ``SITE``. Any other value fails
        the table check with ``sqlite3.IntegrityError``. ``created_at`` is
        the UTC epoch time from the injected clock.
        """
        run = _require_text("run_id", run_id)
        fixture = _require_text("fixture_id", fixture_id)
        origin = _require_text("derived_from", derived_from)
        items = _require_observations(observations)
        self.purge_expired()
        created_at = _require_timestamp(self._clock())
        rows = tuple(_insert_row(item, run, fixture, origin, created_at) for item in items)

        def _insert(connection: sqlite3.Connection) -> None:
            connection.executemany(_INSERT_SQL, rows)

        self._using(_insert)

    def read(self, camera_id: str, run_id: str | None = None) -> list[TrackObservation]:
        """Return one camera's observations, optionally limited to ``run_id``."""
        camera = _require_text("camera_id", camera_id)
        run = _require_text("run_id", run_id) if run_id is not None else None

        def _query(connection: sqlite3.Connection) -> list[TrackObservation]:
            cursor = connection.execute(_READ_SQL, (camera, run, run))
            return [_observation_from_row(row) for row in cursor.fetchall()]

        return self._using(_query)

    def count(
        self,
        *,
        camera_id: str | None = None,
        run_id: str | None = None,
        derived_from: str | None = None,
    ) -> int:
        """Count rows. Omitted filters match every value in that column."""
        camera = _require_text("camera_id", camera_id) if camera_id is not None else None
        run = _require_text("run_id", run_id) if run_id is not None else None
        origin = _require_text("derived_from", derived_from) if derived_from is not None else None

        def _query(connection: sqlite3.Connection) -> int:
            cursor = connection.execute(
                _COUNT_SQL,
                (camera, camera, run, run, origin, origin),
            )
            # COUNT(*) returns one row. The cast records that invariant.
            fetched = cast(sqlite3.Row, cursor.fetchone())
            return int(fetched[0])

        return self._using(_query)

    def delete_where(
        self,
        *,
        derived_from: str | None = None,
        run_id: str | None = None,
    ) -> int:
        """Delete rows matching the given filters and return how many went.

        At least one of ``derived_from`` and ``run_id`` is required. When
        both are set, a row must match both.
        """
        origin = _require_text("derived_from", derived_from) if derived_from is not None else None
        run = _require_text("run_id", run_id) if run_id is not None else None

        def _query(connection: sqlite3.Connection) -> int:
            return _delete_matching(connection, origin, run)

        return self._using(_query)

    def purge_expired(
        self,
        now: float | None = None,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    ) -> int:
        """Delete rows whose ``created_at`` is older than ``max_age_days``.

        The default window is 7 days. ``now`` is a UTC epoch timestamp. When
        it is omitted, the injected clock supplies it. A row is kept when its
        age is exactly the window.
        """
        current = _require_timestamp(self._clock() if now is None else now)
        cutoff = current - float(_retention_seconds(max_age_days))

        def _query(connection: sqlite3.Connection) -> int:
            cursor = connection.execute(_PURGE_SQL, (cutoff,))
            return int(cursor.rowcount)

        return self._using(_query)

    def _ensure_schema(self) -> None:
        def _create(connection: sqlite3.Connection) -> None:
            connection.execute(_CREATE_SQL)
            connection.execute(_INDEX_CREATED_SQL)
            connection.execute(_INDEX_CAMERA_SQL)
            connection.execute(_INDEX_DERIVED_SQL)

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
