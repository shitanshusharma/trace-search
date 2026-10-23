"""SQLite store of per-track colour and bag attributes.

One table, ``track_attributes``, lives in a stdlib SQLite file. The path
must sit outside the repository tree. The guard is the track store helper
``_db_path_outside_repo``. This module calls that helper and does not copy
it. It does not open the track database. Callers that need a time window
pass ``Tracklet`` values in.

``created_at`` is a UTC epoch from the injected clock, or from
``time.time`` when no clock is injected. Rows older than the shipped footage
window (7 days) are deleted on open and before every ``read``, ``query``,
``count``, and ``write``. A row whose age is exactly the shipped footage
window (7 days) is kept. ``delete_where`` does not apply
that window.

The table stores colours, a bag flag, two confidences, and ``n_frames``.
It does not store image bytes or crops. ``write`` appends one row. It does
not replace an older row for the same track.

Ids are refused, not rewritten. ``camera_id``, ``run_id``, and
``derived_from`` raise ``ValueError`` when the value is empty or
whitespace-only, differs from ``str.strip`` of itself, changes under NFKC,
or contains a Unicode format character (category Cf, including U+200B,
U+200C, U+200D, U+2060, and U+FEFF). Nothing is stripped, case-folded, or
NFKC-folded. ``track_id`` is an int, as in ``TrackStore``. A bool, a
negative int, or any other non-int raises ``ValueError``. Text is not parsed.

``upper_color`` and ``lower_color`` must be a ``Color`` member or that
member's exact value. Any other type, any other text, a different case, or
surrounding whitespace raises ``ValueError``. The stored text is the
member value, unchanged.

``upper_confidence`` and ``lower_confidence`` must be a finite int or float
in the closed range 0 to 1. A bool or any other type raises ``TypeError``.
A non-finite value, or a value outside that range, raises ``ValueError``.
The number is not clamped. ``min_confidence`` on a query uses the same rule.

``n_frames`` must be an int greater than or equal to 1. A bool or any other
non-int raises ``TypeError``. A value below 1 raises ``ValueError``.

``bag_on_track`` must be a bool. ``True`` is stored as 1 and ``False`` as
0. Any other value, including the integers 0 and 1, raises ``TypeError``.
A stored value other than 0 or 1 raises ``ValueError`` on read. It is not
coerced.

``derived_from`` uses the same text-id rules, then the table CHECK. The
only stored values are ``SMOKE``, ``CODE_TEST``, and ``SITE``. Any other
text, including ``TRAIN``, raises ``sqlite3.IntegrityError`` from that
CHECK. Python does not rewrite it into an allowed value.

``query`` keeps a row only when three things are true. The row's
``(camera_id, track_id, run_id)`` equals a passed tracklet. That tracklet
overlaps ``t_window`` when a window is passed. ``AttributeQuery.matches``
accepts the row for the filter. The window is inclusive:
``tracklet.t_end >= start`` and ``tracklet.t_start <= end``. ``t_window``
of ``None`` does not apply a time limit. The tracklet join still applies.
``camera_ids`` of ``None`` does not narrow the cameras further. An empty
camera collection raises ``ValueError``. A bare ``str`` raises
``ValueError``, because a ``str`` is not a collection of camera ids.
Unknown filter keys raise ``ValueError``. The keys ``AttributeQuery``
understands are ``upper_color``, ``lower_color``, ``bag_on_track``, and
``min_confidence``.
"""

from __future__ import annotations

import math
import sqlite3
import time
import unicodedata
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

from trace_search.attributes.attributes import AttributeQuery, Color, PersonAttributes
from trace_search.matching.tracklets import Tracklet
from trace_search.retention.config import FOOTAGE_DERIVED_DAYS
from trace_search.tracking.store import _db_path_outside_repo

# Shipped footage-derived window. T-P-7 is the later override.
DEFAULT_MAX_AGE_DAYS = FOOTAGE_DERIVED_DAYS
_SECONDS_PER_DAY = 86400

_COLOR_VALUES = frozenset(member.value for member in Color)
_FILTER_KEYS = frozenset({"upper_color", "lower_color", "bag_on_track", "min_confidence"})

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS track_attributes (
    camera_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    derived_from TEXT NOT NULL CHECK (derived_from IN ('SMOKE','CODE_TEST','SITE')),
    upper_color TEXT NOT NULL,
    lower_color TEXT NOT NULL,
    bag_on_track INTEGER NOT NULL,
    upper_confidence REAL NOT NULL,
    lower_confidence REAL NOT NULL,
    n_frames INTEGER NOT NULL,
    created_at REAL NOT NULL
)
"""

_INDEX_CREATED_SQL = """
CREATE INDEX IF NOT EXISTS idx_track_attributes_created_at
ON track_attributes (created_at)
"""

_INDEX_CAMERA_SQL = """
CREATE INDEX IF NOT EXISTS idx_track_attributes_camera_run
ON track_attributes (camera_id, run_id)
"""

_INDEX_DERIVED_SQL = """
CREATE INDEX IF NOT EXISTS idx_track_attributes_derived_from
ON track_attributes (derived_from)
"""

_INSERT_SQL = """
INSERT INTO track_attributes (
    camera_id,
    track_id,
    run_id,
    derived_from,
    upper_color,
    lower_color,
    bag_on_track,
    upper_confidence,
    lower_confidence,
    n_frames,
    created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_COLUMNS_SQL = """
SELECT
    camera_id,
    track_id,
    run_id,
    derived_from,
    upper_color,
    lower_color,
    bag_on_track,
    upper_confidence,
    lower_confidence,
    n_frames,
    created_at
FROM track_attributes
"""

_READ_SQL = (
    _SELECT_COLUMNS_SQL
    + """
WHERE camera_id = ?
  AND (? IS NULL OR track_id = ?)
  AND (? IS NULL OR run_id = ?)
ORDER BY rowid ASC
"""
)

_QUERY_SQL = _SELECT_COLUMNS_SQL + "\nORDER BY rowid ASC\n"

_COUNT_SQL = """
SELECT COUNT(*)
FROM track_attributes
WHERE (? IS NULL OR camera_id = ?)
  AND (? IS NULL OR track_id = ?)
  AND (? IS NULL OR run_id = ?)
  AND (? IS NULL OR derived_from = ?)
"""

_PURGE_SQL = "DELETE FROM track_attributes WHERE created_at < ?"

_InsertRow = tuple[str, int, str, str, str, str, int, float, float, int, float]
_TrackKey = tuple[str, int, str]
_T = TypeVar("_T")
_Clock = Callable[[], float]


@dataclass(frozen=True)
class TrackAttributeRecord:
    """One stored attribute row.

    ``upper_color`` and ``lower_color`` are ``Color`` values. ``bag_on_track``
    is the bool that was written. ``created_at`` is the UTC epoch from the
    clock at ``write``.
    """

    camera_id: str
    track_id: int
    run_id: str
    derived_from: str
    upper_color: str
    lower_color: str
    bag_on_track: bool
    upper_confidence: float
    lower_confidence: float
    n_frames: int
    created_at: float


def _reject_empty(label: str, value: object) -> str:
    """Reject a missing id. Whitespace-only counts as empty. Do not strip it."""
    if isinstance(value, str) and value.strip() != "":
        return value
    raise ValueError(f"{label} must be a non-empty id, got {value!r}")


def _reject_strip(label: str, value: str) -> None:
    """Refuse surrounding whitespace. Do not strip the id."""
    if value != value.strip():
        raise ValueError(f"{label} must not have leading or trailing whitespace, got {value!r}")


def _reject_nfkc(label: str, value: str) -> None:
    """Refuse an id that NFKC would change. Do not fold the id."""
    if unicodedata.normalize("NFKC", value) != value:
        raise ValueError(f"{label} must already be in NFKC, got {value!r}")


def _reject_format(label: str, value: str) -> None:
    """Refuse Unicode format characters. Do not remove them."""
    for character in value:
        if unicodedata.category(character) == "Cf":
            raise ValueError(f"{label} must not contain a format character, got {value!r}")


def _require_text_id(label: str, value: object) -> str:
    """Return the id unchanged. Never strip, fold, or change case."""
    text = _reject_empty(label, value)
    _reject_strip(label, text)
    _reject_nfkc(label, text)
    _reject_format(label, text)
    return text


def _optional_text_id(label: str, value: str | None) -> str | None:
    if value is None:
        return None
    return _require_text_id(label, value)


def _require_track_id(value: object) -> int:
    """An int >= 0. A bool is refused. Text is not parsed."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"track_id must be an int >= 0, got {value!r}")
    if value < 0:
        raise ValueError(f"track_id must be an int >= 0, got {value!r}")
    return value


def _optional_track_id(value: int | None) -> int | None:
    if value is None:
        return None
    return _require_track_id(value)


def _require_color(label: str, value: object) -> str:
    """A ``Color`` member or its exact value. Case and spaces stay as given."""
    if isinstance(value, Color):
        return value.value
    if isinstance(value, str) and value in _COLOR_VALUES:
        return value
    raise ValueError(f"{label} must be a Color value, got {value!r}")


def _require_confidence(label: str, value: object) -> float:
    """A finite int or float in [0, 1]. A bool is refused. Do not clamp."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number in [0, 1], got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{label} must be in [0, 1], got {value!r}")
    return number


def _require_n_frames(value: object) -> int:
    """An int >= 1. A bool is refused. The value is not coerced."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"n_frames must be an int, got {value!r}")
    if value < 1:
        raise ValueError(f"n_frames must be >= 1, got {value!r}")
    return value


def _require_bag(value: object) -> int:
    """A bool, stored as 0 or 1. The integers 0 and 1 are refused."""
    if isinstance(value, bool):
        return 1 if value else 0
    raise TypeError(f"bag_on_track must be a bool, got {value!r}")


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


def _reject_bare_str(camera_ids: object) -> None:
    """A str would iterate as characters. That is not a list of camera ids."""
    if isinstance(camera_ids, str):
        raise ValueError(
            f"camera_ids must be a collection of camera ids, not a str, got {camera_ids!r}"
        )
    if isinstance(camera_ids, bytes):
        raise ValueError(f"camera_ids must be a collection of camera ids, got {camera_ids!r}")


def _require_camera_ids(camera_ids: object) -> tuple[str, ...] | None:
    """Validate camera ids. ``None`` does not narrow the join."""
    if camera_ids is None:
        return None
    _reject_bare_str(camera_ids)
    if not isinstance(camera_ids, Collection):
        raise ValueError(f"camera_ids must be a collection of camera ids, got {camera_ids!r}")
    if len(camera_ids) == 0:
        raise ValueError("camera_ids must contain at least one camera id")
    unique: dict[str, None] = {}
    for camera_id in camera_ids:
        unique.setdefault(_require_text_id("camera_id", camera_id), None)
    return tuple(unique)


def _require_t_window(window: object) -> tuple[float, float] | None:
    """``None``, or an inclusive ``(start, end)`` pair of finite epochs."""
    if window is None:
        return None
    if isinstance(window, (str, bytes)) or not isinstance(window, Sequence):
        raise TypeError(f"t_window must be a pair of epochs or None, got {window!r}")
    if len(window) != 2:
        raise ValueError(f"t_window must be a pair of epochs, got {window!r}")
    start = _require_timestamp(window[0])
    end = _require_timestamp(window[1])
    if start > end:
        raise ValueError(f"t_window start must be <= end, got {window!r}")
    return (start, end)


def _require_tracklets(value: object) -> tuple[Tracklet, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"tracklets must be a sequence of Tracklet, got {value!r}")
    items: list[Tracklet] = []
    for item in value:
        if not isinstance(item, Tracklet):
            raise TypeError(f"tracklets must be a sequence of Tracklet, got {item!r}")
        items.append(item)
    return tuple(items)


def _reject_unknown_filter_keys(value: Mapping[object, object]) -> None:
    unknown = [str(key) for key in value if key not in _FILTER_KEYS]
    if unknown:
        raise ValueError(f"attribute_filter has unknown keys: {unknown!r}")


def _put_color_filter(source: Mapping[object, object], key: str, built: dict[str, object]) -> None:
    if key in source:
        built[key] = _require_color(key, source[key])


def _put_bag_filter(source: Mapping[object, object], built: dict[str, object]) -> None:
    if "bag_on_track" not in source:
        return
    bag = source["bag_on_track"]
    if not isinstance(bag, bool):
        raise TypeError(f"bag_on_track must be a bool, got {bag!r}")
    built["bag_on_track"] = bag


def _put_confidence_filter(source: Mapping[object, object], built: dict[str, object]) -> None:
    if "min_confidence" not in source:
        return
    built["min_confidence"] = _require_confidence("min_confidence", source["min_confidence"])


def _require_attribute_filter(value: object) -> dict[str, object]:
    """Copy a filter ``AttributeQuery`` can match. Unknown keys are refused."""
    if not isinstance(value, Mapping):
        raise TypeError(f"attribute_filter must be a mapping, got {value!r}")
    _reject_unknown_filter_keys(value)
    built: dict[str, object] = {}
    _put_color_filter(value, "upper_color", built)
    _put_color_filter(value, "lower_color", built)
    _put_bag_filter(value, built)
    _put_confidence_filter(value, built)
    return built


def _window_overlaps(tracklet: Tracklet, window: tuple[float, float] | None) -> bool:
    """Inclusive overlap. ``None`` does not apply a time limit."""
    if window is None:
        return True
    start, end = window
    return tracklet.t_end >= start and tracklet.t_start <= end


def _select_keys(
    tracklets: Sequence[Tracklet],
    camera_ids: tuple[str, ...] | None,
    window: tuple[float, float] | None,
) -> set[_TrackKey]:
    allowed = None if camera_ids is None else set(camera_ids)
    keys: set[_TrackKey] = set()
    for tracklet in tracklets:
        if allowed is not None and tracklet.camera_id not in allowed:
            continue
        if not _window_overlaps(tracklet, window):
            continue
        keys.add((tracklet.camera_id, tracklet.track_id, tracklet.run_id))
    return keys


def _bag_bool(value: object) -> bool:
    """Map a stored 0 or 1 back to a bool. Any other integer is refused."""
    if value == 0:
        return False
    if value == 1:
        return True
    raise ValueError(f"bag_on_track must be stored as 0 or 1, got {value!r}")


def _record_from_row(row: sqlite3.Row) -> TrackAttributeRecord:
    return TrackAttributeRecord(
        camera_id=str(row["camera_id"]),
        track_id=int(row["track_id"]),
        run_id=str(row["run_id"]),
        derived_from=str(row["derived_from"]),
        upper_color=str(row["upper_color"]),
        lower_color=str(row["lower_color"]),
        bag_on_track=_bag_bool(row["bag_on_track"]),
        upper_confidence=float(row["upper_confidence"]),
        lower_confidence=float(row["lower_confidence"]),
        n_frames=int(row["n_frames"]),
        created_at=float(row["created_at"]),
    )


def _attributes_match(record: TrackAttributeRecord, attribute_filter: Mapping[str, object]) -> bool:
    attrs = PersonAttributes(
        upper_clothing_color=Color(record.upper_color),
        lower_clothing_color=Color(record.lower_color),
        bag_on_track=record.bag_on_track,
        upper_confidence=record.upper_confidence,
        lower_confidence=record.lower_confidence,
    )
    return AttributeQuery.matches(attrs, attribute_filter)


def _row_matches_query(
    row: sqlite3.Row,
    keys: set[_TrackKey],
    attribute_filter: Mapping[str, object],
) -> TrackAttributeRecord | None:
    record = _record_from_row(row)
    key = (record.camera_id, record.track_id, record.run_id)
    if key not in keys:
        return None
    if not _attributes_match(record, attribute_filter):
        return None
    return record


def _delete_matching(
    connection: sqlite3.Connection,
    derived_from: str | None,
    run_id: str | None,
) -> int:
    if derived_from is not None and run_id is not None:
        cursor = connection.execute(
            "DELETE FROM track_attributes WHERE derived_from = ? AND run_id = ?",
            (derived_from, run_id),
        )
        return int(cursor.rowcount)
    if derived_from is not None:
        cursor = connection.execute(
            "DELETE FROM track_attributes WHERE derived_from = ?",
            (derived_from,),
        )
        return int(cursor.rowcount)
    if run_id is not None:
        cursor = connection.execute(
            "DELETE FROM track_attributes WHERE run_id = ?",
            (run_id,),
        )
        return int(cursor.rowcount)
    raise ValueError("delete_where requires derived_from or run_id")


class TrackAttributeStore:
    """SQLite store of per-track colour and bag rows.

    Construction opens the file, creates ``track_attributes`` when missing,
    and purges rows older than the shipped footage window (7 days).
    ``read``, ``query``, ``count``, and ``write`` purge again before they
    touch the table. The clock is injectable so retention tests do not sleep.
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
        camera_id: str,
        track_id: int,
        run_id: str,
        derived_from: str,
        upper_color: str,
        lower_color: str,
        bag_on_track: bool,
        upper_confidence: float,
        lower_confidence: float,
        n_frames: int,
    ) -> None:
        """Append one attribute row after purging expired rows.

        ``derived_from`` must be ``SMOKE``, ``CODE_TEST``, or ``SITE``. Any
        other text fails the table CHECK with ``sqlite3.IntegrityError``.
        ``created_at`` is the UTC epoch from the injected clock. A second
        write for the same track appends another row.
        """
        self.purge_expired()
        row = _insert_row(
            camera_id,
            track_id,
            run_id,
            derived_from,
            upper_color,
            lower_color,
            bag_on_track,
            upper_confidence,
            lower_confidence,
            n_frames,
            _require_timestamp(self._clock()),
        )

        def _insert(connection: sqlite3.Connection) -> None:
            connection.execute(_INSERT_SQL, row)

        self._using(_insert)

    def read(
        self,
        camera_id: str,
        track_id: int | None = None,
        run_id: str | None = None,
    ) -> list[TrackAttributeRecord]:
        """Return one camera's rows, optionally limited to track and run.

        Expired rows are purged first. This does not insert or replace a row.
        Omitted ``track_id`` and ``run_id`` match every value in that column.
        Results are in ``rowid`` order.
        """
        self.purge_expired()
        camera = _require_text_id("camera_id", camera_id)
        track = _optional_track_id(track_id)
        run = _optional_text_id("run_id", run_id)

        def _query(connection: sqlite3.Connection) -> list[TrackAttributeRecord]:
            cursor = connection.execute(_READ_SQL, (camera, track, track, run, run))
            return [_record_from_row(row) for row in cursor.fetchall()]

        return self._using(_query)

    def query(
        self,
        attribute_filter: Mapping[str, object],
        camera_ids: Collection[str] | None,
        t_window: tuple[float, float] | None,
        tracklets: Sequence[Tracklet],
    ) -> list[TrackAttributeRecord]:
        """Return rows that join to ``tracklets`` and match the filter.

        Expired rows are purged first. This does not open the track store
        and does not insert or replace a row. ``attribute_filter`` uses the
        ``AttributeQuery`` keys. ``camera_ids`` narrows that join. ``None``
        does not. ``t_window`` is an inclusive overlap on the passed
        tracklets, or ``None`` for no time limit. Results are in ``rowid``
        order.
        """
        self.purge_expired()
        checked_filter = _require_attribute_filter(attribute_filter)
        checked_cameras = _require_camera_ids(camera_ids)
        checked_window = _require_t_window(t_window)
        checked_tracklets = _require_tracklets(tracklets)
        keys = _select_keys(checked_tracklets, checked_cameras, checked_window)

        def _query(connection: sqlite3.Connection) -> list[TrackAttributeRecord]:
            cursor = connection.execute(_QUERY_SQL)
            matched: list[TrackAttributeRecord] = []
            for row in cursor.fetchall():
                record = _row_matches_query(row, keys, checked_filter)
                if record is not None:
                    matched.append(record)
            return matched

        return self._using(_query)

    def count(
        self,
        *,
        camera_id: str | None = None,
        track_id: int | None = None,
        run_id: str | None = None,
        derived_from: str | None = None,
    ) -> int:
        """Count rows after purging expired ones.

        Omitted filters match every value in that column. This does not
        insert or replace a row.
        """
        self.purge_expired()
        camera = _optional_text_id("camera_id", camera_id)
        track = _optional_track_id(track_id)
        run = _optional_text_id("run_id", run_id)
        origin = _optional_text_id("derived_from", derived_from)

        def _query(connection: sqlite3.Connection) -> int:
            cursor = connection.execute(
                _COUNT_SQL,
                (camera, camera, track, track, run, run, origin, origin),
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
        both are set, a row must match both. This does not apply the shipped
        footage window (7 days). A ``CODE_TEST`` row stays when the filter is ``SMOKE``.
        """
        origin = _optional_text_id("derived_from", derived_from)
        run = _optional_text_id("run_id", run_id)

        def _query(connection: sqlite3.Connection) -> int:
            return _delete_matching(connection, origin, run)

        return self._using(_query)

    def purge_expired(
        self,
        now: float | None = None,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    ) -> int:
        """Delete rows whose ``created_at`` is older than ``max_age_days``.

        The default window is the shipped footage window (7 days). ``now``
        is a UTC epoch. When it is omitted, the injected clock supplies it.
        A row is kept when its age is exactly the window.
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


def _insert_row(
    camera_id: object,
    track_id: object,
    run_id: object,
    derived_from: object,
    upper_color: object,
    lower_color: object,
    bag_on_track: object,
    upper_confidence: object,
    lower_confidence: object,
    n_frames: object,
    created_at: float,
) -> _InsertRow:
    return (
        _require_text_id("camera_id", camera_id),
        _require_track_id(track_id),
        _require_text_id("run_id", run_id),
        _require_text_id("derived_from", derived_from),
        _require_color("upper_color", upper_color),
        _require_color("lower_color", lower_color),
        _require_bag(bag_on_track),
        _require_confidence("upper_confidence", upper_confidence),
        _require_confidence("lower_confidence", lower_confidence),
        _require_n_frames(n_frames),
        created_at,
    )
