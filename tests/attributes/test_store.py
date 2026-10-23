"""Tests for the per-track attribute store."""

import inspect
import sqlite3
import time
from pathlib import Path

import pytest

from trace_search.attributes.attributes import Color
from trace_search.attributes.store import (
    DEFAULT_MAX_AGE_DAYS,
    TrackAttributeRecord,
    TrackAttributeStore,
)
from trace_search.detection.types import DetClass
from trace_search.matching.tracklets import Tracklet

_DAY = 86400.0
_EXACT_COLUMNS = (
    ("camera_id", "TEXT"),
    ("track_id", "INTEGER"),
    ("run_id", "TEXT"),
    ("derived_from", "TEXT"),
    ("upper_color", "TEXT"),
    ("lower_color", "TEXT"),
    ("bag_on_track", "INTEGER"),
    ("upper_confidence", "REAL"),
    ("lower_confidence", "REAL"),
    ("n_frames", "INTEGER"),
    ("created_at", "REAL"),
)
# Built from parts so this file does not contain a banned token.
_FORBIDDEN_COLUMNS = (
    "gender",
    "age",
    "face" + "_crop",
    "identity" + "_id",
    "global" + "_id",
    "person" + "_id",
)
_CHECK_SQL = "CHECK (derived_from IN ('SMOKE','CODE_TEST','SITE'))"
_INSERT_COLUMNS = """
INSERT INTO track_attributes (
    camera_id, track_id, run_id, derived_from, upper_color, lower_color,
    bag_on_track, upper_confidence, lower_confidence, n_frames, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class _Clock:
    """Mutable UTC epoch clock. Tests move it; nothing sleeps."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _store(tmp_path: Path, clock: _Clock | None = None) -> TrackAttributeStore:
    return TrackAttributeStore(tmp_path / "attributes.sqlite", clock=clock)


def _write(
    store: TrackAttributeStore,
    *,
    camera_id: str = "cam-1",
    track_id: int = 1,
    run_id: str = "run-1",
    derived_from: str = "CODE_TEST",
    upper_color: str = "red",
    lower_color: str = "blue",
    bag_on_track: bool = False,
    upper_confidence: float = 0.5,
    lower_confidence: float = 0.75,
    n_frames: int = 3,
) -> None:
    store.write(
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
    )


def _expected(
    created_at: float,
    *,
    camera_id: str = "cam-1",
    track_id: int = 1,
    run_id: str = "run-1",
    derived_from: str = "CODE_TEST",
    upper_color: str = "red",
    lower_color: str = "blue",
    bag_on_track: bool = False,
    upper_confidence: float = 0.5,
    lower_confidence: float = 0.75,
    n_frames: int = 3,
) -> TrackAttributeRecord:
    return TrackAttributeRecord(
        camera_id=camera_id,
        track_id=track_id,
        run_id=run_id,
        derived_from=derived_from,
        upper_color=upper_color,
        lower_color=lower_color,
        bag_on_track=bag_on_track,
        upper_confidence=upper_confidence,
        lower_confidence=lower_confidence,
        n_frames=n_frames,
        created_at=created_at,
    )


def _tracklet(
    *,
    camera_id: str = "cam-1",
    track_id: int = 1,
    run_id: str = "run-1",
    t_start: float = 0.0,
    t_end: float = 10.0,
    det_class: DetClass = DetClass.PERSON,
) -> Tracklet:
    box = (0.0, 0.0, 2.0, 4.0)
    return Tracklet(
        camera_id=camera_id,
        track_id=track_id,
        det_class=det_class,
        run_id=run_id,
        t_start=t_start,
        t_end=t_end,
        n_obs=1,
        first_box=box,
        last_box=box,
        frames=(0,),
    )


def _raw_rows(path: Path) -> list[tuple[object, ...]]:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            """
            SELECT camera_id, track_id, run_id, derived_from, upper_color, lower_color,
                   bag_on_track, upper_confidence, lower_confidence, n_frames, created_at
            FROM track_attributes
            ORDER BY rowid ASC
            """
        ).fetchall()
    finally:
        connection.close()
    return [tuple(row) for row in rows]


def _raw_count(path: Path) -> int:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute("SELECT COUNT(*) FROM track_attributes").fetchone()
    finally:
        connection.close()
    assert row is not None
    return int(row[0])


def test_write_read_round_trip(tmp_path: Path) -> None:
    clock = _Clock(42.5)
    store = _store(tmp_path, clock)
    _write(store, bag_on_track=True, n_frames=4)
    expected = _expected(42.5, bag_on_track=True, n_frames=4)

    assert store.read("cam-1") == [expected]
    assert store.read("cam-1", track_id=1, run_id="run-1") == [expected]
    assert store.count(camera_id="cam-1", track_id=1, run_id="run-1", derived_from="CODE_TEST") == 1
    assert store.db_path == (tmp_path / "attributes.sqlite").resolve()
    assert _repo_root() not in store.db_path.parents

    connection = sqlite3.connect(store.db_path)
    try:
        stored = connection.execute(
            """
            SELECT typeof(camera_id), typeof(track_id), typeof(bag_on_track),
                   typeof(upper_confidence), typeof(lower_confidence),
                   typeof(n_frames), typeof(created_at)
            FROM track_attributes
            """
        ).fetchone()
    finally:
        connection.close()
    assert stored == ("text", "integer", "integer", "real", "real", "integer", "real")


def test_color_members_and_exact_values_round_trip(tmp_path: Path) -> None:
    clock = _Clock(5.0)
    store = _store(tmp_path, clock)
    for index, color in enumerate(Color):
        _write(
            store,
            track_id=index,
            upper_color=color,
            lower_color=color.value,
            upper_confidence=0.0,
            lower_confidence=1.0,
            n_frames=1,
        )
    rows = store.read("cam-1")
    assert len(rows) == len(list(Color))
    assert rows[0].upper_color == Color.BLACK.value
    assert rows[0].lower_color == "black"
    assert rows[0].upper_confidence == 0.0
    assert rows[0].lower_confidence == 1.0
    assert rows[-1].upper_color == Color.INVALID_BOX.value


def test_write_appends_a_second_row_for_the_same_track(tmp_path: Path) -> None:
    clock = _Clock(1.0)
    store = _store(tmp_path, clock)
    _write(store, upper_color="red")
    clock.now = 2.0
    _write(store, upper_color="blue")
    found = store.read("cam-1")
    assert [row.upper_color for row in found] == ["red", "blue"]
    assert [row.created_at for row in found] == [1.0, 2.0]


def test_read_filters_by_track_and_run(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(5.0))
    _write(store, camera_id="cam-a", track_id=1, run_id="run-a")
    _write(store, camera_id="cam-a", track_id=2, run_id="run-a", upper_color="green")
    _write(store, camera_id="cam-a", track_id=1, run_id="run-b", upper_color="white")

    assert [row.track_id for row in store.read("cam-a")] == [1, 2, 1]
    assert [row.run_id for row in store.read("cam-a", run_id="run-b")] == ["run-b"]
    assert [row.track_id for row in store.read("cam-a", track_id=2)] == [2]
    assert store.read("cam-a", track_id=1, run_id="missing") == []
    assert store.read("cam-missing") == []


def test_ids_are_stored_exactly(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(3.0))
    _write(store, camera_id="Cam-A", run_id="Run/A", track_id=0)
    _write(store, camera_id="cam 1", run_id="run 1", track_id=4)
    assert store.read("Cam-A")[0].camera_id == "Cam-A"
    assert store.read("Cam-A")[0].run_id == "Run/A"
    assert store.read("cam-a") == []
    assert store.read("cam 1")[0].run_id == "run 1"
    assert store.count(run_id="run-a") == 0
    assert store.count(run_id="Run/A") == 1


def test_allowed_derived_from_values_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(5.0))
    for origin in ("SMOKE", "CODE_TEST", "SITE"):
        _write(store, camera_id=f"cam-{origin}", run_id=f"run-{origin}", derived_from=origin)
    assert store.count() == 3
    with pytest.raises(sqlite3.IntegrityError):
        _write(store, camera_id="cam-lower", derived_from="smoke")
    assert store.count(camera_id="cam-lower") == 0


def test_derived_from_train_raises_integrity_error(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(10.0))
    _write(store)
    with pytest.raises(sqlite3.IntegrityError):
        _write(store, camera_id="cam-train", derived_from="TRAIN")
    assert store.count() == 1
    assert store.read("cam-train") == []
    assert store.read("cam-1") == [_expected(10.0)]


def test_raw_insert_of_train_raises_integrity_error(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(10.0))
    connection = sqlite3.connect(store.db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                _INSERT_COLUMNS,
                (
                    "cam-1",
                    1,
                    "run-raw",
                    "TRAIN",
                    "red",
                    "blue",
                    0,
                    0.5,
                    0.5,
                    1,
                    10.0,
                ),
            )
        connection.rollback()
    finally:
        connection.close()
    assert store.count() == 0


def test_pragma_table_info_exact_columns_and_no_blob(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(0.0))
    connection = sqlite3.connect(store.db_path)
    try:
        info = connection.execute("PRAGMA table_info(track_attributes)").fetchall()
        ddl_row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'track_attributes'"
        ).fetchone()
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    finally:
        connection.close()

    assert ddl_row is not None
    names_and_types = [(row[1], row[2]) for row in info]
    names = [row[1] for row in info]
    assert names_and_types == list(_EXACT_COLUMNS)
    assert set(names) == {name for name, _col_type in _EXACT_COLUMNS}
    assert all(col_type.upper() != "BLOB" for _name, col_type in names_and_types)
    assert "BLOB" not in ddl_row[0].upper()
    assert _CHECK_SQL in ddl_row[0]
    assert set(names).isdisjoint(_FORBIDDEN_COLUMNS)
    assert tables == [("track_attributes",)]


def test_delete_where_smoke_keeps_code_test_rows(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(20.0))
    for index in range(3):
        _write(
            store, camera_id="cam-smoke", track_id=index, run_id="run-smoke", derived_from="SMOKE"
        )
    for index in range(4):
        _write(
            store, camera_id="cam-code", track_id=index, run_id="run-code", derived_from="CODE_TEST"
        )
    _write(store, camera_id="cam-site", track_id=7, run_id="run-site", derived_from="SITE")

    deleted = store.delete_where(derived_from="SMOKE")

    assert deleted == 3
    assert store.count(derived_from="SMOKE") == 0
    assert store.count(derived_from="CODE_TEST") == 4
    assert len(store.read("cam-code")) == 4
    assert len(store.read("cam-site")) == 1
    assert store.read("cam-smoke") == []


def test_delete_where_does_not_purge_other_origins(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    clock = _Clock(now - 8 * _DAY)
    store = _store(tmp_path, clock)
    _write(store, camera_id="cam-smoke", derived_from="SMOKE")
    _write(store, camera_id="cam-code", derived_from="CODE_TEST")
    clock.now = now

    deleted = store.delete_where(derived_from="SMOKE")

    assert deleted == 1
    remaining = _raw_rows(store.db_path)
    assert len(remaining) == 1
    assert remaining[0][0] == "cam-code"
    assert remaining[0][3] == "CODE_TEST"


def test_delete_where_by_run_id_and_both_filters(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(5.0))
    _write(store, camera_id="cam-a", run_id="run-a", derived_from="SMOKE")
    _write(store, camera_id="cam-b", run_id="run-a", derived_from="CODE_TEST")
    _write(store, camera_id="cam-c", run_id="run-b", derived_from="SMOKE")

    removed = store.delete_where(derived_from="SMOKE", run_id="run-a")
    assert removed == 1
    assert store.count(run_id="run-a", derived_from="CODE_TEST") == 1
    assert store.count(run_id="run-b", derived_from="SMOKE") == 1

    removed_run = store.delete_where(run_id="run-b")
    assert removed_run == 1
    assert store.count(run_id="run-b") == 0
    assert len(store.read("cam-b")) == 1


def test_delete_where_requires_a_filter(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(5.0))
    _write(store)
    with pytest.raises(ValueError, match="derived_from or run_id"):
        store.delete_where()
    assert store.count() == 1


def test_count_filters_and_empty_store(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(5.0))
    assert store.count() == 0
    _write(store, camera_id="cam-a", track_id=1, run_id="run-a", derived_from="SMOKE")
    _write(store, camera_id="cam-b", track_id=2, run_id="run-b", derived_from="SITE")
    _write(store, camera_id="cam-b", track_id=3, run_id="run-b", derived_from="SITE")

    assert store.count() == 3
    assert store.count(camera_id="cam-b") == 2
    assert store.count(track_id=3) == 1
    assert store.count(run_id="run-a") == 1
    assert store.count(derived_from="SITE") == 2
    assert store.count(camera_id="cam-b", track_id=2, run_id="run-b", derived_from="SITE") == 1
    assert store.count(camera_id="cam-a", derived_from="SITE") == 0


def test_reopen_reads_persisted_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "attributes.sqlite"
    clock = _Clock(100.0)
    store = TrackAttributeStore(db_path, clock=clock)
    _write(store, track_id=4, n_frames=1)
    reopened = TrackAttributeStore(db_path, clock=clock)
    assert reopened.read("cam-1") == [_expected(100.0, track_id=4, n_frames=1)]
    assert reopened.db_path == db_path.resolve()


def test_default_clock_stores_utc_epoch(tmp_path: Path) -> None:
    before = time.time()
    store = TrackAttributeStore(tmp_path / "attributes.sqlite")
    _write(store)
    after = time.time()
    connection = sqlite3.connect(store.db_path)
    try:
        row = connection.execute("SELECT created_at FROM track_attributes").fetchone()
    finally:
        connection.close()
    assert row is not None
    assert before <= float(row[0]) <= after


def test_integer_clock_is_accepted(tmp_path: Path) -> None:
    store = TrackAttributeStore(tmp_path / "attributes.sqlite", clock=lambda: 15)
    _write(store)
    assert store.read("cam-1")[0].created_at == 15.0


def test_default_max_age_days_is_7() -> None:
    assert DEFAULT_MAX_AGE_DAYS == 7
    default = (
        inspect.signature(TrackAttributeStore.purge_expired).parameters["max_age_days"].default
    )
    assert default == 7


def test_retention_purges_an_8_day_row_and_keeps_a_6_day_row(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    clock = _Clock(now)
    store = _store(tmp_path, clock)
    clock.now = now - 8 * _DAY
    _write(store, camera_id="cam-old", track_id=1)
    clock.now = now - 6 * _DAY
    _write(store, camera_id="cam-fresh", track_id=2)
    assert _raw_rows(store.db_path)[0][0] == "cam-old"

    clock.now = now
    removed = store.purge_expired()

    assert removed == 1
    assert store.read("cam-old") == []
    assert store.read("cam-fresh") == [_expected(now - 6 * _DAY, camera_id="cam-fresh", track_id=2)]


def test_retention_keeps_a_row_aged_exactly_7_days(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    clock = _Clock(now - 7 * _DAY)
    store = _store(tmp_path, clock)
    _write(store, camera_id="cam-exact", track_id=3)
    clock.now = now

    removed = store.purge_expired()

    assert removed == 0
    assert store.read("cam-exact") == [_expected(now - 7 * _DAY, camera_id="cam-exact", track_id=3)]


def test_retention_keeps_a_row_one_second_older_than_6_days(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    created_at = now - 6 * _DAY - 1.0
    clock = _Clock(created_at)
    store = _store(tmp_path, clock)
    _write(store, camera_id="cam-six", track_id=6)
    clock.now = now

    assert store.purge_expired() == 0
    assert store.read("cam-six") == [_expected(created_at, camera_id="cam-six", track_id=6)]


def test_retention_purges_a_row_one_second_older_than_7_days(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    clock = _Clock(now - 7 * _DAY - 1.0)
    store = _store(tmp_path, clock)
    _write(store, camera_id="cam-plus", track_id=4, derived_from="SITE")
    clock.now = now

    removed = store.purge_expired()

    assert removed == 1
    assert store.read("cam-plus") == []


def test_purge_runs_on_open(tmp_path: Path) -> None:
    """Opening the store deletes an already expired row. The check uses raw SQL."""
    now = 1_800_000_000.0
    clock = _Clock(now - 8 * _DAY)
    store = _store(tmp_path, clock)
    _write(store, camera_id="cam-old")
    db_path = store.db_path
    del store
    assert _raw_count(db_path) == 1

    clock.now = now
    TrackAttributeStore(db_path, clock=clock)
    assert _raw_count(db_path) == 0


def test_purge_runs_before_write(tmp_path: Path) -> None:
    start = 2_000_000_000.0
    clock = _Clock(start - 8 * _DAY)
    store = _store(tmp_path, clock)
    _write(store, camera_id="cam-old", track_id=8)
    clock.now = start
    _write(store, camera_id="cam-fresh", track_id=9)

    assert [row[0] for row in _raw_rows(store.db_path)] == ["cam-fresh"]


def test_purge_runs_before_read(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    clock = _Clock(now)
    store = _store(tmp_path, clock)
    clock.now = now - 8 * _DAY
    _write(store, camera_id="cam-old", track_id=1)
    clock.now = now - 6 * _DAY
    _write(store, camera_id="cam-fresh", track_id=2)
    clock.now = now

    found = store.read("cam-old")
    fresh = store.read("cam-fresh")

    assert found == []
    assert fresh == [_expected(now - 6 * _DAY, camera_id="cam-fresh", track_id=2)]
    assert _raw_count(store.db_path) == 1


def test_purge_runs_before_query(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    clock = _Clock(now)
    store = _store(tmp_path, clock)
    clock.now = now - 8 * _DAY
    _write(store, camera_id="cam-old", track_id=1)
    clock.now = now - 6 * _DAY
    _write(store, camera_id="cam-fresh", track_id=2)
    clock.now = now
    tracklets = (
        _tracklet(camera_id="cam-old", track_id=1),
        _tracklet(camera_id="cam-fresh", track_id=2),
    )

    found = store.query({}, None, None, tracklets)

    assert [row.camera_id for row in found] == ["cam-fresh"]
    assert _raw_count(store.db_path) == 1


def test_purge_runs_before_count(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    clock = _Clock(now)
    store = _store(tmp_path, clock)
    clock.now = now - 8 * _DAY
    _write(store, camera_id="cam-old")
    clock.now = now - 6 * _DAY
    _write(store, camera_id="cam-fresh", track_id=2)
    clock.now = now

    assert store.count() == 1
    assert _raw_count(store.db_path) == 1


def test_read_query_and_count_hide_rows_after_the_clock_advances(tmp_path: Path) -> None:
    """Open, move the clock 8 days, and do not write. Reads hide the old row."""
    now = 1_800_000_000.0
    clock = _Clock(now)
    store = _store(tmp_path, clock)
    clock.now = now - 8 * _DAY
    _write(store, camera_id="cam-1", track_id=1, upper_color="red")
    clock.now = now - 6 * _DAY
    _write(store, camera_id="cam-1", track_id=2, upper_color="green")
    clock.now = now
    tracklets = (_tracklet(track_id=1), _tracklet(track_id=2))

    found = store.read("cam-1")
    queried = store.query({}, None, None, tracklets)
    counted = store.count()

    assert [row.track_id for row in found] == [2]
    assert [row.upper_color for row in found] == ["green"]
    assert queried == found
    assert counted == 1


def test_read_query_and_count_leave_fresh_rows_unchanged(tmp_path: Path) -> None:
    clock = _Clock(50.0)
    store = _store(tmp_path, clock)
    _write(store, bag_on_track=True)
    before = _raw_rows(store.db_path)

    assert store.read("cam-1") == [_expected(50.0, bag_on_track=True)]
    assert store.query({}, None, None, (_tracklet(),)) == [_expected(50.0, bag_on_track=True)]
    assert store.count() == 1
    assert _raw_rows(store.db_path) == before


def test_read_query_and_count_drop_only_expired_rows(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    clock = _Clock(now)
    store = _store(tmp_path, clock)
    clock.now = now - 8 * _DAY
    _write(store, camera_id="cam-1", track_id=1, upper_color="red")
    clock.now = now - 6 * _DAY
    _write(store, camera_id="cam-1", track_id=2, upper_color="green")
    clock.now = now
    before = _raw_rows(store.db_path)
    assert len(before) == 2
    tracklets = (_tracklet(track_id=1), _tracklet(track_id=2))

    visible = store.read("cam-1")
    after_read = _raw_rows(store.db_path)
    queried = store.query({}, None, None, tracklets)
    counted = store.count()
    after_rest = _raw_rows(store.db_path)

    assert [row.track_id for row in visible] == [2]
    assert after_read == [before[1]]
    assert queried == visible
    assert counted == 1
    assert after_rest == after_read


def test_query_joins_tracklets_and_filters_attributes(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(9.0))
    _write(
        store, camera_id="cam-1", track_id=1, run_id="run-1", upper_color="red", bag_on_track=True
    )
    _write(
        store,
        camera_id="cam-1",
        track_id=2,
        run_id="run-1",
        upper_color="green",
        lower_color="black",
        bag_on_track=False,
        upper_confidence=0.25,
    )
    _write(store, camera_id="cam-2", track_id=1, run_id="run-1", upper_color="red")
    _write(store, camera_id="cam-1", track_id=1, run_id="run-2", upper_color="red")
    tracklets = (
        _tracklet(camera_id="cam-1", track_id=1, run_id="run-1", t_start=10.0, t_end=20.0),
        _tracklet(camera_id="cam-1", track_id=2, run_id="run-1", t_start=10.0, t_end=20.0),
        _tracklet(camera_id="cam-2", track_id=1, run_id="run-1", t_start=10.0, t_end=20.0),
        _tracklet(camera_id="Cam-1", track_id=1, run_id="run-1", t_start=10.0, t_end=20.0),
    )

    red = store.query({"upper_color": "red"}, None, None, tracklets)
    assert [(row.camera_id, row.track_id, row.run_id) for row in red] == [
        ("cam-1", 1, "run-1"),
        ("cam-2", 1, "run-1"),
    ]

    narrowed = store.query({"upper_color": Color.RED}, ("cam-1", "cam-1"), (10.0, 20.0), tracklets)
    assert [(row.camera_id, row.track_id) for row in narrowed] == [("cam-1", 1)]

    bag = store.query({"bag_on_track": True, "lower_color": "blue"}, None, None, tracklets)
    assert [(row.camera_id, row.track_id) for row in bag] == [("cam-1", 1)]

    confident = store.query(
        {"upper_color": "green", "min_confidence": 0.5},
        None,
        None,
        tracklets,
    )
    assert confident == []
    loose = store.query({"upper_color": "green", "min_confidence": 0.25}, None, None, tracklets)
    assert [row.track_id for row in loose] == [2]
    ignored = store.query({"min_confidence": 0.9}, None, None, tracklets)
    assert len(ignored) == 3


def test_query_window_overlap_is_inclusive(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(4.0))
    _write(store)
    tracklet = _tracklet(t_start=10.0, t_end=20.0)

    assert store.query({}, None, None, (tracklet,)) == [_expected(4.0)]
    assert store.query({}, None, (20.0, 30.0), (tracklet,)) == [_expected(4.0)]
    assert store.query({}, None, (0.0, 10.0), (tracklet,)) == [_expected(4.0)]
    assert store.query({}, None, (20.0, 20.0), (tracklet,)) == [_expected(4.0)]
    assert store.query({}, None, (20.1, 21.0), (tracklet,)) == []
    assert store.query({}, None, (0.0, 9.9), (tracklet,)) == []
    assert store.query({}, None, (10.0, 20.0), ()) == []
    assert store.query({}, ("cam-9",), None, (tracklet,)) == []


def test_query_rejects_a_bad_filter_camera_window_or_tracklet(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(1.0))
    _write(store)
    tracklet = _tracklet()
    with pytest.raises(ValueError, match="unknown keys"):
        store.query({"extra": "red"}, None, None, (tracklet,))
    with pytest.raises(TypeError, match="mapping"):
        store.query([], None, None, (tracklet,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool"):
        store.query({"bag_on_track": 1}, None, None, (tracklet,))
    with pytest.raises(ValueError, match="Color value"):
        store.query({"upper_color": "RED"}, None, None, (tracklet,))
    with pytest.raises(ValueError, match="not a str"):
        store.query({}, "cam-1", None, (tracklet,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="collection"):
        store.query({}, b"cam-1", None, (tracklet,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="collection"):
        store.query({}, 4, None, (tracklet,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        store.query({}, [], None, (tracklet,))
    with pytest.raises(ValueError, match="at least one"):
        store.query({}, set(), None, (tracklet,))
    with pytest.raises(TypeError, match="t_window"):
        store.query({}, None, "10", (tracklet,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="t_window"):
        store.query({}, None, b"10", (tracklet,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pair"):
        store.query({}, None, (1.0,), (tracklet,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="start must be <="):
        store.query({}, None, (3.0, 2.0), (tracklet,))
    with pytest.raises(TypeError, match="timestamp"):
        store.query({}, None, (True, 2.0), (tracklet,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        store.query({}, None, (float("nan"), 1.0), (tracklet,))
    with pytest.raises(TypeError, match="Tracklet"):
        store.query({}, None, None, "cam-1")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Tracklet"):
        store.query({}, None, None, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Tracklet"):
        store.query({}, None, None, (object(),))  # type: ignore[arg-type]
    assert store.count() == 1


def test_query_accepts_a_window_list_and_equal_endpoints(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(2.0))
    _write(store)
    tracklet = _tracklet(t_start=5.0, t_end=5.0)
    assert store.query({}, ["cam-1"], [5.0, 5.0], [tracklet]) == [_expected(2.0)]


@pytest.mark.parametrize(
    "value",
    ["", "   ", "\t", "\n", "\u3000", 4],
    ids=["empty", "spaces", "tab", "newline", "ideographic-space", "int"],
)
def test_id_rejects_empty_or_whitespace(tmp_path: Path, value: object) -> None:
    """A whitespace-only id is empty. The id is not stripped and then accepted."""
    store = _store(tmp_path, _Clock(1.0))
    with pytest.raises(ValueError, match="non-empty id") as caught:
        store.write(value, 1, "run-1", "CODE_TEST", "red", "blue", False, 0.5, 0.5, 1)  # type: ignore[arg-type]
    assert repr(value) in str(caught.value)
    with pytest.raises(ValueError, match="non-empty id"):
        store.write("cam-1", 1, value, "CODE_TEST", "red", "blue", False, 0.5, 0.5, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty id"):
        store.write("cam-1", 1, "run-1", value, "red", "blue", False, 0.5, 0.5, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty id"):
        store.read(value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty id"):
        store.count(camera_id=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty id"):
        store.count(run_id=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty id"):
        store.count(derived_from=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty id"):
        store.delete_where(derived_from=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty id"):
        store.delete_where(run_id=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty id"):
        store.query({}, (value,), None, ())  # type: ignore[arg-type]
    assert store.count() == 0


@pytest.mark.parametrize(
    "value",
    [" cam-1", "cam-1 ", "cam-1\n", "\tcam-1"],
    ids=["leading-space", "trailing-space", "trailing-newline", "leading-tab"],
)
def test_id_rejects_strip_mismatch(tmp_path: Path, value: str) -> None:
    """Surrounding whitespace is refused. The original text is not stored."""
    store = _store(tmp_path, _Clock(1.0))
    with pytest.raises(ValueError, match="whitespace") as caught:
        store.write(value, 1, "run-1", "CODE_TEST", "red", "blue", False, 0.5, 0.5, 1)
    assert repr(value) in str(caught.value)
    with pytest.raises(ValueError, match="whitespace"):
        store.write("cam-1", 1, value, "CODE_TEST", "red", "blue", False, 0.5, 0.5, 1)
    with pytest.raises(ValueError, match="whitespace"):
        store.write("cam-1", 1, "run-1", value, "red", "blue", False, 0.5, 0.5, 1)
    with pytest.raises(ValueError, match="whitespace"):
        store.read(value)
    with pytest.raises(ValueError, match="whitespace"):
        store.query({}, (value,), None, ())
    assert store.count() == 0


def test_id_rejects_nfkc_mismatch(tmp_path: Path) -> None:
    """Fullwidth ``ｃａｍ-1`` is refused. It is not folded."""
    value = "\uff43\uff41\uff4d-1"
    store = _store(tmp_path, _Clock(1.0))
    with pytest.raises(ValueError, match="NFKC") as caught:
        store.write(value, 1, "run-1", "CODE_TEST", "red", "blue", False, 0.5, 0.5, 1)
    assert repr(value) in str(caught.value)
    with pytest.raises(ValueError, match="NFKC"):
        store.write("cam-1", 1, value, "CODE_TEST", "red", "blue", False, 0.5, 0.5, 1)
    with pytest.raises(ValueError, match="NFKC"):
        store.read(value)
    with pytest.raises(ValueError, match="NFKC"):
        store.count(run_id=value)
    assert store.count() == 0


@pytest.mark.parametrize(
    "mark",
    ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"],
    ids=["zwsp", "zwnj", "zwj", "word-joiner", "bom"],
)
def test_id_rejects_format_character(tmp_path: Path, mark: str) -> None:
    """Category Cf is refused, including each zero-width mark."""
    value = f"cam-1{mark}"
    store = _store(tmp_path, _Clock(1.0))
    with pytest.raises(ValueError, match="format character") as caught:
        store.write(value, 1, "run-1", "CODE_TEST", "red", "blue", False, 0.5, 0.5, 1)
    assert repr(value) in str(caught.value)
    with pytest.raises(ValueError, match="format character"):
        store.write("cam-1", 1, value, "CODE_TEST", "red", "blue", False, 0.5, 0.5, 1)
    with pytest.raises(ValueError, match="format character"):
        store.write("cam-1", 1, "run-1", value, "red", "blue", False, 0.5, 0.5, 1)
    with pytest.raises(ValueError, match="format character"):
        store.delete_where(run_id=value)
    assert store.count() == 0


@pytest.mark.parametrize(
    "value",
    [True, False, "1", " 1", "", 1.5, 1.0],
    ids=["true", "false", "text", "padded-text", "empty-text", "float", "one-float"],
)
def test_track_id_rejects_bool_and_non_int(tmp_path: Path, value: object) -> None:
    store = _store(tmp_path, _Clock(1.0))
    with pytest.raises(ValueError, match="track_id"):
        store.write("cam-1", value, "run-1", "CODE_TEST", "red", "blue", False, 0.5, 0.5, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="track_id"):
        store.read("cam-1", track_id=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="track_id"):
        store.count(track_id=value)  # type: ignore[arg-type]
    assert store.count() == 0


def test_required_id_rejects_none(tmp_path: Path) -> None:
    """None is an omitted optional filter. A required id still refuses it."""
    store = _store(tmp_path, _Clock(1.0))
    _write(store)
    with pytest.raises(ValueError, match="non-empty id"):
        store.write(None, 1, "run-1", "CODE_TEST", "red", "blue", False, 0.5, 0.5, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty id"):
        store.write("cam-1", 1, None, "CODE_TEST", "red", "blue", False, 0.5, 0.5, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty id"):
        store.write("cam-1", 1, "run-1", None, "red", "blue", False, 0.5, 0.5, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty id"):
        store.read(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty id"):
        store.query({}, (None,), None, ())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="track_id"):
        store.write("cam-1", None, "run-1", "CODE_TEST", "red", "blue", False, 0.5, 0.5, 1)  # type: ignore[arg-type]
    assert store.count(camera_id=None, track_id=None, run_id=None, derived_from=None) == 1
    assert store.read("cam-1", track_id=None, run_id=None) == [_expected(1.0)]


def test_track_id_rejects_negative(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(1.0))
    with pytest.raises(ValueError, match="track_id"):
        store.write("cam-1", -1, "run-1", "CODE_TEST", "red", "blue", False, 0.5, 0.5, 1)
    with pytest.raises(ValueError, match="track_id"):
        store.read("cam-1", track_id=-1)
    assert store.count() == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("upper_color", "RED"),
        ("upper_color", " red"),
        ("upper_color", "red "),
        ("upper_color", "mauve"),
        ("upper_color", ""),
        ("upper_color", None),
        ("upper_color", 1),
        ("lower_color", "BLUE"),
        ("lower_color", " blue"),
        ("lower_color", "not-a-color"),
        ("upper_color", "\uff52\uff45\uff44"),
    ],
    ids=[
        "upper-case",
        "upper-lead-space",
        "upper-trail-space",
        "upper-unknown",
        "upper-empty",
        "upper-none",
        "upper-int",
        "lower-case",
        "lower-lead-space",
        "lower-unknown",
        "upper-fullwidth",
    ],
)
def test_color_must_be_an_exact_palette_value(tmp_path: Path, field: str, value: object) -> None:
    store = _store(tmp_path, _Clock(1.0))
    kwargs: dict[str, object] = {
        "camera_id": "cam-1",
        "track_id": 1,
        "run_id": "run-1",
        "derived_from": "CODE_TEST",
        "upper_color": "red",
        "lower_color": "blue",
        "bag_on_track": False,
        "upper_confidence": 0.5,
        "lower_confidence": 0.5,
        "n_frames": 1,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="Color value"):
        store.write(**kwargs)  # type: ignore[arg-type]
    assert store.count() == 0


@pytest.mark.parametrize(
    "value",
    [True, False, "0.5", None, "1"],
    ids=["true", "false", "text", "none", "text-one"],
)
def test_confidence_rejects_bool_and_non_numeric(tmp_path: Path, value: object) -> None:
    store = _store(tmp_path, _Clock(1.0))
    with pytest.raises(TypeError, match="real number"):
        store.write("cam-1", 1, "run-1", "CODE_TEST", "red", "blue", False, value, 0.5, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real number"):
        store.write("cam-1", 1, "run-1", "CODE_TEST", "red", "blue", False, 0.5, value, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real number"):
        store.query({"min_confidence": value}, None, None, ())
    assert store.count() == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_confidence_rejects_non_finite(tmp_path: Path, value: float) -> None:
    store = _store(tmp_path, _Clock(1.0))
    with pytest.raises(ValueError, match="finite"):
        store.write("cam-1", 1, "run-1", "CODE_TEST", "red", "blue", False, value, 0.5, 1)
    with pytest.raises(ValueError, match="finite"):
        store.write("cam-1", 1, "run-1", "CODE_TEST", "red", "blue", False, 0.5, value, 1)
    with pytest.raises(ValueError, match="finite"):
        store.query({"min_confidence": value}, None, None, ())
    assert store.count() == 0


@pytest.mark.parametrize("value", [-0.1, 1.1, 2, -1])
def test_confidence_rejects_out_of_range(tmp_path: Path, value: float) -> None:
    store = _store(tmp_path, _Clock(1.0))
    with pytest.raises(ValueError, match="must be in"):
        store.write("cam-1", 1, "run-1", "CODE_TEST", "red", "blue", False, value, 0.5, 1)
    with pytest.raises(ValueError, match="must be in"):
        store.write("cam-1", 1, "run-1", "CODE_TEST", "red", "blue", False, 0.5, value, 1)
    with pytest.raises(ValueError, match="must be in"):
        store.query({"min_confidence": value}, None, None, ())
    assert _raw_count(store.db_path) == 0


def test_confidence_accepts_zero_and_one_including_ints(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(8.0))
    store.write("cam-1", 1, "run-1", "CODE_TEST", "red", "blue", False, 0, 1, 1)
    found = store.read("cam-1")
    assert found[0].upper_confidence == 0.0
    assert found[0].lower_confidence == 1.0


@pytest.mark.parametrize(
    "value",
    [True, False, 1.5, "1", None, 1.0],
    ids=["true", "false", "float", "text", "none", "one-float"],
)
def test_n_frames_rejects_bool_and_non_int(tmp_path: Path, value: object) -> None:
    store = _store(tmp_path, _Clock(1.0))
    with pytest.raises(TypeError, match="must be an int"):
        store.write("cam-1", 1, "run-1", "CODE_TEST", "red", "blue", False, 0.5, 0.5, value)  # type: ignore[arg-type]
    assert store.count() == 0


@pytest.mark.parametrize("value", [0, -1, -5])
def test_n_frames_rejects_below_one(tmp_path: Path, value: int) -> None:
    store = _store(tmp_path, _Clock(1.0))
    with pytest.raises(ValueError, match="must be >= 1"):
        store.write("cam-1", 1, "run-1", "CODE_TEST", "red", "blue", False, 0.5, 0.5, value)
    assert store.count() == 0


@pytest.mark.parametrize(
    "value", [0, 1, "true", None, 2], ids=["zero", "one", "text", "none", "two"]
)
def test_bag_on_track_rejects_non_bool(tmp_path: Path, value: object) -> None:
    store = _store(tmp_path, _Clock(1.0))
    with pytest.raises(TypeError, match="bool"):
        store.write("cam-1", 1, "run-1", "CODE_TEST", "red", "blue", value, 0.5, 0.5, 1)  # type: ignore[arg-type]
    assert store.count() == 0


def test_read_rejects_a_stored_bag_other_than_zero_or_one(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(1.0))
    connection = sqlite3.connect(store.db_path)
    try:
        connection.execute(
            _INSERT_COLUMNS,
            ("cam-1", 1, "run-1", "CODE_TEST", "red", "blue", 2, 0.5, 0.5, 1, 1.0),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ValueError, match="stored as 0 or 1"):
        store.read("cam-1")
    assert _raw_count(store.db_path) == 1


def test_failed_validation_does_not_insert(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(6.0))
    _write(store)
    with pytest.raises(ValueError):
        store.write("cam-1", 1, "run-1", "CODE_TEST", "nope", "blue", False, 0.5, 0.5, 1)
    assert store.count() == 1
    assert _raw_rows(store.db_path)[0][4] == "red"


def test_purge_now_overrides_the_clock(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(0.0))
    _write(store, camera_id="cam-now")
    assert store.count() == 1

    removed = store.purge_expired(now=8 * _DAY)

    assert removed == 1
    assert store.count() == 0


def test_clock_and_age_guards(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(1.0))
    with pytest.raises(TypeError, match="max_age_days"):
        store.purge_expired(max_age_days=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_age_days"):
        store.purge_expired(max_age_days=-1)
    with pytest.raises(TypeError, match="timestamp"):
        store.purge_expired(now="now")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        store.purge_expired(now=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        store.purge_expired(now=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        TrackAttributeStore(tmp_path / "bad-clock.sqlite", clock=lambda: float("nan"))
    with pytest.raises(TypeError, match="timestamp"):
        TrackAttributeStore(tmp_path / "bool-clock.sqlite", clock=lambda: True)  # type: ignore[arg-type,return-value]


def test_repo_root_itself_is_refused() -> None:
    with pytest.raises(ValueError, match="outside the repo"):
        TrackAttributeStore(_repo_root(), clock=lambda: 0.0)


def test_refuses_db_inside_repo() -> None:
    repo = _repo_root()
    targets = (
        repo / "te3-should-not-exist.sqlite",
        repo / "data" / "te3-should-not-exist.sqlite",
        repo / "src" / ".." / "te3-should-not-exist.sqlite",
    )
    for target in targets:
        escaped = target.resolve()
        parent = target.parent
        parent_existed = parent.exists()
        try:
            with pytest.raises(ValueError, match="outside the repo"):
                TrackAttributeStore(target, clock=lambda: 0.0)
            assert not escaped.exists()
        finally:
            if escaped.is_file():
                escaped.unlink()
            if not parent_existed and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()


def test_connect_failure_does_not_leak(tmp_path: Path) -> None:
    folder = tmp_path / "not-a-file"
    folder.mkdir()
    with pytest.raises(sqlite3.OperationalError):
        TrackAttributeStore(folder, clock=lambda: 0.0)


def test_generator_tracklets_are_refused(tmp_path: Path) -> None:
    store = _store(tmp_path, _Clock(1.0))
    _write(store)

    def _one():
        yield _tracklet()

    with pytest.raises(TypeError, match="Tracklet"):
        store.query({}, None, None, _one())
    assert store.count() == 1
