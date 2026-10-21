"""Tests for the local SQLite track store."""

import sqlite3
import time
from pathlib import Path

import pytest

from trace_search.detection.types import DetClass
from trace_search.tracking import store as store_module
from trace_search.tracking.store import (
    TRACK_DB_ENV,
    TrackStore,
    default_db_path,
    load_config,
)
from trace_search.tracking.types import TrackObservation

_DAY = 86400.0
_EXACT_COLUMNS = (
    ("camera_id", "TEXT"),
    ("track_id", "INTEGER"),
    ("det_class", "TEXT"),
    ("frame_number", "INTEGER"),
    ("timestamp_seconds", "REAL"),
    ("box_x1", "REAL"),
    ("box_y1", "REAL"),
    ("box_x2", "REAL"),
    ("box_y2", "REAL"),
    ("score", "REAL"),
    ("run_id", "TEXT"),
    ("source_fixture_id", "TEXT"),
    ("derived_from", "TEXT"),
    ("created_at", "REAL"),
)


class _Clock:
    """Mutable UTC epoch clock. Tests move it; nothing sleeps."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _observation(
    *,
    camera_id: str = "cam-1",
    track_id: int = 1,
    det_class: DetClass = DetClass.PERSON,
    frame_number: int = 0,
    timestamp_seconds: float = 1.0,
    box: tuple[float, float, float, float] = (1.0, 2.0, 5.0, 8.0),
    score: float = 0.5,
) -> TrackObservation:
    return TrackObservation(
        camera_id=camera_id,
        track_id=track_id,
        det_class=det_class,
        frame_number=frame_number,
        timestamp_seconds=timestamp_seconds,
        box=box,
        score=score,
    )


def _observations(count: int, *, camera_id: str = "cam-round") -> list[TrackObservation]:
    classes = (DetClass.PERSON, DetClass.BACKPACK, DetClass.HANDBAG)
    rows: list[TrackObservation] = []
    for index in range(count):
        rows.append(
            _observation(
                camera_id=camera_id,
                track_id=index,
                det_class=classes[index % 3],
                frame_number=index,
                timestamp_seconds=index / 10.0,
                box=(float(index), 0.5, float(index) + 3.0, 4.5),
                score=(index % 11) / 10.0,
            )
        )
    return rows


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_write_read_round_trip_of_100_observations(tmp_path: Path) -> None:
    observations = _observations(100)
    store = TrackStore(tmp_path / "tracks.sqlite", clock=lambda: 1_700_000_000.0)
    store.write("run-100", "FX-CODE-100", "CODE_TEST", observations)

    assert store.read("cam-round") == observations
    assert store.read("cam-round", run_id="run-100") == observations
    assert store.count(camera_id="cam-round", run_id="run-100", derived_from="CODE_TEST") == 100

    connection = sqlite3.connect(store.db_path)
    try:
        stored = connection.execute(
            """
            SELECT typeof(box_x1), typeof(box_y1), typeof(box_x2), typeof(box_y2),
                   typeof(score), typeof(camera_id)
            FROM track_observations
            LIMIT 1
            """
        ).fetchone()
    finally:
        connection.close()
    assert stored == ("real", "real", "real", "real", "real", "text")


def test_derived_from_train_raises_integrity_error(tmp_path: Path) -> None:
    store = TrackStore(tmp_path / "tracks.sqlite", clock=lambda: 10.0)
    with pytest.raises(sqlite3.IntegrityError):
        store.write("run-train", "FX-TRAIN-1", "TRAIN", [_observation()])
    assert store.count() == 0

    connection = sqlite3.connect(store.db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO track_observations (
                    camera_id, track_id, det_class, frame_number, timestamp_seconds,
                    box_x1, box_y1, box_x2, box_y2, score,
                    run_id, source_fixture_id, derived_from, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "cam-1",
                    1,
                    "person",
                    0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.5,
                    "run-raw",
                    "FX-TRAIN-RAW",
                    "TRAIN",
                    10.0,
                ),
            )
        connection.rollback()
    finally:
        connection.close()
    assert store.count() == 0


def test_delete_where_smoke_leaves_code_test_rows(tmp_path: Path) -> None:
    store = TrackStore(tmp_path / "tracks.sqlite", clock=lambda: 20.0)
    smoke = [_observation(camera_id="cam-smoke", track_id=index) for index in range(3)]
    code = [_observation(camera_id="cam-code", track_id=index) for index in range(4)]
    site = [_observation(camera_id="cam-site", track_id=7, det_class=DetClass.BACKPACK)]
    store.write("run-smoke", "FX-SMOKE-1", "SMOKE", smoke)
    store.write("run-code", "FX-CODE-1", "CODE_TEST", code)
    store.write("run-site", "FX-SITE-1", "SITE", site)

    deleted = store.delete_where(derived_from="SMOKE")

    assert deleted == 3
    assert store.count(derived_from="SMOKE") == 0
    assert store.count(derived_from="CODE_TEST") == 4
    assert store.read("cam-code") == code
    assert store.read("cam-site") == site
    assert store.read("cam-smoke") == []


def test_pragma_table_info_exact_columns_and_no_blob(tmp_path: Path) -> None:
    store = TrackStore(tmp_path / "tracks.sqlite", clock=lambda: 0.0)
    connection = sqlite3.connect(store.db_path)
    try:
        info = connection.execute("PRAGMA table_info(track_observations)").fetchall()
        ddl_row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'track_observations'"
        ).fetchone()
    finally:
        connection.close()

    assert ddl_row is not None
    names_and_types = [(row[1], row[2]) for row in info]
    assert names_and_types == list(_EXACT_COLUMNS)
    assert {row[1] for row in info} == {name for name, _col_type in _EXACT_COLUMNS}
    assert all(col_type.upper() != "BLOB" for _name, col_type in names_and_types)
    assert "BLOB" not in ddl_row[0].upper()
    assert "CHECK (derived_from IN ('SMOKE','CODE_TEST','SITE'))" in ddl_row[0]


def test_retention_purges_older_than_7_days(tmp_path: Path) -> None:
    """Rows with created_at 8 days old are purged. Rows 6 days old stay."""
    now = 1_800_000_000.0
    clock = _Clock(now)
    store = TrackStore(tmp_path / "tracks.sqlite", clock=clock)
    old = _observation(camera_id="cam-old", track_id=1)
    fresh = _observation(camera_id="cam-fresh", track_id=2)

    clock.now = now - 8 * _DAY
    store.write("run-old", "FX-CODE-1", "CODE_TEST", [old])
    clock.now = now - 6 * _DAY
    store.write("run-fresh", "FX-CODE-1", "CODE_TEST", [fresh])

    connection = sqlite3.connect(store.db_path)
    try:
        stored = connection.execute(
            "SELECT camera_id, created_at FROM track_observations ORDER BY created_at"
        ).fetchall()
    finally:
        connection.close()
    assert stored == [("cam-old", now - 8 * _DAY), ("cam-fresh", now - 6 * _DAY)]

    clock.now = now
    store.purge_expired()

    assert store.read("cam-old") == []
    assert store.read("cam-fresh") == [fresh]


def test_retention_keeps_a_row_aged_exactly_7_days(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    clock = _Clock(now - 7 * _DAY)
    store = TrackStore(tmp_path / "tracks.sqlite", clock=clock)
    kept = _observation(camera_id="cam-exact", track_id=3)
    store.write("run-exact", "FX-CODE-7", "CODE_TEST", [kept])

    store.purge_expired(now=now, max_age_days=7)

    assert store.read("cam-exact") == [kept]


def test_retention_purges_a_row_one_second_older_than_7_days(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    clock = _Clock(now - 7 * _DAY - 1.0)
    store = TrackStore(tmp_path / "tracks.sqlite", clock=clock)
    aged = _observation(camera_id="cam-plus", track_id=4)
    store.write("run-plus", "FX-CODE-7", "SITE", [aged])

    removed = store.purge_expired(now=now, max_age_days=7)

    assert removed == 1
    assert store.read("cam-plus") == []


def test_purge_runs_on_open(tmp_path: Path) -> None:
    """Opening the store deletes an already expired row. The reopen does no write."""
    now = 1_800_000_000.0
    clock = _Clock(now - 8 * _DAY)
    store = TrackStore(tmp_path / "tracks.sqlite", clock=clock)
    expired = _observation(camera_id="cam-old", track_id=1)
    store.write("run-old", "FX-1", "CODE_TEST", [expired])
    db_path = store.db_path
    del store

    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute("SELECT COUNT(*) FROM track_observations").fetchone()
    finally:
        connection.close()
    assert row == (1,)

    clock.now = now
    reopened = TrackStore(db_path, clock=clock)
    assert reopened.read("cam-old") == []
    assert reopened.count() == 0


def test_purge_runs_before_write(tmp_path: Path) -> None:
    start = 2_000_000_000.0
    clock = _Clock(start - 40 * _DAY)
    store = TrackStore(tmp_path / "tracks.sqlite", clock=clock)
    old = _observation(camera_id="cam-old", track_id=8)
    fresh = _observation(camera_id="cam-fresh", track_id=9)
    store.write("run-old", "FX-1", "CODE_TEST", [old])

    clock.now = start
    store.write("run-fresh", "FX-1", "CODE_TEST", [fresh])

    assert store.read("cam-old") == []
    assert store.read("cam-fresh") == [fresh]
    assert store.count() == 1


def test_read_filters_by_run_id(tmp_path: Path) -> None:
    store = TrackStore(tmp_path / "tracks.sqlite", clock=lambda: 5.0)
    first = _observation(camera_id="cam-a", track_id=1)
    second = _observation(camera_id="cam-a", track_id=2, det_class=DetClass.HANDBAG)
    store.write("run-a", "FX-A", "CODE_TEST", [first])
    store.write("run-b", "FX-B", "CODE_TEST", [second])

    assert store.read("cam-a") == [first, second]
    assert store.read("cam-a", "run-b") == [second]
    assert store.read("cam-a", run_id="missing") == []


def test_count_filters_and_empty_store(tmp_path: Path) -> None:
    store = TrackStore(tmp_path / "tracks.sqlite", clock=lambda: 5.0)
    assert store.count() == 0
    store.write("run-a", "FX-A", "SMOKE", [_observation(camera_id="cam-a", track_id=1)])
    store.write(
        "run-b",
        "FX-B",
        "SITE",
        [_observation(camera_id="cam-b", track_id=2), _observation(camera_id="cam-b", track_id=3)],
    )

    assert store.count() == 3
    assert store.count(camera_id="cam-b") == 2
    assert store.count(run_id="run-a") == 1
    assert store.count(derived_from="SITE") == 2
    assert store.count(camera_id="cam-b", run_id="run-b", derived_from="SITE") == 2
    assert store.count(camera_id="cam-a", derived_from="SITE") == 0


def test_delete_where_by_run_id_and_both_filters(tmp_path: Path) -> None:
    store = TrackStore(tmp_path / "tracks.sqlite", clock=lambda: 5.0)
    store.write("run-a", "FX-A", "SMOKE", [_observation(camera_id="cam-a", track_id=1)])
    store.write("run-a", "FX-A2", "CODE_TEST", [_observation(camera_id="cam-b", track_id=2)])
    store.write("run-b", "FX-B", "SMOKE", [_observation(camera_id="cam-c", track_id=3)])

    removed = store.delete_where(derived_from="SMOKE", run_id="run-a")
    assert removed == 1
    assert store.count(run_id="run-a", derived_from="CODE_TEST") == 1
    assert store.count(run_id="run-b", derived_from="SMOKE") == 1

    removed_run = store.delete_where(run_id="run-b")
    assert removed_run == 1
    assert store.count(run_id="run-b") == 0
    assert store.read("cam-b") == [_observation(camera_id="cam-b", track_id=2)]


def test_delete_where_requires_a_filter(tmp_path: Path) -> None:
    store = TrackStore(tmp_path / "tracks.sqlite", clock=lambda: 5.0)
    store.write("run-a", "FX-A", "CODE_TEST", [_observation()])
    with pytest.raises(ValueError, match="derived_from or run_id"):
        store.delete_where()
    assert store.count() == 1


def test_allowed_derived_from_values_round_trip(tmp_path: Path) -> None:
    store = TrackStore(tmp_path / "tracks.sqlite", clock=lambda: 5.0)
    for origin in ("SMOKE", "CODE_TEST", "SITE"):
        store.write(
            f"run-{origin}",
            f"FX-{origin}",
            origin,
            [_observation(camera_id=f"cam-{origin}", track_id=1)],
        )
    assert store.count() == 3
    with pytest.raises(sqlite3.IntegrityError):
        store.write("run-lower", "FX-lower", "smoke", [_observation(camera_id="cam-lower")])
    assert store.count(camera_id="cam-lower") == 0


def test_metadata_columns_store_run_and_fixture(tmp_path: Path) -> None:
    clock = _Clock(42.5)
    store = TrackStore(tmp_path / "tracks.sqlite", clock=clock)
    store.write("run-meta", "FX-CODE-META", "CODE_TEST", [_observation()])
    connection = sqlite3.connect(store.db_path)
    try:
        row = connection.execute(
            "SELECT run_id, source_fixture_id, derived_from, created_at FROM track_observations"
        ).fetchone()
    finally:
        connection.close()
    assert row == ("run-meta", "FX-CODE-META", "CODE_TEST", 42.5)


def test_empty_write_inserts_nothing(tmp_path: Path) -> None:
    store = TrackStore(tmp_path / "tracks.sqlite", clock=lambda: 5.0)
    store.write("run-empty", "FX-EMPTY", "CODE_TEST", [])
    assert store.count() == 0


def test_reopen_reads_persisted_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "tracks.sqlite"
    clock = _Clock(100.0)
    observations = [_observation(track_id=4, score=1.0), _observation(track_id=5, score=0.0)]
    store = TrackStore(db_path, clock=clock)
    store.write("run-keep", "FX-KEEP", "SITE", observations)

    reopened = TrackStore(db_path, clock=clock)
    assert reopened.read("cam-1") == observations
    assert reopened.db_path == db_path.resolve()


def test_default_clock_stores_utc_epoch(tmp_path: Path) -> None:
    before = time.time()
    store = TrackStore(tmp_path / "tracks.sqlite")
    store.write("run-clock", "FX-CLOCK", "CODE_TEST", [_observation()])
    after = time.time()
    connection = sqlite3.connect(store.db_path)
    try:
        row = connection.execute("SELECT created_at FROM track_observations").fetchone()
    finally:
        connection.close()
    assert row is not None
    assert before <= float(row[0]) <= after


def test_purge_now_overrides_the_clock(tmp_path: Path) -> None:
    store = TrackStore(tmp_path / "tracks.sqlite", clock=lambda: 0.0)
    store.write("run-now", "FX-NOW", "CODE_TEST", [_observation(camera_id="cam-now")])
    assert store.count() == 1

    removed = store.purge_expired(now=8 * _DAY, max_age_days=7)

    assert removed == 1
    assert store.count() == 0


def test_default_db_path_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """TRACE_SEARCH_TRACK_DB wins over XDG_DATA_HOME and over the home fallback."""
    target = tmp_path / "configured.sqlite"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv(TRACK_DB_ENV, str(target))
    assert default_db_path() == target
    assert load_config().db_path == target


def test_xdg_data_home_wins_over_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """XDG_DATA_HOME wins over ~/.local/share when the env db path is unset."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv(TRACK_DB_ENV, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert default_db_path() == tmp_path / "xdg" / "trace-search" / "tracks.sqlite"
    home_fallback = home / ".local" / "share" / "trace-search" / "tracks.sqlite"
    assert default_db_path() != home_fallback


def test_db_path_expands_tilde(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """~ in TRACE_SEARCH_TRACK_DB and XDG_DATA_HOME expands under HOME."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv(TRACK_DB_ENV, "~/x.sqlite")
    assert default_db_path() == home / "x.sqlite"

    monkeypatch.delenv(TRACK_DB_ENV)
    monkeypatch.setenv("XDG_DATA_HOME", "~/xdg")
    assert default_db_path() == home / "xdg" / "trace-search" / "tracks.sqlite"


def test_blank_config_falls_back_to_xdg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(TRACK_DB_ENV, "   ")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    expected = tmp_path / "xdg" / "trace-search" / "tracks.sqlite"
    assert default_db_path() == expected


def test_default_db_path_uses_xdg_outside_repo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(TRACK_DB_ENV, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    path = default_db_path()
    store = TrackStore(path, clock=lambda: 1.0)
    repo = _repo_root()
    assert store.db_path == (tmp_path / "xdg" / "trace-search" / "tracks.sqlite").resolve()
    assert store.db_path.is_file()
    assert repo not in store.db_path.parents
    assert repo != store.db_path


def test_default_db_path_uses_home_when_xdg_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(TRACK_DB_ENV, raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    assert default_db_path() == home / ".local" / "share" / "trace-search" / "tracks.sqlite"


def test_blank_xdg_uses_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(TRACK_DB_ENV, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", "  ")
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    assert default_db_path() == home / ".local" / "share" / "trace-search" / "tracks.sqlite"


def test_repo_root_itself_is_refused() -> None:
    with pytest.raises(ValueError, match="outside the repo"):
        TrackStore(_repo_root(), clock=lambda: 0.0)


def test_path_check_skips_guard_without_repo_markers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(store_module, "__file__", str(tmp_path / "store.py"))
    db_path = tmp_path / "tracks.sqlite"
    assert store_module._db_path_outside_repo(db_path) == db_path.resolve()
    assert store_module._repo_root() is None


def test_refuses_db_inside_repo() -> None:
    repo = _repo_root()
    targets = (
        repo / "t35-should-not-exist.sqlite",
        repo / "data" / "t35-should-not-exist.sqlite",
        repo / "src" / ".." / "t35-should-not-exist.sqlite",
    )
    for target in targets:
        escaped = target.resolve()
        parent = target.parent
        parent_existed = parent.exists()
        try:
            with pytest.raises(ValueError, match="outside the repo"):
                TrackStore(target, clock=lambda: 0.0)
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
        TrackStore(folder, clock=lambda: 0.0)


def test_write_rejects_bad_arguments(tmp_path: Path) -> None:
    store = TrackStore(tmp_path / "tracks.sqlite", clock=lambda: 1.0)
    observation = _observation()
    with pytest.raises(ValueError, match="run_id"):
        store.write("", "FX", "CODE_TEST", [observation])
    with pytest.raises(ValueError, match="fixture_id"):
        store.write("run", "", "CODE_TEST", [observation])
    with pytest.raises(ValueError, match="derived_from"):
        store.write("run", "FX", "", [observation])
    with pytest.raises(TypeError, match="TrackObservation"):
        store.write("run", "FX", "CODE_TEST", "nope")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="TrackObservation"):
        store.write("run", "FX", "CODE_TEST", [object()])  # type: ignore[list-item]
    assert store.count() == 0


def test_read_and_count_reject_blank_filters(tmp_path: Path) -> None:
    store = TrackStore(tmp_path / "tracks.sqlite", clock=lambda: 1.0)
    with pytest.raises(ValueError, match="camera_id"):
        store.read("")
    with pytest.raises(ValueError, match="run_id"):
        store.read("cam-1", run_id="")
    with pytest.raises(ValueError, match="derived_from"):
        store.count(derived_from="")
    with pytest.raises(ValueError, match="camera_id"):
        store.count(camera_id="")
    with pytest.raises(ValueError, match="run_id"):
        store.delete_where(run_id="")


def test_clock_and_age_guards(tmp_path: Path) -> None:
    store = TrackStore(tmp_path / "tracks.sqlite", clock=lambda: 1.0)
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
        TrackStore(tmp_path / "bad-clock.sqlite", clock=lambda: float("nan"))
    with pytest.raises(TypeError, match="timestamp"):
        TrackStore(tmp_path / "bool-clock.sqlite", clock=lambda: True)  # type: ignore[arg-type,return-value]


def test_integer_clock_is_accepted(tmp_path: Path) -> None:
    store = TrackStore(tmp_path / "tracks.sqlite", clock=lambda: 15)
    store.write("run-int", "FX-INT", "CODE_TEST", [_observation()])
    connection = sqlite3.connect(store.db_path)
    try:
        row = connection.execute("SELECT created_at FROM track_observations").fetchone()
    finally:
        connection.close()
    assert row == (15.0,)
