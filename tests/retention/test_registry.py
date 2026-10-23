"""Every footage-derived store with ``created_at`` is a registered ``Purgeable``.

The embedding store (T-R-7) and the export store (T-W-8) are not modules on
main, so this file does not open them. The attribute database filename is
``attributes.sqlite`` under the data root. ``TrackAttributeStore`` has no
default filename of its own.
"""

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

import trace_search.attributes.store as attribute_store
import trace_search.tracking.store as track_store
from trace_search.detection.types import DetClass
from trace_search.retention import retention_seconds
from trace_search.retention.registry import (
    ATTRIBUTE_DB_FILENAME,
    TRACK_DB_FILENAME,
    Purgeable,
    purgeables,
)
from trace_search.tracking.types import TrackObservation

_NOW = 1_800_000_000.0
_DAY = 86400
_SEVEN = 7 * _DAY
_RUN = "run-1"
_FIXTURE = "FX-CODE-1"
_Clock = Callable[[], float]


class _SetClock:
    """Mutable UTC epoch. Tests move it. Nothing sleeps."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _observation(camera_id: str, track_id: int) -> TrackObservation:
    return TrackObservation(
        camera_id=camera_id,
        track_id=track_id,
        det_class=DetClass.PERSON,
        frame_number=0,
        timestamp_seconds=1.0,
        box=(1.0, 2.0, 5.0, 8.0),
        score=0.5,
    )


def _write_track(store: track_store.TrackStore, camera_id: str, track_id: int) -> None:
    store.write(_RUN, _FIXTURE, "CODE_TEST", [_observation(camera_id, track_id)])


def _write_attribute(
    store: attribute_store.TrackAttributeStore,
    camera_id: str,
    track_id: int,
) -> None:
    store.write(
        camera_id,
        track_id,
        _RUN,
        "CODE_TEST",
        "red",
        "blue",
        False,
        0.5,
        0.75,
        1,
    )


def _write_all(stores: list[Purgeable], camera_id: str, track_id: int) -> None:
    for store in stores:
        if isinstance(store, track_store.TrackStore):
            _write_track(store, camera_id, track_id)
        elif isinstance(store, attribute_store.TrackAttributeStore):
            _write_attribute(store, camera_id, track_id)
        else:
            raise AssertionError(type(store).__name__)


def _db_path(store: Purgeable) -> Path:
    if isinstance(store, track_store.TrackStore):
        return store.db_path
    if isinstance(store, attribute_store.TrackAttributeStore):
        return store.db_path
    raise AssertionError(type(store).__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _make_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    return root


def _raw_count(db_path: Path, table: str) -> int:
    if table == "track_observations":
        sql = "SELECT COUNT(*) FROM track_observations"
    elif table == "track_attributes":
        sql = "SELECT COUNT(*) FROM track_attributes"
    else:
        raise AssertionError(table)
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(sql).fetchone()
    finally:
        connection.close()
    assert row is not None
    return int(row[0])


def _created_at_tables(db_path: Path) -> set[str]:
    """Tables that have a ``created_at`` column, from sqlite_master and PRAGMA."""
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
        found: set[str] = set()
        for row in rows:
            table = str(row[0])
            if table.startswith("sqlite_"):
                continue
            if not _safe_table_name(table):
                raise AssertionError(table)
            info = connection.execute("PRAGMA table_info(" + table + ")")
            for column in info.fetchall():
                if str(column[1]) == "created_at":
                    found.add(table)
                    break
        return found
    finally:
        connection.close()


def _safe_table_name(table: str) -> bool:
    if table == "":
        return False
    for character in table:
        if not (character.isascii() and (character.isalnum() or character == "_")):
            return False
    return True


def _module_db_paths(root: Path, clock: _Clock) -> list[Path]:
    """Open each footage-derived store module that exists on main."""
    track = track_store.TrackStore(root / TRACK_DB_FILENAME, clock=clock)
    attributes = attribute_store.TrackAttributeStore(root / ATTRIBUTE_DB_FILENAME, clock=clock)
    return [track.db_path, attributes.db_path]


def _uncovered_created_at_tables(root: Path, clock: _Clock) -> set[str]:
    registered = purgeables(root, clock=clock)
    covered: set[str] = set()
    found: set[str] = set()
    for store in registered:
        covered.update(store.covered_tables)
        found.update(_created_at_tables(_db_path(store)))
    for path in _module_db_paths(root, clock):
        found.update(_created_at_tables(path))
    return found - covered


def _assert_created_at_tables_are_covered(root: Path, clock: _Clock) -> None:
    missing = _uncovered_created_at_tables(root, clock)
    assert missing == set(), missing


def _has_camera(store: Purgeable, camera_id: str) -> bool:
    if isinstance(store, track_store.TrackStore):
        return store.read(camera_id) != []
    if isinstance(store, attribute_store.TrackAttributeStore):
        return store.read(camera_id) != []
    raise AssertionError(type(store).__name__)


def _attribute_row(
    created_at: float,
    camera_id: str,
    track_id: int,
) -> attribute_store.TrackAttributeRecord:
    return attribute_store.TrackAttributeRecord(
        camera_id=camera_id,
        track_id=track_id,
        run_id=_RUN,
        derived_from="CODE_TEST",
        upper_color="red",
        lower_color="blue",
        bag_on_track=False,
        upper_confidence=0.5,
        lower_confidence=0.75,
        n_frames=1,
        created_at=created_at,
    )


def test_retention_seconds_is_whole_days() -> None:
    """A one-second-short window would purge a row aged exactly seven days."""
    assert retention_seconds(0) == 0
    assert retention_seconds(1) == _DAY
    assert retention_seconds(7) == _SEVEN


def test_retention_seconds_rejects_bool_and_negative() -> None:
    with pytest.raises(TypeError, match="max_age_days"):
        retention_seconds(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_age_days"):
        retention_seconds(-1)


def test_stores_do_not_keep_their_own_retention_seconds() -> None:
    assert not hasattr(track_store, "_retention_seconds")
    assert not hasattr(attribute_store, "_retention_seconds")
    assert not hasattr(track_store, "_SECONDS_PER_DAY")
    assert not hasattr(attribute_store, "_SECONDS_PER_DAY")


def test_registry_filenames_match_the_stores_on_main() -> None:
    """``attributes.sqlite`` is the registry's name. The attribute store has none."""
    assert TRACK_DB_FILENAME == "tracks.sqlite"
    assert TRACK_DB_FILENAME == track_store._DB_FILENAME
    assert ATTRIBUTE_DB_FILENAME == "attributes.sqlite"


def test_registry_registers_the_track_and_attribute_stores(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    stores = purgeables(root, clock=lambda: 1.0)
    assert [store.label for store in stores] == ["tracks", "attributes"]
    assert stores[0].covered_tables == ("track_observations",)
    assert stores[1].covered_tables == ("track_attributes",)


def test_protocol_accepts_both_real_stores(tmp_path: Path) -> None:
    """``isinstance`` is the runtime check. The annotations are the mypy assignment."""
    root = _make_root(tmp_path)
    stores = purgeables(root, clock=lambda: 1.0)
    track: Purgeable = stores[0]
    attributes: Purgeable = stores[1]
    assert isinstance(track, Purgeable)
    assert isinstance(attributes, Purgeable)
    assert isinstance(track, track_store.TrackStore)
    assert isinstance(attributes, attribute_store.TrackAttributeStore)
    assert track.label == "tracks"
    assert attributes.label == "attributes"


def test_created_at_tables_are_covered_by_a_registered_purgeable(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    _assert_created_at_tables_are_covered(root, lambda: 1.0)


def test_a_table_without_created_at_does_not_need_a_purgeable(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    clock = _SetClock(1.0)
    stores = purgeables(root, clock=clock)
    connection = sqlite3.connect(_db_path(stores[0]))
    try:
        connection.execute("CREATE TABLE notes (id INTEGER NOT NULL, body TEXT NOT NULL)")
        connection.commit()
    finally:
        connection.close()
    _assert_created_at_tables_are_covered(root, clock)


def test_planted_unregistered_created_at_table_fails_the_coverage_check(
    tmp_path: Path,
) -> None:
    root = _make_root(tmp_path)
    clock = _SetClock(1.0)
    stores = purgeables(root, clock=clock)
    connection = sqlite3.connect(_db_path(stores[0]))
    try:
        connection.execute(
            "CREATE TABLE planted_unregistered (id INTEGER NOT NULL, created_at REAL NOT NULL)"
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(AssertionError, match="planted_unregistered"):
        _assert_created_at_tables_are_covered(root, clock)


def test_purgeables_never_returns_a_path_outside_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "data").resolve()
    root.mkdir()
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    stores = purgeables(root, clock=lambda: 1.0)
    assert stores
    for store in stores:
        path = _db_path(store).resolve()
        assert root in path.parents
        assert outside not in path.parents
        assert path.is_file()
    assert list(outside.iterdir()) == []

    monkeypatch.setattr(
        "trace_search.retention.registry.TRACK_DB_FILENAME",
        "../escaped.sqlite",
    )
    with pytest.raises(ValueError, match="data root"):
        purgeables(root, clock=lambda: 1.0)
    assert not (tmp_path / "escaped.sqlite").exists()


def test_dotdot_filename_stays_inside_the_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _make_root(tmp_path)
    monkeypatch.setattr("trace_search.retention.registry.TRACK_DB_FILENAME", "..")
    with pytest.raises(ValueError, match="data root"):
        purgeables(root, clock=lambda: 1.0)
    assert not (tmp_path / "tracks.sqlite").exists()


def test_empty_filename_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _make_root(tmp_path)
    monkeypatch.setattr("trace_search.retention.registry.TRACK_DB_FILENAME", "")
    with pytest.raises(ValueError, match="data root"):
        purgeables(root, clock=lambda: 1.0)


def test_purgeables_keeps_the_outside_repo_guard() -> None:
    repo = _repo_root()
    target = repo / "t-p-2-should-not-exist"
    try:
        with pytest.raises(ValueError, match="outside the repo"):
            purgeables(target, clock=lambda: 0.0)
        assert not target.exists()
    finally:
        if target.is_dir():
            for path in target.iterdir():
                if path.is_file():
                    path.unlink()
            target.rmdir()


def test_data_root_must_be_a_directory(tmp_path: Path) -> None:
    target = tmp_path / "not-a-directory"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        purgeables(target, clock=lambda: 1.0)


def test_data_root_must_be_a_path() -> None:
    with pytest.raises(TypeError, match="data_root"):
        purgeables("relative")  # type: ignore[arg-type]


def test_missing_data_root_is_created_under_the_resolved_path(tmp_path: Path) -> None:
    root = tmp_path / "missing" / "data"
    stores = purgeables(root, clock=lambda: 1.0)
    resolved = root.resolve()
    assert root.is_dir()
    for store in stores:
        assert resolved in _db_path(store).resolve().parents


def test_relative_data_root_is_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.chdir(tmp_path)
    stores = purgeables(Path("data"), clock=lambda: 1.0)
    for store in stores:
        assert root.resolve() in _db_path(store).resolve().parents


def test_data_root_expands_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    root = home / "data"
    root.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    stores = purgeables(Path("~/data"), clock=lambda: 1.0)
    for store in stores:
        assert root.resolve() in _db_path(store).resolve().parents


def test_purgeables_ignores_the_track_db_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "from-env"
    outside.mkdir()
    monkeypatch.setenv("TRACE_SEARCH_TRACK_DB", str(outside / "tracks.sqlite"))
    root = _make_root(tmp_path)
    stores = purgeables(root, clock=lambda: 1.0)
    for store in stores:
        assert root.resolve() in _db_path(store).resolve().parents
    assert not (outside / "tracks.sqlite").exists()


def test_default_clock_still_registers_both_stores(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    stores = purgeables(root)
    assert [store.label for store in stores] == ["tracks", "attributes"]


def test_count_older_than_matches_the_deleted_count(tmp_path: Path) -> None:
    """The count is taken before any delete. It equals the purge return value."""
    clock = _SetClock(_NOW - (_SEVEN + 1))
    root = _make_root(tmp_path)
    stores = purgeables(root, clock=clock)
    _write_all(stores, "cam-old", 1)
    clock.now = _NOW - (_SEVEN - 1)
    _write_all(stores, "cam-young", 2)
    clock.now = _NOW
    for store in stores:
        table = store.covered_tables[0]
        path = _db_path(store)
        before = _raw_count(path, table)
        counted = store.count_older_than()
        assert _raw_count(path, table) == before
        deleted = store.purge_expired()
        after = _raw_count(path, table)
        assert counted == deleted
        assert deleted == before - after
        assert deleted == 1
        assert store.count_older_than() == 0


def test_count_and_purge_are_zero_when_nothing_is_expired(tmp_path: Path) -> None:
    clock = _SetClock(_NOW)
    root = _make_root(tmp_path)
    stores = purgeables(root, clock=clock)
    _write_all(stores, "cam-fresh", 1)
    for store in stores:
        assert store.count_older_than(now=_NOW) == 0
        assert store.purge_expired(now=_NOW) == 0
        assert _raw_count(_db_path(store), store.covered_tables[0]) == 1


def test_explicit_now_counts_without_moving_the_injected_clock(tmp_path: Path) -> None:
    clock = _SetClock(_NOW - (_SEVEN + 1))
    root = _make_root(tmp_path)
    stores = purgeables(root, clock=clock)
    _write_all(stores, "cam-old", 1)
    for store in stores:
        counted = store.count_older_than(now=_NOW, max_age_days=7)
        assert counted == 1
        assert _raw_count(_db_path(store), store.covered_tables[0]) == 1
        assert store.purge_expired(now=_NOW, max_age_days=7) == 1


def test_registry_keeps_7d_minus_1s_and_purges_7d_plus_1s(tmp_path: Path) -> None:
    """Open through the registry. Exactly 7 days stays, so a short window fails."""
    clock = _SetClock(_NOW - (_SEVEN + 1))
    root = _make_root(tmp_path)
    stores = purgeables(root, clock=clock)
    _write_all(stores, "cam-purged", 1)
    clock.now = _NOW - _SEVEN
    _write_all(stores, "cam-exact", 2)
    clock.now = _NOW - (_SEVEN - 1)
    _write_all(stores, "cam-kept", 3)
    clock.now = _NOW
    for store in stores:
        assert store.count_older_than() == 1

    reopened = purgeables(root, clock=clock)

    for store in reopened:
        assert _has_camera(store, "cam-kept")
        assert _has_camera(store, "cam-exact")
        assert not _has_camera(store, "cam-purged")
        if isinstance(store, track_store.TrackStore):
            assert store.read("cam-kept") == [_observation("cam-kept", 3)]
        elif isinstance(store, attribute_store.TrackAttributeStore):
            assert store.read("cam-kept") == [_attribute_row(_NOW - (_SEVEN - 1), "cam-kept", 3)]


def test_attribute_count_still_purges_before_it_counts(tmp_path: Path) -> None:
    """``count`` still deletes first. ``count_older_than`` is the count that remains."""
    clock = _SetClock(_NOW - (_SEVEN + 1))
    root = _make_root(tmp_path)
    store = attribute_store.TrackAttributeStore(root / ATTRIBUTE_DB_FILENAME, clock=clock)
    _write_attribute(store, "cam-old", 1)
    clock.now = _NOW
    assert store.count_older_than() == 1
    assert _raw_count(store.db_path, "track_attributes") == 1
    assert store.count() == 0
    assert _raw_count(store.db_path, "track_attributes") == 0


def test_track_read_and_count_still_run_after_an_explicit_purge(tmp_path: Path) -> None:
    clock = _SetClock(_NOW - (_SEVEN + 1))
    root = _make_root(tmp_path)
    store = track_store.TrackStore(root / TRACK_DB_FILENAME, clock=clock)
    _write_track(store, "cam-old", 1)
    clock.now = _NOW - (_SEVEN - 1)
    _write_track(store, "cam-kept", 2)
    clock.now = _NOW
    assert store.count_older_than() == 1
    assert store.purge_expired() == 1
    assert store.read("cam-kept") == [_observation("cam-kept", 2)]
    assert store.count(camera_id="cam-kept") == 1
    assert store.read("cam-old") == []
