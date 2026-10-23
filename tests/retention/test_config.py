"""Shipped retention is 7 days and 365 days, and every store reads that default."""

import inspect
import sqlite3
from pathlib import Path

import pytest

from trace_search.attributes.store import (
    DEFAULT_MAX_AGE_DAYS as ATTRIBUTE_MAX_AGE_DAYS,
)
from trace_search.attributes.store import (
    TrackAttributeRecord,
    TrackAttributeStore,
)
from trace_search.detection.types import DetClass
from trace_search.matching.tracklets import load_tracklets
from trace_search.retention import AUDIT_DAYS, FOOTAGE_DERIVED_DAYS, RetentionConfig, load
from trace_search.tracking.store import DEFAULT_MAX_AGE_DAYS as TRACK_MAX_AGE_DAYS
from trace_search.tracking.store import TrackStore
from trace_search.tracking.types import TrackObservation

_NOW = 1_800_000_000.0
_DAY_SECONDS = 86400
# Literal 7 days. Do not derive this from FOOTAGE_DERIVED_DAYS: a store
# default of 6 or 8 must fail the boundary below.
_SEVEN_DAYS = 7 * _DAY_SECONDS
_RUN = "run-1"
_FIXTURE = "FX-CODE-1"


class _Clock:
    """Mutable UTC epoch. Tests move it. Nothing sleeps."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _young_created_at() -> float:
    """7 days minus 1 second before ``_NOW``."""
    return _NOW - (_SEVEN_DAYS - 1)


def _old_created_at() -> float:
    """7 days plus 1 second before ``_NOW``."""
    return _NOW - (_SEVEN_DAYS + 1)


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


def _write_attribute(store: TrackAttributeStore, camera_id: str, track_id: int) -> None:
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


def _attribute_row(created_at: float, camera_id: str, track_id: int) -> TrackAttributeRecord:
    return TrackAttributeRecord(
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


def _count_rows(db_path: Path, table: str) -> int:
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


def _write_aged_pair(store: TrackStore | TrackAttributeStore, clock: _Clock) -> None:
    """Write the older row first, then the younger one; each is fresh when written."""
    clock.now = _old_created_at()
    if isinstance(store, TrackStore):
        store.write(_RUN, _FIXTURE, "CODE_TEST", [_observation("cam-purged", 1)])
        clock.now = _young_created_at()
        store.write(_RUN, _FIXTURE, "CODE_TEST", [_observation("cam-kept", 2)])
        return
    _write_attribute(store, "cam-purged", 1)
    clock.now = _young_created_at()
    _write_attribute(store, "cam-kept", 2)


def test_shipped_retention_is_exactly_7_and_365() -> None:
    """AC-14a. Any other shipped value fails."""
    assert FOOTAGE_DERIVED_DAYS == 7
    assert AUDIT_DAYS == 365
    shipped = load()
    assert shipped.footage_days == 7
    assert shipped.audit_days == 365
    assert shipped == RetentionConfig(footage_days=7, audit_days=365)


def test_load_uses_only_the_shipped_constants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env var and no file. ``load`` takes no override argument."""
    monkeypatch.setenv("FOOTAGE_DERIVED_DAYS", "1")
    monkeypatch.setenv("AUDIT_DAYS", "2")
    monkeypatch.setenv("TRACE_SEARCH_FOOTAGE_DAYS", "3")
    monkeypatch.setenv("TRACE_SEARCH_AUDIT_DAYS", "4")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "retention.json").write_text(
        '{"footage_days": 1, "audit_days": 2}\n',
        encoding="utf-8",
    )
    assert list(inspect.signature(load).parameters) == []
    shipped = load()
    assert shipped.footage_days == FOOTAGE_DERIVED_DAYS
    assert shipped.audit_days == AUDIT_DAYS


def test_injected_config_does_not_change_shipped_days() -> None:
    """A caller-built config is the only way a test uses another window."""
    injected = RetentionConfig(footage_days=1, audit_days=2)
    assert injected.footage_days == 1
    assert injected.audit_days == 2
    assert injected != load()
    assert load() == RetentionConfig(
        footage_days=FOOTAGE_DERIVED_DAYS,
        audit_days=AUDIT_DAYS,
    )


@pytest.mark.parametrize(
    ("footage_days", "audit_days", "label"),
    [
        (True, 365, "footage_days"),
        (7, False, "audit_days"),
        (1.5, 365, "footage_days"),
        (7, "365", "audit_days"),
    ],
)
def test_day_counts_reject_bools_and_non_ints(
    footage_days: object,
    audit_days: object,
    label: str,
) -> None:
    with pytest.raises(TypeError, match=label):
        RetentionConfig(footage_days, audit_days)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("footage_days", "audit_days", "label"),
    [
        (-1, 365, "footage_days"),
        (7, -5, "audit_days"),
    ],
)
def test_day_counts_reject_negatives(footage_days: int, audit_days: int, label: str) -> None:
    with pytest.raises(ValueError, match=label):
        RetentionConfig(footage_days=footage_days, audit_days=audit_days)


def test_zero_day_config_is_allowed_and_leaves_the_shipped_values() -> None:
    injected = RetentionConfig(footage_days=0, audit_days=0)
    assert injected.footage_days == 0
    assert injected.audit_days == 0
    assert load().footage_days == FOOTAGE_DERIVED_DAYS
    assert load().audit_days == AUDIT_DAYS


def test_config_equality_hash_and_frozen_fields() -> None:
    first = RetentionConfig(footage_days=1, audit_days=2)
    second = RetentionConfig(footage_days=1, audit_days=2)
    assert first == second
    assert hash(first) == hash(second)
    assert first != RetentionConfig(footage_days=9, audit_days=2)
    assert first != RetentionConfig(footage_days=1, audit_days=9)
    assert first != object()
    with pytest.raises(AttributeError):
        setattr(first, "footage_days", 9)
    with pytest.raises(AttributeError):
        setattr(first, "audit_days", 9)
    with pytest.raises(AttributeError):
        setattr(first, "_footage_days", 9)
    with pytest.raises(AttributeError):
        delattr(first, "_audit_days")
    assert first.footage_days == 1
    assert first.audit_days == 2


def test_store_defaults_alias_the_shipped_footage_window() -> None:
    """Each store imports the footage default. It does not keep its own number."""
    assert TRACK_MAX_AGE_DAYS == FOOTAGE_DERIVED_DAYS
    assert ATTRIBUTE_MAX_AGE_DAYS == FOOTAGE_DERIVED_DAYS
    track_default = inspect.signature(TrackStore.purge_expired).parameters["max_age_days"].default
    attribute_default = (
        inspect.signature(TrackAttributeStore.purge_expired).parameters["max_age_days"].default
    )
    assert track_default == FOOTAGE_DERIVED_DAYS
    assert attribute_default == FOOTAGE_DERIVED_DAYS


def test_track_store_default_keeps_7_days_minus_1s_and_purges_plus_1s(tmp_path: Path) -> None:
    """Open purges. Read then returns the row that is 7 days minus 1 second old."""
    clock = _Clock(_old_created_at())
    db_path = tmp_path / "tracks.sqlite"
    _write_aged_pair(TrackStore(db_path, clock=clock), clock)
    clock.now = _NOW

    reopened = TrackStore(db_path, clock=clock)

    assert _count_rows(db_path, "track_observations") == 1
    assert reopened.read("cam-kept") == [_observation("cam-kept", 2)]
    assert reopened.read("cam-purged") == []
    assert reopened.count(camera_id="cam-kept") == 1
    assert reopened.count(camera_id="cam-purged") == 0


def test_attribute_store_default_keeps_7_days_minus_1s_and_purges_plus_1s(
    tmp_path: Path,
) -> None:
    """Open purges before read. The 7 days minus 1 second row is the one left."""
    clock = _Clock(_old_created_at())
    db_path = tmp_path / "attributes.sqlite"
    _write_aged_pair(TrackAttributeStore(db_path, clock=clock), clock)
    clock.now = _NOW

    reopened = TrackAttributeStore(db_path, clock=clock)

    assert _count_rows(db_path, "track_attributes") == 1
    assert reopened.read("cam-kept") == [_attribute_row(_young_created_at(), "cam-kept", 2)]
    assert reopened.read("cam-purged") == []
    assert reopened.count(camera_id="cam-kept") == 1
    assert reopened.count(camera_id="cam-purged") == 0


def test_tracklet_summary_keeps_7_days_minus_1s_and_purges_plus_1s(tmp_path: Path) -> None:
    """T-E-2 reads through TrackStore.purge_expired and has no window of its own."""
    clock = _Clock(_old_created_at())
    store = TrackStore(tmp_path / "tracks.sqlite", clock=clock)
    _write_aged_pair(store, clock)
    clock.now = _NOW

    loaded = load_tracklets(store, _RUN)

    assert [item.camera_id for item in loaded] == ["cam-kept"]
