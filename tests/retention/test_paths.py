"""Purge entry point refuses an audit root that the data root can reach.

The audit directory comes from ``AUDIT_LOG_DIR``. These tests do not sleep.
The clock is injected, and every path lives under ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

import pytest

import trace_search.attributes.store as attribute_store
import trace_search.retention.paths as retention_paths
import trace_search.tracking.store as track_store
from trace_search.audit.log import AUDIT_LOG_DIR, is_audit_path
from trace_search.detection.types import DetClass
from trace_search.retention import retention_seconds
from trace_search.retention.config import FOOTAGE_DERIVED_DAYS
from trace_search.retention.registry import Purgeable, purgeables
from trace_search.tracking.types import TrackObservation

_NOW = 1_800_000_000.0
_RUN = "run-1"
_FIXTURE = "FX-CODE-1"


class _SetClock:
    """Mutable UTC epoch. Tests move it. Nothing sleeps."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _snapshot(root: Path) -> tuple[tuple[str, str], ...]:
    """Hash every file under ``root``. Symlinks contribute their target text."""
    rows: list[tuple[str, str]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        current = Path(directory)
        for dirname in dirnames:
            path = current / dirname
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                rows.append((relative, "symlink:" + os.readlink(path)))
            else:
                rows.append((relative, "dir"))
        for filename in filenames:
            path = current / filename
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                rows.append((relative, "symlink:" + os.readlink(path)))
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append((relative, digest))
    rows.sort()
    return tuple(rows)


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


def _table_count(db_path: Path, table: str) -> int:
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
    if row is None:
        raise AssertionError(db_path)
    return int(row[0])


def _row_count(stores: list[Purgeable]) -> int:
    total = 0
    for store in stores:
        total += _table_count(_db_path(store), store.covered_tables[0])
    return total


def _seed_old_and_young(data: Path, clock: _SetClock) -> list[Purgeable]:
    window = retention_seconds(FOOTAGE_DERIVED_DAYS)
    clock.now = _NOW - (window + 1)
    stores = purgeables(data, clock=clock)
    _write_all(stores, "cam-old", 1)
    clock.now = _NOW - (window - 1)
    _write_all(stores, "cam-young", 2)
    clock.now = _NOW
    return stores


def _expired_count(data: Path, clock: _SetClock) -> int:
    total = 0
    for store in purgeables(data, clock=clock):
        total += store.count_older_than()
    return total


def _audit_tree(root: Path) -> None:
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "audit.jsonl").write_bytes(b'{"kept":true}\n')
    (root / "audit.jsonl.head").write_bytes(b"anchor-bytes")


def test_retention_paths_does_not_repeat_the_audit_directory_literal() -> None:
    """The purge module has no audit-directory literal and no cwd-relative default."""
    source = Path(retention_paths.__file__).read_text(encoding="utf-8")
    assert AUDIT_LOG_DIR not in source
    assert "default_audit_root" not in source


def test_purge_from_another_cwd_leaves_the_audit_root_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A purge started in some other directory does not touch the audit root."""
    audit = tmp_path / "configured-audit"
    _audit_tree(audit)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.bin").write_bytes(b"do-not-touch")
    data = tmp_path / "data"
    data.mkdir()
    clock = _SetClock(_NOW)
    _seed_old_and_young(data, clock)
    audit_before = _snapshot(audit)
    outside_before = _snapshot(outside)
    assert audit_before
    assert outside_before
    expected = _expired_count(data, clock)
    assert expected > 0
    before_rows = _row_count(purgeables(data, clock=clock))
    assert before_rows > expected

    other = tmp_path / "other-cwd"
    other.mkdir()
    monkeypatch.chdir(other)
    deleted = retention_paths.purge_expired(data, audit_root=audit, clock=clock)

    assert deleted == expected
    assert _snapshot(audit) == audit_before
    assert _snapshot(outside) == outside_before
    after_rows = _row_count(purgeables(data, clock=clock))
    assert after_rows == before_rows - deleted
    assert after_rows > 0
    assert _expired_count(data, clock) == 0


def test_explicit_now_is_counted_without_moving_the_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    window = retention_seconds(FOOTAGE_DERIVED_DAYS)
    clock = _SetClock(_NOW - (window + 1))
    stores = purgeables(data, clock=clock)
    _write_all(stores, "cam-old", 1)
    other = tmp_path / "other-cwd"
    other.mkdir()
    monkeypatch.chdir(other)
    expected = 0
    for store in purgeables(data, clock=clock):
        expected += store.count_older_than(now=_NOW)
    deleted = retention_paths.purge_expired(data, audit_root=audit, clock=clock, now=_NOW)
    assert expected > 0
    assert deleted == expected
    remaining = 0
    for store in purgeables(data, clock=clock):
        remaining += store.count_older_than(now=_NOW)
    assert remaining == 0


def test_data_root_equal_to_the_audit_root_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "same"
    root.mkdir()
    (root / "audit.jsonl").write_bytes(b"audit")
    before = _snapshot(root)
    alias = root / ".." / root.name
    with pytest.raises(ValueError, match="equal"):
        retention_paths.validate_roots(alias, audit_root=root)
    with pytest.raises(ValueError, match="equal"):
        retention_paths.purge_expired(root, audit_root=root)
    assert _snapshot(root) == before
    assert not (root / "tracks.sqlite").exists()
    assert not (root / "attributes.sqlite").exists()


def test_data_root_containing_the_audit_root_is_refused(tmp_path: Path) -> None:
    data = tmp_path / "data"
    audit = data / "secure-audit"
    _audit_tree(audit)
    before = _snapshot(audit)
    with pytest.raises(ValueError, match="contain"):
        retention_paths.validate_roots(data, audit_root=audit)
    with pytest.raises(ValueError, match="contain"):
        retention_paths.purge_expired(data, audit_root=audit)
    assert _snapshot(audit) == before
    assert not (data / "tracks.sqlite").exists()
    assert not (data / "attributes.sqlite").exists()


def test_data_root_symlink_into_the_audit_root_is_refused(tmp_path: Path) -> None:
    audit = tmp_path / "audit"
    nested = audit / "nested"
    nested.mkdir(parents=True)
    (audit / "audit.jsonl").write_bytes(b"audit")
    link = tmp_path / "data-link"
    link.symlink_to(nested, target_is_directory=True)
    before = _snapshot(audit)
    with pytest.raises(ValueError, match="contain"):
        retention_paths.validate_roots(link, audit_root=audit)
    with pytest.raises(ValueError, match="contain"):
        retention_paths.purge_expired(link, audit_root=audit)
    assert _snapshot(audit) == before
    assert not (nested / "tracks.sqlite").exists()
    assert not (nested / "attributes.sqlite").exists()


def test_data_root_symlink_onto_the_audit_root_is_refused(tmp_path: Path) -> None:
    audit = tmp_path / "audit"
    _audit_tree(audit)
    link = tmp_path / "onto"
    link.symlink_to(audit, target_is_directory=True)
    before = _snapshot(audit)
    with pytest.raises(ValueError, match="equal"):
        retention_paths.validate_roots(link, audit_root=audit)
    with pytest.raises(ValueError, match="equal"):
        retention_paths.purge_expired(link, audit_root=audit)
    assert _snapshot(audit) == before
    assert not (audit / "tracks.sqlite").exists()


def test_folder_named_like_the_audit_dir_elsewhere_is_not_exempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A directory that only shares the audit directory's name is purged."""
    audit = tmp_path / "configured-audit"
    _audit_tree(audit)
    audit_before = _snapshot(audit)
    data = tmp_path / "warehouse" / AUDIT_LOG_DIR
    data.mkdir(parents=True)
    decoy = data / "note.txt"
    decoy.write_bytes(b"not-the-audit-log")
    other_named = tmp_path / "clips" / AUDIT_LOG_DIR
    other_named.mkdir(parents=True)
    (other_named / "crop.json").write_bytes(b"crop")
    other_before = _snapshot(other_named)
    clock = _SetClock(_NOW)
    _seed_old_and_young(data, clock)
    expected = _expired_count(data, clock)
    other = tmp_path / "other-cwd"
    other.mkdir()
    monkeypatch.chdir(other)

    deleted = retention_paths.purge_expired(data, audit_root=audit, clock=clock)

    assert expected > 0
    assert deleted == expected
    assert decoy.read_bytes() == b"not-the-audit-log"
    assert _snapshot(audit) == audit_before
    assert _snapshot(other_named) == other_before
    assert not is_audit_path(decoy, audit_root=audit)
    assert not is_audit_path(other_named / "crop.json", audit_root=audit)
    assert is_audit_path(audit / "nested" / "audit.jsonl", audit_root=audit)


def test_omitted_audit_root_is_a_type_error(tmp_path: Path) -> None:
    """Omitting audit_root is a TypeError and opens no store."""
    data = tmp_path / "data"
    data.mkdir()
    audit = tmp_path / "audit"
    _audit_tree(audit)
    audit_before = _snapshot(audit)
    with pytest.raises(TypeError, match="audit_root"):
        retention_paths.purge_expired(data)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="audit_root"):
        retention_paths.validate_roots(data)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="positional"):
        retention_paths.purge_expired(data, audit)  # type: ignore[misc]
    with pytest.raises(TypeError, match="positional"):
        retention_paths.validate_roots(data, audit)  # type: ignore[misc]
    assert _snapshot(audit) == audit_before
    assert not (data / "tracks.sqlite").exists()
    assert not (data / "attributes.sqlite").exists()
    assert not (audit / "tracks.sqlite").exists()
    assert not (audit / "attributes.sqlite").exists()


def test_relative_roots_are_refused_from_another_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative audit root or data root is refused, and nothing is deleted."""
    audit = tmp_path / "configured-audit"
    _audit_tree(audit)
    data = tmp_path / "data"
    data.mkdir()
    clock = _SetClock(_NOW)
    seeded = _seed_old_and_young(data, clock)
    audit_before = _snapshot(audit)
    data_before = _snapshot(data)
    rows_before = _row_count(seeded)
    expired_before = _expired_count(data, clock)
    assert rows_before > expired_before > 0

    other = tmp_path / "other-cwd"
    other.mkdir()
    monkeypatch.chdir(other)

    with pytest.raises(ValueError, match="audit_root is relative"):
        retention_paths.validate_roots(data, audit_root=Path(AUDIT_LOG_DIR))
    with pytest.raises(ValueError, match="audit_root is relative"):
        retention_paths.purge_expired(data, audit_root=Path(AUDIT_LOG_DIR), clock=clock)
    with pytest.raises(ValueError, match="data_root is relative"):
        retention_paths.validate_roots(Path("data"), audit_root=audit)
    with pytest.raises(ValueError, match="data_root is relative"):
        retention_paths.purge_expired(Path("data"), audit_root=audit, clock=clock)

    assert _snapshot(audit) == audit_before
    assert _snapshot(data) == data_before
    assert _row_count(purgeables(data, clock=clock)) == rows_before
    assert _expired_count(data, clock) == expired_before
    assert list(other.iterdir()) == []


def test_roots_must_be_paths(tmp_path: Path) -> None:
    audit = tmp_path / "audit"
    audit.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(TypeError):
        retention_paths.validate_roots("data", audit_root=audit)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        retention_paths.validate_roots(data, audit_root="audit")  # type: ignore[arg-type]


def test_file_root_is_refused(tmp_path: Path) -> None:
    data = tmp_path / "not-a-directory"
    data.write_bytes(b"x")
    audit = tmp_path / "audit"
    audit.mkdir()
    with pytest.raises(ValueError, match="directory"):
        retention_paths.validate_roots(data, audit_root=audit)
    with pytest.raises(ValueError, match="directory"):
        retention_paths.validate_roots(audit, audit_root=data)


def test_symlink_loop_is_refused(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.symlink_to(right)
    right.symlink_to(left)
    audit = tmp_path / "audit"
    audit.mkdir()
    with pytest.raises(ValueError, match="resolved"):
        retention_paths.validate_roots(left, audit_root=audit)


def test_home_relative_data_root_is_expanded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    data = home / "data"
    data.mkdir(parents=True)
    audit = tmp_path / "audit"
    audit.mkdir()
    monkeypatch.setenv("HOME", str(home))
    retention_paths.validate_roots(Path("~/data"), audit_root=audit)
    with pytest.raises(ValueError, match="equal"):
        retention_paths.validate_roots(Path("~/data"), audit_root=data)
