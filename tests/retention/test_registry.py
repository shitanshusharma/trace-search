"""Every footage-derived store with ``created_at`` is a registered ``Purgeable``.

The export store (T-W-8) is not a module on main, so this file does not open
it. The attribute database filename is ``attributes.sqlite`` under the data
root. ``TrackAttributeStore`` has no default filename of its own. The
embedding database filename is ``embeddings.sqlite``. ``SqliteVectorStore``
has no default filename of its own.
"""

import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

import trace_search.attributes.store as attribute_store
import trace_search.reid.sqlite_vector_store as embedding_store
import trace_search.tracking.store as track_store
from trace_search.detection.types import DetClass
from trace_search.reid.types import AppearanceEmbedding
from trace_search.retention import retention_seconds
from trace_search.retention.registry import (
    ATTRIBUTE_DB_FILENAME,
    EMBEDDING_DB_FILENAME,
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


def _axis_vector(dim: int) -> tuple[float, ...]:
    return (1.0,) + (0.0,) * (dim - 1)


def _write_embedding(
    store: embedding_store.SqliteVectorStore, camera_id: str, track_id: int
) -> None:
    embedding = AppearanceEmbedding(
        camera_id=camera_id,
        track_id=track_id,
        frame_number=0,
        vector=_axis_vector(store.dim),
        embedder_name="registry",
        placeholder=True,
    )
    store.add(_RUN, "CODE_TEST", [embedding])


def _write_all(stores: list[Purgeable], camera_id: str, track_id: int) -> None:
    for store in stores:
        if isinstance(store, track_store.TrackStore):
            _write_track(store, camera_id, track_id)
        elif isinstance(store, attribute_store.TrackAttributeStore):
            _write_attribute(store, camera_id, track_id)
        elif isinstance(store, embedding_store.SqliteVectorStore):
            _write_embedding(store, camera_id, track_id)
        else:
            raise AssertionError(type(store).__name__)


def _db_path(store: Purgeable) -> Path:
    if isinstance(store, track_store.TrackStore):
        return store.db_path
    if isinstance(store, attribute_store.TrackAttributeStore):
        return store.db_path
    if isinstance(store, embedding_store.SqliteVectorStore):
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
    elif table == "appearance_embeddings":
        sql = "SELECT COUNT(*) FROM appearance_embeddings"
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


def _package_source() -> Path:
    """The ``trace_search`` package tree. The scan reads this source only."""
    return _repo_root() / "src" / "trace_search"


def _is_ident_char(character: str) -> bool:
    return character.isascii() and (character.isalnum() or character == "_")


def _skip_ws(source: str, index: int) -> int:
    while index < len(source) and source[index].isspace():
        index += 1
    return index


def _skip_quoted(source: str, index: int) -> int:
    quote = source[index]
    index += 1
    while index < len(source):
        if source[index] == quote:
            return index + 1
        index += 1
    return index


def _read_identifier(source: str, index: int) -> tuple[str, int] | None:
    if index >= len(source) or not _is_ident_char(source[index]):
        return None
    if not (source[index].isalpha() or source[index] == "_"):
        return None
    end = index + 1
    while end < len(source) and _is_ident_char(source[end]):
        end += 1
    return source[index:end], end


def _skip_if_not_exists(source: str, index: int) -> int:
    ident = _read_identifier(source, index)
    if ident is None or ident[0].upper() != "IF":
        return index
    cursor = _skip_ws(source, ident[1])
    ident = _read_identifier(source, cursor)
    if ident is None or ident[0].upper() != "NOT":
        return index
    cursor = _skip_ws(source, ident[1])
    ident = _read_identifier(source, cursor)
    if ident is None or ident[0].upper() != "EXISTS":
        return index
    return ident[1]


def _read_paren_body(source: str, open_index: int) -> tuple[str, int]:
    depth = 0
    index = open_index
    start = open_index + 1
    while index < len(source):
        char = source[index]
        if char in ("'", '"'):
            index = _skip_quoted(source, index)
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return source[start:index], index + 1
        index += 1
    return source[start:], len(source)


def _body_has_created_at(body: str) -> bool:
    index = 0
    while index < len(body):
        char = body[index]
        if char in ("'", '"'):
            index = _skip_quoted(body, index)
            continue
        ident = _read_identifier(body, index)
        if ident is None:
            index += 1
            continue
        if ident[0].lower() == "created_at":
            return True
        index = ident[1]
    return False


def _keyword_at(source: str, keyword: str, index: int) -> bool:
    end = index + len(keyword)
    if end > len(source):
        return False
    return source[index:end].upper() == keyword.upper()


def _keyword_ends_ident(source: str, end: int) -> bool:
    if end >= len(source):
        return True
    return not _is_ident_char(source[end])


def _create_table_names_with_created_at(source: str) -> set[str]:
    """Table names from ``CREATE TABLE`` DDL whose body names ``created_at``."""
    found: set[str] = set()
    cursor = 0
    while cursor < len(source):
        if not _keyword_at(source, "CREATE", cursor):
            cursor += 1
            continue
        if cursor > 0 and _is_ident_char(source[cursor - 1]):
            cursor += len("CREATE")
            continue
        after_create = _skip_ws(source, cursor + len("CREATE"))
        table_end = after_create + len("TABLE")
        table_here = _keyword_at(source, "TABLE", after_create)
        if not table_here or not _keyword_ends_ident(source, table_end):
            cursor += len("CREATE")
            continue
        after_table = _skip_ws(source, table_end)
        name_at = _skip_ws(source, _skip_if_not_exists(source, after_table))
        ident = _read_identifier(source, name_at)
        if ident is None:
            cursor += len("CREATE")
            continue
        paren_at = _skip_ws(source, ident[1])
        if paren_at >= len(source) or source[paren_at] != "(":
            cursor += len("CREATE")
            continue
        body, cursor = _read_paren_body(source, paren_at)
        if _body_has_created_at(body):
            found.add(ident[0])
    return found


def _created_at_tables_in_source(package_root: Path) -> set[str]:
    """Static read of ``*.py`` under ``package_root``. No database is opened."""
    found: set[str] = set()
    pending = [package_root]
    while pending:
        current = pending.pop()
        if current.is_symlink() or not current.is_dir():
            continue
        for child in current.iterdir():
            if child.is_symlink():
                continue
            if child.is_dir():
                if child.name == "__pycache__":
                    continue
                pending.append(child)
                continue
            if child.suffix == ".py" and child.is_file():
                text = child.read_text(encoding="utf-8")
                found.update(_create_table_names_with_created_at(text))
    return found


def _uncovered_created_at_tables(
    root: Path,
    clock: _Clock,
    package_root: Path,
) -> set[str]:
    registered = purgeables(root, clock=clock)
    covered: set[str] = set()
    found: set[str] = set()
    for store in registered:
        covered.update(store.covered_tables)
        found.update(_created_at_tables(_db_path(store)))
    scanned = _created_at_tables_in_source(package_root)
    assert scanned, "source scan found no created_at tables"
    found.update(scanned)
    return found - covered


def _assert_created_at_tables_are_covered(
    root: Path,
    clock: _Clock,
    package_root: Path | None = None,
) -> None:
    if package_root is None:
        package_root = _package_source()
    missing = _uncovered_created_at_tables(root, clock, package_root)
    assert missing == set(), missing


def _has_camera(store: Purgeable, camera_id: str) -> bool:
    if isinstance(store, track_store.TrackStore):
        return store.read(camera_id) != []
    if isinstance(store, attribute_store.TrackAttributeStore):
        return store.read(camera_id) != []
    if isinstance(store, embedding_store.SqliteVectorStore):
        found = store.search(_axis_vector(store.dim), k=1, camera_ids=[camera_id])
        return found != []
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
    assert not hasattr(embedding_store, "_retention_seconds")
    assert not hasattr(track_store, "_SECONDS_PER_DAY")
    assert not hasattr(attribute_store, "_SECONDS_PER_DAY")
    assert not hasattr(embedding_store, "_SECONDS_PER_DAY")


def test_registry_filenames_match_the_stores_on_main() -> None:
    """``attributes.sqlite`` is the registry's name. The attribute store has none."""
    assert TRACK_DB_FILENAME == "tracks.sqlite"
    assert TRACK_DB_FILENAME == track_store._DB_FILENAME
    assert ATTRIBUTE_DB_FILENAME == "attributes.sqlite"
    assert EMBEDDING_DB_FILENAME == "embeddings.sqlite"


def test_registry_registers_the_track_attribute_and_embedding_stores(tmp_path: Path) -> None:
    """The registry returns the track, attribute, and embedding stores, in that order."""
    root = _make_root(tmp_path)
    stores = purgeables(root, clock=lambda: 1.0)
    assert [store.label for store in stores] == ["tracks", "attributes", "embeddings"]
    assert stores[0].covered_tables == ("track_observations",)
    assert stores[1].covered_tables == ("track_attributes",)
    assert stores[2].covered_tables == ("appearance_embeddings",)


def test_protocol_accepts_the_track_attribute_and_embedding_stores(tmp_path: Path) -> None:
    """``isinstance`` is the runtime check. The annotations are the mypy assignment."""
    root = _make_root(tmp_path)
    stores = purgeables(root, clock=lambda: 1.0)
    track: Purgeable = stores[0]
    attributes: Purgeable = stores[1]
    embeddings: Purgeable = stores[2]
    assert isinstance(track, Purgeable)
    assert isinstance(attributes, Purgeable)
    assert isinstance(embeddings, Purgeable)
    assert isinstance(track, track_store.TrackStore)
    assert isinstance(attributes, attribute_store.TrackAttributeStore)
    assert isinstance(embeddings, embedding_store.SqliteVectorStore)
    assert track.label == "tracks"
    assert attributes.label == "attributes"
    assert embeddings.label == "embeddings"


def test_registry_counts_and_purges_embeddings_of_another_width(tmp_path: Path) -> None:
    """Opening at dim 4 still counts, then purges, rows stored at another width.

    ``purgeables`` passes ``purge_on_open=False``, so ``count_older_than``
    sees the expired row before ``purge_expired`` deletes it.
    """
    root = _make_root(tmp_path)
    path = root / EMBEDDING_DB_FILENAME
    clock = _SetClock(_NOW - 8 * _DAY)
    writer = embedding_store.SqliteVectorStore(path, 8, clock=clock, purge_on_open=False)
    writer.add(
        _RUN,
        "CODE_TEST",
        [
            AppearanceEmbedding(
                camera_id="cam-old",
                track_id=1,
                frame_number=0,
                vector=_axis_vector(8),
                embedder_name="registry",
                placeholder=True,
            )
        ],
    )
    clock.now = _NOW - 6 * _DAY
    writer.add(
        _RUN,
        "CODE_TEST",
        [
            AppearanceEmbedding(
                camera_id="cam-young",
                track_id=2,
                frame_number=0,
                vector=_axis_vector(8),
                embedder_name="registry",
                placeholder=True,
            )
        ],
    )
    del writer
    clock.now = _NOW
    stores = purgeables(root, clock=clock)
    store = stores[2]
    assert isinstance(store, embedding_store.SqliteVectorStore)
    assert store.dim == 4
    assert store.count_older_than() == 1
    assert _raw_count(path, "appearance_embeddings") == 2
    assert store.purge_expired() == 1
    assert store.count_older_than() == 0
    assert _raw_count(path, "appearance_embeddings") == 1


def test_registry_registers_the_embedding_store(tmp_path: Path) -> None:
    """The embedding file is ``embeddings.sqlite`` under the data root, not the track file."""
    root = _make_root(tmp_path)
    stores = purgeables(root, clock=lambda: 1.0)
    embeddings = [store for store in stores if isinstance(store, embedding_store.SqliteVectorStore)]
    assert len(embeddings) == 1
    store = embeddings[0]
    assert isinstance(store, Purgeable)
    assert store.label == "embeddings"
    assert store.covered_tables == ("appearance_embeddings",)
    assert store.db_path.name == EMBEDDING_DB_FILENAME
    assert store.db_path.name != TRACK_DB_FILENAME
    assert root.resolve() in store.db_path.resolve().parents


def test_created_at_tables_are_covered_by_a_registered_purgeable(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    _assert_created_at_tables_are_covered(root, lambda: 1.0)


def test_source_scan_reads_the_create_table_statements_on_main() -> None:
    """The scan sees the footage tables, including DDL that nests parentheses."""
    found = _created_at_tables_in_source(_package_source())
    assert found == {"track_observations", "track_attributes", "appearance_embeddings"}


def test_source_scan_ignores_a_table_without_created_at(tmp_path: Path) -> None:
    package = tmp_path / "trace_search"
    package.mkdir()
    (package / "notes.py").write_text(
        "CREATE TABLE notes (id INTEGER, body TEXT)\n",
        encoding="utf-8",
    )
    assert _created_at_tables_in_source(package) == set()


def test_source_scan_fails_on_an_unregistered_module(tmp_path: Path) -> None:
    """A copied package plus one unregistered CREATE TABLE fails the check.

    The scan reads that copy. It does not register the new table, and the
    registry does not glob the data root or the package.
    """
    package = tmp_path / "trace_search"
    shutil.copytree(_package_source(), package)
    extra = package / "planted_extra.py"
    extra.write_text(
        "CREATE TABLE planted_store (id INTEGER, created_at REAL)\n",
        encoding="utf-8",
    )
    root = _make_root(tmp_path)
    with pytest.raises(AssertionError, match="planted_store"):
        _assert_created_at_tables_are_covered(root, lambda: 1.0, package_root=package)


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


def test_default_clock_still_registers_the_track_attribute_and_embedding_stores(
    tmp_path: Path,
) -> None:
    """Omitting the clock still registers the track, attribute, and embedding stores."""
    root = _make_root(tmp_path)
    stores = purgeables(root)
    assert [store.label for store in stores] == ["tracks", "attributes", "embeddings"]
    assert stores[0].covered_tables == ("track_observations",)
    assert stores[1].covered_tables == ("track_attributes",)
    assert stores[2].covered_tables == ("appearance_embeddings",)


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


def test_fresh_purgeables_counts_and_purges_an_old_row(tmp_path: Path) -> None:
    """A new registry must still see a row that an open-time purge would drop."""
    clock = _SetClock(_NOW - (_SEVEN + 1))
    root = _make_root(tmp_path)
    stores = purgeables(root, clock=clock)
    _write_all(stores, "cam-old", 1)
    clock.now = _NOW
    fresh = purgeables(root, clock=clock)
    assert fresh
    for store in fresh:
        assert store.count_older_than() == 1
        assert store.purge_expired() == 1
        assert store.count_older_than() == 0


def test_registry_keeps_7d_minus_1s_and_purges_7d_plus_1s(tmp_path: Path) -> None:
    """A fresh registry still counts the old row. Exactly 7 days stays."""
    clock = _SetClock(_NOW - (_SEVEN + 1))
    root = _make_root(tmp_path)
    stores = purgeables(root, clock=clock)
    _write_all(stores, "cam-purged", 1)
    clock.now = _NOW - _SEVEN
    _write_all(stores, "cam-exact", 2)
    clock.now = _NOW - (_SEVEN - 1)
    _write_all(stores, "cam-kept", 3)
    clock.now = _NOW
    fresh = purgeables(root, clock=clock)
    for store in fresh:
        assert store.count_older_than() == 1
        assert store.purge_expired() == 1
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
