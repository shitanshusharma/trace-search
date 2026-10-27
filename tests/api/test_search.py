"""SearchService: attributes, example crops, and click-to-search.

Windows use inclusive overlap, not containment. Data is synthetic and
lives under tmp_path. Placeholder scores are not treated as accuracy.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from trace_search.api import search as search_module
from trace_search.api.search import SearchService, TrackRef
from trace_search.attributes.store import TrackAttributeStore
from trace_search.audit.log import AuditLog, init_hmac_key
from trace_search.detection.types import DetClass
from trace_search.reid.types import AppearanceEmbedding
from trace_search.reid.vector_store import InMemoryVectorStore
from trace_search.tracking.store import TrackStore
from trace_search.tracking.types import TrackObservation

_AXIS = (1.0, 0.0, 0.0)
_DIAG = (1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0)
_ORTH = (0.0, 1.0, 0.0)
_ALLOWED_FILES = frozenset(
    {
        "tracks.sqlite",
        "attributes.sqlite",
        "audit.jsonl",
        "audit.jsonl.head",
        "audit.jsonl.auditlock",
        "audit.hmac",
    }
)


@pytest.fixture(autouse=True)
def _refuse_socket_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests refuse socket.socket.connect. Local SQLite does not connect."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("search tests refuse socket.socket.connect")

    monkeypatch.setattr(socket.socket, "connect", refuse)


class _FixedEmbedder:
    """Returns one hand-built unit vector. Records the crop it was given."""

    dim = 3
    placeholder = True

    def __init__(self, vector: tuple[float, ...]) -> None:
        self._vector = vector
        self.calls = 0
        self.crops: list[npt.NDArray[np.uint8]] = []
        self.rows: npt.NDArray[np.float32] | None = None

    @property
    def name(self) -> str:
        """Read-only embedder label. It is not a person name."""
        return "fixed-test-embedder"

    def embed(self, crops: Sequence[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        self.calls += 1
        self.crops.append(np.array(crops[0], copy=True))
        if self.rows is not None:
            return self.rows
        row = np.asarray(self._vector, dtype=np.float32)
        stacked = np.stack([row])
        return stacked


class _TrackSpy:
    """Counts track-store use. Writes go to the inner store directly."""

    def __init__(self, inner: TrackStore) -> None:
        self.inner = inner
        self.touches = 0
        self.db_hits = 0

    @property
    def db_path(self) -> Path:
        self.db_hits += 1
        return self.inner.db_path

    def read(self, *args: object, **kwargs: object) -> list[TrackObservation]:
        self.touches += 1
        return self.inner.read(*args, **kwargs)

    def camera_ids(self, *args: object, **kwargs: object) -> tuple[str, ...]:
        self.touches += 1
        return self.inner.camera_ids(*args, **kwargs)

    def purge_expired(self, *args: object, **kwargs: object) -> int:
        self.touches += 1
        return self.inner.purge_expired(*args, **kwargs)


class _AttrSpy:
    """Records attribute queries. The inner store still answers them."""

    def __init__(self, inner: TrackAttributeStore) -> None:
        self.inner = inner
        self.queries: list[tuple[object, object]] = []

    def query(
        self,
        attribute_filter: object,
        camera_ids: object,
        t_window: object,
        tracklets: object,
    ) -> list[object]:
        self.queries.append((camera_ids, t_window))
        return self.inner.query(attribute_filter, camera_ids, t_window, tracklets)


class _VectorSpy:
    """Records vector reads. ``camera_ids`` stays visible to the test."""

    def __init__(self, inner: InMemoryVectorStore) -> None:
        self.inner = inner
        self.searches: list[dict[str, object]] = []
        self.counts = 0

    def count(self, derived_from: str | None = None, run_id: str | None = None) -> int:
        self.counts += 1
        return self.inner.count(derived_from=derived_from, run_id=run_id)

    def search(
        self,
        vector: object,
        k: int,
        camera_ids: object = None,
        *,
        run_id: str | None = None,
        derived_from: str | None = None,
    ) -> list[tuple[AppearanceEmbedding, float]]:
        self.searches.append(
            {
                "k": k,
                "camera_ids": camera_ids,
                "run_id": run_id,
                "derived_from": derived_from,
            }
        )
        return self.inner.search(
            vector,
            k,
            camera_ids,
            run_id=run_id,
            derived_from=derived_from,
        )


@dataclass
class _World:
    tmp: Path
    tracks: TrackStore
    attrs: TrackAttributeStore
    vectors: InMemoryVectorStore
    embedder: _FixedEmbedder
    audit: AuditLog
    track_spy: _TrackSpy
    attr_spy: _AttrSpy
    vector_spy: _VectorSpy
    service: SearchService


def _open_world(tmp_path: Path, vector: tuple[float, ...] = _AXIS) -> _World:
    """Stores, audit log, and spies under tmp_path. No search has run."""
    key_path = tmp_path / "audit.hmac"
    init_hmac_key(key_path)
    audit = AuditLog(tmp_path / "audit.jsonl", key_path=key_path)
    audit.initialize()
    tracks = TrackStore(tmp_path / "tracks.sqlite")
    attrs = TrackAttributeStore(tmp_path / "attributes.sqlite")
    vectors = InMemoryVectorStore(3)
    embedder = _FixedEmbedder(vector)
    track_spy = _TrackSpy(tracks)
    attr_spy = _AttrSpy(attrs)
    vector_spy = _VectorSpy(vectors)
    service = SearchService(track_spy, attr_spy, vector_spy, embedder, audit)
    return _World(
        tmp=tmp_path,
        tracks=tracks,
        attrs=attrs,
        vectors=vectors,
        embedder=embedder,
        audit=audit,
        track_spy=track_spy,
        attr_spy=attr_spy,
        vector_spy=vector_spy,
        service=service,
    )


def _observation(
    camera_id: str,
    track_id: int,
    frame_number: int,
    timestamp: float,
    box: tuple[float, float, float, float] = (0.0, 0.0, 4.0, 4.0),
) -> TrackObservation:
    return TrackObservation(
        camera_id=camera_id,
        track_id=track_id,
        det_class=DetClass.PERSON,
        frame_number=frame_number,
        timestamp_seconds=timestamp,
        box=box,
        score=0.9,
    )


def _add(
    world: _World,
    *,
    camera_id: str,
    track_id: int,
    t_start: float,
    t_end: float | None = None,
    upper: str = "red",
    lower: str = "blue",
    bag: bool = False,
    vector: tuple[float, ...] | None = None,
    box: tuple[float, float, float, float] = (0.0, 0.0, 4.0, 4.0),
    frame_number: int = 0,
    attributes: bool = True,
) -> None:
    end = t_start if t_end is None else t_end
    observations = [_observation(camera_id, track_id, frame_number, t_start, box)]
    if end != t_start:
        observations.append(_observation(camera_id, track_id, frame_number + 1, end, box))
    world.tracks.write("run-1", "code-fixture", "CODE_TEST", observations)
    if attributes:
        world.attrs.write(
            camera_id,
            track_id,
            "run-1",
            "CODE_TEST",
            upper,
            lower,
            bag,
            0.8,
            0.8,
            1,
        )
    if vector is not None:
        _add_vector(world, camera_id, track_id, frame_number, vector)


def _add_vector(
    world: _World,
    camera_id: str,
    track_id: int,
    frame_number: int,
    vector: tuple[float, ...],
) -> None:
    world.vectors.add(
        "run-1",
        "CODE_TEST",
        [
            AppearanceEmbedding(
                camera_id=camera_id,
                track_id=track_id,
                frame_number=frame_number,
                vector=vector,
                embedder_name="hand-built",
                placeholder=True,
            )
        ],
    )


def _assert_no_search(world: _World) -> None:
    assert world.audit.count() == 0
    assert world.audit.read_all() == []
    assert world.embedder.calls == 0
    assert world.attr_spy.queries == []
    assert world.vector_spy.searches == []
    assert world.vector_spy.counts == 0
    assert world.track_spy.touches == 0
    assert world.track_spy.db_hits == 0


def _assert_only_db_and_audit(tmp_path: Path) -> None:
    found = {
        path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()
    }
    assert found == set(_ALLOWED_FILES)


def _base(method: str) -> dict[str, object]:
    crop = np.zeros((2, 2, 3), dtype=np.uint8)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    common: dict[str, object] = {
        "operator": "ada",
        "case_id": "CASE-1",
        "purpose": "find the coat",
        "t_window": (0, 10),
        "camera_ids": ["cam-a"],
    }
    if method == "by_attributes":
        return {**common, "query": {"upper_color": "red"}, "limit": 5}
    if method == "by_example_crop":
        return {**common, "crop": crop, "k": 5}
    return {
        **common,
        "camera_id": "cam-a",
        "track_id": 1,
        "frame_number": 0,
        "frame": frame,
        "k": 5,
    }


def _call(world: _World, method: str, **overrides: object) -> list[TrackRef]:
    kwargs = _base(method)
    kwargs.update(overrides)
    called = getattr(world.service, method)
    result: list[TrackRef] = called(**kwargs)
    return result


_METHODS = ("by_attributes", "by_example_crop", "by_track")

_TEXT_CASES = (
    ("operator", None),
    ("operator", ""),
    ("operator", " "),
    ("operator", "\n"),
    ("operator", "\t"),
    ("operator", 1),
    ("operator", True),
    ("case_id", None),
    ("case_id", ""),
    ("case_id", " "),
    ("case_id", 2),
    ("purpose", None),
    ("purpose", ""),
    ("purpose", " "),
    ("purpose", b"why"),
)

_WINDOW_CASES = (
    (None, TypeError),
    ((1.0,), ValueError),
    ((1.0, 2.0, 3.0), ValueError),
    ("1-2", TypeError),
    (b"12", TypeError),
    ((float("nan"), 1.0), ValueError),
    ((1.0, float("inf")), ValueError),
    ((float("-inf"), 1.0), ValueError),
    ((2.0, 1.0), ValueError),
    ((True, 5.0), TypeError),
    ((1.0, False), TypeError),
    (("1", 2.0), TypeError),
    ((1.0, "2"), TypeError),
    ({0: 1.0, 1: 2.0}, TypeError),
)

_CAMERA_CASES = (
    None,
    [],
    "",
    "cam-a",
    b"cam-a",
    [1],
    ["cam-a", None],
    ["cam-a", ""],
    ["cam-a", " "],
    ["cam-a", 3],
    ["cam-a", b"cam-b"],
)


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_contract_is_documented() -> None:
    """Docstrings state overlap, sort order, and that scores are not accuracy."""
    module_doc = _flat(search_module.__doc__ or "")
    assert "inclusive overlap, not containment" in module_doc
    assert "t_end >= window start" in module_doc
    assert "t_start <= window end" in module_doc
    assert (
        "Results are sorted by score descending, then by t_start ascending, "
        "then by camera_id ascending, then by track_id ascending."
    ) in module_doc
    assert "placeholder similarity has no accuracy meaning" in module_doc
    crop_doc = _flat(SearchService.by_example_crop.__doc__ or "")
    track_doc = _flat(SearchService.by_track.__doc__ or "")
    assert "placeholder similarity has no accuracy meaning" in crop_doc
    assert "placeholder similarity has no accuracy meaning" in track_doc
    assert "Scores are not evidence" in crop_doc
    assert "crop is written" in track_doc


def test_track_ref_is_frozen() -> None:
    ref = TrackRef("cam-a", 1, 0.0, 1.0, None)
    with pytest.raises(AttributeError):
        ref.score = 1.0  # type: ignore[misc]


def test_socket_connect_is_refused() -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(OSError, match="search tests refuse socket.socket.connect"):
        probe.connect(("127.0.0.1", 9))
    probe.close()


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize(("field", "value"), _TEXT_CASES)
def test_text_rejected_before_search(
    tmp_path: Path, method: str, field: str, value: object
) -> None:
    world = _open_world(tmp_path)
    with pytest.raises(ValueError):
        _call(world, method, **{field: value})
    _assert_no_search(world)


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize(("window", "exc"), _WINDOW_CASES)
def test_window_rejected_before_search(
    tmp_path: Path, method: str, window: object, exc: type[BaseException]
) -> None:
    world = _open_world(tmp_path)
    with pytest.raises(exc):
        _call(world, method, t_window=window)
    _assert_no_search(world)


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("camera_ids", _CAMERA_CASES)
def test_cameras_rejected_before_search(tmp_path: Path, method: str, camera_ids: object) -> None:
    world = _open_world(tmp_path)
    with pytest.raises(ValueError):
        _call(world, method, camera_ids=camera_ids)
    _assert_no_search(world)


def test_camera_generator_rejected_before_search(tmp_path: Path) -> None:
    world = _open_world(tmp_path)
    cameras: Iterator[str] = (item for item in ["cam-a"])
    with pytest.raises(ValueError):
        _call(world, "by_attributes", camera_ids=cameras)
    _assert_no_search(world)


@pytest.mark.parametrize("limit", (0, -1, True, False, 1.5, "1", None))
def test_limit_rejected_before_search(tmp_path: Path, limit: object) -> None:
    world = _open_world(tmp_path)
    with pytest.raises(ValueError):
        _call(world, "by_attributes", limit=limit)
    _assert_no_search(world)


@pytest.mark.parametrize("method", ("by_example_crop", "by_track"))
@pytest.mark.parametrize("k", (0, -1, True, False, 1.5, "1", None))
def test_k_rejected_before_search(tmp_path: Path, method: str, k: object) -> None:
    world = _open_world(tmp_path)
    with pytest.raises(ValueError):
        _call(world, method, k=k)
    _assert_no_search(world)


@pytest.mark.parametrize(
    ("query", "exc"),
    (
        ({"upper_color": "crimson"}, ValueError),
        ({"upper_color": "Red"}, ValueError),
        ({"lower_color": "plaid"}, ValueError),
        ({"bag_on_track": 1}, TypeError),
        ({"bag_on_track": 0}, TypeError),
        ({"bag_on_track": "yes"}, TypeError),
        ({"min_confidence": True}, TypeError),
        ({"not_a_key": True}, ValueError),
        (None, TypeError),
        ([], TypeError),
    ),
)
def test_attribute_query_rejected_before_search(
    tmp_path: Path, query: object, exc: type[BaseException]
) -> None:
    world = _open_world(tmp_path)
    with pytest.raises(exc):
        _call(world, "by_attributes", query=query)
    _assert_no_search(world)


@pytest.mark.parametrize(
    "crop",
    (
        [1, 2, 3],
        None,
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 3, 1), dtype=np.uint8),
        np.zeros((0, 2, 3), dtype=np.uint8),
        np.zeros((2, 0, 3), dtype=np.uint8),
        np.zeros((2, 2, 0), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((2, 2, 3), dtype=np.uint16),
    ),
)
def test_crop_rejected_before_search(tmp_path: Path, crop: object) -> None:
    world = _open_world(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        _call(world, "by_example_crop", crop=crop)
    _assert_no_search(world)


@pytest.mark.parametrize(
    "frame",
    (
        np.zeros((4, 4), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((0, 2, 3), dtype=np.uint8),
        None,
    ),
)
def test_frame_rejected_before_lookup(tmp_path: Path, frame: object) -> None:
    world = _open_world(tmp_path)
    _add(world, camera_id="cam-a", track_id=1, t_start=1.0, vector=_AXIS)
    with pytest.raises((TypeError, ValueError)):
        _call(world, "by_track", frame=frame)
    _assert_no_search(world)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("track_id", True),
        ("track_id", -1),
        ("track_id", 1.0),
        ("frame_number", True),
        ("frame_number", -1),
        ("camera_id", ""),
        ("camera_id", None),
    ),
)
def test_track_selector_rejected_before_lookup(tmp_path: Path, field: str, value: object) -> None:
    world = _open_world(tmp_path)
    _add(world, camera_id="cam-a", track_id=1, t_start=1.0, vector=_AXIS)
    with pytest.raises(ValueError):
        _call(world, "by_track", **{field: value})
    _assert_no_search(world)


def test_missing_observation_writes_no_audit(tmp_path: Path) -> None:
    world = _open_world(tmp_path)
    _add(world, camera_id="cam-a", track_id=1, t_start=1.0, vector=_AXIS)
    with pytest.raises(ValueError, match="no stored observation"):
        _call(world, "by_track", track_id=99)
    assert world.audit.count() == 0
    assert world.embedder.calls == 0
    assert world.vector_spy.searches == []


@pytest.mark.parametrize(
    "box",
    (
        (-10.0, -10.0, -1.0, -1.0),
        (-1.0, 0.0, 0.0, 2.0),
        (0.0, -1.0, 2.0, 0.0),
    ),
)
def test_box_outside_frame_writes_no_audit(
    tmp_path: Path, box: tuple[float, float, float, float]
) -> None:
    world = _open_world(tmp_path)
    _add(
        world,
        camera_id="cam-a",
        track_id=1,
        t_start=1.0,
        vector=_AXIS,
        box=box,
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not overlap"):
        _call(world, "by_track", frame=frame)
    assert world.audit.count() == 0
    assert world.embedder.calls == 0
    assert world.vector_spy.searches == []


def test_attribute_query_returns_overlap_inside_cameras(tmp_path: Path) -> None:
    """Matching tracks only. Edges touch the window. A covering track is inside."""
    world = _open_world(tmp_path)
    # Included: contained, ends on the start, starts on the end, covers the window.
    _add(world, camera_id="cam-a", track_id=1, t_start=12.0, t_end=18.0, lower="black")
    _add(world, camera_id="cam-a", track_id=2, t_start=0.0, t_end=10.0)
    _add(world, camera_id="cam-a", track_id=3, t_start=20.0, t_end=25.0)
    _add(world, camera_id="cam-a", track_id=6, t_start=0.0, t_end=30.0, lower="green")
    # Excluded: just outside, wrong colour, bag set, other camera.
    _add(world, camera_id="cam-a", track_id=4, t_start=0.0, t_end=9.0)
    _add(world, camera_id="cam-a", track_id=5, t_start=21.0, t_end=30.0)
    _add(world, camera_id="cam-a", track_id=7, t_start=12.0, t_end=18.0, upper="blue")
    _add(world, camera_id="cam-a", track_id=8, t_start=12.0, t_end=18.0, bag=True)
    _add(world, camera_id="cam-b", track_id=9, t_start=12.0, t_end=18.0)
    query = {"upper_color": "red", "bag_on_track": False}
    refs = world.service.by_attributes(
        "ada",
        "CASE-1",
        "find the coat",
        query,
        (10, 20),
        ["cam-a"],
    )
    assert refs == [
        TrackRef("cam-a", 2, 0.0, 10.0, None),
        TrackRef("cam-a", 6, 0.0, 30.0, None),
        TrackRef("cam-a", 1, 12.0, 18.0, None),
        TrackRef("cam-a", 3, 20.0, 25.0, None),
    ]
    assert all(item.score is None for item in refs)
    assert world.attr_spy.queries == [(None, None)]
    assert world.audit.count() == 1
    entry = world.audit.read_all()[0]
    assert entry.action == "search"
    assert entry.operator_username == "ada"
    assert entry.case_id == "CASE-1"
    assert entry.purpose == "find the coat"
    assert entry.query == '{"bag_on_track":false,"upper_color":"red"}'
    assert entry.result_ids == ["cam-a:2", "cam-a:6", "cam-a:1", "cam-a:3"]
    assert entry.details is None


def test_point_window_includes_a_track_on_that_instant(tmp_path: Path) -> None:
    world = _open_world(tmp_path)
    _add(world, camera_id="cam-a", track_id=1, t_start=10.0, t_end=10.0)
    _add(world, camera_id="cam-a", track_id=2, t_start=9.0, t_end=9.0)
    _add(world, camera_id="cam-a", track_id=3, t_start=0.0, t_end=12.0)
    refs = world.service.by_attributes(
        "ada",
        "CASE-1",
        "find the coat",
        {"upper_color": "red"},
        (10, 10),
        ["cam-a"],
    )
    assert [(item.track_id, item.t_start, item.t_end) for item in refs] == [
        (3, 0.0, 12.0),
        (1, 10.0, 10.0),
    ]


def test_attribute_limit_keeps_the_earliest(tmp_path: Path) -> None:
    world = _open_world(tmp_path)
    _add(world, camera_id="cam-a", track_id=1, t_start=30.0)
    _add(world, camera_id="cam-a", track_id=2, t_start=10.0)
    _add(world, camera_id="cam-a", track_id=3, t_start=20.0)
    refs = world.service.by_attributes(
        "ada",
        "CASE-1",
        "find the coat",
        {"upper_color": "red"},
        (0, 40),
        ["cam-a"],
        limit=2,
    )
    assert [item.track_id for item in refs] == [2, 3]
    assert world.audit.read_all()[0].result_ids == ["cam-a:2", "cam-a:3"]


def test_duplicate_attribute_rows_collapse(tmp_path: Path) -> None:
    world = _open_world(tmp_path)
    _add(world, camera_id="cam-a", track_id=1, t_start=1.0, t_end=4.0)
    world.attrs.write(
        "cam-a",
        1,
        "run-1",
        "CODE_TEST",
        "red",
        "blue",
        False,
        0.8,
        0.8,
        1,
    )
    refs = world.service.by_attributes(
        "ada",
        "CASE-1",
        "find the coat",
        {"upper_color": "red"},
        (0, 10),
        ["cam-a"],
    )
    assert refs == [TrackRef("cam-a", 1, 1.0, 4.0, None)]


def test_same_track_in_two_runs_keeps_the_first_span(tmp_path: Path) -> None:
    """Two runs share a camera and track id. The first span is the one returned."""
    world = _open_world(tmp_path)
    world.tracks.write(
        "run-1",
        "code-fixture",
        "CODE_TEST",
        [
            _observation("cam-a", 1, 0, 1.0),
            _observation("cam-a", 1, 1, 4.0),
        ],
    )
    world.tracks.write(
        "run-2",
        "code-fixture",
        "CODE_TEST",
        [
            _observation("cam-a", 1, 0, 6.0),
            _observation("cam-a", 1, 1, 9.0),
        ],
    )
    world.attrs.write("cam-a", 1, "run-1", "CODE_TEST", "red", "blue", False, 0.8, 0.8, 1)
    world.attrs.write("cam-a", 1, "run-2", "CODE_TEST", "red", "blue", False, 0.8, 0.8, 1)
    refs = world.service.by_attributes(
        "ada",
        "CASE-1",
        "find the coat",
        {"upper_color": "red"},
        (0, 10),
        ["cam-a"],
    )
    assert refs == [TrackRef("cam-a", 1, 1.0, 4.0, None)]


def test_attribute_ties_use_camera_then_track(tmp_path: Path) -> None:
    world = _open_world(tmp_path)
    _add(world, camera_id="cam-a", track_id=4, t_start=0.0)
    _add(world, camera_id="cam-a", track_id=3, t_start=5.0)
    _add(world, camera_id="cam-a", track_id=2, t_start=5.0)
    _add(world, camera_id="cam-b", track_id=1, t_start=5.0)
    _add(world, camera_id="cam-a", track_id=1, t_start=10.0)
    refs = world.service.by_attributes(
        "ada",
        "CASE-1",
        "find the coat",
        {"upper_color": "red"},
        (0, 20),
        ["cam-a", "cam-b"],
    )
    assert [(item.camera_id, item.track_id) for item in refs] == [
        ("cam-a", 4),
        ("cam-a", 2),
        ("cam-a", 3),
        ("cam-b", 1),
        ("cam-a", 1),
    ]


def _crop_world(tmp_path: Path) -> _World:
    """Five tracks. Score 0 has the earliest time so a time-only sort would promote it."""
    world = _open_world(tmp_path, _AXIS)
    _add(world, camera_id="cam-a", track_id=1, t_start=10.0, vector=_AXIS)
    _add(world, camera_id="cam-a", track_id=2, t_start=1.0, vector=_AXIS)
    _add(world, camera_id="cam-a", track_id=3, t_start=1.0, vector=_AXIS)
    _add(world, camera_id="cam-b", track_id=1, t_start=1.0, vector=_AXIS)
    _add(world, camera_id="cam-a", track_id=4, t_start=0.0, vector=_ORTH)
    return world


def test_crop_order_is_score_then_time_then_ids(tmp_path: Path) -> None:
    world = _crop_world(tmp_path)
    crop = np.zeros((4, 4, 3), dtype=np.uint8)
    crop.fill(7)
    refs = world.service.by_example_crop(
        "ada",
        "CASE-1",
        "find the coat",
        crop,
        (0, 20),
        ["cam-a", "cam-b"],
        k=5,
    )
    assert [(item.camera_id, item.track_id) for item in refs] == [
        ("cam-a", 2),
        ("cam-a", 3),
        ("cam-b", 1),
        ("cam-a", 1),
        ("cam-a", 4),
    ]
    assert refs[0].score == pytest.approx(1.0)
    assert refs[-1].score == pytest.approx(0.0)
    assert refs[0].t_start == 1.0
    entry = world.audit.read_all()[0]
    assert entry.query == '{"crop_dtype":"uint8","crop_shape":[4,4,3],"kind":"example_crop"}'
    assert "7,7" not in (entry.query or "")
    assert entry.result_ids == ["cam-a:2", "cam-a:3", "cam-b:1", "cam-a:1", "cam-a:4"]
    raw = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "7,7" not in raw


def test_crop_k_truncates_after_the_sort(tmp_path: Path) -> None:
    world = _crop_world(tmp_path)
    crop = np.ones((2, 2, 3), dtype=np.uint8)
    refs = world.service.by_example_crop(
        "ada",
        "CASE-1",
        "find the coat",
        crop,
        (0, 20),
        ["cam-a", "cam-b"],
        k=2,
    )
    assert [(item.camera_id, item.track_id) for item in refs] == [
        ("cam-a", 2),
        ("cam-a", 3),
    ]


def test_nearest_placeholder_vector_is_first(tmp_path: Path) -> None:
    world = _open_world(tmp_path, _AXIS)
    _add(world, camera_id="cam-a", track_id=1, t_start=5.0, vector=_DIAG)
    _add(world, camera_id="cam-a", track_id=2, t_start=1.0, vector=_ORTH)
    _add_vector(world, "cam-a", 1, 2, _AXIS)
    crop = np.zeros((2, 2, 3), dtype=np.uint8)
    refs = world.service.by_example_crop(
        "ada",
        "CASE-1",
        "find the coat",
        crop,
        (0, 10),
        ["cam-a"],
        k=2,
    )
    assert [item.track_id for item in refs] == [1, 2]
    assert refs[0].score == pytest.approx(1.0)
    assert refs[1].score == pytest.approx(0.0)
    assert refs[0].t_start == 5.0


def test_stray_vector_that_would_win_is_scoped_out(tmp_path: Path) -> None:
    """The stray row is the store's top hit and would fill k=1 by itself."""
    world = _open_world(tmp_path, _AXIS)
    _add(world, camera_id="cam-a", track_id=1, t_start=5.0, vector=_DIAG)
    _add_vector(world, "cam-a", 999, 0, _AXIS)
    crop = np.zeros((2, 2, 3), dtype=np.uint8)
    refs = world.service.by_example_crop(
        "ada",
        "CASE-1",
        "find the coat",
        crop,
        (0, 10),
        ["cam-a"],
        k=1,
    )
    assert [(item.camera_id, item.track_id) for item in refs] == [("cam-a", 1)]
    assert refs[0].score == pytest.approx(float(np.dot(_AXIS, _DIAG)))
    assert world.vector_spy.searches
    call = world.vector_spy.searches[0]
    assert call["camera_ids"] is None
    assert call["k"] >= 2
    assert call["run_id"] is None
    assert call["derived_from"] is None


def test_better_vector_outside_the_window_is_excluded(tmp_path: Path) -> None:
    world = _open_world(tmp_path, _AXIS)
    _add(world, camera_id="cam-a", track_id=1, t_start=5.0, vector=_DIAG)
    _add(world, camera_id="cam-a", track_id=2, t_start=100.0, vector=_AXIS)
    crop = np.zeros((2, 2, 3), dtype=np.uint8)
    refs = world.service.by_example_crop(
        "ada",
        "CASE-1",
        "find the coat",
        crop,
        (0, 10),
        ["cam-a"],
        k=1,
    )
    assert [(item.camera_id, item.track_id) for item in refs] == [("cam-a", 1)]


def test_better_vector_on_another_camera_is_excluded(tmp_path: Path) -> None:
    world = _open_world(tmp_path, _AXIS)
    _add(world, camera_id="cam-a", track_id=1, t_start=5.0, vector=_DIAG)
    _add(world, camera_id="cam-b", track_id=1, t_start=5.0, vector=_AXIS)
    crop = np.zeros((2, 2, 3), dtype=np.uint8)
    refs = world.service.by_example_crop(
        "ada",
        "CASE-1",
        "find the coat",
        crop,
        (0, 10),
        ["cam-a"],
        k=1,
    )
    assert [(item.camera_id, item.track_id) for item in refs] == [("cam-a", 1)]


def test_vectors_and_no_tracks_return_empty(tmp_path: Path) -> None:
    world = _open_world(tmp_path, _AXIS)
    _add_vector(world, "cam-a", 1, 0, _AXIS)
    crop = np.zeros((2, 2, 3), dtype=np.uint8)
    refs = world.service.by_example_crop(
        "ada",
        "CASE-1",
        "find the coat",
        crop,
        (0, 10),
        ["cam-a"],
    )
    assert refs == []
    assert world.audit.count() == 1
    assert world.audit.read_all()[0].result_ids == []


def test_empty_vector_store_audits_an_empty_result(tmp_path: Path) -> None:
    world = _open_world(tmp_path, _AXIS)
    _add(world, camera_id="cam-a", track_id=1, t_start=1.0)
    crop = np.zeros((2, 2, 3), dtype=np.uint8)
    refs = world.service.by_example_crop(
        "ada",
        "CASE-1",
        "find the coat",
        crop,
        (0, 10),
        ["cam-a"],
    )
    assert refs == []
    assert world.vector_spy.counts == 1
    assert world.vector_spy.searches == []
    assert world.audit.count() == 1


@pytest.mark.parametrize(
    "rows",
    (
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((1,), dtype=np.float32),
        np.zeros((1, 3, 1), dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
    ),
)
def test_embed_must_return_one_row(tmp_path: Path, rows: npt.NDArray[np.float32]) -> None:
    world = _open_world(tmp_path, _AXIS)
    world.embedder.rows = rows
    crop = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="one row"):
        world.service.by_example_crop(
            "ada",
            "CASE-1",
            "find the coat",
            crop,
            (0, 10),
            ["cam-a"],
        )
    assert world.audit.count() == 0
    assert world.vector_spy.searches == []


def test_by_track_cuts_the_box_in_memory(tmp_path: Path) -> None:
    world = _open_world(tmp_path, _AXIS)
    box = (1.0, 2.0, 4.0, 5.0)
    other = (0.0, 0.0, 1.0, 1.0)
    # An earlier frame of the same track must be skipped. The requested frame is 4.
    world.tracks.write(
        "run-1",
        "code-fixture",
        "CODE_TEST",
        [_observation("cam-a", 7, 1, 3.0, box)],
    )
    _add(
        world,
        camera_id="cam-a",
        track_id=7,
        t_start=4.0,
        vector=_DIAG,
        box=box,
        frame_number=4,
    )
    _add(
        world,
        camera_id="cam-a",
        track_id=8,
        t_start=4.0,
        vector=_ORTH,
        box=other,
        frame_number=0,
    )
    _add(world, camera_id="cam-b", track_id=1, t_start=4.0, vector=_AXIS)
    frame = np.zeros((6, 6, 3), dtype=np.uint8)
    frame[2:5, 1:4] = 7
    refs = world.service.by_track(
        "ada",
        "CASE-9",
        "click",
        "cam-a",
        7,
        4,
        frame,
        (0, 10),
        ["cam-a"],
        k=1,
    )
    assert world.embedder.calls == 1
    seen = world.embedder.crops[0]
    assert seen.shape == (3, 3, 3)
    assert int(seen.min()) == 7
    assert int(seen.max()) == 7
    assert [(item.camera_id, item.track_id) for item in refs] == [("cam-a", 7)]
    entry = world.audit.read_all()[0]
    assert entry.case_id == "CASE-9"
    assert entry.purpose == "click"
    expected_query = (
        '{"camera_id":"cam-a","crop_dtype":"uint8","crop_shape":[3,3,3],'
        '"frame_number":4,"kind":"by_track","track_id":7}'
    )
    assert entry.query == expected_query
    assert "7,7" not in (entry.query or "")
    _assert_only_db_and_audit(tmp_path)


def test_track_id_zero_and_frame_zero_are_valid(tmp_path: Path) -> None:
    world = _open_world(tmp_path, _AXIS)
    _add(world, camera_id="cam-a", track_id=0, t_start=1.0, vector=_AXIS, frame_number=0)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    refs = world.service.by_track(
        "ada",
        "CASE-1",
        "click",
        "cam-a",
        0,
        0,
        frame,
        (0, 5),
        ["cam-a"],
    )
    assert [(item.camera_id, item.track_id) for item in refs] == [("cam-a", 0)]


def test_audit_failure_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = _open_world(tmp_path, _AXIS)
    _add(
        world,
        camera_id="cam-a",
        track_id=1,
        t_start=1.0,
        t_end=2.0,
        vector=_AXIS,
        box=(0.0, 0.0, 2.0, 2.0),
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    crop = np.zeros((2, 2, 3), dtype=np.uint8)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("audit append failed")

    monkeypatch.setattr(world.audit, "append", boom)
    with pytest.raises(RuntimeError, match="audit append failed"):
        world.service.by_attributes(
            "ada",
            "CASE-1",
            "find the coat",
            {"upper_color": "red"},
            (0, 10),
            ["cam-a"],
        )
    with pytest.raises(RuntimeError, match="audit append failed"):
        world.service.by_example_crop(
            "ada",
            "CASE-1",
            "find the coat",
            crop,
            (0, 10),
            ["cam-a"],
        )
    with pytest.raises(RuntimeError, match="audit append failed"):
        world.service.by_track(
            "ada",
            "CASE-1",
            "click",
            "cam-a",
            1,
            0,
            frame,
            (0, 10),
            ["cam-a"],
        )
    assert world.audit.count() == 0
    assert world.audit.read_all() == []


def test_ten_mixed_calls_verify_and_skip_a_missing_case(tmp_path: Path) -> None:
    world = _open_world(tmp_path, _AXIS)
    _add(
        world,
        camera_id="cam-a",
        track_id=1,
        t_start=1.0,
        t_end=3.0,
        vector=_AXIS,
        box=(0.0, 0.0, 2.0, 2.0),
    )
    crop = np.zeros((2, 2, 3), dtype=np.uint8)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    calls = (
        lambda: world.service.by_attributes(
            "ada", "CASE-1", "find the coat", {"upper_color": "red"}, (0, 10), ["cam-a"]
        ),
        lambda: world.service.by_attributes(
            "ada", "CASE-1", "find the coat", {"upper_color": "blue"}, (0, 10), ["cam-a"]
        ),
        lambda: world.service.by_attributes(
            "ada", "CASE-1", "find the coat", {"upper_color": "red"}, (0, 10), ["cam-a"], limit=1
        ),
        lambda: world.service.by_example_crop(
            "ada", "CASE-1", "find the coat", crop, (0, 10), ["cam-a"]
        ),
        lambda: world.service.by_example_crop(
            "ada", "CASE-1", "find the coat", crop, (0, 10), ["cam-a"], k=1
        ),
        lambda: world.service.by_track(
            "ada", "CASE-1", "click", "cam-a", 1, 0, frame, (0, 10), ["cam-a"]
        ),
        lambda: world.service.by_track(
            "ada", "CASE-2", "click", "cam-a", 1, 0, frame, (0, 10), ["cam-a"], k=1
        ),
        lambda: world.service.by_attributes(
            "bea", "CASE-2", "again", {"upper_color": "red"}, (0, 10), ["cam-a"]
        ),
        lambda: world.service.by_example_crop("bea", "CASE-2", "again", crop, (0, 10), ["cam-a"]),
        lambda: world.service.by_attributes(
            "ada", "CASE-3", "third", {"lower_color": "blue"}, (0, 10), ["cam-a"]
        ),
    )
    for index, call in enumerate(calls, start=1):
        call()
        assert world.audit.count() == index
        if index == 3:
            with pytest.raises(ValueError):
                world.service.by_attributes(
                    "ada",
                    None,  # type: ignore[arg-type]
                    "find the coat",
                    {"upper_color": "red"},
                    (0, 10),
                    ["cam-a"],
                )
            assert world.audit.count() == 3
    entries = world.audit.read_all()
    assert len(entries) == 10
    assert [entry.action for entry in entries] == ["search"] * 10
    assert all(entry.case_id for entry in entries)
    assert all(entry.purpose for entry in entries)
    checked = world.audit.verify()
    assert checked.valid is True
    assert checked.total_entries == 10
    assert checked.broken_index is None
    _assert_only_db_and_audit(tmp_path)


def test_tmp_path_holds_only_db_and_audit_after_crop(tmp_path: Path) -> None:
    world = _open_world(tmp_path, _AXIS)
    _add(world, camera_id="cam-a", track_id=1, t_start=1.0, vector=_AXIS)
    crop = np.full((3, 3, 3), 7, dtype=np.uint8)
    world.service.by_example_crop(
        "ada",
        "CASE-1",
        "find the coat",
        crop,
        (0, 10),
        ["cam-a"],
    )
    _assert_only_db_and_audit(tmp_path)
