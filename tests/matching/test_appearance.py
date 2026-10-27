"""Track appearance pass: colours, bag flag, and one placeholder vector.

The video source is an in-memory numpy fake. No media file is opened.
SQLite files sit under pytest's tmp_path, outside the repo.
"""

import math
import socket
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from trace_search.attributes.attributes import AttributeExtractor, Color
from trace_search.attributes.store import TrackAttributeStore
from trace_search.detection.types import DetClass
from trace_search.ingest.video_source import Frame, VideoSource
from trace_search.matching.appearance import AppearanceStats, describe_tracks
from trace_search.matching.tracklets import Tracklet, load_tracklets
from trace_search.reid.placeholder import UntrainedPlaceholderEmbedder
from trace_search.reid.sqlite_vector_store import SqliteVectorStore
from trace_search.reid.types import AppearanceEmbedder
from trace_search.reid.vector_store import InMemoryVectorStore, VectorStore
from trace_search.tracking.store import TrackStore
from trace_search.tracking.types import TrackObservation

_CAMERA = "cam-1"
_OTHER_CAMERA = "cam-other"
_RUN = "run-appearance-1"
_ORIGIN = "CODE_TEST"
_FIXTURE = "FX-CODE-1"
_NOW = 1_700_000_000.0
_WIDTH = 240
_HEIGHT = 260
_RED = (0, 0, 220)
_BLUE = (220, 0, 0)
_GREEN = (0, 200, 0)
_YELLOW = (0, 220, 220)
_GRAY = (100, 100, 100)
_BOX_RED = (10.0, 20.0, 60.0, 220.0)
_BOX_WITH_BAG = (120.0, 30.0, 190.0, 230.0)
_BOX_BAG = (160.0, 100.0, 210.0, 160.0)
_BOX_PACK = (200.0, 5.0, 230.0, 35.0)
_BOX_TALL = (10.0, 20.0, 60.0, 220.0)
_BOX_SHORT = (80.0, 20.0, 110.0, 100.0)
_BOX_OUTSIDE = (1000.0, 1000.0, 1100.0, 1200.0)
_SAMPLES = 5
_MEDIA_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".npy",
    ".npz",
    ".tif",
    ".tiff",
    ".gif",
}
_Key = tuple[str, int, int]


class ArraySource(VideoSource):
    """Frames kept in a list. Nothing is read from disk."""

    def __init__(
        self,
        camera_id: str,
        images: Sequence[npt.NDArray[np.uint8]],
        frame_numbers: Sequence[int] | None = None,
    ) -> None:
        super().__init__(camera_id)
        self._images = list(images)
        self._numbers = list(range(len(self._images)) if frame_numbers is None else frame_numbers)
        self._index = 0
        self.reads = 0

    def open(self) -> None:
        self._index = 0
        self._is_open = True

    def close(self) -> None:
        self._is_open = False

    def read_frame(self) -> Frame | None:
        self.reads += 1
        if self._index >= len(self._images):
            return None
        image = self._images[self._index]
        number = self._numbers[self._index]
        self._index += 1
        return Frame(
            image=image,
            timestamp_seconds=float(number),
            frame_number=number,
            camera_id=self.camera_id,
        )

    def get_fps(self) -> float:
        return 10.0

    def get_frame_count(self) -> int | None:
        return len(self._images)


class _EmbedSpy:
    """Counts embed calls per crop shape. It does not keep the crops."""

    placeholder = True

    def __init__(self, inner: UntrainedPlaceholderEmbedder) -> None:
        self._inner = inner
        self.dim = inner.dim
        self.calls_by_shape: dict[tuple[int, int], int] = {}
        self.crops_by_shape: dict[tuple[int, int], int] = {}

    @property
    def name(self) -> str:
        return self._inner.name

    def embed(self, crops: Sequence[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        shapes = [(int(crop.shape[0]), int(crop.shape[1])) for crop in crops]
        for shape in set(shapes):
            self.calls_by_shape[shape] = self.calls_by_shape.get(shape, 0) + 1
        for shape in shapes:
            self.crops_by_shape[shape] = self.crops_by_shape.get(shape, 0) + 1
        return self._inner.embed(crops)


@dataclass
class _PersonPaint:
    track_id: int
    box: tuple[float, float, float, float]
    upper: tuple[int, int, int]
    lower: tuple[int, int, int]
    frames: tuple[int, ...]


@dataclass
class _BagPaint:
    track_id: int
    box: tuple[float, float, float, float]
    frames: tuple[int, ...]
    det_class: DetClass


@dataclass
class _Scene:
    source: ArraySource
    tracklets: list[Tracklet]
    bag_tracklets: list[Tracklet]
    store: TrackStore
    embedder: UntrainedPlaceholderEmbedder
    extractor: AttributeExtractor


def _clock() -> float:
    return _NOW


@pytest.fixture(autouse=True)
def _refuse_outbound_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests refuse socket.socket.connect. Local SQLite does not connect."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("outbound network is refused")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _embedder() -> UntrainedPlaceholderEmbedder:
    return UntrainedPlaceholderEmbedder(dim=16, seed=0)


def _blank() -> npt.NDArray[np.uint8]:
    return np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)


def _paint(
    image: npt.NDArray[np.uint8],
    box: tuple[float, float, float, float],
    upper: tuple[int, int, int],
    lower: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    height = y2 - y1
    crop = image[y1:y2, x1:x2]
    head = int(0.25 * height)
    torso = int(0.55 * height)
    legs = int(0.90 * height)
    crop[:head] = _GRAY
    crop[head:torso] = upper
    crop[torso:legs] = lower
    crop[legs:] = _GRAY


def _observation(
    track_id: int,
    frame_number: int,
    box: tuple[float, float, float, float],
    det_class: DetClass = DetClass.PERSON,
) -> TrackObservation:
    return TrackObservation(
        camera_id=_CAMERA,
        track_id=track_id,
        det_class=det_class,
        frame_number=frame_number,
        timestamp_seconds=float(frame_number),
        box=box,
        score=0.9,
    )


def _manual(
    camera_id: str,
    track_id: int,
    frames: tuple[int, ...],
    det_class: DetClass = DetClass.PERSON,
) -> Tracklet:
    box = (1.0, 1.0, 30.0, 80.0)
    start = 0.0 if not frames else float(frames[0])
    end = start if not frames else float(frames[-1])
    return Tracklet(
        camera_id=camera_id,
        track_id=track_id,
        det_class=det_class,
        run_id=_RUN,
        t_start=start,
        t_end=end,
        n_obs=len(frames),
        first_box=box,
        last_box=box,
        frames=frames,
    )


def _build(
    tmp_path: Path,
    n_frames: int,
    people: Sequence[_PersonPaint],
    bags: Sequence[_BagPaint],
) -> _Scene:
    images = [_blank() for _ in range(n_frames)]
    observations: list[TrackObservation] = []
    for person in people:
        for frame_number in person.frames:
            _paint(images[frame_number], person.box, person.upper, person.lower)
            observations.append(_observation(person.track_id, frame_number, person.box))
    for bag in bags:
        for frame_number in bag.frames:
            observations.append(_observation(bag.track_id, frame_number, bag.box, bag.det_class))
    store = TrackStore(tmp_path / "tracks.sqlite", clock=_clock)
    store.write(_RUN, _FIXTURE, _ORIGIN, observations)
    loaded = load_tracklets(store, _RUN, det_class=DetClass.PERSON)
    bag_tracklets: list[Tracklet] = []
    bag_tracklets.extend(load_tracklets(store, _RUN, det_class=DetClass.HANDBAG))
    bag_tracklets.extend(load_tracklets(store, _RUN, det_class=DetClass.BACKPACK))
    return _Scene(
        source=ArraySource(_CAMERA, images),
        tracklets=loaded,
        bag_tracklets=bag_tracklets,
        store=store,
        embedder=_embedder(),
        extractor=AttributeExtractor(),
    )


def _parade(tmp_path: Path) -> _Scene:
    """Nine frames. Five even indexes are sampled. The handbag is on three of them."""
    frames = tuple(range(9))
    return _build(
        tmp_path,
        9,
        (
            _PersonPaint(1, _BOX_RED, _RED, _BLUE, frames),
            _PersonPaint(2, _BOX_WITH_BAG, _GREEN, _YELLOW, frames),
        ),
        (
            _BagPaint(7, _BOX_BAG, (0, 4, 8), DetClass.HANDBAG),
            _BagPaint(8, _BOX_PACK, (0,), DetClass.BACKPACK),
        ),
    )


def _attributes(tmp_path: Path, leaf: str = "attributes.sqlite") -> TrackAttributeStore:
    return TrackAttributeStore(tmp_path / leaf, clock=_clock)


def _describe(
    scene: _Scene,
    vectors: VectorStore,
    attr_store: TrackAttributeStore,
    embedder: AppearanceEmbedder | None = None,
    samples_per_track: int = _SAMPLES,
    derived_from: str = _ORIGIN,
) -> AppearanceStats:
    return describe_tracks(
        scene.source,
        scene.tracklets,
        scene.bag_tracklets,
        scene.store,
        _RUN,
        scene.extractor,
        scene.embedder if embedder is None else embedder,
        attr_store,
        vectors,
        derived_from,
        samples_per_track,
    )


def _rows(attr_store: TrackAttributeStore) -> dict[int, tuple[object, ...]]:
    found: dict[int, tuple[object, ...]] = {}
    for record in attr_store.read(_CAMERA, run_id=_RUN):
        found[record.track_id] = (
            record.upper_color,
            record.lower_color,
            record.bag_on_track,
            record.upper_confidence,
            record.lower_confidence,
            record.n_frames,
            record.derived_from,
            record.created_at,
        )
    return found


def _probe(embedder: UntrainedPlaceholderEmbedder) -> npt.NDArray[np.float32]:
    crop = np.zeros((64, 32, 3), dtype=np.uint8)
    crop[:, :] = _RED
    return embedder.embed([crop])[0]


def _norm(vector: tuple[float, ...]) -> float:
    return math.sqrt(sum(item * item for item in vector))


def _score_map(vectors: VectorStore, probe: npt.NDArray[np.float32]) -> dict[_Key, float]:
    """Key and score only. The vector itself is not part of the map."""
    hits = vectors.search(probe, k=10)
    scored: dict[_Key, float] = {}
    for embedding, score in hits:
        key = (embedding.camera_id, embedding.track_id, embedding.frame_number)
        assert key not in scored
        scored[key] = float(score)
    return scored


def test_socket_connect_is_refused() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(OSError, match="outbound network is refused"):
        sock.connect(("127.0.0.1", 9))
    sock.close()


def test_scripted_handbag_sits_beside_the_person() -> None:
    """The bag extends past the person and still clears the containment bar."""
    assigned = AttributeExtractor.assign_bags([_BOX_RED, _BOX_WITH_BAG], [_BOX_BAG])
    assert assigned == [False, True]
    ratio = AttributeExtractor._containment_ratio(_BOX_WITH_BAG, _BOX_BAG)
    assert ratio > 0.5


def test_red_over_blue_and_handbag_on_three_of_five_sampled_frames(tmp_path: Path) -> None:
    scene = _parade(tmp_path)
    vectors = InMemoryVectorStore(scene.embedder.dim)
    attr_store = _attributes(tmp_path)
    stats = _describe(scene, vectors, attr_store)

    assert scene.source.reads == 10
    assert stats.frames_read == 9
    assert stats.tracks_described == 2
    assert not scene.source.is_open()
    rows = _rows(attr_store)
    assert rows[1][0] == Color.RED.value
    assert rows[1][1] == Color.BLUE.value
    assert rows[1][2] is False
    assert rows[1][5] == _SAMPLES
    assert float(rows[1][3]) > 0.5
    assert float(rows[1][4]) > 0.5
    assert rows[2][0] == Color.GREEN.value
    assert rows[2][1] == Color.YELLOW.value
    assert rows[2][2] is True
    assert rows[2][5] == _SAMPLES
    assert rows[1][6] == _ORIGIN
    assert rows[1][7] == _NOW


def test_bag_only_on_unsampled_frames_stays_off(tmp_path: Path) -> None:
    """Bags on frames 1, 3, 5 and 7 miss the five even samples."""
    frames = tuple(range(9))
    scene = _build(
        tmp_path,
        9,
        (_PersonPaint(1, _BOX_WITH_BAG, _RED, _BLUE, frames),),
        (_BagPaint(7, _BOX_BAG, (1, 3, 5, 7), DetClass.HANDBAG),),
    )
    vectors = InMemoryVectorStore(scene.embedder.dim)
    attr_store = _attributes(tmp_path)
    _describe(scene, vectors, attr_store)
    row = _rows(attr_store)[1]
    assert row[0] == Color.RED.value
    assert row[1] == Color.BLUE.value
    assert row[2] is False
    assert row[5] == _SAMPLES


def test_person_tracks_get_one_unit_placeholder_vector_and_bag_tracks_get_none(
    tmp_path: Path,
) -> None:
    scene = _parade(tmp_path)
    vectors = InMemoryVectorStore(scene.embedder.dim)
    attr_store = _attributes(tmp_path)
    stats = _describe(scene, vectors, attr_store)

    assert stats.embeddings_stored == 2
    assert vectors.count(run_id=_RUN) == 2
    scored = _score_map(vectors, _probe(scene.embedder))
    assert {key[1] for key in scored} == {1, 2}
    hits = vectors.search(_probe(scene.embedder), k=2)
    for embedding, _score in hits:
        assert embedding.placeholder is True
        assert embedding.embedder_name == scene.embedder.name
        assert embedding.camera_id == _CAMERA
        assert embedding.frame_number == 0
        assert _norm(embedding.vector) == pytest.approx(1.0, abs=1e-5)
        assert embedding.track_id not in {7, 8}


def test_tied_colours_are_unknown(tmp_path: Path) -> None:
    paints = [
        (_RED, _BLUE),
        (_RED, _BLUE),
        (_BLUE, _RED),
        (_BLUE, _RED),
    ]
    scene = _one_person(tmp_path, paints)
    vectors = InMemoryVectorStore(scene.embedder.dim)
    attr_store = _attributes(tmp_path)
    _describe(scene, vectors, attr_store, samples_per_track=4)
    upper, lower, _bag, upper_confidence, lower_confidence, n_frames, *_rest = _rows(attr_store)[1]
    assert upper == Color.UNKNOWN.value
    assert lower == Color.UNKNOWN.value
    assert upper_confidence == 0.0
    assert lower_confidence == 0.0
    assert n_frames == 4


def test_single_sample_uses_the_middle_frame(tmp_path: Path) -> None:
    paints = [
        (_GREEN, _YELLOW),
        (_GREEN, _YELLOW),
        (_RED, _BLUE),
        (_GREEN, _YELLOW),
        (_GREEN, _YELLOW),
    ]
    scene = _one_person(tmp_path, paints)
    vectors = InMemoryVectorStore(scene.embedder.dim)
    attr_store = _attributes(tmp_path)
    _describe(scene, vectors, attr_store, samples_per_track=1)
    upper, lower, _bag, _upper_confidence, _lower_confidence, n_frames, *_rest = _rows(attr_store)[
        1
    ]
    assert upper == Color.RED.value
    assert lower == Color.BLUE.value
    assert n_frames == 1
    hits = vectors.search(_probe(scene.embedder), k=1)
    assert hits[0][0].frame_number == 2


def test_spy_writes_no_crop_or_frame_and_limits_embed_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _build(
        tmp_path,
        9,
        (
            _PersonPaint(1, _BOX_TALL, _RED, _BLUE, tuple(range(9))),
            _PersonPaint(2, _BOX_SHORT, _GREEN, _YELLOW, (0, 1, 2)),
        ),
        (),
    )
    opened: list[str] = []
    _forbid_media_writes(monkeypatch, opened)
    spy = _EmbedSpy(scene.embedder)
    vectors = SqliteVectorStore(tmp_path / "vectors.sqlite", spy.dim, clock=_clock)
    attr_store = _attributes(tmp_path)
    _describe(scene, vectors, attr_store, embedder=spy)

    assert spy.calls_by_shape[(_BOX_TALL_SHAPE)] == _SAMPLES
    assert spy.crops_by_shape[_BOX_TALL_SHAPE] == _SAMPLES
    assert spy.calls_by_shape[_BOX_SHORT_SHAPE] == 3
    assert spy.crops_by_shape[_BOX_SHORT_SHAPE] == 3
    assert all(count <= _SAMPLES for count in spy.calls_by_shape.values())
    assert all(count <= _SAMPLES for count in spy.crops_by_shape.values())
    files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert {path.suffix for path in files} == {".sqlite"}
    assert {path.name for path in files} == {
        "tracks.sqlite",
        "attributes.sqlite",
        "vectors.sqlite",
    }
    assert not any(Path(path).suffix.lower() in _MEDIA_SUFFIXES for path in opened)


_BOX_TALL_SHAPE = (200, 50)
_BOX_SHORT_SHAPE = (80, 30)


def _forbid_media_writes(monkeypatch: pytest.MonkeyPatch, opened: list[str]) -> None:
    real_open = open

    def _open(file: object, mode: str = "r", *args: object, **kwargs: object) -> object:
        path = str(file)
        opened.append(path)
        if Path(path).suffix.lower() in _MEDIA_SUFFIXES:
            raise AssertionError(path)
        return real_open(file, mode, *args, **kwargs)

    def _refuse(path: object, *_args: object, **_kwargs: object) -> None:
        opened.append(str(path))
        raise AssertionError(str(path))

    monkeypatch.setattr("builtins.open", _open)
    monkeypatch.setattr(cv2, "imwrite", _refuse)
    monkeypatch.setattr(np, "save", _refuse)
    monkeypatch.setattr(np, "savez", _refuse)
    monkeypatch.setattr(np, "savez_compressed", _refuse)


def test_same_input_is_identical_across_two_runs(tmp_path: Path) -> None:
    scene = _parade(tmp_path)
    first_vectors = InMemoryVectorStore(scene.embedder.dim)
    second_vectors = InMemoryVectorStore(scene.embedder.dim)
    first_attr = _attributes(tmp_path, "attributes-a.sqlite")
    second_attr = _attributes(tmp_path, "attributes-b.sqlite")
    first_stats = _describe(scene, first_vectors, first_attr)
    second_stats = _describe(scene, second_vectors, second_attr)

    assert scene.source.reads == 20
    assert first_stats == second_stats
    assert _rows(first_attr) == _rows(second_attr)
    probe = _probe(scene.embedder)
    assert _vector_map(first_vectors, probe) == _vector_map(second_vectors, probe)


def _vector_map(
    vectors: VectorStore,
    probe: npt.NDArray[np.float32],
) -> dict[_Key, tuple[bool, str, tuple[float, ...]]]:
    """In-memory runs keep the original tuple, so these two maps can be equal."""
    found: dict[_Key, tuple[bool, str, tuple[float, ...]]] = {}
    for embedding, _score in vectors.search(probe, k=10):
        key = (embedding.camera_id, embedding.track_id, embedding.frame_number)
        found[key] = (embedding.placeholder, embedding.embedder_name, embedding.vector)
    return found


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_rerun_keeps_one_vector_per_key(tmp_path: Path, backend: str) -> None:
    scene = _parade(tmp_path)
    if backend == "memory":
        vectors: VectorStore = InMemoryVectorStore(scene.embedder.dim)
    else:
        vectors = SqliteVectorStore(tmp_path / "vectors.sqlite", scene.embedder.dim, clock=_clock)
    attr_store = _attributes(tmp_path)
    _describe(scene, vectors, attr_store)
    _describe(scene, vectors, attr_store)

    assert vectors.count(run_id=_RUN) == 2
    assert attr_store.count(run_id=_RUN) == 2
    probe = _probe(scene.embedder)
    hits = vectors.search(probe, k=vectors.count(run_id=_RUN))
    keys = [(item.camera_id, item.track_id, item.frame_number) for item, _score in hits]
    assert len(keys) == len(set(keys)) == 2
    for embedding, _score in hits:
        assert embedding.placeholder is True
        assert _norm(embedding.vector) == pytest.approx(1.0, abs=1e-5)


def test_cross_store_results_match_by_key_and_score(tmp_path: Path) -> None:
    """SqliteVectorStore rounds vectors to float32 on read.

    Compare the two stores by key and score. Do not compare embedding.vector.
    """
    scene = _parade(tmp_path)
    memory = InMemoryVectorStore(scene.embedder.dim)
    sqlite = SqliteVectorStore(tmp_path / "vectors.sqlite", scene.embedder.dim, clock=_clock)
    memory_attr = _attributes(tmp_path, "attributes-memory.sqlite")
    sqlite_attr = _attributes(tmp_path, "attributes-sqlite.sqlite")
    _describe(scene, memory, memory_attr)
    _describe(scene, sqlite, sqlite_attr)
    probe = _probe(scene.embedder)
    memory_scores = _score_map(memory, probe)
    sqlite_scores = _score_map(sqlite, probe)

    assert set(memory_scores) == set(sqlite_scores)
    assert len(memory_scores) == 2
    for key, score in memory_scores.items():
        assert sqlite_scores[key] == pytest.approx(score, abs=1e-5)
    assert _nearest(memory_scores) == _nearest(sqlite_scores)


def _nearest(scored: dict[_Key, float]) -> _Key:
    return max(scored, key=lambda key: scored[key])


def test_tracks_outside_this_pass_are_skipped(tmp_path: Path) -> None:
    scene = _parade(tmp_path)
    scene.bag_tracklets = [*scene.bag_tracklets, scene.tracklets[0]]
    scene.tracklets = [
        *scene.tracklets,
        scene.bag_tracklets[0],
        _manual(_CAMERA, 99, ()),
        _manual(_CAMERA, 98, (0, 1)),
        _manual(_OTHER_CAMERA, 3, (0, 1, 2)),
    ]
    scene.bag_tracklets.append(_manual(_CAMERA, 77, (3,), DetClass.HANDBAG))
    vectors = InMemoryVectorStore(scene.embedder.dim)
    attr_store = _attributes(tmp_path)
    stats = _describe(scene, vectors, attr_store)

    assert stats.tracks_described == 2
    assert stats.embeddings_stored == 2
    assert set(_rows(attr_store)) == {1, 2}
    assert attr_store.read(_OTHER_CAMERA, run_id=_RUN) == []
    assert {key[1] for key in _score_map(vectors, _probe(scene.embedder))} == {1, 2}


def test_embedding_uses_the_lowest_sampled_frame_even_if_seen_later(tmp_path: Path) -> None:
    """Frame 4 is yielded before frame 0. The stored frame number is still 0."""
    images: list[npt.NDArray[np.uint8]] = []
    observations: list[TrackObservation] = []
    for frame_number in (0, 4):
        image = _blank()
        _paint(image, _BOX_RED, _RED, _BLUE)
        images.append(image)
        observations.append(_observation(1, frame_number, _BOX_RED))
    store = TrackStore(tmp_path / "tracks.sqlite", clock=_clock)
    store.write(_RUN, _FIXTURE, _ORIGIN, observations)
    scene = _Scene(
        source=ArraySource(_CAMERA, [images[1], images[0]], frame_numbers=(4, 0)),
        tracklets=load_tracklets(store, _RUN, det_class=DetClass.PERSON),
        bag_tracklets=[],
        store=store,
        embedder=_embedder(),
        extractor=AttributeExtractor(),
    )
    vectors = InMemoryVectorStore(scene.embedder.dim)
    attr_store = _attributes(tmp_path)
    _describe(scene, vectors, attr_store, samples_per_track=2)
    hits = vectors.search(_probe(scene.embedder), k=1)
    assert hits[0][0].frame_number == 0


def test_out_of_frame_box_stores_attributes_without_a_vector(tmp_path: Path) -> None:
    image = _blank()
    store = TrackStore(tmp_path / "tracks.sqlite", clock=_clock)
    store.write(_RUN, _FIXTURE, _ORIGIN, [_observation(4, 0, _BOX_OUTSIDE)])
    scene = _Scene(
        source=ArraySource(_CAMERA, [image]),
        tracklets=load_tracklets(store, _RUN, det_class=DetClass.PERSON),
        bag_tracklets=[],
        store=store,
        embedder=_embedder(),
        extractor=AttributeExtractor(),
    )
    vectors = InMemoryVectorStore(scene.embedder.dim)
    attr_store = _attributes(tmp_path)
    stats = _describe(scene, vectors, attr_store, samples_per_track=1)

    assert stats.tracks_described == 1
    assert stats.embeddings_stored == 0
    assert vectors.count(run_id=_RUN) == 0
    upper, lower, bag, *_rest = _rows(attr_store)[4]
    assert upper == Color.INVALID_BOX.value
    assert lower == Color.INVALID_BOX.value
    assert bag is False


def test_already_open_source_is_not_restarted(tmp_path: Path) -> None:
    scene = _parade(tmp_path)
    scene.source.open()
    assert scene.source.reads == 0
    consumed = list(scene.source.frames())
    assert len(consumed) == 9
    vectors = InMemoryVectorStore(scene.embedder.dim)
    attr_store = _attributes(tmp_path)
    stats = _describe(scene, vectors, attr_store)

    assert stats.frames_read == 0
    assert stats.tracks_described == 0
    assert scene.source.is_open()
    assert vectors.count() == 0


def test_bad_arguments_do_not_purge_a_finished_run(tmp_path: Path) -> None:
    scene = _parade(tmp_path)
    vectors = InMemoryVectorStore(scene.embedder.dim)
    attr_store = _attributes(tmp_path)
    _describe(scene, vectors, attr_store)
    with pytest.raises(ValueError, match="samples_per_track"):
        _describe(scene, vectors, attr_store, samples_per_track=0)
    with pytest.raises(ValueError, match="samples_per_track"):
        _describe(scene, vectors, attr_store, samples_per_track=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="derived_from"):
        _describe(scene, vectors, attr_store, derived_from="TRAIN")
    assert vectors.count(run_id=_RUN) == 2
    assert attr_store.count(run_id=_RUN) == 2


def test_cancelling_vectors_are_refused_before_write(tmp_path: Path) -> None:
    scene = _one_person(tmp_path, [(_RED, _BLUE), (_RED, _BLUE)])
    vectors = InMemoryVectorStore(2)
    attr_store = _attributes(tmp_path)
    with pytest.raises(ValueError, match="non-zero"):
        _describe(
            scene,
            vectors,
            attr_store,
            embedder=_SignedEmbedder(((1.0, 0.0), (-1.0, 0.0))),
            samples_per_track=2,
        )
    assert attr_store.count(run_id=_RUN) == 0
    assert vectors.count() == 0


def test_non_finite_mean_is_refused(tmp_path: Path) -> None:
    scene = _one_person(tmp_path, [(_RED, _BLUE)])
    vectors = InMemoryVectorStore(2)
    attr_store = _attributes(tmp_path)
    with pytest.raises(ValueError, match="finite"):
        _describe(
            scene,
            vectors,
            attr_store,
            embedder=_SignedEmbedder(((math.nan, 0.0),)),
            samples_per_track=1,
        )
    assert vectors.count() == 0


def test_embed_must_return_one_row(tmp_path: Path) -> None:
    scene = _one_person(tmp_path, [(_RED, _BLUE)])
    vectors = InMemoryVectorStore(2)
    attr_store = _attributes(tmp_path)
    with pytest.raises(ValueError, match="one row"):
        _describe(
            scene,
            vectors,
            attr_store,
            embedder=_EmptyEmbedder(),
            samples_per_track=1,
        )
    assert attr_store.count(run_id=_RUN) == 0


class _SignedEmbedder:
    placeholder = True
    dim = 2

    def __init__(self, rows: Sequence[Sequence[float]]) -> None:
        self._rows = tuple(tuple(row) for row in rows)
        self._index = 0

    @property
    def name(self) -> str:
        return "signed-placeholder"

    def embed(self, crops: Sequence[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        chosen: list[tuple[float, ...]] = []
        for _crop in crops:
            chosen.append(self._rows[self._index])
            self._index += 1
        return np.asarray(chosen, dtype=np.float32)


class _EmptyEmbedder:
    placeholder = True
    dim = 2

    @property
    def name(self) -> str:
        return "empty-placeholder"

    def embed(self, crops: Sequence[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        del crops
        return np.zeros((0, 2), dtype=np.float32)


def _one_person(
    tmp_path: Path,
    paints: Sequence[tuple[tuple[int, int, int], tuple[int, int, int]]],
) -> _Scene:
    images: list[npt.NDArray[np.uint8]] = []
    observations: list[TrackObservation] = []
    for frame_number, (upper, lower) in enumerate(paints):
        image = _blank()
        _paint(image, _BOX_RED, upper, lower)
        images.append(image)
        observations.append(_observation(1, frame_number, _BOX_RED))
    store = TrackStore(tmp_path / "tracks.sqlite", clock=_clock)
    store.write(_RUN, _FIXTURE, _ORIGIN, observations)
    return _Scene(
        source=ArraySource(_CAMERA, images),
        tracklets=load_tracklets(store, _RUN, det_class=DetClass.PERSON),
        bag_tracklets=[],
        store=store,
        embedder=_embedder(),
        extractor=AttributeExtractor(),
    )
