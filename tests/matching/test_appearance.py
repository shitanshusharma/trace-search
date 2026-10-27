"""Track appearance pass: colours, bag flag, and one placeholder vector.

The video source is an in-memory numpy fake. No media file is opened.
SQLite files sit under pytest's tmp_path, outside the repo.
"""

import math
import os
import re
import socket
import sqlite3
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
    """Counts embed calls per track. It does not keep the crops."""

    placeholder = True

    def __init__(self, inner: UntrainedPlaceholderEmbedder) -> None:
        self._inner = inner
        self.dim = inner.dim
        self.calls_by_track: dict[int, int] = {}
        self.crops_by_track: dict[int, int] = {}

    @property
    def name(self) -> str:
        return self._inner.name

    def embed(self, crops: Sequence[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        seen: set[int] = set()
        for crop in crops:
            track_id = _track_id_from_crop(crop)
            self.crops_by_track[track_id] = self.crops_by_track.get(track_id, 0) + 1
            if track_id not in seen:
                self.calls_by_track[track_id] = self.calls_by_track.get(track_id, 0) + 1
                seen.add(track_id)
        return self._inner.embed(crops)


def _track_id_from_crop(crop: npt.NDArray[np.uint8]) -> int:
    """The painted upper colour is the track id. Shape is not."""
    row = int(crop.shape[0] * 0.4)
    pixel = tuple(int(channel) for channel in crop[row, int(crop.shape[1]) // 2])
    if pixel == _RED:
        return 1
    if pixel == _GREEN:
        return 2
    raise AssertionError(pixel)


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
    camera_id: str = _CAMERA,
) -> TrackObservation:
    return TrackObservation(
        camera_id=camera_id,
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
    """The bag extends past the person and is assigned through the public extractor."""
    frame = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    extracted = AttributeExtractor().extract_frame(
        frame,
        [_BOX_RED, _BOX_WITH_BAG],
        [_BOX_BAG],
    )
    assert [item.bag_on_track for item in extracted] == [False, True]


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


def test_bag_inside_unsampled_person_stays_with_that_person(tmp_path: Path) -> None:
    """A owns the only sample on the shared frame. The bag sits inside B.

    A is on frames 0-4, so one sample is frame 2. B is on frames 0-6, so
    one sample is frame 3. The bag is on both of those frames, fully inside
    B and three quarters inside A.
    """
    scene = _build(
        tmp_path,
        7,
        (
            _PersonPaint(1, _BOX_PARTIAL, _RED, _BLUE, tuple(range(5))),
            _PersonPaint(2, _BOX_OWNER, _GREEN, _YELLOW, tuple(range(7))),
        ),
        (_BagPaint(7, _BOX_SHARED_BAG, (2, 3), DetClass.HANDBAG),),
    )
    vectors = InMemoryVectorStore(scene.embedder.dim)
    attr_store = _attributes(tmp_path)
    _describe(scene, vectors, attr_store, samples_per_track=1)
    rows = _rows(attr_store)
    assert rows[1][2] is False
    assert rows[2][2] is True
    assert rows[1][5] == 1
    assert rows[2][5] == 1
    frames = {
        item.track_id: item.frame_number
        for item, _score in vectors.search(_probe(scene.embedder), k=2)
    }
    assert frames == {1: 2, 2: 3}


_BOX_PARTIAL = (70.0, 100.0, 130.0, 180.0)
_BOX_OWNER = (100.0, 100.0, 180.0, 180.0)
_BOX_SHARED_BAG = (100.0, 100.0, 140.0, 140.0)


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


def test_mean_of_different_samples_is_normalised(tmp_path: Path) -> None:
    """Two different unit rows. Their mean has norm below 1. The store keeps the unit mean."""
    scene = _one_person(tmp_path, [(_RED, _BLUE), (_GREEN, _YELLOW)])
    rows = ((1.0, 0.0), (0.0, 1.0))
    vectors = InMemoryVectorStore(2)
    attr_store = _attributes(tmp_path)
    _describe(
        scene,
        vectors,
        attr_store,
        embedder=_SignedEmbedder(rows),
        samples_per_track=2,
    )
    mean = np.mean(np.asarray(rows, dtype=np.float64), axis=0)
    raw_norm = float(np.linalg.norm(mean))
    assert raw_norm < 1.0
    expected = mean / raw_norm
    hits = vectors.search(np.asarray(expected, dtype=np.float32), k=1)
    stored = hits[0][0].vector
    assert _norm(stored) == pytest.approx(1.0, abs=1e-5)
    assert np.asarray(stored, dtype=np.float64) == pytest.approx(expected, abs=1e-5)


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
    tmp_path_factory: pytest.TempPathFactory,
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
    _forbid_media_writes(monkeypatch, opened, tmp_path)
    scene.source._images = [image.view(_RaisingToFile) for image in scene.source._images]
    crop = scene.source._images[0][0:4, 0:4].copy()
    with pytest.raises(AssertionError, match="/tmp/x"):
        crop.tofile("/tmp/x")
    outside = tmp_path_factory.mktemp("outside") / "x.bin"
    outside_match = re.escape(os.fspath(outside))
    with pytest.raises(AssertionError, match=outside_match):
        _probe_os_open(outside)
    with pytest.raises(AssertionError, match=outside_match):
        _probe_builtin_open(outside)
    png_path = tmp_path / "crop.png"
    with pytest.raises(AssertionError, match=re.escape(os.fspath(png_path))):
        _probe_builtin_open(png_path)
    assert not outside.is_file()
    assert not png_path.is_file()
    opened.clear()  # the probes above are not part of the pass
    spy = _EmbedSpy(scene.embedder)
    vectors = SqliteVectorStore(tmp_path / "vectors.sqlite", spy.dim, clock=_clock)
    attr_store = _attributes(tmp_path)
    _describe(scene, vectors, attr_store, embedder=spy)

    assert spy.calls_by_track[1] == _SAMPLES
    assert spy.crops_by_track[1] == _SAMPLES
    assert spy.calls_by_track[2] == 3
    assert spy.crops_by_track[2] == 3
    assert all(count <= _SAMPLES for count in spy.calls_by_track.values())
    assert all(count <= _SAMPLES for count in spy.crops_by_track.values())
    files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert {path.suffix for path in files} == {".sqlite"}
    assert {path.name for path in files} == {
        "tracks.sqlite",
        "attributes.sqlite",
        "vectors.sqlite",
    }
    assert not any(Path(path).suffix.lower() in _MEDIA_SUFFIXES for path in opened)


def _forbid_media_writes(
    monkeypatch: pytest.MonkeyPatch,
    opened: list[str],
    root: Path,
) -> None:
    """Refuse crop writes, including writes that never touch ``root``."""
    real_open = open
    real_os_open = os.open
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

    def _open(file: object, mode: str = "r", *args: object, **kwargs: object) -> object:
        if isinstance(file, int):
            return real_open(file, mode, *args, **kwargs)
        path = os.fsdecode(file) if isinstance(file, bytes) else os.fspath(file)
        opened.append(path)
        _refuse_blocked_write(path, root, writing=_mode_writes(mode))
        return real_open(file, mode, *args, **kwargs)

    def _os_open(
        path: object, flags: int, mode: int = 0o600, *args: object, **kwargs: object
    ) -> int:
        if isinstance(path, int):
            return real_os_open(path, flags, mode, *args, **kwargs)
        text = os.fsdecode(path) if isinstance(path, bytes) else os.fspath(path)
        opened.append(text)
        _refuse_blocked_write(text, root, writing=bool(flags & write_flags))
        return real_os_open(path, flags, mode, *args, **kwargs)

    def _refuse(path: object, *_args: object, **_kwargs: object) -> None:
        opened.append(str(path))
        raise AssertionError(str(path))

    monkeypatch.setattr("builtins.open", _open)
    monkeypatch.setattr(os, "open", _os_open)
    monkeypatch.setattr(cv2, "imwrite", _refuse)
    monkeypatch.setattr(np, "save", _refuse)
    monkeypatch.setattr(np, "savez", _refuse)
    monkeypatch.setattr(np, "savez_compressed", _refuse)


class _RaisingToFile(np.ndarray):
    """``ndarray.tofile`` is a C slot and cannot be replaced on the base type."""

    def tofile(self, fid: object, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError(str(fid))


def _probe_os_open(path: Path) -> None:
    """Write-open then close. The spy must raise before a descriptor exists."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        return
    finally:
        os.close(descriptor)


def _probe_builtin_open(path: Path) -> None:
    """Write-open then close. The spy must raise before a file object exists."""
    with open(path, "wb"):
        return


def _mode_writes(mode: str) -> bool:
    return any(mark in mode for mark in ("w", "a", "x", "+"))


def _refuse_blocked_write(path: str, root: Path, *, writing: bool) -> None:
    candidate = Path(path)
    if candidate.suffix.lower() in _MEDIA_SUFFIXES:
        raise AssertionError(path)
    if writing and not _is_under(candidate, root):
        raise AssertionError(path)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


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


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_second_camera_in_the_same_run_keeps_both(tmp_path: Path, backend: str) -> None:
    source_a, source_b, tracklets, store, extractor, embedder = _pair(tmp_path)
    vectors = _backend(tmp_path, backend, embedder.dim)
    attr_store = _attributes(tmp_path)
    _describe_camera(source_a, tracklets, store, extractor, embedder, vectors, attr_store)
    _describe_camera(source_b, tracklets, store, extractor, embedder, vectors, attr_store)

    assert vectors.count(run_id=_RUN) == 2
    assert attr_store.count(run_id=_RUN) == 2
    assert _colours(attr_store, _CAM_A) == (Color.RED.value, Color.BLUE.value, False)
    assert _colours(attr_store, _CAM_B) == (Color.GREEN.value, Color.YELLOW.value, False)
    keys = _camera_keys(vectors, _probe(embedder))
    assert keys == {(_CAM_A, 1, 0), (_CAM_B, 1, 0)}


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_rerun_replaces_only_that_camera(tmp_path: Path, backend: str) -> None:
    source_a, source_b, tracklets, store, extractor, embedder = _pair(tmp_path)
    vectors = _backend(tmp_path, backend, embedder.dim)
    attr_store = _attributes(tmp_path)
    _describe_camera(source_a, tracklets, store, extractor, embedder, vectors, attr_store)
    _describe_camera(source_b, tracklets, store, extractor, embedder, vectors, attr_store)
    kept_attributes = _rowids(attr_store.db_path)[_CAM_B]
    kept_vector = _cam_b_vector(vectors, embedder)
    _describe_camera(source_a, tracklets, store, extractor, embedder, vectors, attr_store)

    assert vectors.count(run_id=_RUN) == 2
    assert attr_store.count(camera_id=_CAM_A, run_id=_RUN) == 1
    assert attr_store.count(camera_id=_CAM_B, run_id=_RUN) == 1
    assert _rowids(attr_store.db_path)[_CAM_B] == kept_attributes
    assert _same_cam_b(vectors, embedder, kept_vector)
    assert _colours(attr_store, _CAM_B) == (Color.GREEN.value, Color.YELLOW.value, False)
    keys = _camera_keys(vectors, _probe(embedder))
    assert len(keys) == len(set(keys)) == 2
    assert (_CAM_B, 1, 0) in keys
    assert (_CAM_A, 1, 0) in keys


_CAM_A = "cam-1"
_CAM_B = "cam-2"


def _backend(tmp_path: Path, backend: str, dim: int) -> VectorStore:
    if backend == "memory":
        return InMemoryVectorStore(dim)
    return SqliteVectorStore(tmp_path / "vectors.sqlite", dim, clock=_clock)


def _describe_camera(
    source: ArraySource,
    tracklets: Sequence[Tracklet],
    store: TrackStore,
    extractor: AttributeExtractor,
    embedder: AppearanceEmbedder,
    vectors: VectorStore,
    attr_store: TrackAttributeStore,
) -> AppearanceStats:
    return describe_tracks(
        source,
        tracklets,
        [],
        store,
        _RUN,
        extractor,
        embedder,
        attr_store,
        vectors,
        _ORIGIN,
        3,
    )


def _pair(
    tmp_path: Path,
) -> tuple[
    ArraySource,
    ArraySource,
    list[Tracklet],
    TrackStore,
    AttributeExtractor,
    UntrainedPlaceholderEmbedder,
]:
    observations: list[TrackObservation] = []
    images_a: list[npt.NDArray[np.uint8]] = []
    images_b: list[npt.NDArray[np.uint8]] = []
    for frame_number in range(3):
        image_a = _blank()
        _paint(image_a, _BOX_RED, _RED, _BLUE)
        images_a.append(image_a)
        observations.append(_observation(1, frame_number, _BOX_RED, camera_id=_CAM_A))
        image_b = _blank()
        _paint(image_b, _BOX_WITH_BAG, _GREEN, _YELLOW)
        images_b.append(image_b)
        observations.append(_observation(1, frame_number, _BOX_WITH_BAG, camera_id=_CAM_B))
    store = TrackStore(tmp_path / "tracks.sqlite", clock=_clock)
    store.write(_RUN, _FIXTURE, _ORIGIN, observations)
    return (
        ArraySource(_CAM_A, images_a),
        ArraySource(_CAM_B, images_b),
        load_tracklets(store, _RUN, det_class=DetClass.PERSON),
        store,
        AttributeExtractor(),
        _embedder(),
    )


def _colours(
    attr_store: TrackAttributeStore,
    camera_id: str,
) -> tuple[str, str, bool]:
    found = attr_store.read(camera_id, run_id=_RUN)
    assert len(found) == 1
    record = found[0]
    return (record.upper_color, record.lower_color, record.bag_on_track)


def _camera_keys(vectors: VectorStore, probe: npt.NDArray[np.float32]) -> set[_Key]:
    return {
        (item.camera_id, item.track_id, item.frame_number)
        for item, _score in vectors.search(probe, k=10)
    }


_ATTRIBUTE_ROWS = "SELECT rowid, camera_id FROM track_attributes"
_VECTOR_ROWS = "SELECT rowid, camera_id FROM appearance_embeddings"


def _rowids(db_path: Path) -> dict[str, int]:
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(_ATTRIBUTE_ROWS).fetchall()
    finally:
        connection.close()
    return {str(camera_id): int(rowid) for rowid, camera_id in rows}


def _vector_rowids(db_path: Path) -> dict[str, int]:
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(_VECTOR_ROWS).fetchall()
    finally:
        connection.close()
    return {str(camera_id): int(rowid) for rowid, camera_id in rows}


def _cam_b_vector(vectors: VectorStore, embedder: UntrainedPlaceholderEmbedder) -> object:
    if isinstance(vectors, SqliteVectorStore):
        return _vector_rowids(vectors.db_path)[_CAM_B]
    found = {item.camera_id: item for item, _score in vectors.search(_probe(embedder), k=10)}
    return found[_CAM_B]


def _same_cam_b(
    vectors: VectorStore,
    embedder: UntrainedPlaceholderEmbedder,
    kept: object,
) -> bool:
    current = _cam_b_vector(vectors, embedder)
    if isinstance(vectors, SqliteVectorStore):
        return current == kept
    return current is kept


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
    """Every sampled box is outside the frame. The track is stored with no vector."""
    images = [_blank(), _blank()]
    store = TrackStore(tmp_path / "tracks.sqlite", clock=_clock)
    store.write(
        _RUN,
        _FIXTURE,
        _ORIGIN,
        [_observation(4, 0, _BOX_OUTSIDE), _observation(4, 1, _BOX_OUTSIDE)],
    )
    scene = _Scene(
        source=ArraySource(_CAMERA, images),
        tracklets=load_tracklets(store, _RUN, det_class=DetClass.PERSON),
        bag_tracklets=[],
        store=store,
        embedder=_embedder(),
        extractor=AttributeExtractor(),
    )
    vectors = InMemoryVectorStore(scene.embedder.dim)
    attr_store = _attributes(tmp_path)
    stats = _describe(scene, vectors, attr_store, samples_per_track=2)

    assert stats.tracks_described == 1
    assert stats.embeddings_stored == 0
    assert vectors.count(run_id=_RUN) == 0
    upper, lower, bag, _upper_confidence, _lower_confidence, n_frames, *_rest = _rows(attr_store)[4]
    assert upper == Color.INVALID_BOX.value
    assert lower == Color.INVALID_BOX.value
    assert bag is False
    assert n_frames == 2


def test_already_open_source_is_not_restarted(tmp_path: Path) -> None:
    scene = _parade(tmp_path)
    scene.source.open()
    assert scene.source.reads == 0
    consumed = list(scene.source.frames())
    assert len(consumed) == 9
    vectors = InMemoryVectorStore(scene.embedder.dim)
    attr_store = _attributes(tmp_path)
    with pytest.raises(ValueError, match="no frames"):
        _describe(scene, vectors, attr_store)

    assert scene.source.is_open()
    assert scene.source.reads == 11
    assert vectors.count() == 0
    assert attr_store.count() == 0


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_empty_source_raises_before_purge(tmp_path: Path, backend: str) -> None:
    scene = _parade(tmp_path)
    vectors = _backend(tmp_path, backend, scene.embedder.dim)
    attr_store = _attributes(tmp_path)
    _describe(scene, vectors, attr_store)
    kept_rows = _rowids(attr_store.db_path)
    kept_vectors = _vector_identity(vectors, scene.embedder)
    empty = ArraySource(_CAMERA, [])
    with pytest.raises(ValueError, match="no frames"):
        describe_tracks(
            empty,
            scene.tracklets,
            scene.bag_tracklets,
            scene.store,
            _RUN,
            scene.extractor,
            scene.embedder,
            attr_store,
            vectors,
            _ORIGIN,
        )
    assert not empty.is_open()
    assert _rowids(attr_store.db_path) == kept_rows
    assert _vector_identity(vectors, scene.embedder) == kept_vectors
    assert vectors.count(run_id=_RUN) == 2
    assert attr_store.count(run_id=_RUN) == 2


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_source_past_every_planned_sample_raises_before_purge(tmp_path: Path, backend: str) -> None:
    """An already-open source positioned after every planned sample must not delete."""
    scene = _parade(tmp_path)
    vectors = _backend(tmp_path, backend, scene.embedder.dim)
    attr_store = _attributes(tmp_path)
    _describe(scene, vectors, attr_store)
    kept_rows = _attribute_identity(attr_store.db_path)
    kept_vectors = _vector_identity(vectors, scene.embedder)
    kept_attributes = _rows(attr_store)
    late = ArraySource(_CAMERA, [_blank() for _ in range(11)])
    late.open()
    for _ in range(9):
        assert late.read_frame() is not None
    with pytest.raises(ValueError, match="planned sample"):
        describe_tracks(
            late,
            scene.tracklets,
            scene.bag_tracklets,
            scene.store,
            _RUN,
            scene.extractor,
            scene.embedder,
            attr_store,
            vectors,
            _ORIGIN,
        )
    assert late.is_open()
    assert _attribute_identity(attr_store.db_path) == kept_rows
    assert _vector_identity(vectors, scene.embedder) == kept_vectors
    assert _rows(attr_store) == kept_attributes
    assert vectors.count(run_id=_RUN) == 2
    assert attr_store.count(run_id=_RUN) == 2


_ATTRIBUTE_IDENTITY = "SELECT rowid, camera_id, track_id FROM track_attributes"
_VECTOR_IDENTITY = "SELECT rowid, camera_id, track_id, frame_number FROM appearance_embeddings"


def _attribute_identity(db_path: Path) -> tuple[tuple[int, str, int], ...]:
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(_ATTRIBUTE_IDENTITY).fetchall()
    finally:
        connection.close()
    return tuple(
        sorted((int(rowid), str(camera_id), int(track_id)) for rowid, camera_id, track_id in rows)
    )


def _vector_identity(
    vectors: VectorStore,
    embedder: UntrainedPlaceholderEmbedder,
) -> tuple[tuple[object, ...], ...]:
    """Row identity. A rewritten row has a new rowid or a new object."""
    if isinstance(vectors, SqliteVectorStore):
        connection = sqlite3.connect(vectors.db_path)
        try:
            rows = connection.execute(_VECTOR_IDENTITY).fetchall()
        finally:
            connection.close()
        return tuple(
            sorted(
                (int(rowid), str(camera_id), int(track_id), int(frame))
                for rowid, camera_id, track_id, frame in rows
            )
        )
    probe = _probe(embedder)
    return tuple(
        sorted(
            (item.camera_id, item.track_id, item.frame_number, id(item))
            for item, _score in vectors.search(probe, k=10)
        )
    )


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
