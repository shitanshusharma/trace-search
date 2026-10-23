"""Tracklet summaries: one frozen record per per-camera track."""

import socket
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from trace_search.detection.types import DetClass
from trace_search.matching.tracklets import Tracklet, load_tracklets
from trace_search.tracking.store import TrackStore
from trace_search.tracking.types import TrackObservation

_RUN = "run-1"
_OTHER_RUN = "run-2"
_FIXTURE = "FX-CODE-1"

# Inserted out of timestamp order on purpose. first/last follow time, then frame.
_END_BOX = (1.0, 1.0, 5.0, 5.0)
_FRAME_4_BOX = (4.0, 4.0, 8.0, 8.0)
_FIRST_BOX = (2.0, 2.0, 6.0, 6.0)
_SECOND_BOX = (2.5, 2.5, 6.5, 6.5)
_SINGLE_BOX = (9.0, 1.0, 12.0, 4.0)
_CAM_B_LAST = (6.0, 6.0, 9.0, 9.0)
_CAM_B_FIRST = (0.0, 0.0, 3.0, 3.0)
_BAG_BOX = (20.0, 20.0, 24.0, 28.0)
_BAG_ONLY_BOX = (30.0, 30.0, 35.0, 40.0)
_HANDBAG_BOX = (14.0, 14.0, 18.0, 22.0)
_CAM_C_BOX = (7.0, 7.0, 11.0, 11.0)
_OTHER_BOX = (99.0, 99.0, 120.0, 140.0)


@pytest.fixture(autouse=True)
def _refuse_outbound_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """T6 keeps these tests off the network. Local SQLite does not connect."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("outbound network is refused")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def store(tmp_path: Path) -> TrackStore:
    """SQLite file under pytest's tmp_path, which sits outside the repo."""
    return TrackStore(tmp_path / "tracks.sqlite", clock=lambda: 1_700_000_000.0)


def _row(
    camera_id: str,
    track_id: int,
    frame_number: int,
    timestamp_seconds: float,
    box: tuple[float, float, float, float],
    det_class: DetClass = DetClass.PERSON,
) -> TrackObservation:
    return TrackObservation(
        camera_id=camera_id,
        track_id=track_id,
        det_class=det_class,
        frame_number=frame_number,
        timestamp_seconds=timestamp_seconds,
        box=box,
        score=0.5,
    )


def _tracklet(
    camera_id: str,
    track_id: int,
    t_start: float,
    t_end: float,
    n_obs: int,
    first_box: tuple[float, float, float, float],
    last_box: tuple[float, float, float, float],
    frames: tuple[int, ...],
    det_class: DetClass = DetClass.PERSON,
    run_id: str = _RUN,
) -> Tracklet:
    return Tracklet(
        camera_id=camera_id,
        track_id=track_id,
        det_class=det_class,
        run_id=run_id,
        t_start=t_start,
        t_end=t_end,
        n_obs=n_obs,
        first_box=first_box,
        last_box=last_box,
        frames=frames,
    )


def _expected_three() -> list[Tracklet]:
    """cam-a track 2, cam-a track 5, then cam-b track 2. Same track id, two cameras."""
    return [
        _tracklet("cam-a", 2, 10.0, 50.0, 4, _FIRST_BOX, _END_BOX, (2, 2, 4, 1)),
        _tracklet("cam-a", 5, 10.0, 10.0, 1, _SINGLE_BOX, _SINGLE_BOX, (3,)),
        _tracklet("cam-b", 2, 1.0, 8.0, 2, _CAM_B_FIRST, _CAM_B_LAST, (0, 6)),
    ]


def _write_acceptance(store: TrackStore) -> None:
    """Two cameras, three person tracks, plus bags, another camera, and another run."""
    store.write(
        _RUN,
        _FIXTURE,
        "CODE_TEST",
        [
            _row("cam-b", 2, 6, 8.0, _CAM_B_LAST),
            _row("cam-b", 2, 0, 1.0, _CAM_B_FIRST),
            _row("cam-a", 5, 3, 10.0, _SINGLE_BOX),
            _row("cam-a", 2, 1, 50.0, _END_BOX),
            _row("cam-a", 2, 4, 10.0, _FRAME_4_BOX),
            _row("cam-a", 2, 2, 10.0, _FIRST_BOX),
            _row("cam-a", 2, 2, 10.0, _SECOND_BOX),
            _row("cam-a", 8, 1, 3.0, _BAG_BOX, DetClass.BACKPACK),
            _row("cam-b", 3, 4, 6.0, _HANDBAG_BOX, DetClass.HANDBAG),
            _row("cam-c", 1, 1, 2.0, _CAM_C_BOX),
            _row("cam-d", 4, 2, 4.0, _BAG_ONLY_BOX, DetClass.BACKPACK),
        ],
    )
    store.write(
        _OTHER_RUN,
        "FX-CODE-2",
        "CODE_TEST",
        [_row("cam-a", 2, 9, 99.0, _OTHER_BOX)],
    )


def _write_out_of_order(store: TrackStore) -> None:
    """One track whose row order, frame order, and timestamp order all disagree."""
    store.write(
        _RUN,
        _FIXTURE,
        "CODE_TEST",
        [
            _row("cam-a", 2, 1, 50.0, _END_BOX),
            _row("cam-a", 2, 4, 10.0, _FRAME_4_BOX),
            _row("cam-a", 2, 2, 10.0, _FIRST_BOX),
            _row("cam-a", 2, 2, 10.0, _SECOND_BOX),
        ],
    )


def test_tracklet_fields() -> None:
    assert [field.name for field in fields(Tracklet)] == [
        "camera_id",
        "track_id",
        "det_class",
        "run_id",
        "t_start",
        "t_end",
        "n_obs",
        "first_box",
        "last_box",
        "frames",
    ]


def test_tracklet_is_frozen() -> None:
    item = _tracklet("cam-a", 1, 1.0, 2.0, 1, _SINGLE_BOX, _SINGLE_BOX, (0,))
    with pytest.raises(FrozenInstanceError):
        item.n_obs = 9  # type: ignore[misc]


def test_database_file_stays_outside_the_repo(store: TrackStore, tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    db_path = store.db_path.resolve()
    assert db_path == (tmp_path / "tracks.sqlite").resolve()
    assert repo != db_path
    assert repo not in db_path.parents


def test_hand_built_rows_for_two_cameras_and_three_tracks(store: TrackStore) -> None:
    """10.0 and 50.0 are the stored timestamps. The loader does not add a clock offset."""
    _write_acceptance(store)
    loaded = load_tracklets(store, _RUN, camera_ids=("cam-a", "cam-b"))
    assert loaded == _expected_three()


def test_backpack_tracks_are_left_out_by_default(store: TrackStore) -> None:
    store.write(
        _RUN,
        _FIXTURE,
        "CODE_TEST",
        [
            _row("cam-a", 1, 0, 1.0, (0.0, 0.0, 2.0, 2.0)),
            _row("cam-a", 2, 0, 1.0, _BAG_BOX, DetClass.BACKPACK),
            _row("cam-a", 3, 0, 1.0, _HANDBAG_BOX, DetClass.HANDBAG),
        ],
    )
    loaded = load_tracklets(store, _RUN)
    assert [(item.track_id, item.det_class) for item in loaded] == [(1, DetClass.PERSON)]


def test_backpack_tracks_are_returned_for_that_class(store: TrackStore) -> None:
    _write_acceptance(store)
    loaded = load_tracklets(store, _RUN, det_class=DetClass.BACKPACK)
    found = [
        (item.camera_id, item.track_id, item.det_class, item.first_box, item.n_obs)
        for item in loaded
    ]
    assert found == [
        ("cam-a", 8, DetClass.BACKPACK, _BAG_BOX, 1),
        ("cam-d", 4, DetClass.BACKPACK, _BAG_ONLY_BOX, 1),
    ]


def test_handbag_filter_returns_only_handbags(store: TrackStore) -> None:
    _write_acceptance(store)
    loaded = load_tracklets(store, _RUN, det_class=DetClass.HANDBAG)
    assert len(loaded) == 1
    assert loaded[0].det_class is DetClass.HANDBAG
    assert loaded[0].camera_id == "cam-b"
    assert loaded[0].track_id == 3
    assert loaded[0].first_box == _HANDBAG_BOX
    assert loaded[0].frames == (4,)


def test_rows_from_another_run_id_are_ignored(store: TrackStore) -> None:
    _write_acceptance(store)
    loaded = load_tracklets(store, _RUN, camera_ids=("cam-a", "cam-b"))
    assert loaded == _expected_three()
    assert all(item.t_end != 99.0 for item in loaded)
    assert all(9 not in item.frames for item in loaded)


def test_camera_ids_filter_limits_cameras(store: TrackStore) -> None:
    store.write(
        _RUN,
        _FIXTURE,
        "CODE_TEST",
        [
            _row("cam-a", 1, 0, 1.0, (0.0, 0.0, 1.0, 2.0)),
            _row("cam-b", 1, 0, 1.0, (1.0, 1.0, 2.0, 3.0)),
        ],
    )
    loaded = load_tracklets(store, _RUN, camera_ids=("cam-a",))
    assert [item.camera_id for item in loaded] == ["cam-a"]


def test_unknown_camera_id_returns_no_tracklets(store: TrackStore) -> None:
    store.write(
        _RUN,
        _FIXTURE,
        "CODE_TEST",
        [_row("cam-a", 1, 0, 1.0, (0.0, 0.0, 1.0, 2.0))],
    )
    assert load_tracklets(store, _RUN, camera_ids=("cam-missing",)) == []
    mixed = load_tracklets(store, _RUN, camera_ids=("cam-missing", "cam-a"))
    assert [item.camera_id for item in mixed] == ["cam-a"]


def test_camera_ids_none_includes_every_camera_in_the_run(store: TrackStore) -> None:
    _write_acceptance(store)
    loaded = load_tracklets(store, _RUN)
    assert {(item.camera_id, item.track_id, item.det_class) for item in loaded} == {
        ("cam-a", 2, DetClass.PERSON),
        ("cam-a", 5, DetClass.PERSON),
        ("cam-b", 2, DetClass.PERSON),
        ("cam-c", 1, DetClass.PERSON),
    }


def test_run_with_no_rows_returns_no_tracklets(store: TrackStore) -> None:
    assert load_tracklets(store, "run-missing") == []


def test_sort_puts_camera_id_before_an_earlier_start(store: TrackStore) -> None:
    store.write(
        _RUN,
        _FIXTURE,
        "CODE_TEST",
        [
            _row("cam-b", 1, 0, 1.0, (0.0, 0.0, 1.0, 2.0)),
            _row("cam-a", 1, 0, 50.0, (1.0, 1.0, 2.0, 3.0)),
        ],
    )
    # Request cam-b first. Without the sort, that read order would win.
    loaded = load_tracklets(store, _RUN, camera_ids=("cam-b", "cam-a"))
    assert [(item.camera_id, item.t_start, item.track_id) for item in loaded] == [
        ("cam-a", 50.0, 1),
        ("cam-b", 1.0, 1),
    ]


def test_sort_puts_t_start_before_track_id(store: TrackStore) -> None:
    store.write(
        _RUN,
        _FIXTURE,
        "CODE_TEST",
        [
            _row("cam-a", 2, 0, 40.0, (0.0, 0.0, 1.0, 2.0)),
            _row("cam-a", 9, 0, 1.0, (1.0, 1.0, 2.0, 3.0)),
        ],
    )
    loaded = load_tracklets(store, _RUN)
    assert [(item.camera_id, item.t_start, item.track_id) for item in loaded] == [
        ("cam-a", 1.0, 9),
        ("cam-a", 40.0, 2),
    ]


def test_sort_uses_track_id_when_camera_and_start_tie(store: TrackStore) -> None:
    store.write(
        _RUN,
        _FIXTURE,
        "CODE_TEST",
        [
            _row("cam-a", 5, 0, 10.0, (0.0, 0.0, 1.0, 2.0)),
            _row("cam-a", 3, 0, 10.0, (1.0, 1.0, 2.0, 3.0)),
        ],
    )
    loaded = load_tracklets(store, _RUN)
    assert [item.track_id for item in loaded] == [3, 5]


def test_first_and_last_box_follow_timestamp_then_frame(store: TrackStore) -> None:
    """Row order would pick the t=50 box first. Frame order would too. Time does not."""
    _write_out_of_order(store)
    loaded = load_tracklets(store, _RUN)
    assert len(loaded) == 1
    assert loaded[0].first_box == _FIRST_BOX
    assert loaded[0].last_box == _END_BOX
    assert loaded[0].t_start == 10.0
    assert loaded[0].t_end == 50.0


def test_frames_follow_timestamp_then_frame_then_row_order(store: TrackStore) -> None:
    """Duplicate frame 2 keeps insert order. That is the store's row order."""
    _write_out_of_order(store)
    loaded = load_tracklets(store, _RUN)
    assert len(loaded) == 1
    assert loaded[0].frames == (2, 2, 4, 1)


def test_n_obs_counts_stored_rows(store: TrackStore) -> None:
    """Repeated frames still count. n_obs is the row count, not a unique-frame count."""
    _write_out_of_order(store)
    loaded = load_tracklets(store, _RUN)
    assert len(loaded) == 1
    assert loaded[0].n_obs == 4


def test_same_track_id_does_not_mix_classes(store: TrackStore) -> None:
    person_box = (0.0, 0.0, 2.0, 2.0)
    bag_box = (3.0, 3.0, 6.0, 6.0)
    store.write(
        _RUN,
        _FIXTURE,
        "CODE_TEST",
        [
            _row("cam-a", 1, 0, 1.0, person_box),
            _row("cam-a", 1, 1, 2.0, bag_box, DetClass.BACKPACK),
        ],
    )
    people = load_tracklets(store, _RUN)
    bags = load_tracklets(store, _RUN, det_class=DetClass.BACKPACK)
    assert len(people) == 1
    assert people[0].n_obs == 1
    assert people[0].first_box == person_box
    assert people[0].det_class is DetClass.PERSON
    assert len(bags) == 1
    assert bags[0].n_obs == 1
    assert bags[0].first_box == bag_box
    assert bags[0].det_class is DetClass.BACKPACK


def test_repeated_camera_ids_are_read_once(store: TrackStore) -> None:
    store.write(
        _RUN,
        _FIXTURE,
        "CODE_TEST",
        [_row("cam-a", 1, 0, 1.0, (0.0, 0.0, 1.0, 2.0))],
    )
    once = load_tracklets(store, _RUN, camera_ids=("cam-a",))
    twice = load_tracklets(store, _RUN, camera_ids=("cam-a", "cam-a"))
    assert twice == once
    assert len(twice) == 1


def test_ids_are_matched_exactly_without_case_folding(store: TrackStore) -> None:
    store.write(
        "Run-A",
        _FIXTURE,
        "CODE_TEST",
        [_row("Cam-A", 1, 0, 5.0, (0.0, 0.0, 1.0, 2.0))],
    )
    found = load_tracklets(store, "Run-A", camera_ids=("Cam-A",))
    assert len(found) == 1
    assert found[0].run_id == "Run-A"
    assert found[0].camera_id == "Cam-A"
    assert found[0].t_start == 5.0
    assert load_tracklets(store, "run-a", camera_ids=("Cam-A",)) == []
    assert load_tracklets(store, "Run-A", camera_ids=("cam-a",)) == []


@pytest.mark.parametrize(
    "value",
    ["", "   ", "\t", "\n", "\u3000", None, 4],
    ids=["empty", "spaces", "tab", "newline", "ideographic-space", "none", "int"],
)
def test_id_rejects_empty_or_whitespace(store: TrackStore, value: object) -> None:
    """A whitespace-only id is empty. The id is not stripped and then accepted."""
    with pytest.raises(ValueError, match="non-empty"):
        load_tracklets(store, value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        load_tracklets(store, _RUN, camera_ids=(value,))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [" cam-a", "cam-a ", "cam-a\n", "\tcam-a"],
    ids=["leading-space", "trailing-space", "trailing-newline", "leading-tab"],
)
def test_id_rejects_strip_mismatch(store: TrackStore, value: str) -> None:
    """Surrounding whitespace is refused. The stored lookup uses the original id."""
    with pytest.raises(ValueError, match="whitespace") as caught:
        load_tracklets(store, value)
    assert repr(value) in str(caught.value)
    with pytest.raises(ValueError, match="whitespace") as caught:
        load_tracklets(store, _RUN, camera_ids=(value,))
    assert repr(value) in str(caught.value)


def test_id_rejects_nfkc_mismatch(store: TrackStore) -> None:
    """Fullwidth ``ｃａｍ-a`` is refused. It is not folded to ``cam-a``."""
    value = "\uff43\uff41\uff4d-a"
    with pytest.raises(ValueError, match="NFKC") as caught:
        load_tracklets(store, value)
    assert repr(value) in str(caught.value)
    with pytest.raises(ValueError, match="NFKC") as caught:
        load_tracklets(store, _RUN, camera_ids=(value,))
    assert repr(value) in str(caught.value)


@pytest.mark.parametrize(
    "mark",
    ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"],
    ids=["zwsp", "zwnj", "zwj", "word-joiner", "bom"],
)
def test_id_rejects_format_character(store: TrackStore, mark: str) -> None:
    """Category Cf is refused, including each zero-width mark."""
    value = f"cam-a{mark}"
    with pytest.raises(ValueError, match="format character") as caught:
        load_tracklets(store, value)
    assert repr(value) in str(caught.value)
    with pytest.raises(ValueError, match="format character") as caught:
        load_tracklets(store, _RUN, camera_ids=(value,))
    assert repr(value) in str(caught.value)


def test_camera_ids_rejects_bare_str(store: TrackStore) -> None:
    """A str is not a collection of camera ids. Its characters are not ids."""
    _write_acceptance(store)
    with pytest.raises(ValueError, match="not a str"):
        load_tracklets(store, _RUN, camera_ids="cam-a")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "camera_ids",
    [[], (), set()],
    ids=["list", "tuple", "set"],
)
def test_camera_ids_rejects_empty_collection(
    store: TrackStore,
    camera_ids: list[str] | tuple[str, ...] | set[str],
) -> None:
    """An empty collection is refused. It does not mean every camera, or no rows."""
    _write_acceptance(store)
    with pytest.raises(ValueError, match="at least one"):
        load_tracklets(store, _RUN, camera_ids=camera_ids)


def test_camera_ids_rejects_a_non_collection(store: TrackStore) -> None:
    with pytest.raises(ValueError, match="must be a collection"):
        load_tracklets(store, _RUN, camera_ids=4)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    ["person", None, 1, True, DetClass.PERSON.value],
    ids=["string", "none", "int", "bool", "enum-value"],
)
def test_det_class_rejects_non_det_class(store: TrackStore, value: object) -> None:
    with pytest.raises(ValueError, match="DetClass"):
        load_tracklets(store, _RUN, det_class=value)  # type: ignore[arg-type]
