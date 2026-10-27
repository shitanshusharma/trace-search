"""Cross-camera linker on hand-built tracklets and unit vectors.

No footage, no files written into the repo tree, and no network.
"""

from __future__ import annotations

import socket
import sys
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from trace_search.attributes.attributes import Color, PersonAttributes
from trace_search.attributes.store import TrackAttributeStore
from trace_search.detection.types import DetClass
from trace_search.ingest.camera_registry import CameraConfig, CameraRegistry
from trace_search.matching.linker import (
    ATTR_WEIGHT,
    MIN_SCORE,
    Link,
    _assign,
    _cost_matrix,
    link_tracklets,
    topology_mask,
)
from trace_search.matching.route import build_route
from trace_search.matching.topology import CameraTopology, TransitEdge
from trace_search.matching.tracklets import Tracklet
from trace_search.reid.ac06_harness import run
from trace_search.reid.eval import EvalItem, rank1_map
from trace_search.reid.types import AppearanceEmbedding
from trace_search.reid.vector_store import InMemoryVectorStore

_BOX = (0.0, 0.0, 1.0, 2.0)
_EAST = (1.0, 0.0)
_NORTH = (0.0, 1.0)
_WEST = (-1.0, 0.0)
_RUN = "run-1"
Pair = tuple[str, int, str, int]
TrackKey = tuple[str, int]
AttrMap = dict[str, object]


@pytest.fixture(autouse=True)
def _refuse_outbound_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests refuse socket.socket.connect. Local SQLite does not connect."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("outbound network is refused")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_socket_connect_is_refused() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.2)
    try:
        with pytest.raises(OSError, match="outbound network is refused"):
            sock.connect(("192.0.2.1", 1))
    finally:
        sock.close()


def _tracklet(
    camera_id: str,
    track_id: int,
    t_start: float,
    t_end: float,
    *,
    run_id: str = _RUN,
    det_class: DetClass = DetClass.PERSON,
) -> Tracklet:
    return Tracklet(
        camera_id=camera_id,
        track_id=track_id,
        det_class=det_class,
        run_id=run_id,
        t_start=t_start,
        t_end=t_end,
        n_obs=1,
        first_box=_BOX,
        last_box=_BOX,
        frames=(0,),
    )


def _attrs(upper: object, lower: object, bag: object) -> AttrMap:
    return {"upper": upper, "lower": lower, "bag": bag}


def _agree() -> AttrMap:
    return _attrs("red", "blue", True)


def _cameras(*camera_ids: str) -> CameraRegistry:
    registry = CameraRegistry()
    for index, camera_id in enumerate(camera_ids):
        registry.register(CameraConfig(camera_id, "fixed", float(index), 0.0, 0.0))
    return registry


def _topology(
    edges: Sequence[tuple[str, str, float, float]],
    *extra_cameras: str,
) -> CameraTopology:
    names: list[str] = list(extra_cameras)
    for origin, dest, _low, _high in edges:
        if origin not in names:
            names.append(origin)
        if dest not in names:
            names.append(dest)
    built = tuple(TransitEdge(origin, dest, low, high) for origin, dest, low, high in edges)
    return CameraTopology(_cameras(*names), built)


def _window(low: float = 10.0, high: float = 20.0) -> CameraTopology:
    return _topology((("cam-a", "cam-b", low, high),))


def _pairs(links: Sequence[Link]) -> set[Pair]:
    return {(link.from_camera, link.from_track, link.to_camera, link.to_track) for link in links}


def _one_gap(gap: float) -> list[Link]:
    tracklets = (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-b", 2, gap, gap),
    )
    vectors = {("cam-a", 1): _EAST, ("cam-b", 2): _EAST}
    attributes = {("cam-a", 1): _agree(), ("cam-b", 2): _agree()}
    return link_tracklets(tracklets, _window(), vectors, attributes)


@pytest.mark.parametrize("gap", [9.9, 20.1])
def test_gap_outside_the_window_by_a_tenth_is_not_linked(gap: float) -> None:
    assert _pairs(_one_gap(gap)) == set()


@pytest.mark.parametrize("gap", [10.0, 20.0])
def test_gap_on_the_window_boundary_is_linked(gap: float) -> None:
    links = _one_gap(gap)
    assert _pairs(links) == {("cam-a", 1, "cam-b", 2)}
    assert links[0].gap_seconds == gap
    assert links[0].score == 1.0


def test_missing_edge_with_identical_vectors_is_not_linked() -> None:
    tracklets = (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-c", 2, 15.0, 15.0),
    )
    vectors = {("cam-a", 1): _EAST, ("cam-c", 2): _EAST}
    attributes = {("cam-a", 1): _agree(), ("cam-c", 2): _agree()}
    topology = _topology((), "cam-a", "cam-c")
    assert _pairs(link_tracklets(tracklets, topology, vectors, attributes)) == set()


def test_reverse_edge_with_identical_vectors_is_not_linked() -> None:
    tracklets = (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-b", 2, 15.0, 15.0),
    )
    vectors = {("cam-a", 1): _EAST, ("cam-b", 2): _EAST}
    attributes = {("cam-a", 1): _agree(), ("cam-b", 2): _agree()}
    topology = _topology((("cam-b", "cam-a", 10.0, 20.0),))
    assert _pairs(link_tracklets(tracklets, topology, vectors, attributes)) == set()


def test_two_exits_and_two_entries_assign_the_expected_pairs() -> None:
    tracklets = (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-a", 2, 0.0, 0.0),
        _tracklet("cam-b", 1, 15.0, 15.0),
        _tracklet("cam-b", 2, 15.0, 15.0),
    )
    vectors = {
        ("cam-a", 1): _EAST,
        ("cam-a", 2): _NORTH,
        ("cam-b", 1): _EAST,
        ("cam-b", 2): _NORTH,
    }
    links = link_tracklets(tracklets, _window(), vectors, {}, attr_weight=0.0)
    assert _pairs(links) == {
        ("cam-a", 1, "cam-b", 1),
        ("cam-a", 2, "cam-b", 2),
    }
    assert len(links) == len(_pairs(links))
    assert len({link.from_track for link in links}) == 2
    assert len({link.to_track for link in links}) == 2


def test_input_order_does_not_change_the_pairs() -> None:
    forward = (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-a", 2, 0.0, 0.0),
        _tracklet("cam-b", 1, 15.0, 15.0),
        _tracklet("cam-b", 2, 15.0, 15.0),
    )
    backward = tuple(reversed(forward))
    vectors = {
        ("cam-a", 1): _EAST,
        ("cam-a", 2): _NORTH,
        ("cam-b", 1): _EAST,
        ("cam-b", 2): _NORTH,
    }
    first = link_tracklets(forward, _window(), vectors, {}, attr_weight=0.0)
    second = link_tracklets(backward, _window(), vectors, {}, attr_weight=0.0)
    assert first == second


def test_greedy_assignment_differs_from_hungarian() -> None:
    """Greedy keeps the first perfect pair. Hungarian crosses for a higher sum.

    e1 agrees with both entries on the known colours. e2 agrees with n1 on
    two of three attributes and with n2 on one of two. The crossed sum is
    higher, so the pairs are (1, 2) and (2, 1), not (1, 1) and (2, 2).
    """
    tracklets = (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-a", 2, 1.0, 1.0),
        _tracklet("cam-b", 1, 15.0, 15.0),
        _tracklet("cam-b", 2, 16.0, 16.0),
    )
    vectors = {
        ("cam-a", 1): _EAST,
        ("cam-a", 2): _EAST,
        ("cam-b", 1): _EAST,
        ("cam-b", 2): _EAST,
    }
    attributes: dict[TrackKey, AttrMap] = {
        ("cam-a", 1): _attrs("red", "red", False),
        ("cam-a", 2): _attrs("red", "red", True),
        ("cam-b", 1): _attrs("red", "red", False),
        ("cam-b", 2): _attrs("red", None, False),
    }
    links = link_tracklets(tracklets, _window(), vectors, attributes)
    assert _pairs(links) == {
        ("cam-a", 1, "cam-b", 2),
        ("cam-a", 2, "cam-b", 1),
    }


def test_equal_scores_break_ties_on_camera_and_track_id() -> None:
    """Both exits look the same. The total is equal either way.

    The smaller ``(camera_id, track_id)`` pairing is (1, 1) and (2, 2).
    """
    tracklets = (
        _tracklet("cam-a", 2, 0.0, 0.0),
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-b", 2, 15.0, 15.0),
        _tracklet("cam-b", 1, 15.0, 15.0),
    )
    vectors = {key: _EAST for key in (("cam-a", 1), ("cam-a", 2), ("cam-b", 1), ("cam-b", 2))}
    attributes = {
        ("cam-a", 1): _attrs("red", "red", False),
        ("cam-a", 2): _attrs("red", "red", False),
        ("cam-b", 1): _attrs("red", "red", True),
        ("cam-b", 2): _attrs("red", "blue", True),
    }
    links = link_tracklets(tracklets, _window(), vectors, attributes)
    assert _pairs(links) == {
        ("cam-a", 1, "cam-b", 1),
        ("cam-a", 2, "cam-b", 2),
    }


def test_colour_mismatch_with_equal_vectors_scores_lower_than_a_match() -> None:
    matched = _one_gap(15.0)
    mismatched = link_tracklets(
        (
            _tracklet("cam-a", 1, 0.0, 0.0),
            _tracklet("cam-b", 2, 15.0, 15.0),
        ),
        _window(),
        {("cam-a", 1): _EAST, ("cam-b", 2): _EAST},
        {
            ("cam-a", 1): _agree(),
            ("cam-b", 2): _attrs("blue", "blue", True),
        },
    )
    assert len(matched) == 1
    assert len(mismatched) == 1
    assert mismatched[0].score < matched[0].score
    assert matched[0].score == 1.0
    assert mismatched[0].score == (1.0 + (2.0 / 3.0)) / 2.0


def test_no_track_has_two_links_on_the_same_edge() -> None:
    tracklets = (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-a", 2, 0.0, 0.0),
        _tracklet("cam-a", 3, 0.0, 0.0),
        _tracklet("cam-b", 1, 15.0, 15.0),
        _tracklet("cam-b", 2, 15.0, 15.0),
    )
    vectors = {
        ("cam-a", 1): _EAST,
        ("cam-a", 2): _NORTH,
        ("cam-a", 3): _WEST,
        ("cam-b", 1): _EAST,
        ("cam-b", 2): _NORTH,
    }
    links = link_tracklets(tracklets, _window(), vectors, {}, attr_weight=0.0)
    assert _pairs(links) == {
        ("cam-a", 1, "cam-b", 1),
        ("cam-a", 2, "cam-b", 2),
    }
    assert all(link.from_camera != link.to_camera for link in links)


def test_one_exit_may_link_out_on_two_edges() -> None:
    topology = _topology(
        (
            ("cam-a", "cam-b", 10.0, 20.0),
            ("cam-a", "cam-c", 10.0, 20.0),
        )
    )
    tracklets = (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-b", 2, 15.0, 15.0),
        _tracklet("cam-c", 3, 15.0, 15.0),
    )
    vectors = {("cam-a", 1): _EAST, ("cam-b", 2): _EAST, ("cam-c", 3): _EAST}
    attributes = {key: _agree() for key in vectors}
    assert _pairs(link_tracklets(tracklets, topology, vectors, attributes)) == {
        ("cam-a", 1, "cam-b", 2),
        ("cam-a", 1, "cam-c", 3),
    }


def test_two_runs_return_the_same_links() -> None:
    first = _one_gap(15.0)
    second = _one_gap(15.0)
    assert first == second


def test_score_is_one_for_identical_vectors_and_full_agreement() -> None:
    left = PersonAttributes(Color.RED, Color.BLUE, True, 1.0, 1.0)
    right = PersonAttributes(Color.RED, Color.BLUE, True, 1.0, 1.0)
    links = link_tracklets(
        (
            _tracklet("cam-a", 1, 0.0, 0.0),
            _tracklet("cam-b", 2, 15.0, 15.0),
        ),
        _window(),
        {("cam-a", 1): _EAST, ("cam-b", 2): _EAST},
        {("cam-a", 1): left, ("cam-b", 2): right},
    )
    assert len(links) == 1
    assert links[0].score == 1.0


def test_score_is_zero_for_opposite_vectors_and_full_disagreement() -> None:
    links = link_tracklets(
        (
            _tracklet("cam-a", 1, 0.0, 0.0),
            _tracklet("cam-b", 2, 15.0, 15.0),
        ),
        _window(),
        {("cam-a", 1): _EAST, ("cam-b", 2): _WEST},
        {
            ("cam-a", 1): _attrs("red", "blue", True),
            ("cam-b", 2): _attrs("green", "black", False),
        },
        min_score=0.0,
    )
    assert len(links) == 1
    assert links[0].score == 0.0


def test_unknown_attributes_have_share_zero() -> None:
    unknown = _attrs(Color.UNKNOWN, "unknown", None)
    links = link_tracklets(
        (
            _tracklet("cam-a", 1, 0.0, 0.0),
            _tracklet("cam-b", 2, 15.0, 15.0),
        ),
        _window(),
        {("cam-a", 1): _EAST, ("cam-b", 2): _EAST},
        {("cam-a", 1): unknown, ("cam-b", 2): unknown},
    )
    assert len(links) == 1
    assert links[0].score == 0.5
    omitted = link_tracklets(
        (
            _tracklet("cam-a", 1, 0.0, 0.0),
            _tracklet("cam-b", 2, 15.0, 15.0),
        ),
        _window(),
        {("cam-a", 1): _EAST, ("cam-b", 2): _EAST},
        {},
    )
    assert omitted[0].score == 0.5
    opposite = link_tracklets(
        (
            _tracklet("cam-a", 1, 0.0, 0.0),
            _tracklet("cam-b", 2, 15.0, 15.0),
        ),
        _window(),
        {("cam-a", 1): _EAST, ("cam-b", 2): _WEST},
        {("cam-a", 1): unknown, ("cam-b", 2): unknown},
        min_score=0.0,
    )
    assert opposite[0].score == 0.0


def test_unknown_attribute_is_not_a_disagreement() -> None:
    links = link_tracklets(
        (
            _tracklet("cam-a", 1, 0.0, 0.0),
            _tracklet("cam-b", 2, 15.0, 15.0),
        ),
        _window(),
        {("cam-a", 1): _EAST, ("cam-b", 2): _EAST},
        {
            ("cam-a", 1): _attrs(Color.UNKNOWN, "blue", True),
            ("cam-b", 2): _attrs("red", Color.BLUE, True),
        },
    )
    assert links[0].score == 1.0


def test_invalid_box_is_a_known_colour() -> None:
    agreed = link_tracklets(
        (
            _tracklet("cam-a", 1, 0.0, 0.0),
            _tracklet("cam-b", 2, 15.0, 15.0),
        ),
        _window(),
        {("cam-a", 1): _EAST, ("cam-b", 2): _EAST},
        {
            ("cam-a", 1): _attrs(Color.INVALID_BOX, None, None),
            ("cam-b", 2): _attrs("invalid_box", None, None),
        },
    )
    differed = link_tracklets(
        (
            _tracklet("cam-a", 1, 0.0, 0.0),
            _tracklet("cam-b", 2, 15.0, 15.0),
        ),
        _window(),
        {("cam-a", 1): _EAST, ("cam-b", 2): _EAST},
        {
            ("cam-a", 1): _attrs(Color.INVALID_BOX, None, None),
            ("cam-b", 2): _attrs("red", None, None),
        },
    )
    assert agreed[0].score == 1.0
    assert differed[0].score == 0.5


def test_track_without_a_vector_is_not_linked() -> None:
    tracklets = (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-a", 2, 0.0, 0.0),
        _tracklet("cam-b", 3, 15.0, 15.0),
    )
    vectors = {("cam-a", 1): _EAST, ("cam-b", 3): _EAST}
    attributes = {key: _agree() for key in (("cam-a", 1), ("cam-a", 2), ("cam-b", 3))}
    links = link_tracklets(tracklets, _window(), vectors, attributes)
    assert _pairs(links) == {("cam-a", 1, "cam-b", 3)}
    assert all(link.from_track != 2 and link.to_track != 2 for link in links)


def test_tracks_on_one_camera_are_not_linked() -> None:
    """Two exits on one camera are not linked to each other.

    The edge has one entry, so the tie-break keeps the smaller track id.
    """
    tracklets = (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-a", 2, 1.0, 1.0),
        _tracklet("cam-b", 3, 15.0, 15.0),
    )
    vectors = {("cam-a", 1): _EAST, ("cam-a", 2): _EAST, ("cam-b", 3): _EAST}
    attributes = {key: _agree() for key in vectors}
    links = link_tracklets(tracklets, _window(), vectors, attributes)
    assert _pairs(links) == {("cam-a", 1, "cam-b", 3)}
    assert all(link.from_camera != link.to_camera for link in links)


def test_min_score_drops_the_weaker_assigned_pair() -> None:
    tracklets = (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-a", 2, 0.0, 0.0),
        _tracklet("cam-b", 1, 15.0, 15.0),
        _tracklet("cam-b", 2, 15.0, 15.0),
    )
    vectors = {key: _EAST for key in (("cam-a", 1), ("cam-a", 2), ("cam-b", 1), ("cam-b", 2))}
    attributes = {
        ("cam-a", 1): _attrs("red", "red", False),
        ("cam-a", 2): _attrs("red", "red", False),
        ("cam-b", 1): _attrs("red", "red", False),
        ("cam-b", 2): _attrs("red", "red", True),
    }
    links = link_tracklets(tracklets, _window(), vectors, attributes, min_score=0.9)
    assert _pairs(links) == {("cam-a", 1, "cam-b", 1)}


def test_score_equal_to_min_score_is_kept() -> None:
    tracklets = (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-b", 2, 15.0, 15.0),
    )
    vectors = {("cam-a", 1): _EAST, ("cam-b", 2): _EAST}
    kept = link_tracklets(tracklets, _window(), vectors, {}, min_score=0.5)
    dropped = link_tracklets(tracklets, _window(), vectors, {}, min_score=0.51)
    assert len(kept) == 1
    assert kept[0].score == 0.5
    assert dropped == []


def test_defaults_match_the_module_constants() -> None:
    assert MIN_SCORE == 0.0
    assert ATTR_WEIGHT == 1.0
    tracklets = (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-b", 2, 15.0, 15.0),
    )
    vectors = {("cam-a", 1): _EAST, ("cam-b", 2): _EAST}
    attributes = {("cam-a", 1): _agree(), ("cam-b", 2): _agree()}
    implicit = link_tracklets(tracklets, _window(), vectors, attributes)
    explicit = link_tracklets(
        tracklets,
        _window(),
        vectors,
        attributes,
        min_score=MIN_SCORE,
        attr_weight=ATTR_WEIGHT,
    )
    assert implicit == explicit


def test_build_route_accepts_linker_scores_at_the_bounds() -> None:
    perfect = _one_gap(15.0)
    route = build_route(
        _tracklet("cam-a", 1, 0.0, 0.0),
        _gap_tracklets(15.0),
        perfect,
    )
    assert [(item.camera_id, item.track_id) for item in route.segments] == [
        ("cam-a", 1),
        ("cam-b", 2),
    ]
    zero = link_tracklets(
        _gap_tracklets(15.0),
        _window(),
        {("cam-a", 1): _EAST, ("cam-b", 2): _WEST},
        {
            ("cam-a", 1): _attrs("red", "blue", True),
            ("cam-b", 2): _attrs("green", "black", False),
        },
        min_score=0.0,
    )
    assert zero[0].score == 0.0
    walked = build_route(_tracklet("cam-a", 1, 0.0, 0.0), _gap_tracklets(15.0), zero)
    assert [(item.camera_id, item.track_id) for item in walked.segments] == [
        ("cam-a", 1),
        ("cam-b", 2),
    ]


def _gap_tracklets(gap: float) -> tuple[Tracklet, Tracklet]:
    return (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-b", 2, gap, gap),
    )


def test_link_fields_are_only_the_six_link_columns() -> None:
    assert [item.name for item in fields(Link)] == [
        "from_camera",
        "from_track",
        "to_camera",
        "to_track",
        "gap_seconds",
        "score",
    ]
    link = _one_gap(15.0)[0]
    with pytest.raises(FrozenInstanceError):
        link.score = 0.2  # type: ignore[misc]


def test_link_rejects_a_bad_field() -> None:
    with pytest.raises(ValueError):
        Link("cam-a", 1, "cam-a", 2, 1.0, 1.0)
    with pytest.raises(ValueError):
        Link("cam-a", 1, "cam-b", 2, 1.0, 1.1)
    with pytest.raises(ValueError):
        Link("cam-a", 1, "cam-b", 2, 1.0, -0.1)
    with pytest.raises(ValueError):
        Link("cam-a", True, "cam-b", 2, 1.0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Link("cam-a", 1, "cam-b", 2, float("nan"), 1.0)


def test_refuses_bad_weights_and_vectors() -> None:
    tracklets = _gap_tracklets(15.0)
    vectors = {("cam-a", 1): _EAST, ("cam-b", 2): _EAST}
    with pytest.raises(ValueError):
        link_tracklets(tracklets, _window(), vectors, {}, attr_weight=-0.1)
    with pytest.raises(ValueError):
        link_tracklets(tracklets, _window(), vectors, {}, attr_weight=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        link_tracklets(tracklets, _window(), vectors, {}, min_score=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        link_tracklets(
            tracklets,
            _window(),
            {("cam-a", 1): (0.0, 0.0), ("cam-b", 2): _EAST},
            {},
        )
    with pytest.raises(ValueError):
        link_tracklets(
            tracklets,
            _window(),
            {("cam-a", 1): (1.0, 0.0, 0.0), ("cam-b", 2): _EAST},
            {},
        )
    with pytest.raises(TypeError):
        link_tracklets(tracklets, _window(), ["nope"], {})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        link_tracklets(
            tracklets,
            _window(),
            vectors,
            {("cam-a", 1): _attrs("red", "blue", 1)},
        )
    with pytest.raises(ValueError):
        link_tracklets((_tracklet("cam-a", 1, 0.0, 0.0),) * 2, _window(), vectors, {})


def test_reads_vectors_and_attributes_from_the_stores(tmp_path: Path) -> None:
    tracklets = (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-a", 9, 0.0, 0.0),
        _tracklet("cam-b", 2, 15.0, 15.0),
    )
    vectors = InMemoryVectorStore(2)
    vectors.add(
        _RUN,
        "CODE_TEST",
        (
            _embedding("cam-a", 1, 5, _EAST),
            _embedding("cam-a", 1, 1, _NORTH),
            _embedding("cam-b", 2, 0, _NORTH),
        ),
    )
    attributes = TrackAttributeStore(tmp_path / "attrs.sqlite")
    _write_attr(attributes, "cam-a", 1, "red", "blue", True)
    _write_attr(attributes, "cam-a", 1, "green", "blue", True)
    _write_attr(attributes, "cam-b", 2, "green", "blue", True)
    links = link_tracklets(
        tracklets,
        _window(),
        vectors,
        attributes,
        run_id=_RUN,
        derived_from="CODE_TEST",
    )
    assert _pairs(links) == {("cam-a", 1, "cam-b", 2)}
    assert links[0].score == 1.0
    empty = InMemoryVectorStore(2)
    assert (
        link_tracklets(
            tracklets,
            _window(),
            empty,
            attributes,
            run_id=_RUN,
            derived_from="CODE_TEST",
        )
        == []
    )


def test_module_docstring_states_the_score_limits() -> None:
    text = " ".join((sys.modules["trace_search.matching.linker"].__doc__ or "").split())
    assert "placeholder similarity has no accuracy meaning" in text
    assert "((cosine + 1) / 2 + attr_weight * share) / (1 + attr_weight)" in text
    assert "not evidence" in text
    assert "share is 0" in text
    assert "keeps every mapped score, including 0" in text
    assert "no tie-break penalty" in text
    assert "1e-12" in text
    assert "longer side of the grid" in text
    assert "differ in ``run_id`` or ``det_class`` is skipped" in text
    assert "not linked to a handbag track" in text
    mask_text = " ".join((topology_mask.__doc__ or "").split())
    assert "maps a query item to its EXIT time and a gallery item to its ENTRY time" in mask_text
    source = Path(link_tracklets.__code__.co_filename).read_text(encoding="utf-8")
    assert "describe_tracks" not in source
    lowered = source.lower()
    for token in ("urllib", "requests.", "http.client", "socket", "https://", "http://"):
        assert token not in lowered


def test_topology_mask_excludes_a_pair_from_the_gallery() -> None:
    topology = _window(1.0, 5.0)
    times = {("cam-a", 1): 0.0, ("cam-b", 2): 0.5, ("cam-b", 3): 2.0}

    def time_of(item: EvalItem) -> float:
        return times[(item.camera_id, item.track_id)]

    mask = topology_mask(topology, time_of)
    query = [EvalItem("cam-a", 1, 1, _EAST)]
    outside = EvalItem("cam-b", 2, 1, _EAST)
    inside_other = EvalItem("cam-b", 3, 2, _EAST)
    gallery = [outside, inside_other]
    assert mask(query[0], outside) is False
    assert mask(query[0], inside_other) is True
    plain = rank1_map(query, gallery)
    masked = rank1_map(query, gallery, gallery_mask=mask)
    assert plain.rank1 == 1.0
    assert plain.num_valid_query == 1
    assert plain.num_masked_out == 0
    assert masked.rank1 == 0.0
    assert masked.num_valid_query == 1
    assert masked.num_invalid_query == 0
    assert masked.num_masked_out == 1
    assert masked.mean_ap == 0.0
    assert masked.num_gallery == 2
    result = run("FX-HOLD-001", query=query, gallery=gallery, gallery_mask=mask)
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert result["evidence"] is False
    assert result["status"] == "DEFERRED"
    assert result["ac06_score"] is None
    assert diagnostics["rank1"] == 0.0
    assert diagnostics["num_gallery"] == 2
    assert "not evidence" in str(diagnostics["note"])
    assert "not evidence" in str(diagnostics["topology_time_limits"])
    early = EvalItem("cam-b", 4, 1, _EAST)
    late = EvalItem("cam-b", 5, 1, _EAST)
    on_low = EvalItem("cam-b", 6, 1, _EAST)
    on_high = EvalItem("cam-b", 7, 1, _EAST)
    wide = _window(10.0, 20.0)
    wide_times = {
        ("cam-a", 1): 0.0,
        ("cam-b", 4): 9.9,
        ("cam-b", 5): 20.1,
        ("cam-b", 6): 10.0,
        ("cam-b", 7): 20.0,
    }

    def wide_time(item: EvalItem) -> float:
        return wide_times[(item.camera_id, item.track_id)]

    wide_mask = topology_mask(wide, wide_time)
    anchor = query[0]
    assert wide_mask(anchor, early) is False
    assert wide_mask(anchor, late) is False
    assert wide_mask(anchor, on_low) is True
    assert wide_mask(anchor, on_high) is True


def _embedding(
    camera_id: str,
    track_id: int,
    frame_number: int,
    vector: tuple[float, ...],
) -> AppearanceEmbedding:
    return AppearanceEmbedding(
        camera_id=camera_id,
        track_id=track_id,
        frame_number=frame_number,
        vector=vector,
        embedder_name="untrained-placeholder-v1",
        placeholder=True,
    )


def _write_attr(
    store: TrackAttributeStore,
    camera_id: str,
    track_id: int,
    upper: str,
    lower: str,
    bag: bool,
    *,
    run_id: str = _RUN,
    derived_from: str = "CODE_TEST",
) -> None:
    store.write(
        camera_id,
        track_id,
        run_id,
        derived_from,
        upper,
        lower,
        bag,
        1.0,
        1.0,
        1,
    )


def test_an_exit_skips_the_entry_outside_the_window() -> None:
    tracklets = (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-b", 2, 15.0, 15.0),
        _tracklet("cam-b", 3, 20.1, 20.1),
    )
    vectors = {("cam-a", 1): _EAST, ("cam-b", 2): _EAST, ("cam-b", 3): _EAST}
    attributes = {key: _agree() for key in vectors}
    links = link_tracklets(tracklets, _window(), vectors, attributes)
    assert _pairs(links) == {("cam-a", 1, "cam-b", 2)}


def test_store_read_uses_only_the_requested_run_and_origin(tmp_path: Path) -> None:
    """A second run and a SMOKE row with the same key must not enter a CODE_TEST link."""
    tracklets = (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-b", 2, 15.0, 15.0),
    )
    vectors = InMemoryVectorStore(2)
    vectors.add("run-1", "SMOKE", (_embedding("cam-a", 1, 0, _WEST),))
    vectors.add("run-2", "CODE_TEST", (_embedding("cam-a", 1, 1, _EAST),))
    vectors.add("run-1", "CODE_TEST", (_embedding("cam-a", 1, 5, _NORTH),))
    vectors.add("run-2", "CODE_TEST", (_embedding("cam-b", 2, 0, _WEST),))
    vectors.add("run-1", "SMOKE", (_embedding("cam-b", 2, 0, _EAST),))
    vectors.add("run-1", "CODE_TEST", (_embedding("cam-b", 2, 5, _NORTH),))
    attributes = TrackAttributeStore(tmp_path / "attrs.sqlite")
    _write_attr(attributes, "cam-a", 1, "red", "blue", True)
    _write_attr(attributes, "cam-a", 1, "green", "black", False, derived_from="SMOKE")
    _write_attr(attributes, "cam-a", 1, "green", "black", False, run_id="run-2")
    _write_attr(attributes, "cam-b", 2, "red", "blue", True)
    links = link_tracklets(
        tracklets,
        _window(),
        vectors,
        attributes,
        run_id=_RUN,
        derived_from="CODE_TEST",
    )
    assert _pairs(links) == {("cam-a", 1, "cam-b", 2)}
    assert links[0].score == 1.0


def test_store_read_requires_run_and_derived_from() -> None:
    vectors = InMemoryVectorStore(2)
    with pytest.raises(ValueError, match="required when reading a store"):
        link_tracklets((), _window(), vectors, {})


def test_a_cross_run_pair_is_not_linked() -> None:
    """The foreign entry is skipped. The same-run entry is still linked."""
    tracklets = (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-b", 2, 15.0, 15.0, run_id="run-2"),
        _tracklet("cam-b", 3, 15.0, 15.0),
    )
    vectors = {
        ("cam-a", 1): _EAST,
        ("cam-b", 2): _EAST,
        ("cam-b", 3): _EAST,
    }
    attributes = {key: _agree() for key in vectors}
    links = link_tracklets(tracklets, _window(), vectors, attributes)
    assert _pairs(links) == {("cam-a", 1, "cam-b", 3)}


def test_a_different_det_class_is_not_linked() -> None:
    """Person and handbag are not linked. The other person entry still is."""
    tracklets = (
        _tracklet("cam-a", 1, 0.0, 0.0),
        _tracklet("cam-b", 2, 15.0, 15.0, det_class=DetClass.HANDBAG),
        _tracklet("cam-b", 3, 15.0, 15.0),
    )
    vectors = {
        ("cam-a", 1): _EAST,
        ("cam-b", 2): _EAST,
        ("cam-b", 3): _EAST,
    }
    attributes = {key: _agree() for key in vectors}
    links = link_tracklets(tracklets, _window(), vectors, attributes)
    assert _pairs(links) == {("cam-a", 1, "cam-b", 3)}


def test_a_zero_length_store_vector_is_refused() -> None:
    store = _FixedVectorStore(())
    with pytest.raises(ValueError, match="vector must be non-empty"):
        link_tracklets(
            (_tracklet("cam-a", 1, 0.0, 0.0),),
            _window(),
            store,
            {},
            run_id=_RUN,
            derived_from="CODE_TEST",
        )


def test_a_zero_store_vector_is_refused() -> None:
    store = _FixedVectorStore((0.0, 0.0))
    with pytest.raises(ValueError, match="vector must be non-zero"):
        link_tracklets(
            (_tracklet("cam-a", 1, 0.0, 0.0),),
            _window(),
            store,
            {},
            run_id=_RUN,
            derived_from="CODE_TEST",
        )


def test_a_1e_5_total_gap_beats_identity_on_2x2_and_3x3() -> None:
    """The higher total wins by 1e-5. Identity is the tie-break order, and it loses."""
    two = [
        [0.9, 0.9 + 5e-6],
        [0.9 + 5e-6, 0.9],
    ]
    assert set(_assign(_cost_matrix(two))) == {(0, 1), (1, 0)}
    base = 0.9
    bumped = 0.9 + 5e-6
    three = [
        [base, base, bumped],
        [base, base, base],
        [bumped, base, base],
    ]
    assert set(_assign(_cost_matrix(three))) == {(0, 2), (1, 1), (2, 0)}


def test_review_gaps_of_1e_4_and_1_2e_4_beat_identity() -> None:
    """The re-review matrices: a 1e-4 total and a 1.2e-4 total beat the diagonal."""
    two = [
        [0.9, 0.90005],
        [0.90005, 0.9],
    ]
    assert set(_assign(_cost_matrix(two))) == {(0, 1), (1, 0)}
    base = 0.9
    bumped = 0.90006
    three = [
        [base, base, bumped],
        [base, base, base],
        [bumped, base, base],
    ]
    assert set(_assign(_cost_matrix(three))) == {(0, 2), (1, 1), (2, 0)}


def test_exact_score_tie_picks_sorted_track_order() -> None:
    tied = [
        [0.5, 0.5],
        [0.5, 0.5],
    ]
    assert _assign(_cost_matrix(tied)) == [(0, 0), (1, 1)]
    wide = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    assert _assign(_cost_matrix(wide)) == [(0, 0), (1, 1)]


def test_large_exact_tie_follows_track_id_order() -> None:
    """200 by 200 equal scores pair the sorted track ids in order."""
    count = 200
    exits = tuple(_tracklet("cam-a", track_id, 0.0, 0.0) for track_id in range(count, 0, -1))
    entries = tuple(_tracklet("cam-b", track_id, 15.0, 15.0) for track_id in range(count, 0, -1))
    vectors = {("cam-a", track_id): _EAST for track_id in range(1, count + 1)}
    vectors.update({("cam-b", track_id): _EAST for track_id in range(1, count + 1)})
    links = link_tracklets((*exits, *entries), _window(), vectors, {})
    assert len(links) == count
    assert _pairs(links) == {
        ("cam-a", track_id, "cam-b", track_id) for track_id in range(1, count + 1)
    }


class _FixedVectorStore:
    """One embedding, including a vector a real store would refuse."""

    def __init__(self, vector: tuple[float, ...]) -> None:
        self.dim = 2
        self._vector = vector

    def count(self, derived_from: str | None = None, run_id: str | None = None) -> int:
        return 1

    def search(
        self,
        vector: object,
        k: int,
        camera_ids: object = None,
        *,
        run_id: str | None = None,
        derived_from: str | None = None,
    ) -> list[tuple[AppearanceEmbedding, float]]:
        embedding = object.__new__(AppearanceEmbedding)
        object.__setattr__(embedding, "camera_id", "cam-a")
        object.__setattr__(embedding, "track_id", 1)
        object.__setattr__(embedding, "frame_number", 0)
        object.__setattr__(embedding, "vector", self._vector)
        object.__setattr__(embedding, "embedder_name", "untrained-placeholder-v1")
        object.__setattr__(embedding, "placeholder", True)
        return [(embedding, 0.0)]
