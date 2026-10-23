"""Route builder on hand-built tracklets and links.

``Link`` is not on main. ``FakeLink`` has T-E-5's fields and nothing else.
No footage, no files written into the repo tree, and no stored route.
"""

from __future__ import annotations

import ast
import inspect
import random
import socket
import sys
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, dataclass, fields
from pathlib import Path

import pytest

from trace_search.detection.types import DetClass
from trace_search.matching.route import LinkLike, Route, RouteSegment, _walk, build_route
from trace_search.matching.route_metrics import (
    RouteLike,
    RouteSegmentLike,
    TruthSegment,
    false_merge_rate,
    segment_precision,
    segment_recall,
)
from trace_search.matching.tracklets import Tracklet

_BOX = (0.0, 0.0, 1.0, 2.0)
_CAMERA_SITES = ("seed", "tracklet", "from_camera", "to_camera", "camera_ids")
_TRACK_SITES = ("seed_track", "tracklet_track", "from_track", "to_track")
_ZERO_WIDTH = ("\u200b", "\u200c", "\u200d", "\u2060", "\ufeff")


@dataclass(frozen=True)
class FakeLink:
    """Hand-built stand-in for T-E-5's Link. Not a Link class."""

    from_camera: str
    from_track: int
    to_camera: str
    to_track: int
    gap_seconds: float
    score: float


@pytest.fixture(autouse=True)
def _refuse_outbound_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests stay off the network."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("outbound network is refused")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _tracklet(camera_id: str, track_id: int, t_entry: float, t_exit: float) -> Tracklet:
    return _raw_tracklet(camera_id, track_id, t_entry, t_exit)


def _raw_tracklet(
    camera_id: object,
    track_id: object,
    t_entry: object,
    t_exit: object,
) -> Tracklet:
    return Tracklet(
        camera_id=camera_id,  # type: ignore[arg-type]
        track_id=track_id,  # type: ignore[arg-type]
        det_class=DetClass.PERSON,
        run_id="run-1",
        t_start=t_entry,  # type: ignore[arg-type]
        t_end=t_exit,  # type: ignore[arg-type]
        n_obs=1,
        first_box=_BOX,
        last_box=_BOX,
        frames=(0,),
    )


def _raw_link(
    from_camera: object,
    from_track: object,
    to_camera: object,
    to_track: object,
) -> FakeLink:
    return FakeLink(
        from_camera,  # type: ignore[arg-type]
        from_track,  # type: ignore[arg-type]
        to_camera,  # type: ignore[arg-type]
        to_track,  # type: ignore[arg-type]
        1.0,
        1.0,
    )


def _line(
    cameras: tuple[str, ...],
    *,
    start: float = 0.0,
    step: float = 2.0,
    span: float = 1.0,
) -> tuple[tuple[Tracklet, ...], tuple[FakeLink, ...]]:
    tracklets = tuple(
        _tracklet(camera, index + 1, start + step * index, start + step * index + span)
        for index, camera in enumerate(cameras)
    )
    links = tuple(
        FakeLink(
            left.camera_id,
            left.track_id,
            right.camera_id,
            right.track_id,
            step - span,
            1.0,
        )
        for left, right in zip(tracklets, tracklets[1:])
    )
    return tracklets, links


def _segment(camera_id: str, track_id: int, t_entry: float, t_exit: float) -> RouteSegment:
    return RouteSegment(camera_id, track_id, t_entry, t_exit)


def _protocol_attr_names(protocol: type[object]) -> tuple[str, ...]:
    """Public read-only protocol attributes, in class-body order."""
    declared = tuple(
        name
        for name in vars(protocol)
        if not name.startswith("_") and isinstance(inspect.getattr_static(protocol, name), property)
    )
    proto_attrs = getattr(protocol, "__protocol_attrs__", None)
    if proto_attrs is not None:
        assert set(declared) == set(proto_attrs)
    return declared


def _source() -> str:
    return Path(build_route.__code__.co_filename).read_text(encoding="utf-8")


def _build_at(site: str, value: object) -> Route:
    """One otherwise valid route with ``value`` planted at ``site``."""
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    other = _tracklet("cam-b", 2, 2.0, 3.0)
    link: FakeLink | None = _raw_link("cam-a", 1, "cam-b", 2)
    camera_ids: Sequence[object] | None = None
    if site == "seed":
        seed = _raw_tracklet(value, 1, 0.0, 1.0)
    elif site == "tracklet":
        other = _raw_tracklet(value, 2, 2.0, 3.0)
    elif site == "from_camera":
        link = _raw_link(value, 1, "cam-b", 2)
    elif site == "to_camera":
        link = _raw_link("cam-a", 1, value, 2)
    elif site == "camera_ids":
        camera_ids = (value,)
    elif site == "seed_track":
        seed = _raw_tracklet("cam-a", value, 0.0, 1.0)
    elif site == "tracklet_track":
        other = _raw_tracklet("cam-b", value, 2.0, 3.0)
    elif site == "from_track":
        link = _raw_link("cam-a", value, "cam-b", 2)
    elif site == "to_track":
        link = _raw_link("cam-a", 1, "cam-b", value)
    else:
        raise AssertionError(site)
    return build_route(
        seed,
        (seed, other),
        (link,),
        camera_ids=camera_ids,  # type: ignore[arg-type]
    )


def _refuse(site: str, value: object, match: str) -> None:
    with pytest.raises(ValueError, match=match) as caught:
        _build_at(site, value)
    assert repr(value) in str(caught.value)


def test_route_fields_are_exact_and_frozen() -> None:
    assert tuple(field.name for field in fields(RouteSegment)) == (
        "camera_id",
        "track_id",
        "t_entry",
        "t_exit",
    )
    assert tuple(field.name for field in fields(Route)) == (
        "seed_camera",
        "seed_track",
        "segments",
    )
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    route = build_route(seed, (seed,), ())
    assert isinstance(route.segments, tuple)
    assert not hasattr(route, "id")
    with pytest.raises(FrozenInstanceError):
        route.seed_camera = "cam-b"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        route.segments[0].t_exit = 9.0  # type: ignore[misc]


def test_fake_link_fields_match_link_like() -> None:
    """Renaming a field on FakeLink or LinkLike fails this test."""
    fake = tuple(field.name for field in fields(FakeLink))
    assert fake == (
        "from_camera",
        "from_track",
        "to_camera",
        "to_track",
        "gap_seconds",
        "score",
    )
    assert fake == _protocol_attr_names(LinkLike)
    for name in fake:
        prop = inspect.getattr_static(LinkLike, name)
        assert isinstance(prop, property)
        assert prop.fset is None
        assert prop.fget is not None
        with pytest.raises(NotImplementedError):
            prop.fget(object())


def test_module_does_not_define_a_link_class_or_call_itself() -> None:
    tree = ast.parse(_source())
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    assert "LinkLike" in classes
    assert "Link" not in classes
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                assert child.func.id != node.name


def test_route_source_has_no_network_or_loader_hooks() -> None:
    text = _source()
    for banned in (
        "urllib",
        "requests",
        "socket",
        "http://",
        "https://",
        "load_reid_samples",
        "LoaderContext",
    ):
        assert banned not in text


def test_four_camera_chain_from_a_middle_seed_is_time_ordered() -> None:
    tracklets, links = _line(("cam-a", "cam-b", "cam-c", "cam-d"))
    route = build_route(tracklets[1], tracklets, links)
    assert route.seed_camera == "cam-b"
    assert route.seed_track == 2
    assert route.segments == (
        _segment("cam-a", 1, 0.0, 1.0),
        _segment("cam-b", 2, 2.0, 3.0),
        _segment("cam-c", 3, 4.0, 5.0),
        _segment("cam-d", 4, 6.0, 7.0),
    )


def test_cycle_stops_without_repeating_a_track() -> None:
    tracklets, links = _line(("cam-a", "cam-b", "cam-c"))
    cycled = (*links, FakeLink("cam-c", 3, "cam-a", 1, 1.0, 1.0))
    route = build_route(tracklets[1], tracklets, cycled)
    keys = tuple((item.camera_id, item.track_id) for item in route.segments)
    assert keys == (("cam-a", 1), ("cam-b", 2), ("cam-c", 3))
    assert len(keys) == len(set(keys))


def test_self_loop_does_not_repeat_the_seed() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    link = FakeLink("cam-a", 1, "cam-a", 1, 0.0, 1.0)
    route = build_route(seed, (seed,), (link,))
    assert route.segments == (_segment("cam-a", 1, 0.0, 1.0),)


def test_window_drops_segments_outside_the_chain() -> None:
    tracklets, links = _line(("cam-a", "cam-b", "cam-c", "cam-d"))
    route = build_route(tracklets[1], tracklets, links, t_window=(1.5, 5.5))
    assert tuple(item.camera_id for item in route.segments) == ("cam-b", "cam-c")


def test_window_stops_at_the_edge_and_does_not_jump() -> None:
    """A later track inside the window is not reached through an outside track."""
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    outside = _tracklet("cam-b", 2, 10.0, 11.0)
    inside = _tracklet("cam-c", 3, 2.0, 3.0)
    links = (
        FakeLink("cam-a", 1, "cam-b", 2, 1.0, 1.0),
        FakeLink("cam-b", 2, "cam-c", 3, 1.0, 1.0),
    )
    route = build_route(seed, (seed, outside, inside), links, t_window=(0.0, 5.0))
    assert tuple(item.camera_id for item in route.segments) == ("cam-a",)


def test_window_edges_are_inclusive() -> None:
    early = _tracklet("cam-a", 1, 1.0, 2.0)
    seed = _tracklet("cam-b", 2, 2.0, 4.0)
    late = _tracklet("cam-c", 3, 4.0, 5.0)
    links = (
        FakeLink("cam-a", 1, "cam-b", 2, 0.0, 1.0),
        FakeLink("cam-b", 2, "cam-c", 3, 0.0, 1.0),
    )
    route = build_route(seed, (early, seed, late), links, t_window=(2.0, 4.0))
    assert tuple(item.camera_id for item in route.segments) == ("cam-a", "cam-b", "cam-c")


def test_window_just_outside_the_edge_is_dropped() -> None:
    early = _tracklet("cam-a", 1, 0.0, 1.9)
    seed = _tracklet("cam-b", 2, 2.0, 4.0)
    late = _tracklet("cam-c", 3, 4.1, 5.0)
    links = (
        FakeLink("cam-a", 1, "cam-b", 2, 0.1, 1.0),
        FakeLink("cam-b", 2, "cam-c", 3, 0.1, 1.0),
    )
    route = build_route(seed, (early, seed, late), links, t_window=(2.0, 4.0))
    assert tuple(item.camera_id for item in route.segments) == ("cam-b",)


def test_track_that_covers_the_window_is_kept() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 10.0)
    route = build_route(seed, (seed,), (), t_window=(2.0, 3.0))
    assert route.segments == (_segment("cam-a", 1, 0.0, 10.0),)


def test_point_window_keeps_a_track_that_touches_it() -> None:
    seed = _tracklet("cam-a", 1, 1.0, 1.0)
    route = build_route(seed, (seed,), (), t_window=(1.0, 1.0))
    assert route.segments == (_segment("cam-a", 1, 1.0, 1.0),)


def test_seed_with_no_links_is_one_segment() -> None:
    seed = _tracklet("cam-a", 7, 1.0, 2.0)
    route = build_route(seed, (seed,), ())
    assert route.seed_camera == "cam-a"
    assert route.seed_track == 7
    assert route.segments == (_segment("cam-a", 7, 1.0, 2.0),)


def test_links_that_do_not_touch_the_seed_are_not_used() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    left = _tracklet("cam-b", 2, 2.0, 3.0)
    right = _tracklet("cam-c", 3, 4.0, 5.0)
    link = FakeLink("cam-b", 2, "cam-c", 3, 1.0, 1.0)
    route = build_route(seed, (seed, left, right), (link,))
    assert tuple(item.camera_id for item in route.segments) == ("cam-a",)


def test_camera_ids_stop_traversal_and_do_not_jump() -> None:
    tracklets, links = _line(("cam-a", "cam-b", "cam-c", "cam-d"))
    route = build_route(
        tracklets[0],
        tracklets,
        links,
        camera_ids=("cam-a", "cam-b", "cam-d"),
    )
    assert tuple(item.camera_id for item in route.segments) == ("cam-a", "cam-b")


def test_camera_ids_are_exact_and_duplicates_are_one_camera() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    other = _tracklet("Cam-B", 2, 2.0, 3.0)
    link = FakeLink("cam-a", 1, "Cam-B", 2, 1.0, 1.0)
    route = build_route(
        seed,
        (seed, other),
        (link,),
        camera_ids=("cam-a", "cam-a", "cam-b"),
    )
    assert tuple(item.camera_id for item in route.segments) == ("cam-a",)
    as_set = build_route(seed, (seed, other), (link,), camera_ids={"cam-a", "Cam-B"})
    assert tuple(item.camera_id for item in as_set.segments) == ("cam-a", "Cam-B")


def test_tie_break_picks_the_smaller_camera_id() -> None:
    """Equal scores ignore gap. The smaller camera id wins."""
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    loser = _tracklet("cam-c", 1, 2.0, 3.0)
    winner = _tracklet("cam-b", 2, 2.0, 3.0)
    further = _tracklet("cam-d", 1, 4.0, 5.0)
    links = (
        FakeLink("cam-a", 1, "cam-c", 1, 0.1, 1.0),
        FakeLink("cam-a", 1, "cam-b", 2, 9.0, 1.0),
        FakeLink("cam-c", 1, "cam-d", 1, 1.0, 1.0),
    )
    catalogue = (seed, loser, winner, further)
    route = build_route(seed, catalogue, links)
    assert tuple((item.camera_id, item.track_id) for item in route.segments) == (
        ("cam-a", 1),
        ("cam-b", 2),
    )
    assert build_route(seed, catalogue, tuple(reversed(links))) == route


def test_tie_break_picks_the_smaller_track_id() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    loser = _tracklet("cam-b", 9, 2.0, 3.0)
    winner = _tracklet("cam-b", 2, 2.0, 3.0)
    further = _tracklet("cam-c", 1, 4.0, 5.0)
    links = (
        FakeLink("cam-a", 1, "cam-b", 9, 0.1, 1.0),
        FakeLink("cam-a", 1, "cam-b", 2, 9.0, 1.0),
        FakeLink("cam-b", 9, "cam-c", 1, 1.0, 1.0),
    )
    route = build_route(seed, (seed, loser, winner, further), links)
    assert tuple((item.camera_id, item.track_id) for item in route.segments) == (
        ("cam-a", 1),
        ("cam-b", 2),
    )


def test_backward_tie_break_picks_the_smaller_camera_id() -> None:
    seed = _tracklet("cam-d", 1, 10.0, 11.0)
    loser = _tracklet("cam-c", 1, 0.0, 1.0)
    winner = _tracklet("cam-b", 2, 0.0, 1.0)
    further = _tracklet("cam-e", 1, 0.0, 0.5)
    links = (
        FakeLink("cam-c", 1, "cam-d", 1, 0.1, 1.0),
        FakeLink("cam-b", 2, "cam-d", 1, 9.0, 1.0),
        FakeLink("cam-e", 1, "cam-c", 1, 1.0, 1.0),
    )
    route = build_route(seed, (seed, loser, winner, further), links)
    assert tuple(item.camera_id for item in route.segments) == ("cam-b", "cam-d")


def test_same_route_on_repeat_and_shuffled_inputs() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    loser = _tracklet("cam-c", 1, 2.0, 3.0)
    winner = _tracklet("cam-b", 2, 2.0, 3.0)
    tracklets = [seed, loser, winner]
    links = [
        FakeLink("cam-a", 1, "cam-c", 1, 1.0, 0.9),
        FakeLink("cam-a", 1, "cam-b", 2, 1.0, 0.1),
    ]
    first = build_route(seed, tracklets, links)
    second = build_route(seed, tracklets, links)
    assert first == second
    assert first is not second
    rng = random.Random(2026)
    for _index in range(8):
        shuffled_tracks = list(tracklets)
        shuffled_links = list(links)
        rng.shuffle(shuffled_tracks)
        rng.shuffle(shuffled_links)
        assert build_route(seed, shuffled_tracks, shuffled_links) == first
    assert tuple(item.camera_id for item in first.segments) == ("cam-a", "cam-c")


def test_higher_score_beats_the_smaller_camera_id() -> None:
    """A higher score wins even when the other camera id sorts first."""
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    dead = _tracklet("cam-b", 1, 2.0, 3.0)
    live = _tracklet("cam-c", 1, 2.0, 3.0)
    later = _tracklet("cam-d", 1, 4.0, 5.0)
    last = _tracklet("cam-e", 1, 6.0, 7.0)
    links = (
        FakeLink("cam-a", 1, "cam-b", 1, 1.0, 0.01),
        FakeLink("cam-a", 1, "cam-c", 1, 9.0, 0.99),
        FakeLink("cam-c", 1, "cam-d", 1, 1.0, 1.0),
        FakeLink("cam-d", 1, "cam-e", 1, 1.0, 1.0),
    )
    catalogue = (seed, dead, live, later, last)
    route = build_route(seed, catalogue, links)
    assert tuple(item.camera_id for item in route.segments) == ("cam-a", "cam-c", "cam-d", "cam-e")
    assert build_route(seed, catalogue, tuple(reversed(links))) == route


def _route_with_score(score: object) -> Route:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    other = _tracklet("cam-b", 2, 2.0, 3.0)
    link = FakeLink("cam-a", 1, "cam-b", 2, 1.0, score)  # type: ignore[arg-type]
    return build_route(seed, (seed, other), (link,))


@pytest.mark.parametrize(
    "score",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "pos-inf", "neg-inf"],
)
def test_non_finite_scores_are_refused(score: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _route_with_score(score)


@pytest.mark.parametrize(
    "score",
    ["0.5", None, True, False],
    ids=["str", "none", "true", "false"],
)
def test_non_numeric_scores_are_refused(score: object) -> None:
    """A bool is not a score. It is not read as 0 or 1, and a string is not parsed."""
    with pytest.raises(ValueError, match="finite number") as caught:
        _route_with_score(score)
    assert repr(score) in str(caught.value)


@pytest.mark.parametrize(
    "score",
    [-0.01, -1.0, 1.01, 2, 5.0],
    ids=["just-under", "negative", "just-over", "int-two", "five"],
)
def test_out_of_range_scores_are_refused(score: float) -> None:
    """Scores outside the closed interval ``[0, 1]`` are refused."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _route_with_score(score)


def test_scores_at_zero_and_one_are_kept() -> None:
    """``0`` and ``1`` are inside the range, including whole-number ints."""
    for score in (0.0, 1.0, 0, 1):
        route = _route_with_score(score)
        assert tuple(item.camera_id for item in route.segments) == ("cam-a", "cam-b")


def test_duplicate_link_keeps_the_higher_score() -> None:
    """Two links for one track pair keep the higher score, not the first or the lower."""
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    paired = _tracklet("cam-b", 1, 2.0, 3.0)
    rival = _tracklet("cam-c", 1, 2.0, 3.0)
    higher_second = (
        FakeLink("cam-a", 1, "cam-b", 1, 1.0, 0.2),
        FakeLink("cam-a", 1, "cam-c", 1, 1.0, 0.5),
        FakeLink("cam-a", 1, "cam-b", 1, 1.0, 0.9),
    )
    higher_first = (
        FakeLink("cam-a", 1, "cam-b", 1, 1.0, 0.9),
        FakeLink("cam-a", 1, "cam-c", 1, 1.0, 0.5),
        FakeLink("cam-a", 1, "cam-b", 1, 1.0, 0.2),
    )
    catalogue = (seed, paired, rival)
    route = build_route(seed, catalogue, higher_second)
    assert tuple(item.camera_id for item in route.segments) == ("cam-a", "cam-b")
    assert build_route(seed, catalogue, higher_first) == route


def test_blocked_candidate_falls_through_to_the_next() -> None:
    """A blocked first choice is skipped. The next open candidate is used."""
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    excluded = _tracklet("cam-aa", 1, 2.0, 3.0)
    allowed = _tracklet("cam-c", 1, 2.0, 3.0)
    by_camera = build_route(
        seed,
        (seed, excluded, allowed),
        (
            FakeLink("cam-a", 1, "cam-aa", 1, 1.0, 0.99),
            FakeLink("cam-a", 1, "cam-c", 1, 1.0, 0.1),
        ),
        camera_ids=("cam-a", "cam-c"),
    )
    assert tuple(item.camera_id for item in by_camera.segments) == ("cam-a", "cam-c")

    outside = _tracklet("cam-b", 1, 50.0, 51.0)
    inside = _tracklet("cam-c", 1, 2.0, 3.0)
    by_window = build_route(
        seed,
        (seed, outside, inside),
        (
            FakeLink("cam-a", 1, "cam-b", 1, 1.0, 0.99),
            FakeLink("cam-a", 1, "cam-c", 1, 1.0, 0.1),
        ),
        t_window=(0.0, 10.0),
    )
    assert tuple(item.camera_id for item in by_window.segments) == ("cam-a", "cam-c")

    earlier = _tracklet("cam-a", 1, 0.0, 1.0)
    middle = _tracklet("cam-b", 1, 2.0, 3.0)
    later = _tracklet("cam-c", 1, 4.0, 5.0)
    by_visit = build_route(
        middle,
        (earlier, middle, later),
        (
            FakeLink("cam-a", 1, "cam-b", 1, 1.0, 1.0),
            FakeLink("cam-b", 1, "cam-a", 1, 1.0, 0.99),
            FakeLink("cam-b", 1, "cam-c", 1, 1.0, 0.1),
        ),
    )
    assert tuple(item.camera_id for item in by_visit.segments) == ("cam-a", "cam-b", "cam-c")


def test_segments_follow_time_when_a_forward_link_goes_earlier() -> None:
    late = _tracklet("cam-b", 1, 10.0, 11.0)
    early = _tracklet("cam-a", 1, 0.0, 1.0)
    link = FakeLink("cam-b", 1, "cam-a", 1, 0.0, 1.0)
    route = build_route(late, (late, early), (link,))
    assert tuple(item.camera_id for item in route.segments) == ("cam-a", "cam-b")


def test_equal_times_break_on_camera_id_then_track_id() -> None:
    seed = _tracklet("cam-m", 5, 0.0, 1.0)
    earlier_camera = _tracklet("cam-b", 9, 0.0, 1.0)
    same_camera = _tracklet("cam-m", 1, 0.0, 1.0)
    links = (
        FakeLink("cam-m", 5, "cam-b", 9, 0.0, 1.0),
        FakeLink("cam-b", 9, "cam-m", 1, 0.0, 1.0),
    )
    route = build_route(seed, (seed, earlier_camera, same_camera), links)
    assert tuple((item.camera_id, item.track_id) for item in route.segments) == (
        ("cam-b", 9),
        ("cam-m", 1),
        ("cam-m", 5),
    )


def test_chain_of_at_least_2000_segments_is_iterative() -> None:
    count = max(2001, sys.getrecursionlimit() + 50)
    cameras = tuple(f"c{index:04d}" for index in range(count))
    tracklets, links = _line(cameras, start=0.0, step=1.0, span=0.5)
    route = build_route(tracklets[0], tracklets, links)
    assert len(route.segments) == count
    assert route.segments[0].camera_id == cameras[0]
    assert route.segments[-1].camera_id == cameras[-1]
    assert route.segments[0].t_entry < route.segments[-1].t_entry
    keys = [(item.camera_id, item.track_id) for item in route.segments]
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys))


def test_real_route_satisfies_route_metrics_protocols() -> None:
    """T-E-7's protocols accept a real route, and the metrics run on it."""
    tracklets, links = _line(("cam-a", "cam-b"))
    route = build_route(tracklets[0], tracklets, links)
    assert tuple(field.name for field in fields(RouteSegment)) == _protocol_attr_names(
        RouteSegmentLike
    )
    assert tuple(field.name for field in fields(Route)) == _protocol_attr_names(RouteLike)
    checked: RouteLike = route
    segment: RouteSegmentLike = route.segments[0]
    assert checked.seed_camera == route.seed_camera == segment.camera_id
    assert checked.seed_track == route.seed_track == segment.track_id
    assert segment.t_exit == route.segments[0].t_exit
    ownership = {(item.camera_id, item.track_id): "fig-a" for item in route.segments}
    ownership[(route.seed_camera, route.seed_track)] = "fig-a"
    truth = tuple(
        TruthSegment("fig-a", item.camera_id, item.t_entry, item.t_exit) for item in route.segments
    )
    assert segment_recall(route.segments, truth, ownership) == 1.0
    assert segment_precision(route.segments, truth, ownership) == 1.0
    assert false_merge_rate((route,), ownership) == 0.0


def test_route_is_not_stored_and_the_caller_lists_are_not_kept() -> None:
    route_mod = sys.modules["trace_search.matching.route"]

    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    other = _tracklet("cam-b", 2, 2.0, 3.0)
    tracklets = [seed, other]
    links = [FakeLink("cam-a", 1, "cam-b", 2, 1.0, 1.0)]
    route = build_route(seed, tracklets, links)
    tracklets.clear()
    links.clear()
    again = build_route(seed, (seed, other), (FakeLink("cam-a", 1, "cam-b", 2, 1.0, 1.0),))
    assert route == again
    assert route is not again
    assert not hasattr(build_route, "cache_info")
    for name, value in vars(route_mod).items():
        if name.startswith("__"):
            continue
        assert type(value) not in (list, dict, set), name


def test_whole_number_times_are_stored_as_floats() -> None:
    seed = _raw_tracklet("cam-a", 1, 0, 2)
    route = build_route(seed, (seed,), ())
    assert route.segments[0].t_entry == 0.0
    assert isinstance(route.segments[0].t_entry, float)
    assert isinstance(route.segments[0].t_exit, float)


def test_seed_may_be_a_separate_equal_tracklet() -> None:
    listed = _tracklet("cam-a", 1, 0.0, 1.0)
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    route = build_route(seed, (listed,), ())
    assert route.segments == (_segment("cam-a", 1, 0.0, 1.0),)


@pytest.mark.parametrize(
    ("t_entry", "t_exit"),
    [(0.5, 1.0), (0.0, 1.5)],
    ids=["start", "end"],
)
def test_seed_times_must_match_the_catalogue(t_entry: float, t_exit: float) -> None:
    listed = _tracklet("cam-a", 1, 0.0, 1.0)
    seed = _tracklet("cam-a", 1, t_entry, t_exit)
    with pytest.raises(ValueError, match="do not match"):
        build_route(seed, (listed,), ())


def test_seed_missing_from_tracklets_is_refused() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    other = _tracklet("cam-b", 2, 2.0, 3.0)
    with pytest.raises(ValueError, match="not in tracklets") as caught:
        build_route(seed, (other,), ())
    assert "cam-a" in str(caught.value)


def test_duplicate_tracklet_is_refused() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    copy = _tracklet("cam-a", 1, 0.0, 1.0)
    with pytest.raises(ValueError, match="duplicate tracklet"):
        build_route(seed, (seed, copy), ())


def test_tracklets_must_be_tracklet_records() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    with pytest.raises(ValueError, match="Tracklet"):
        build_route(seed, (seed, object()), ())  # type: ignore[arg-type]


def test_records_reject_a_str_and_a_non_iterable() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    with pytest.raises(ValueError, match="iterable of records"):
        build_route(seed, "cam-a", ())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="iterable of records"):
        build_route(seed, (seed,), 5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="iterable of records"):
        build_route(seed, (seed,), b"ab")  # type: ignore[arg-type]


def test_walk_with_an_empty_catalogue_adds_nothing() -> None:
    """The step loop can be entered and it can be skipped."""
    assert _walk(("cam-a", 1), {}, set(), {}, None, None) == ()


def test_followed_link_without_a_tracklet_is_refused() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    link = FakeLink("cam-a", 1, "cam-z", 9, 1.0, 1.0)
    with pytest.raises(ValueError, match="no tracklet") as caught:
        build_route(seed, (seed,), (link,))
    assert "cam-z" in str(caught.value)


def test_unfollowed_link_may_name_a_missing_tracklet() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    other = _tracklet("cam-b", 2, 2.0, 3.0)
    link = FakeLink("cam-b", 2, "cam-z", 9, 1.0, 1.0)
    route = build_route(seed, (seed, other), (link,))
    assert tuple(item.camera_id for item in route.segments) == ("cam-a",)


def test_excluded_camera_without_a_tracklet_stops_without_raising() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    link = FakeLink("cam-a", 1, "cam-z", 9, 1.0, 1.0)
    route = build_route(seed, (seed,), (link,), camera_ids=("cam-a",))
    assert route.segments == (_segment("cam-a", 1, 0.0, 1.0),)


def test_seed_outside_camera_ids_is_refused() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    with pytest.raises(ValueError, match="outside camera_ids"):
        build_route(seed, (seed,), (), camera_ids=("cam-b",))


def test_seed_outside_the_window_is_refused() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    with pytest.raises(ValueError, match="outside t_window"):
        build_route(seed, (seed,), (), t_window=(2.0, 3.0))


def test_tracklet_interval_must_run_forward() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    bad = _tracklet("cam-b", 2, 5.0, 4.0)
    with pytest.raises(ValueError, match="t_end"):
        build_route(seed, (seed, bad), ())


@pytest.mark.parametrize(
    "t_entry",
    [float("nan"), float("inf"), True, "now"],
    ids=["nan", "inf", "bool", "text"],
)
def test_tracklet_times_must_be_finite_numbers(t_entry: object) -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    bad = _raw_tracklet("cam-b", 2, t_entry, 1.0)
    with pytest.raises(ValueError, match="finite"):
        build_route(seed, (seed, bad), ())


def test_window_rejects_bad_pairs() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    with pytest.raises(ValueError, match="or None"):
        build_route(seed, (seed,), (), t_window="ab")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="or None"):
        build_route(seed, (seed,), (), t_window=5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pair of finite times"):
        build_route(seed, (seed,), (), t_window=(0.0,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pair of finite times"):
        build_route(seed, (seed,), (), t_window=(0.0, 1.0, 2.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite number"):
        build_route(seed, (seed,), (), t_window=(True, 1.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be finite"):
        build_route(seed, (seed,), (), t_window=(0.0, float("nan")))
    with pytest.raises(ValueError, match="must be >="):
        build_route(seed, (seed,), (), t_window=(3.0, 1.0))


def test_camera_ids_rejects_a_str_a_non_collection_and_empty() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    with pytest.raises(ValueError, match="not a str"):
        build_route(seed, (seed,), (), camera_ids="cam-a")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="collection of camera ids"):
        build_route(seed, (seed,), (), camera_ids=4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        build_route(seed, (seed,), (), camera_ids=())
    with pytest.raises(ValueError, match="at least one"):
        build_route(seed, (seed,), (), camera_ids=[])


@pytest.mark.parametrize("site", _CAMERA_SITES)
@pytest.mark.parametrize(
    "value",
    [" cam-a", "cam-a ", "cam-a\n", "\tcam-a", "\u3000cam-a"],
    ids=["leading-space", "trailing-space", "trailing-newline", "leading-tab", "ideographic"],
)
def test_padded_camera_ids_are_refused(site: str, value: str) -> None:
    _refuse(site, value, "whitespace")


@pytest.mark.parametrize("site", _TRACK_SITES)
@pytest.mark.parametrize(
    "value",
    [" 1", "1 ", "1\n", "\t1"],
    ids=["leading-space", "trailing-space", "trailing-newline", "leading-tab"],
)
def test_padded_track_ids_are_refused(site: str, value: str) -> None:
    _refuse(site, value, "whitespace")


@pytest.mark.parametrize("site", _CAMERA_SITES)
def test_nfkc_camera_ids_are_refused(site: str) -> None:
    _refuse(site, "\uff43am-a", "NFKC")


@pytest.mark.parametrize("site", _TRACK_SITES)
def test_nfkc_track_ids_are_refused(site: str) -> None:
    _refuse(site, "\uff11", "NFKC")


@pytest.mark.parametrize("site", _CAMERA_SITES)
@pytest.mark.parametrize("mark", _ZERO_WIDTH, ids=["zwsp", "zwnj", "zwj", "word-joiner", "bom"])
def test_zero_width_camera_ids_are_refused(site: str, mark: str) -> None:
    _refuse(site, f"cam-a{mark}", "format character")


@pytest.mark.parametrize("site", _TRACK_SITES)
@pytest.mark.parametrize("mark", _ZERO_WIDTH, ids=["zwsp", "zwnj", "zwj", "word-joiner", "bom"])
def test_zero_width_track_ids_are_refused(site: str, mark: str) -> None:
    _refuse(site, f"1{mark}", "format character")


@pytest.mark.parametrize(
    "value",
    ["", "   ", "\t", "\n", "\u3000", None, 4],
    ids=["empty", "spaces", "tab", "newline", "ideographic-space", "none", "int"],
)
def test_empty_camera_ids_are_refused(value: object) -> None:
    _refuse("seed", value, "non-empty")
    _refuse("camera_ids", value, "non-empty")


@pytest.mark.parametrize(
    "value",
    ["", "   ", "1", True, False, 1.5, None],
    ids=["empty", "spaces", "digit-text", "true", "false", "float", "none"],
)
def test_bad_track_ids_are_refused(value: object) -> None:
    match = "non-empty" if value in ("", "   ") else "int track id"
    _refuse("seed_track", value, match)
    if value == "1":
        _refuse("from_track", value, "int track id")


def test_route_segment_refuses_a_padded_camera_id() -> None:
    with pytest.raises(ValueError, match="whitespace") as caught:
        RouteSegment(" cam-a", 1, 0.0, 1.0)
    assert repr(" cam-a") in str(caught.value)


def test_route_segments_must_be_a_tuple_of_segments() -> None:
    segment = _segment("cam-a", 1, 0.0, 1.0)
    assert Route("cam-a", 1, ()).segments == ()
    with pytest.raises(ValueError, match="tuple of RouteSegment"):
        Route("cam-a", 1, "ab")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tuple of RouteSegment"):
        Route("cam-a", 1, [segment])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tuple of RouteSegment"):
        Route("cam-a", 1, (object(),))  # type: ignore[arg-type]


def test_bytes_camera_ids_are_refused() -> None:
    seed = _tracklet("cam-a", 1, 0.0, 1.0)
    with pytest.raises(ValueError, match="not a str"):
        build_route(seed, (seed,), (), camera_ids=b"ab")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="or None"):
        build_route(seed, (seed,), (), t_window=b"ab")  # type: ignore[arg-type]
