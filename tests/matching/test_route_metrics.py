"""Route metrics on hand-built records.

No footage, no T5 route objects, and no files written into the repo tree.
``Route`` and ``RouteSegment`` are not defined here. The stand-ins are frozen
dataclasses with T-E-6's field names.
"""

from __future__ import annotations

import inspect
import math
import socket
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, dataclass, fields
from fractions import Fraction
from pathlib import Path

import pytest

from trace_search.matching.route_metrics import (
    RouteLike,
    RouteSegmentLike,
    TruthSegment,
    false_merge_rate,
    matched_pairs,
    segment_precision,
    segment_recall,
)

_ABS = 1e-12


@dataclass(frozen=True)
class _HandSegment:
    """Stand-in for a T-E-6 segment. Not a RouteSegment."""

    camera_id: str
    track_id: int
    t_entry: float
    t_exit: float


@dataclass(frozen=True)
class _HandRoute:
    """Stand-in for a T-E-6 route. Not a Route."""

    seed_camera: str
    seed_track: int
    segments: tuple[_HandSegment, ...]


def _exact(actual: object, numerator: int, denominator: int) -> None:
    expected = Fraction(numerator, denominator)
    assert isinstance(actual, float)
    assert abs(actual - float(expected)) < _ABS
    assert Fraction(actual).limit_denominator(denominator) == expected


def _chain(steps: tuple[tuple[str, int, str], ...]) -> _HandRoute:
    segments = tuple(
        _HandSegment(camera_id, track_id, float(index), float(index) + 1.0)
        for index, (camera_id, track_id, _figure) in enumerate(steps)
    )
    camera_id, track_id, _figure = steps[0]
    return _HandRoute(camera_id, track_id, segments)


def _owns(*groups: tuple[tuple[str, int, str], ...]) -> dict[tuple[str, int], str]:
    owned: dict[tuple[str, int], str] = {}
    for steps in groups:
        for camera_id, track_id, figure in steps:
            owned[(camera_id, track_id)] = figure
    return owned


def _recall_case() -> tuple[
    tuple[_HandSegment, ...], tuple[TruthSegment, ...], dict[tuple[str, int], str]
]:
    """Three predictions cover three of four truth segments."""
    truth = tuple(
        TruthSegment("fig-a", camera_id, 0.0, 10.0)
        for camera_id in ("cam-a", "cam-b", "cam-c", "cam-d")
    )
    predicted = (
        _HandSegment("cam-a", 1, 0.0, 10.0),
        _HandSegment("cam-b", 2, 0.0, 10.0),
        _HandSegment("cam-c", 3, 0.0, 10.0),
    )
    ownership = {("cam-a", 1): "fig-a", ("cam-b", 2): "fig-a", ("cam-c", 3): "fig-a"}
    return predicted, truth, ownership


def _precision_case() -> tuple[
    tuple[_HandSegment, ...], tuple[TruthSegment, ...], dict[tuple[str, int], str]
]:
    """Three of four predictions match. The fourth is outside the tolerance."""
    truth = tuple(
        TruthSegment("fig-a", camera_id, 0.0, 10.0) for camera_id in ("cam-a", "cam-b", "cam-c")
    )
    predicted = (
        _HandSegment("cam-a", 1, 0.0, 10.0),
        _HandSegment("cam-b", 2, 0.0, 10.0),
        _HandSegment("cam-c", 3, 0.0, 10.0),
        _HandSegment("cam-a", 4, 100.0, 110.0),
    )
    ownership = {
        ("cam-a", 1): "fig-a",
        ("cam-b", 2): "fig-a",
        ("cam-c", 3): "fig-a",
        ("cam-a", 4): "fig-a",
    }
    return predicted, truth, ownership


def test_truth_segment_fields_and_frozen() -> None:
    segment = TruthSegment("fig-a", "cam-a", 0.0, 1.0)
    assert [field.name for field in fields(TruthSegment)] == [
        "figure",
        "camera_id",
        "t_entry",
        "t_exit",
    ]
    with pytest.raises(FrozenInstanceError):
        segment.figure = "fig-b"  # type: ignore[misc]


def test_protocols_use_t_e_6_field_names_and_are_read_only() -> None:
    metrics = inspect.getmodule(TruthSegment)
    assert metrics is not None
    assert not hasattr(metrics, "Route")
    assert not hasattr(metrics, "RouteSegment")
    assert "trace_search.matching.route" not in metrics.__dict__
    for cls, names in (
        (RouteSegmentLike, ("camera_id", "track_id", "t_entry", "t_exit")),
        (RouteLike, ("seed_camera", "seed_track", "segments")),
    ):
        for name in names:
            prop = inspect.getattr_static(cls, name)
            assert isinstance(prop, property)
            assert prop.fget is not None
            with pytest.raises(NotImplementedError):
                prop.fget(object())


def test_three_of_four_truth_segments_give_recall_0_75() -> None:
    predicted, truth, ownership = _recall_case()
    assert matched_pairs(predicted, truth, ownership) == ((0, 0), (1, 1), (2, 2))
    _exact(segment_recall(predicted, truth, ownership), 3, 4)


def test_one_wrong_segment_of_four_gives_precision_0_75() -> None:
    predicted, truth, ownership = _precision_case()
    assert matched_pairs(predicted, truth, ownership) == ((0, 0), (1, 1), (2, 2))
    _exact(segment_precision(predicted, truth, ownership), 3, 4)


def test_false_merge_rate_is_one_bad_cross_camera_link_of_four() -> None:
    """1/4 = 0.25. Four same-camera links are outside the denominator.

    Counting those four would make the rate 1/8.
    """
    good_a = (("cam-a", 1, "fig-a"), ("cam-b", 2, "fig-a"))
    good_b = (("cam-c", 3, "fig-a"), ("cam-d", 4, "fig-a"))
    good_c = (("cam-e", 5, "fig-a"), ("cam-f", 6, "fig-a"))
    bad = (("cam-g", 7, "fig-a"), ("cam-h", 8, "fig-b"))
    same = (
        ("cam-i", 9, "fig-c"),
        ("cam-i", 10, "fig-c"),
        ("cam-i", 11, "fig-c"),
        ("cam-i", 12, "fig-c"),
        ("cam-i", 13, "fig-c"),
    )
    routes = (_chain(good_a), _chain(good_b), _chain(good_c), _chain(bad), _chain(same))
    ownership = _owns(good_a, good_b, good_c, bad, same)
    _exact(false_merge_rate(routes, ownership), 1, 4)


def test_same_camera_link_between_different_figures_is_not_a_false_merge() -> None:
    """The cross-camera link is clean, so the rate is a real 0.0, not an empty rate."""
    cross = (("cam-a", 1, "fig-a"), ("cam-b", 2, "fig-a"))
    same = (("cam-c", 3, "fig-a"), ("cam-c", 4, "fig-b"))
    routes = (_chain(cross), _chain(same))
    _exact(false_merge_rate(routes, _owns(cross, same)), 0, 1)


def test_all_bad_cross_camera_links_are_a_real_one() -> None:
    bad = (("cam-a", 1, "fig-a"), ("cam-b", 2, "fig-b"))
    _exact(false_merge_rate((_chain(bad),), _owns(bad)), 1, 1)


def test_false_merge_counts_only_adjacent_pairs() -> None:
    """The ends share a figure. Only the adjacent cross-camera link is bad."""
    steps = (
        ("cam-a", 1, "fig-a"),
        ("cam-a", 2, "fig-b"),
        ("cam-b", 3, "fig-a"),
    )
    _exact(false_merge_rate((_chain(steps),), _owns(steps)), 1, 1)


def test_ten_second_gap_matches_on_both_sides() -> None:
    truth = (TruthSegment("fig-a", "cam-a", 0.0, 10.0),)
    ownership = {("cam-a", 1): "fig-a"}
    early = (_HandSegment("cam-a", 1, -20.0, -10.0),)
    late = (_HandSegment("cam-a", 1, 20.0, 30.0),)
    _exact(segment_recall(early, truth, ownership), 1, 1)
    _exact(segment_recall(late, truth, ownership), 1, 1)
    _exact(segment_precision(early, truth, ownership), 1, 1)
    _exact(segment_precision(late, truth, ownership), 1, 1)


def test_ten_point_one_second_gap_does_not_match_on_both_sides() -> None:
    truth = (TruthSegment("fig-a", "cam-a", 0.0, 10.0),)
    ownership = {("cam-a", 1): "fig-a"}
    early = (_HandSegment("cam-a", 1, -20.1, -10.1),)
    late = (_HandSegment("cam-a", 1, 20.1, 30.0),)
    _exact(segment_recall(early, truth, ownership), 0, 1)
    _exact(segment_recall(late, truth, ownership), 0, 1)
    _exact(segment_precision(early, truth, ownership), 0, 1)
    _exact(segment_precision(late, truth, ownership), 0, 1)


def test_overlap_and_touching_intervals_match() -> None:
    truth = (TruthSegment("fig-a", "cam-a", 0.0, 10.0),)
    ownership = {("cam-a", 1): "fig-a"}
    overlap = (_HandSegment("cam-a", 1, 5.0, 15.0),)
    touch = (_HandSegment("cam-a", 1, 10.0, 12.0),)
    contained = (_HandSegment("cam-a", 1, -5.0, 20.0),)
    _exact(segment_recall(overlap, truth, ownership), 1, 1)
    _exact(segment_recall(touch, truth, ownership), 1, 1)
    _exact(segment_recall(contained, truth, ownership), 1, 1)


def test_different_camera_does_not_match() -> None:
    truth = (TruthSegment("fig-a", "cam-a", 0.0, 10.0),)
    predicted = (_HandSegment("cam-b", 1, 0.0, 10.0),)
    ownership = {("cam-b", 1): "fig-a"}
    assert matched_pairs(predicted, truth, ownership) == ()
    _exact(segment_recall(predicted, truth, ownership), 0, 1)


def test_different_figure_does_not_match() -> None:
    truth = (TruthSegment("fig-a", "cam-a", 0.0, 10.0),)
    predicted = (_HandSegment("cam-a", 1, 0.0, 10.0),)
    ownership = {("cam-a", 1): "Fig-A"}
    assert matched_pairs(predicted, truth, ownership) == ()
    _exact(segment_precision(predicted, truth, ownership), 0, 1)


def test_matching_is_one_to_one() -> None:
    """Many-to-one would report precision 1 and recall 1 here. Both stay at 1/2."""
    one_truth = (TruthSegment("fig-a", "cam-a", 0.0, 10.0),)
    two_preds = (
        _HandSegment("cam-a", 1, 0.0, 10.0),
        _HandSegment("cam-a", 2, 0.0, 10.0),
    )
    two_owner = {("cam-a", 1): "fig-a", ("cam-a", 2): "fig-a"}
    assert matched_pairs(two_preds, one_truth, two_owner) == ((0, 0),)
    _exact(segment_precision(two_preds, one_truth, two_owner), 1, 2)
    _exact(segment_recall(two_preds, one_truth, two_owner), 1, 1)

    two_truths = (
        TruthSegment("fig-a", "cam-a", 0.0, 10.0),
        TruthSegment("fig-a", "cam-a", 0.0, 10.0),
    )
    one_pred = (_HandSegment("cam-a", 1, 0.0, 10.0),)
    one_owner = {("cam-a", 1): "fig-a"}
    assert matched_pairs(one_pred, two_truths, one_owner) == ((0, 0),)
    _exact(segment_recall(one_pred, two_truths, one_owner), 1, 2)
    _exact(segment_precision(one_pred, two_truths, one_owner), 1, 1)


def test_tighter_gap_is_assigned_before_a_looser_one() -> None:
    truth = (TruthSegment("fig-a", "cam-a", 0.0, 10.0),)
    predicted = (
        _HandSegment("cam-a", 1, 20.0, 30.0),
        _HandSegment("cam-a", 2, 0.0, 10.0),
    )
    ownership = {("cam-a", 1): "fig-a", ("cam-a", 2): "fig-a"}
    assert matched_pairs(predicted, truth, ownership) == ((1, 0),)


def test_zero_matches_are_a_real_zero() -> None:
    truth = (TruthSegment("fig-a", "cam-a", 0.0, 1.0),)
    predicted = (_HandSegment("cam-b", 1, 0.0, 1.0),)
    ownership = {("cam-b", 1): "fig-a"}
    _exact(segment_recall(predicted, truth, ownership), 0, 1)
    _exact(segment_precision(predicted, truth, ownership), 0, 1)


def test_empty_truth_does_not_invent_recall() -> None:
    predicted = (_HandSegment("cam-a", 1, 0.0, 1.0),)
    with pytest.raises(ValueError, match="segment_recall denominator is empty"):
        segment_recall(predicted, (), {("cam-a", 1): "fig-a"})
    with pytest.raises(ValueError, match="segment_recall denominator is empty"):
        segment_recall((), (), {})


def test_empty_predicted_does_not_invent_precision() -> None:
    truth = (TruthSegment("fig-a", "cam-a", 0.0, 1.0),)
    with pytest.raises(ValueError, match="segment_precision denominator is empty"):
        segment_precision((), truth, {})
    with pytest.raises(ValueError, match="segment_precision denominator is empty"):
        segment_precision((), (), {})


def test_no_cross_camera_link_does_not_invent_false_merge() -> None:
    same = (("cam-a", 1, "fig-a"), ("cam-a", 2, "fig-b"))
    with pytest.raises(ValueError, match="false_merge_rate denominator is empty"):
        false_merge_rate((_chain(same),), _owns(same))
    with pytest.raises(ValueError, match="false_merge_rate denominator is empty"):
        false_merge_rate((), {})
    single = (("cam-a", 1, "fig-a"),)
    with pytest.raises(ValueError, match="false_merge_rate denominator is empty"):
        false_merge_rate((_chain(single),), _owns(single))


def test_negative_and_zero_length_intervals_are_allowed() -> None:
    segment = TruthSegment("fig-a", "cam-a", -2.0, -1.0)
    assert segment.t_entry == -2.0
    point = TruthSegment("fig-a", "cam-a", 5.0, 5.0)
    assert point.t_exit == 5.0


def test_whole_number_times_are_stored_as_float() -> None:
    segment = TruthSegment("fig-a", "cam-a", 1, 2)  # type: ignore[arg-type]
    assert segment.t_entry == 1.0
    assert segment.t_exit == 2.0
    assert isinstance(segment.t_entry, float)
    assert isinstance(segment.t_exit, float)


@pytest.mark.parametrize(
    ("figure", "camera_id"),
    [
        ("", "cam-a"),
        ("fig-a", ""),
        ("   ", "cam-a"),
        ("fig-a", "\t"),
        (None, "cam-a"),
        ("fig-a", 4),
    ],
    ids=[
        "figure-empty",
        "camera-empty",
        "figure-spaces",
        "camera-tab",
        "figure-none",
        "camera-int",
    ],
)
def test_truth_segment_rejects_empty_ids(figure: object, camera_id: object) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TruthSegment(figure, camera_id, 0.0, 1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [" fig-a", "fig-a ", "fig-a\n", "\tfig-a"],
    ids=["leading-space", "trailing-space", "trailing-newline", "leading-tab"],
)
def test_ids_with_surrounding_whitespace_are_refused(value: str) -> None:
    with pytest.raises(ValueError, match="whitespace"):
        TruthSegment(value, "cam-a", 0.0, 1.0)
    with pytest.raises(ValueError, match="whitespace"):
        TruthSegment("fig-a", value, 0.0, 1.0)
    predicted = (_HandSegment(value, 1, 0.0, 1.0),)
    with pytest.raises(ValueError, match="whitespace"):
        segment_recall(predicted, (), {})
    route = _HandRoute(value, 1, ())
    with pytest.raises(ValueError, match="whitespace"):
        false_merge_rate((route,), {("cam-a", 1): "fig-a"})


def test_nfkc_mismatch_is_refused_not_folded() -> None:
    camera_id = "\uff43\uff41\uff4d-a"
    figure = "\uff46\uff49\uff47-a"
    with pytest.raises(ValueError, match="NFKC"):
        TruthSegment("fig-a", camera_id, 0.0, 1.0)
    with pytest.raises(ValueError, match="NFKC"):
        TruthSegment(figure, "cam-a", 0.0, 1.0)
    predicted = (_HandSegment(camera_id, 1, 0.0, 1.0),)
    with pytest.raises(ValueError, match="NFKC"):
        segment_precision(predicted, (), {})
    with pytest.raises(ValueError, match="NFKC"):
        segment_recall((), (), {("cam-a", 1): figure})


@pytest.mark.parametrize(
    "mark",
    ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"],
    ids=["zwsp", "zwnj", "zwj", "word-joiner", "bom"],
)
def test_format_characters_are_refused(mark: str) -> None:
    with pytest.raises(ValueError, match="format character"):
        TruthSegment(f"fig-a{mark}", "cam-a", 0.0, 1.0)
    with pytest.raises(ValueError, match="format character"):
        TruthSegment("fig-a", f"cam-a{mark}", 0.0, 1.0)
    predicted = (_HandSegment(f"cam-a{mark}", 1, 0.0, 1.0),)
    with pytest.raises(ValueError, match="format character"):
        segment_recall(predicted, (), {})


@pytest.mark.parametrize("track_id", [True, False], ids=["true", "false"])
def test_bool_track_id_is_refused(track_id: bool) -> None:
    predicted = (_HandSegment("cam-a", track_id, 0.0, 1.0),)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not a bool"):
        segment_recall(predicted, (), {})
    route = _HandRoute("cam-a", track_id, ())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not a bool"):
        false_merge_rate((route,), {})
    with pytest.raises(ValueError, match="not a bool"):
        segment_precision((), (), {("cam-a", track_id): "fig-a"})  # type: ignore[dict-item]


@pytest.mark.parametrize("track_id", [1.5, "1", None], ids=["float", "str", "none"])
def test_non_int_track_id_is_refused(track_id: object) -> None:
    predicted = (_HandSegment("cam-a", track_id, 0.0, 1.0),)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="int track id"):
        segment_recall(predicted, (), {})


@pytest.mark.parametrize(
    "value",
    [math.nan, math.inf, -math.inf],
    ids=["nan", "inf", "neg-inf"],
)
def test_non_finite_times_are_refused(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        TruthSegment("fig-a", "cam-a", value, 1.0)
    with pytest.raises(ValueError, match="must be finite"):
        TruthSegment("fig-a", "cam-a", 0.0, value)
    predicted = (_HandSegment("cam-a", 1, value, 1.0),)
    with pytest.raises(ValueError, match="must be finite"):
        segment_recall(predicted, (), {("cam-a", 1): "fig-a"})


@pytest.mark.parametrize("value", [True, False, "0", None], ids=["true", "false", "str", "none"])
def test_non_numeric_times_are_refused(value: object) -> None:
    with pytest.raises(ValueError, match="finite number"):
        TruthSegment("fig-a", "cam-a", value, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite number"):
        TruthSegment("fig-a", "cam-a", 0.0, value)  # type: ignore[arg-type]


def test_exit_before_entry_is_refused() -> None:
    with pytest.raises(ValueError, match="t_exit"):
        TruthSegment("fig-a", "cam-a", 2.0, 1.0)
    predicted = (_HandSegment("cam-a", 1, 2.0, 1.0),)
    with pytest.raises(ValueError, match="t_exit"):
        segment_precision(predicted, (), {("cam-a", 1): "fig-a"})


def test_missing_figure_is_refused() -> None:
    predicted = (_HandSegment("cam-a", 1, 0.0, 1.0),)
    truth = (TruthSegment("fig-a", "cam-a", 0.0, 1.0),)
    with pytest.raises(ValueError, match="no figure"):
        segment_recall(predicted, truth, {})


def test_seed_track_must_have_a_figure() -> None:
    steps = (("cam-a", 1, "fig-a"), ("cam-b", 2, "fig-a"))
    route = _HandRoute("cam-z", 9, _chain(steps).segments)
    with pytest.raises(ValueError, match="no figure"):
        false_merge_rate((route,), _owns(steps))


def test_truth_items_must_be_truth_segments() -> None:
    predicted = (_HandSegment("cam-a", 1, 0.0, 1.0),)
    with pytest.raises(ValueError, match="TruthSegment"):
        segment_recall(predicted, (predicted[0],), {("cam-a", 1): "fig-a"})  # type: ignore[arg-type]


def test_sequences_reject_str_bytes_and_non_sequences() -> None:
    with pytest.raises(ValueError, match="sequence"):
        segment_recall("cam-a", (), {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sequence"):
        segment_recall(b"ab", (), {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sequence"):
        false_merge_rate(4, {})  # type: ignore[arg-type]
    route = _HandRoute("cam-a", 1, "nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sequence"):
        false_merge_rate((route,), {("cam-a", 1): "fig-a"})


def test_ownership_must_be_a_mapping_of_pairs() -> None:
    with pytest.raises(ValueError, match="mapping"):
        segment_recall((), (), [("cam-a", 1)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ownership key"):
        segment_recall((), (), {"cam-a": "fig-a"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="ownership key"):
        segment_precision((), (), {("cam-a",): "fig-a"})  # type: ignore[dict-item]


def test_missing_record_field_is_refused() -> None:
    class _Bare:
        camera_id = "cam-a"

    with pytest.raises(ValueError, match="missing"):
        segment_recall((_Bare(),), (), {})  # type: ignore[arg-type]


def test_hand_built_segment_satisfies_the_route_protocol() -> None:
    segment: RouteSegmentLike = _HandSegment("cam-a", 1, 0.0, 1.0)
    route: RouteLike = _HandRoute("cam-a", 1, (segment,))  # type: ignore[arg-type]
    assert route.seed_camera == "cam-a"
    assert route.segments[0].track_id == 1


def test_metrics_do_not_open_files_or_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def refuse_open(*_args: object, **_kwargs: object) -> object:
        calls.append("open")
        raise AssertionError("metrics opened a file")

    def refuse_connect(*_args: object, **_kwargs: object) -> None:
        calls.append("connect")
        raise OSError("refused")

    monkeypatch.setattr("builtins.open", refuse_open)
    monkeypatch.setattr(socket.socket, "connect", refuse_connect)
    predicted, truth, ownership = _recall_case()
    _exact(segment_recall(predicted, truth, ownership), 3, 4)
    assert calls == []


def test_metric_sources_do_not_name_loader_contexts_or_external_metrics() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "src/trace_search/matching/route_metrics.py").read_text(encoding="utf-8")
    for banned in ("load_reid_samples", "LoaderContext", "motmetrics", "trackeval", "smoke-clips"):
        assert banned not in text


def test_routes_argument_is_a_sequence_of_routes() -> None:
    steps = (("cam-a", 1, "fig-a"), ("cam-b", 2, "fig-a"))
    routes: Sequence[RouteLike] = (_chain(steps),)
    _exact(false_merge_rate(routes, _owns(steps)), 0, 1)
