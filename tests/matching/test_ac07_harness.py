"""AC-07 harness. Hand-built records only. No score is produced.

Smoke ids raise in every mode. ``code_test`` may carry diagnostic numbers
and still cannot return a pass or an AC-07 score. ``eval`` and ``unit_test``
are fail-closed on the raw HOLD id.
"""

from __future__ import annotations

import json
import socket
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest

from trace_search.fixtures.guards import FixtureSeparationError
from trace_search.matching.ac07_harness import run
from trace_search.matching.route_metrics import (
    TruthSegment,
    false_merge_rate,
    segment_precision,
    segment_recall,
)

_DIAGNOSTICS_NOTE = (
    "Not an AC-07 result. AC-07 is DEFERRED in Cut 01; these numbers are not evidence."
)
_THRESHOLDS = {
    "segment_recall": 0.80,
    "segment_precision": 0.90,
    "false_merge_rate": 0.05,
}
_TOP_LEVEL_KEYS = (
    "criterion",
    "fixture_id",
    "status",
    "evidence",
    "ac07_score",
    "thresholds",
    "diagnostics",
)
_DIAGNOSTIC_KEYS = (
    "note",
    "segment_recall",
    "segment_precision",
    "false_merge_rate",
)
_MODES = ("eval", "code_test", "unit_test")
_SMOKE_IDS = (
    "fx-smoke-syn-001",
    " FX_SMOKE_001 ",
    "\uff26\uff38-SMOKE-001",
    "FX-SMOKE-001",
)
_HOLD_MUTANTS = (
    "FX-HOLD-0012",
    "FX-HOLD-001-X",
    "X-FX-HOLD-001",
    "FX-HOLD-001\n",
    "fx-hold-001",
)
_ABS = 1e-12


@dataclass(frozen=True)
class _HandSegment:
    camera_id: str
    track_id: int
    t_entry: float
    t_exit: float


@dataclass(frozen=True)
class _HandRoute:
    seed_camera: str
    seed_track: int
    segments: tuple[_HandSegment, ...]


def _exact(actual: object, numerator: int, denominator: int) -> None:
    expected = Fraction(numerator, denominator)
    assert isinstance(actual, float)
    assert abs(actual - float(expected)) < _ABS


def _perfect() -> tuple[
    tuple[_HandRoute, ...], tuple[TruthSegment, ...], dict[tuple[str, int], str]
]:
    """Two segments, one clean cross-camera link. Metrics would clear AC-07."""
    segments = (
        _HandSegment("cam-a", 1, 0.0, 10.0),
        _HandSegment("cam-b", 2, 20.0, 30.0),
    )
    route = _HandRoute("cam-a", 1, segments)
    truth = (
        TruthSegment("fig-a", "cam-a", 0.0, 10.0),
        TruthSegment("fig-a", "cam-b", 20.0, 30.0),
    )
    ownership = {("cam-a", 1): "fig-a", ("cam-b", 2): "fig-a"}
    return (route,), truth, ownership


def _recall_inputs() -> tuple[
    tuple[_HandRoute, ...], tuple[TruthSegment, ...], dict[tuple[str, int], str]
]:
    """Three of four truth segments, one clean cross-camera chain."""
    segments = (
        _HandSegment("cam-a", 1, 0.0, 10.0),
        _HandSegment("cam-b", 2, 0.0, 10.0),
        _HandSegment("cam-c", 3, 0.0, 10.0),
    )
    route = _HandRoute("cam-a", 1, segments)
    truth = tuple(
        TruthSegment("fig-a", camera_id, 0.0, 10.0)
        for camera_id in ("cam-a", "cam-b", "cam-c", "cam-d")
    )
    ownership = {("cam-a", 1): "fig-a", ("cam-b", 2): "fig-a", ("cam-c", 3): "fig-a"}
    return (route,), truth, ownership


def _accepted_calls() -> tuple[Callable[[], dict[str, object]], ...]:
    """Every mode and accepted id that returns a payload."""
    routes, truth, ownership = _perfect()
    recall_routes, recall_truth, recall_owner = _recall_inputs()
    return (
        lambda: run("FX-HOLD-001"),
        lambda: run("FX-HOLD-000", mode="eval"),
        lambda: run("FX-HOLD-999", mode="eval"),
        lambda: run(
            "FX-HOLD-001",
            routes=routes,
            truth=truth,
            ownership=ownership,
            mode="eval",
        ),
        lambda: run(
            "FX-HOLD-001",
            routes=routes,
            truth=truth,
            ownership=ownership,
            mode="unit_test",
        ),
        lambda: run("FX-CODE-101", mode="code_test"),
        lambda: run("FX-CODE-000", mode="code_test"),
        lambda: run(
            "FX-CODE-101",
            routes=routes,
            truth=truth,
            ownership=ownership,
            mode="code_test",
        ),
        lambda: run(
            "FX-CODE-101",
            routes=recall_routes,
            truth=recall_truth,
            ownership=recall_owner,
            mode="code_test",
        ),
    )


def _assert_deferred(result: Mapping[str, object]) -> None:
    encoded = json.dumps(result)
    assert result["status"] == "DEFERRED"
    assert result["status"] != "PASS"
    assert result["evidence"] is False
    assert result["ac07_score"] is None
    assert result["thresholds"] == _THRESHOLDS
    assert '"status": "PASS"' not in encoded
    assert '"ac07_score": null' in encoded
    assert result["criterion"] == "AC-07"
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["note"] == _DIAGNOSTICS_NOTE
    assert "Not an AC-07 result" in diagnostics["note"]
    assert "not evidence" in diagnostics["note"]


@pytest.mark.parametrize("mode", [*_MODES, "EVAL", "nope"])
@pytest.mark.parametrize("fixture_id", _SMOKE_IDS)
def test_smoke_ids_raise_in_every_mode(fixture_id: str, mode: str) -> None:
    with pytest.raises(FixtureSeparationError, match="FX-SMOKE") as caught:
        run(fixture_id, mode=mode)  # type: ignore[arg-type]
    message = str(caught.value)
    assert fixture_id in message
    assert "normalised id starts with FX-SMOKE" in message


def test_fullwidth_smoke_normalises_only_for_the_refusal() -> None:
    fixture_id = "\uff26\uff38-SMOKE-001"
    assert unicodedata.normalize("NFKC", fixture_id) == "FX-SMOKE-001"
    with pytest.raises(FixtureSeparationError, match="FX-SMOKE"):
        run(fixture_id, mode="code_test")


def test_fx_code_is_refused_in_eval() -> None:
    with pytest.raises(FixtureSeparationError, match="id starts with FX-CODE") as caught:
        run("FX-CODE-101")
    assert "FX-CODE-101" in str(caught.value)
    with pytest.raises(FixtureSeparationError, match="id starts with FX-CODE"):
        run("FX-CODE-101", mode="eval")


def test_fx_code_is_refused_in_unit_test() -> None:
    with pytest.raises(FixtureSeparationError, match="id starts with FX-CODE"):
        run("FX-CODE-101", mode="unit_test")


def test_fx_code_code_test_is_deferred_with_diagnostics() -> None:
    result = run("FX-CODE-101", mode="code_test")
    _assert_deferred(result)
    assert result["fixture_id"] == "FX-CODE-101"
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["segment_recall"] is None
    assert diagnostics["segment_precision"] is None
    assert diagnostics["false_merge_rate"] is None


def test_code_test_diagnostics_match_the_metric_functions() -> None:
    routes, truth, ownership = _recall_inputs()
    result = run(
        "FX-CODE-101",
        routes=routes,
        truth=truth,
        ownership=ownership,
        mode="code_test",
    )
    predicted = tuple(segment for route in routes for segment in route.segments)
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics, dict)
    _assert_deferred(result)
    _exact(diagnostics["segment_recall"], 3, 4)
    _exact(diagnostics["segment_precision"], 1, 1)
    _exact(diagnostics["false_merge_rate"], 0, 1)
    assert diagnostics["segment_recall"] == segment_recall(predicted, truth, ownership)
    assert diagnostics["segment_precision"] == segment_precision(predicted, truth, ownership)
    assert diagnostics["false_merge_rate"] == false_merge_rate(routes, ownership)


def test_perfect_code_test_numbers_stay_deferred() -> None:
    routes, truth, ownership = _perfect()
    result = run(
        "FX-CODE-101",
        routes=routes,
        truth=truth,
        ownership=ownership,
        mode="code_test",
    )
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics, dict)
    _exact(diagnostics["segment_recall"], 1, 1)
    _exact(diagnostics["segment_precision"], 1, 1)
    _exact(diagnostics["false_merge_rate"], 0, 1)
    _assert_deferred(result)


@pytest.mark.parametrize(
    "fixture_id",
    _HOLD_MUTANTS,
    ids=["four-digits", "suffix", "prefix", "trailing-newline", "lowercase"],
)
@pytest.mark.parametrize("mode", ["eval", "unit_test"])
def test_hold_anchor_mutants_raise_in_eval(fixture_id: str, mode: str) -> None:
    """A trailing newline must not match. ``search`` with ``$`` would accept it."""
    with pytest.raises(FixtureSeparationError, match="fullmatch") as caught:
        run(fixture_id, mode=mode)  # type: ignore[arg-type]
    assert fixture_id in str(caught.value)


@pytest.mark.parametrize(
    "fixture_id",
    [
        " FX-HOLD-001 ",
        "FX_HOLD_001",
        "\uff26\uff38-HOLD-\uff10\uff10\uff11",
        "FX-HOLD-\u0660\u0660\u0661",
    ],
    ids=["padded", "underscores", "fullwidth", "arabic-indic"],
)
def test_hold_acceptance_does_not_use_the_normalised_id(fixture_id: str) -> None:
    with pytest.raises(FixtureSeparationError, match="fullmatch"):
        run(fixture_id, mode="eval")
    with pytest.raises(FixtureSeparationError, match="fullmatch"):
        run(fixture_id, mode="unit_test")


@pytest.mark.parametrize(
    "fixture_id",
    ["fx-code-101", " FX-CODE-101 ", "FX_CODE_101", "\uff26\uff38-CODE-101"],
    ids=["lower", "padded", "underscores", "fullwidth"],
)
def test_code_test_does_not_normalise_ids_into_acceptance(fixture_id: str) -> None:
    with pytest.raises(FixtureSeparationError):
        run(fixture_id, mode="code_test")


def test_code_test_refuses_a_hold_id() -> None:
    with pytest.raises(FixtureSeparationError, match="starts with FX-CODE"):
        run("FX-HOLD-001", mode="code_test")


def test_unit_test_hold_is_deferred_and_does_not_compute(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args: object, **_kwargs: object) -> float:
        raise AssertionError("unit_test computed a metric")

    monkeypatch.setattr("trace_search.matching.ac07_harness.segment_recall", explode)
    monkeypatch.setattr("trace_search.matching.ac07_harness.segment_precision", explode)
    monkeypatch.setattr("trace_search.matching.ac07_harness.false_merge_rate", explode)
    routes, truth, ownership = _perfect()
    result = run(
        "FX-HOLD-014",
        routes=routes,
        truth=truth,
        ownership=ownership,
        mode="unit_test",
    )
    _assert_deferred(result)
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["segment_recall"] is None
    assert diagnostics["segment_precision"] is None
    assert diagnostics["false_merge_rate"] is None


def test_eval_does_not_compute_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args: object, **_kwargs: object) -> float:
        raise AssertionError("eval computed a metric")

    monkeypatch.setattr("trace_search.matching.ac07_harness.segment_recall", explode)
    monkeypatch.setattr("trace_search.matching.ac07_harness.segment_precision", explode)
    monkeypatch.setattr("trace_search.matching.ac07_harness.false_merge_rate", explode)
    routes, truth, ownership = _perfect()
    result = run("FX-HOLD-001", routes=routes, truth=truth, ownership=ownership, mode="eval")
    _assert_deferred(result)
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["segment_recall"] is None


def test_no_accepted_branch_returns_pass_or_score() -> None:
    """Sweep every mode and accepted id. None of them may pass or carry a score."""
    payloads = [call() for call in _accepted_calls()]
    assert len(payloads) >= 3
    for result in payloads:
        _assert_deferred(result)
        assert result["status"] != "PASS"
        assert result["ac07_score"] is None
    fixture_ids = [result["fixture_id"] for result in payloads]
    assert "FX-HOLD-001" in fixture_ids
    assert "FX-HOLD-000" in fixture_ids
    assert "FX-HOLD-999" in fixture_ids
    assert "FX-CODE-101" in fixture_ids
    assert "FX-CODE-000" in fixture_ids


def test_code_test_omitted_inputs_are_not_measured() -> None:
    truth = (TruthSegment("fig-a", "cam-a", 0.0, 1.0),)
    routes = (_HandRoute("cam-a", 1, (_HandSegment("cam-a", 1, 0.0, 1.0),)),)
    omitted: tuple[dict[str, object], ...] = (
        {},
        {"routes": routes},
        {"routes": routes, "truth": truth},
        {"truth": truth, "ownership": {("cam-a", 1): "fig-a"}},
    )
    for kwargs in omitted:
        result = run("FX-CODE-101", mode="code_test", **kwargs)  # type: ignore[arg-type]
        diagnostics = result["diagnostics"]
        assert isinstance(diagnostics, dict)
        assert diagnostics["segment_recall"] is None
        assert diagnostics["segment_precision"] is None
        assert diagnostics["false_merge_rate"] is None
        _assert_deferred(result)


def test_code_test_empty_recall_denominator_is_not_hidden() -> None:
    route = _HandRoute("cam-a", 1, (_HandSegment("cam-a", 1, 0.0, 1.0),))
    with pytest.raises(ValueError, match="segment_recall denominator is empty"):
        run(
            "FX-CODE-101",
            routes=(route,),
            truth=(),
            ownership={("cam-a", 1): "fig-a"},
            mode="code_test",
        )


def test_code_test_empty_precision_denominator_is_not_hidden() -> None:
    truth = (TruthSegment("fig-a", "cam-a", 0.0, 1.0),)
    with pytest.raises(ValueError, match="segment_precision denominator is empty"):
        run("FX-CODE-101", routes=(), truth=truth, ownership={}, mode="code_test")


def test_code_test_empty_false_merge_denominator_is_not_hidden() -> None:
    segment = _HandSegment("cam-a", 1, 0.0, 1.0)
    route = _HandRoute("cam-a", 1, (segment, segment))
    truth = (TruthSegment("fig-a", "cam-a", 0.0, 1.0),)
    with pytest.raises(ValueError, match="false_merge_rate denominator is empty"):
        run(
            "FX-CODE-101",
            routes=(route,),
            truth=truth,
            ownership={("cam-a", 1): "fig-a"},
            mode="code_test",
        )


def test_payload_keys_and_a_fresh_thresholds_dict() -> None:
    result = run("FX-HOLD-001")
    assert tuple(result) == _TOP_LEVEL_KEYS
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert tuple(diagnostics) == _DIAGNOSTIC_KEYS
    result["thresholds"]["segment_recall"] = 0.0
    again = run("FX-HOLD-001")
    assert again["thresholds"]["segment_recall"] == 0.80


def test_fixture_id_must_be_a_non_empty_string() -> None:
    with pytest.raises(ValueError, match="fixture_id"):
        run("")
    with pytest.raises(ValueError, match="fixture_id"):
        run(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["EVAL", "code-test", "unit-test", ""])
def test_mode_rejects_lookalike_values(mode: str) -> None:
    with pytest.raises(ValueError, match="unit_test"):
        run("FX-HOLD-001", mode=mode)  # type: ignore[arg-type]


def test_harness_does_not_open_files_or_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def refuse_open(*_args: object, **_kwargs: object) -> object:
        calls.append("open")
        raise AssertionError("harness opened a file")

    def refuse_connect(*_args: object, **_kwargs: object) -> None:
        calls.append("connect")
        raise OSError("refused")

    monkeypatch.setattr("builtins.open", refuse_open)
    monkeypatch.setattr(socket.socket, "connect", refuse_connect)
    routes, truth, ownership = _perfect()
    assert run("FX-HOLD-001")["ac07_score"] is None
    measured = run(
        "FX-CODE-101",
        routes=routes,
        truth=truth,
        ownership=ownership,
        mode="code_test",
    )
    assert measured["status"] == "DEFERRED"
    assert calls == []


def test_sources_are_hand_built_and_do_not_name_forbidden_loaders() -> None:
    root = Path(__file__).resolve().parents[2] / "src/trace_search/matching"
    text = (root / "route_metrics.py").read_text(encoding="utf-8")
    text += (root / "ac07_harness.py").read_text(encoding="utf-8")
    for banned in (
        "load_reid_samples",
        "LoaderContext",
        "motmetrics",
        "trackeval",
        "smoke-clips",
        ".mp4",
        "urllib",
        "http.client",
        "socket",
    ):
        assert banned not in text


def test_accepted_ids_cover_every_mode() -> None:
    """The sweep above is the pass-ban. This names the three modes directly."""
    routes, truth, ownership = _perfect()
    payloads: Sequence[Mapping[str, object]] = (
        run("FX-HOLD-007", mode="eval"),
        run("FX-HOLD-007", routes=routes, truth=truth, ownership=ownership, mode="unit_test"),
        run("FX-CODE-007", routes=routes, truth=truth, ownership=ownership, mode="code_test"),
    )
    for result in payloads:
        _assert_deferred(result)
        assert result["ac07_score"] is None
        assert result["status"] != "PASS"
