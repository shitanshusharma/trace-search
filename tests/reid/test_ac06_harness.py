"""AC-06 harness behaviour.

Numbers come from caller-supplied arrays. The harness does not read a
file. ``status`` stays ``DEFERRED`` and ``ac06_score`` stays null.
These diagnostics are not an AC-06 result.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Callable
from fractions import Fraction

import numpy as np
import pytest

from trace_search.fixtures.guards import FixtureSeparationError
from trace_search.reid.ac06_harness import run
from trace_search.reid.eval import EvalItem, rank1_map

_DIAGNOSTICS_NOTE = (
    "Not an AC-06 result. AC-06 is DEFERRED in Cut 01; these numbers are not evidence."
)
_TOPOLOGY = "not applied (T6 supplies gallery_mask)"
_TOP_LEVEL_KEYS = (
    "criterion",
    "fixture_id",
    "status",
    "evidence",
    "ac06_score",
    "thresholds",
    "placeholder_embedder",
    "diagnostics",
)
_DIAGNOSTIC_KEYS = (
    "note",
    "rank1",
    "mAP",
    "num_query",
    "num_valid_query",
    "num_gallery",
    "topology_time_limits",
)
_NULL_DIAGNOSTIC_FIELDS = (
    "rank1",
    "mAP",
    "num_query",
    "num_valid_query",
    "num_gallery",
)
# Fullwidth FX and digits. NFKC folds these; Arabic-Indic digits stay.
_FULLWIDTH_HOLD = "\uff26\uff38-HOLD-\uff10\uff10\uff11"
_FULLWIDTH_SMOKE = "\uff26\uff38-SMOKE-SYN-001"
_ARABIC_INDIC_HOLD = "FX-HOLD-\u0660\u0660\u0661"


def _axes() -> tuple[tuple[float, ...], ...]:
    """Four orthogonal directions, built in the test. Same case as T-R-12."""
    eye = np.eye(4, dtype=np.float64)
    return tuple(tuple(float(value) for value in eye[index]) for index in range(4))


def _hand_built() -> tuple[list[EvalItem], list[EvalItem]]:
    """3 queries and 5 gallery items from ``tests/reid/test_eval.py``.

    Rank-1 is 2/3. mAP is (1/1 + 1/2 + 1/1) / 3 = 5/6.
    """
    e1, e2, e3, e4 = _axes()
    query = [
        EvalItem("q-0", 10, 1, e1),
        EvalItem("q-1", 11, 2, e2),
        EvalItem("q-2", 12, 3, e3),
    ]
    gallery = [
        EvalItem("g-0", 1, 1, e1),
        EvalItem("g-1", 2, 1, e2),
        EvalItem("g-2", 3, 2, e2),
        EvalItem("g-3", 4, 3, e3),
        EvalItem("g-4", 5, 4, e4),
    ]
    return query, gallery


def _perfect() -> tuple[list[EvalItem], list[EvalItem]]:
    vector = (1.0, 0.0)
    return [EvalItem("q-0", 1, 1, vector)], [EvalItem("g-0", 2, 1, vector)]


def _exact(actual: object, expected: Fraction) -> None:
    assert isinstance(actual, float)
    assert abs(actual - float(expected)) < 1e-12
    assert Fraction(actual).limit_denominator(expected.denominator) == expected


def _accept_pair(_query: EvalItem, _gallery: EvalItem) -> bool:
    """Keep every gallery item. The arguments are the mask signature."""
    return True


def _success_payloads() -> list[dict[str, object]]:
    """Every branch that returns a payload. None of them may pass."""
    query, gallery = _hand_built()
    perfect_query, perfect_gallery = _perfect()
    calls: tuple[Callable[[], dict[str, object]], ...] = (
        lambda: run("FX-HOLD-001"),
        lambda: run("FX-HOLD-001", query=query, gallery=gallery),
        lambda: run("FX-HOLD-001", query=query, gallery=gallery, mode="eval"),
        lambda: run("FX-HOLD-001", query=query, gallery=gallery, gallery_mask=_accept_pair),
        lambda: run("FX-HOLD-001", query=query, gallery=gallery, mode="unit_test"),
        lambda: run("FX-SMOKE-SYN-001", query=query, gallery=gallery, mode="unit_test"),
        lambda: run("FX-CODE-001", query=query, gallery=gallery, mode="unit_test"),
        lambda: run("fx-hold-001"),
        lambda: run(" FX-HOLD-001 "),
        lambda: run("FX_HOLD_001"),
        lambda: run(_FULLWIDTH_HOLD),
        lambda: run("anything", mode="unit_test"),
        lambda: run("FX-HOLD-001", query=perfect_query, gallery=perfect_gallery),
        lambda: run("FX-HOLD-001", query=[], gallery=[]),
        lambda: run("FX-HOLD-001", query=query, gallery=None),
        lambda: run("FX-HOLD-001", query=None, gallery=gallery),
    )
    return [call() for call in calls]


def _assert_null_diagnostics(result: dict[str, object]) -> None:
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics, dict)
    for field in _NULL_DIAGNOSTIC_FIELDS:
        assert diagnostics[field] is None
    assert diagnostics["note"] == _DIAGNOSTICS_NOTE
    assert diagnostics["topology_time_limits"] == _TOPOLOGY


@pytest.mark.parametrize(
    ("fixture_id", "phrase"),
    [
        ("FX-SMOKE-SYN-001", "starts with FX-SMOKE"),
        ("fx-smoke-syn-001", "starts with FX-SMOKE"),
        (" FX_SMOKE_SYN_001 ", "starts with FX-SMOKE"),
        (_FULLWIDTH_SMOKE, "starts with FX-SMOKE"),
        ("Fx-Code-001", "starts with FX-CODE"),
    ],
)
def test_blocked_prefixes_raise_with_original_text(fixture_id: str, phrase: str) -> None:
    with pytest.raises(FixtureSeparationError) as caught:
        run(fixture_id)
    message = str(caught.value)
    assert fixture_id in message
    assert phrase in message


@pytest.mark.parametrize(
    "fixture_id",
    [
        "FX-DEV-001",
        "X-FX-SMOKE-001",
        "anything",
        "FX-HOLD-1",
        "FX-HOLD-0012",
        "FX-HOLD-001-X",
        "X-FX-HOLD-001",
    ],
)
def test_non_hold_ids_are_refused(fixture_id: str) -> None:
    with pytest.raises(FixtureSeparationError) as caught:
        run(fixture_id)
    message = str(caught.value)
    assert fixture_id in message
    assert "does not match" in message
    assert "starts with FX-SMOKE" not in message
    assert "starts with FX-CODE" not in message


def test_arabic_indic_digits_are_refused() -> None:
    """NFKC leaves Arabic-Indic digits unchanged. ``\\d`` would accept them."""
    with pytest.raises(FixtureSeparationError) as caught:
        run(_ARABIC_INDIC_HOLD)
    assert _ARABIC_INDIC_HOLD in str(caught.value)


def test_nfkc_folds_fullwidth_hold_id() -> None:
    result = run(_FULLWIDTH_HOLD)
    assert result["status"] == "DEFERRED"
    assert result["fixture_id"] == _FULLWIDTH_HOLD
    _assert_null_diagnostics(result)


def test_strip_allows_padded_hold_id() -> None:
    result = run(" FX-HOLD-001 ")
    assert result["fixture_id"] == " FX-HOLD-001 "
    assert result["status"] == "DEFERRED"


def test_upper_allows_lower_hold_id() -> None:
    result = run("fx-hold-001")
    assert result["fixture_id"] == "fx-hold-001"
    assert result["status"] == "DEFERRED"


def test_underscore_allows_hold_id() -> None:
    result = run("FX_HOLD_001")
    assert result["fixture_id"] == "FX_HOLD_001"
    assert result["status"] == "DEFERRED"


@pytest.mark.parametrize("mode", ["UNIT_TEST", "unit-test"])
def test_mode_rejects_lookalike_values(mode: str) -> None:
    with pytest.raises(ValueError, match="unit_test"):
        run("FX-HOLD-001", mode=mode)  # type: ignore[arg-type]


def test_unit_test_nulls_diagnostics_even_with_arrays() -> None:
    query, gallery = _hand_built()
    result = run("FX-SMOKE-SYN-001", query=query, gallery=gallery, mode="unit_test")
    assert result["fixture_id"] == "FX-SMOKE-SYN-001"
    assert result["status"] == "DEFERRED"
    assert result["evidence"] is False
    assert result["ac06_score"] is None
    _assert_null_diagnostics(result)


def test_unit_test_nulls_hold_diagnostics_with_arrays() -> None:
    query, gallery = _hand_built()
    result = run("FX-HOLD-001", query=query, gallery=gallery, mode="unit_test")
    assert result["status"] == "DEFERRED"
    assert result["ac06_score"] is None
    _assert_null_diagnostics(result)


def test_hold_without_arrays_has_null_diagnostics() -> None:
    result = run("FX-HOLD-001")
    assert result["fixture_id"] == "FX-HOLD-001"
    assert result["status"] == "DEFERRED"
    assert result["evidence"] is False
    assert result["ac06_score"] is None
    _assert_null_diagnostics(result)


def test_partial_arrays_are_not_measured() -> None:
    query, gallery = _hand_built()
    _assert_null_diagnostics(run("FX-HOLD-001", query=query, gallery=None))
    _assert_null_diagnostics(run("FX-HOLD-001", query=None, gallery=gallery))


def test_hold_hand_built_diagnostics_match_rank1_and_map() -> None:
    query, gallery = _hand_built()
    result = run("FX-HOLD-001", query=query, gallery=gallery)
    expected = rank1_map(query, gallery)
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert result["status"] == "DEFERRED"
    assert result["evidence"] is False
    assert result["ac06_score"] is None
    assert result["placeholder_embedder"] is True
    assert result["thresholds"] == {"rank1": 0.90, "mAP": 0.70}
    assert diagnostics["note"] == _DIAGNOSTICS_NOTE
    assert diagnostics["topology_time_limits"] == _TOPOLOGY
    assert diagnostics["rank1"] == expected.rank1
    assert diagnostics["mAP"] == expected.mean_ap
    assert diagnostics["num_query"] == 3
    assert diagnostics["num_valid_query"] == 3
    assert diagnostics["num_gallery"] == 5
    _exact(diagnostics["rank1"], Fraction(2, 3))
    _exact(diagnostics["mAP"], Fraction(5, 6))
    assert expected.num_query == 3
    assert expected.num_gallery == 5


def test_perfect_scores_stay_deferred() -> None:
    query, gallery = _perfect()
    result = run("FX-HOLD-001", query=query, gallery=gallery)
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["rank1"] == 1.0
    assert diagnostics["mAP"] == 1.0
    assert result["status"] == "DEFERRED"
    assert result["ac06_score"] is None
    assert result["evidence"] is False


def test_empty_arrays_are_measured_as_zeros() -> None:
    result = run("FX-HOLD-001", query=[], gallery=[])
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["rank1"] == 0.0
    assert diagnostics["mAP"] == 0.0
    assert diagnostics["num_query"] == 0
    assert diagnostics["num_valid_query"] == 0
    assert diagnostics["num_gallery"] == 0
    assert result["ac06_score"] is None


def test_status_is_deferred_on_every_branch() -> None:
    for result in _success_payloads():
        encoded = json.dumps(result)
        assert result["status"] == "DEFERRED"
        assert '"status": "DEFERRED"' in encoded
        assert '"status": "PASS"' not in encoded


def test_ac06_score_is_null_on_every_branch() -> None:
    for result in _success_payloads():
        encoded = json.dumps(result)
        assert result["ac06_score"] is None
        assert '"ac06_score": null' in encoded


def test_evidence_is_false_on_every_branch() -> None:
    for result in _success_payloads():
        assert result["evidence"] is False
        assert json.dumps(result).count('"evidence": false') == 1


def test_gallery_mask_is_refused_until_t6() -> None:
    """Eval with both arrays applies the mask. ``unit_test`` still refuses it."""
    query, gallery = _hand_built()
    calls: list[tuple[int, int]] = []

    def mask(left: EvalItem, right: EvalItem) -> bool:
        calls.append((left.track_id, right.track_id))
        return not (left.track_id == 11 and right.track_id == 2)

    result = run("FX-HOLD-001", query=query, gallery=gallery, gallery_mask=mask)
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert (11, 2) in calls
    assert result["evidence"] is False
    assert result["status"] == "DEFERRED"
    assert result["ac06_score"] is None
    assert diagnostics["rank1"] == 1.0
    assert "not evidence" in str(diagnostics["note"])
    assert diagnostics["topology_time_limits"] == "applied via gallery_mask; not evidence"
    unit_calls: list[tuple[EvalItem, EvalItem]] = []

    def unit_mask(left: EvalItem, right: EvalItem) -> bool:
        unit_calls.append((left, right))
        return True

    with pytest.raises(ValueError, match="T6"):
        run("FX-HOLD-001", query=query, gallery=gallery, gallery_mask=unit_mask, mode="unit_test")
    assert unit_calls == []
    with pytest.raises(ValueError, match="T6"):
        run("FX-HOLD-001", gallery_mask=unit_mask, mode="unit_test")


def test_gallery_mask_is_refused_in_eval_without_arrays() -> None:
    calls: list[tuple[EvalItem, EvalItem]] = []

    def mask(left: EvalItem, right: EvalItem) -> bool:
        calls.append((left, right))
        return True

    with pytest.raises(ValueError, match="T6"):
        run("FX-HOLD-001", gallery_mask=mask, mode="eval")
    assert calls == []


def test_payload_keys_and_thresholds() -> None:
    result = run("FX-HOLD-001")
    assert tuple(result) == _TOP_LEVEL_KEYS
    assert result["criterion"] == "AC-06"
    assert result["placeholder_embedder"] is True
    assert result["thresholds"] == {"rank1": 0.90, "mAP": 0.70}
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert tuple(diagnostics) == _DIAGNOSTIC_KEYS


def test_fixture_id_must_be_a_non_empty_string() -> None:
    with pytest.raises(ValueError):
        run("")
    with pytest.raises(ValueError):
        run(None)  # type: ignore[arg-type]


def test_harness_does_not_open_files(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse_open(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("harness opened a file")

    monkeypatch.setattr("builtins.open", refuse_open)
    query, gallery = _hand_built()
    assert run("FX-HOLD-001")["ac06_score"] is None
    measured = run("FX-HOLD-001", query=query, gallery=gallery)
    assert measured["diagnostics"]["num_query"] == 3  # type: ignore[index]


def test_harness_does_not_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def refuse_create_connection(*_args: object, **_kwargs: object) -> None:
        calls.append("create_connection")
        raise OSError("refused")

    monkeypatch.setattr(socket, "create_connection", refuse_create_connection)
    query, gallery = _hand_built()
    assert run("FX-HOLD-001", query=query, gallery=gallery)["status"] == "DEFERRED"
    assert calls == []
