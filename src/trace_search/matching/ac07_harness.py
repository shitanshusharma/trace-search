"""AC-07 harness for segment recall, precision, and false-merge rate.

The metric functions live in ``route_metrics`` and measure caller-supplied
records. ``run`` does not produce an AC-07 result in Cut 01. ``status`` is
``DEFERRED``, ``evidence`` is false, and ``ac07_score`` is null. Thresholds
are copied from AC-07 and are not compared to a measured number.

``FX-SMOKE`` is refused first, in every mode, including an unknown mode.
That check normalises the id (NFKC, strip, upper-case, ``_`` read as ``-``)
and raises when the result starts with ``FX-SMOKE``. The error keeps the
original id. Eval does not accept a normalised id.

``eval`` and ``unit_test`` are fail-closed. A raw id that starts with
``FX-CODE`` raises in both. Any other id must full-match
``^FX-HOLD-[0-9]{3}$`` on the raw text. The check uses ``fullmatch``, not
``search``, and it does not use the normalised form. ``unit_test`` does not
skip that gate and does not compute numbers.

``code_test`` fullmatches ``^FX-CODE-[0-9]+$`` after NFKC, upper-case, and
``_`` read as ``-``. The digit class is ASCII ``[0-9]``. Strip is not
applied on this anchor: ``str.strip`` would delete a trailing newline and
the id would match. ``FX-SMOKE`` is still refused first, in every mode.
When routes, truth, and ownership are all supplied, the three metrics sit
under ``diagnostics`` with a note that they are not an AC-07 result.
Omitted inputs are not measured. ``ac07_score`` stays null on that path too.

This module does not read a fixture file, a label file, or a media file.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict, cast

from trace_search.fixtures.guards import FixtureSeparationError
from trace_search.matching.route_metrics import (
    RouteLike,
    RouteSegmentLike,
    TruthSegment,
    false_merge_rate,
    segment_precision,
    segment_recall,
)

# ASCII digits. A unicode digit class would accept Arabic-Indic digits,
# which NFKC leaves unchanged.
_HOLD_ID = re.compile(r"^FX-HOLD-[0-9]{3}$")
_CODE_ID = re.compile(r"^FX-CODE-[0-9]+$")
_SMOKE_PREFIX = "FX-SMOKE"
_CODE_PREFIX = "FX-CODE"
_EVAL_MODE = "eval"
_CODE_TEST_MODE = "code_test"
_UNIT_TEST_MODE = "unit_test"
_RECALL_THRESHOLD = 0.80
_PRECISION_THRESHOLD = 0.90
_FALSE_MERGE_THRESHOLD = 0.05
_DIAGNOSTICS_NOTE = (
    "Not an AC-07 result. AC-07 is DEFERRED in Cut 01; these numbers are not evidence."
)

_Mode = Literal["eval", "code_test", "unit_test"]

__all__ = [
    "Ac07RunResult",
    "Diagnostics",
    "Thresholds",
    "run",
]


class Thresholds(TypedDict):
    """AC-07 bars. They are copied onto the payload and not applied."""

    segment_recall: float
    segment_precision: float
    false_merge_rate: float


class Diagnostics(TypedDict):
    """Measured numbers. The note says they are not an AC-07 result."""

    note: str
    segment_recall: float | None
    segment_precision: float | None
    false_merge_rate: float | None


class Ac07RunResult(TypedDict):
    """Cut 01 AC-07 payload. ``ac07_score`` is always null."""

    criterion: Literal["AC-07"]
    fixture_id: str
    status: Literal["DEFERRED"]
    evidence: Literal[False]
    ac07_score: None
    thresholds: Thresholds
    diagnostics: Diagnostics


def run(
    fixture_id: str,
    routes: Sequence[RouteLike] | None = None,
    truth: Sequence[TruthSegment] | None = None,
    mode: _Mode = "eval",
    *,
    ownership: Mapping[tuple[str, int], str] | None = None,
) -> Ac07RunResult:
    """Return a Cut 01 AC-07 payload. The payload is never a pass.

    ``mode`` must be ``eval``, ``code_test``, or ``unit_test``. Smoke ids
    raise before the mode check. ``eval`` and ``unit_test`` do not compute
    the metrics. ``code_test`` puts them under ``diagnostics`` only when
    routes, truth, and ownership were all passed, and labels them as not
    an AC-07 result.
    """
    original = _require_fixture_id(fixture_id)
    _refuse_smoke(original)
    checked = _require_mode(mode)
    if checked == _CODE_TEST_MODE:
        return _run_code_test(original, routes, truth, ownership)
    if checked == _UNIT_TEST_MODE:
        return _run_unit_test(original)
    return _run_eval(original)


def _require_fixture_id(fixture_id: object) -> str:
    if isinstance(fixture_id, str) and fixture_id != "":
        return fixture_id
    raise ValueError(f"fixture_id must be a non-empty string, got {fixture_id!r}")


def _require_mode(mode: str) -> _Mode:
    if mode == _EVAL_MODE:
        return "eval"
    if mode == _CODE_TEST_MODE:
        return "code_test"
    if mode == _UNIT_TEST_MODE:
        return "unit_test"
    raise ValueError(f"mode must be 'eval', 'code_test', or 'unit_test', got {mode!r}")


def _refuse_smoke(original: str) -> None:
    """Raise when the normalised id starts with ``FX-SMOKE``. Every mode."""
    if _normalise_fixture_id(original).startswith(_SMOKE_PREFIX):
        _reject_smoke(original)


def _reject_smoke(original: str) -> None:
    raise FixtureSeparationError(f"{original} is refused: normalised id starts with FX-SMOKE")


def _normalise_fixture_id(fixture_id: str) -> str:
    """NFKC, then strip, upper-case, and read ``_`` as ``-``. Smoke gate only."""
    folded = _apply_nfkc(fixture_id)
    stripped = _apply_strip(folded)
    upper = _apply_upper(stripped)
    return _apply_underscores(upper)


def _apply_nfkc(fixture_id: str) -> str:
    return unicodedata.normalize("NFKC", fixture_id)


def _apply_strip(fixture_id: str) -> str:
    return fixture_id.strip()


def _apply_upper(fixture_id: str) -> str:
    return fixture_id.upper()


def _apply_underscores(fixture_id: str) -> str:
    return fixture_id.replace("_", "-")


def _run_eval(original: str) -> Ac07RunResult:
    """Fail-closed eval. Raw HOLD ids only. Metrics are not computed."""
    return _closed_payload(original)


def _run_unit_test(original: str) -> Ac07RunResult:
    """Fail-closed unit_test. Same id gate as eval. Metrics are not computed."""
    return _closed_payload(original)


def _closed_payload(original: str) -> Ac07RunResult:
    _refuse_code(original)
    _require_hold(original)
    return _null_result(original)


def _refuse_code(original: str) -> None:
    """Refuse a raw ``FX-CODE`` id. Normalisation is not used here."""
    if original.startswith(_CODE_PREFIX):
        raise FixtureSeparationError(f"{original} is refused: id starts with FX-CODE")


def _require_hold(original: str) -> None:
    """Accept only a raw id that full-matches the HOLD anchor."""
    if _HOLD_ID.fullmatch(original) is not None:
        return
    raise FixtureSeparationError(
        f"{original} is refused: id does not fullmatch ^FX-HOLD-[0-9]{{3}}$"
    )


def _run_code_test(
    original: str,
    routes: Sequence[RouteLike] | None,
    truth: Sequence[TruthSegment] | None,
    ownership: Mapping[tuple[str, int], str] | None,
) -> Ac07RunResult:
    _require_code(original)
    return _result(original, _code_diagnostics(routes, truth, ownership))


def _require_code(original: str) -> None:
    """Accept a folded id that full-matches ``^FX-CODE-[0-9]+$``. No strip."""
    if _CODE_ID.fullmatch(_code_match_text(original)) is not None:
        return
    raise FixtureSeparationError(
        f"{original} is refused: normalised id does not fullmatch ^FX-CODE-[0-9]+$"
    )


def _code_match_text(original: str) -> str:
    """NFKC, upper-case, and ``_`` to ``-``. No strip, so a newline stays."""
    return _apply_underscores(_apply_upper(_apply_nfkc(original)))


def _code_diagnostics(
    routes: Sequence[RouteLike] | None,
    truth: Sequence[TruthSegment] | None,
    ownership: Mapping[tuple[str, int], str] | None,
) -> Diagnostics:
    """Measure only when every input was supplied. Omitted inputs stay null."""
    if routes is None or truth is None or ownership is None:
        return _blank_diagnostics()
    predicted = _flatten(routes)
    return _measured_diagnostics(predicted, truth, routes, ownership)


def _flatten(routes: Sequence[RouteLike]) -> tuple[RouteSegmentLike, ...]:
    predicted: list[RouteSegmentLike] = []
    for route in routes:
        predicted.extend(route.segments)
    return tuple(predicted)


def _measured_diagnostics(
    predicted: Sequence[RouteSegmentLike],
    truth: Sequence[TruthSegment],
    routes: Sequence[RouteLike],
    ownership: Mapping[tuple[str, int], str],
) -> Diagnostics:
    """Caller-supplied numbers. They are not copied into ``ac07_score``."""
    recall = segment_recall(predicted, truth, ownership)
    precision = segment_precision(predicted, truth, ownership)
    merges = false_merge_rate(routes, ownership)
    return _filled_diagnostics(recall, precision, merges)


def _null_result(fixture_id: str) -> Ac07RunResult:
    return _result(fixture_id, _blank_diagnostics())


def _result(fixture_id: str, diagnostics: Diagnostics) -> Ac07RunResult:
    return {
        "criterion": "AC-07",
        "fixture_id": fixture_id,
        "status": _status(),
        "evidence": _evidence(),
        "ac07_score": cast(None, _score()),
        "thresholds": _thresholds(),
        "diagnostics": diagnostics,
    }


def _status() -> Literal["DEFERRED"]:
    """Cut 01 has no pass branch. Thresholds are not applied."""
    return "DEFERRED"


def _evidence() -> Literal[False]:
    """Diagnostics are not evidence for AC-07."""
    return False


def _score() -> object:
    """No AC-07 score is produced. The payload casts this to null."""
    return None


def _thresholds() -> Thresholds:
    return {
        "segment_recall": _RECALL_THRESHOLD,
        "segment_precision": _PRECISION_THRESHOLD,
        "false_merge_rate": _FALSE_MERGE_THRESHOLD,
    }


def _blank_diagnostics() -> Diagnostics:
    return _filled_diagnostics(None, None, None)


def _filled_diagnostics(
    recall: float | None,
    precision: float | None,
    merges: float | None,
) -> Diagnostics:
    return {
        "note": _DIAGNOSTICS_NOTE,
        "segment_recall": recall,
        "segment_precision": precision,
        "false_merge_rate": merges,
    }
