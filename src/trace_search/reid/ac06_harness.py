"""AC-06 harness for cross-camera Rank-1 and mAP.

``rank1_map`` measures caller-supplied arrays. ``run`` does not produce
an AC-06 result in Cut 01. ``status`` is ``DEFERRED``, ``evidence`` is
false, and ``ac06_score`` is null. Measured numbers live under
``diagnostics`` with a note that they are not an AC-06 result.

A fixture id is normalised with NFKC, strip, upper-case, and ``_`` read
as ``-`` before the gate. ``FX-SMOKE`` and ``FX-CODE`` then raise
``FixtureSeparationError``. The error keeps the original id. Any other
id must full-match ``^FX-HOLD-[0-9]{3}$``. The digit class is ASCII
``[0-9]`` because ``\\d`` also matches non-ASCII digits that NFKC does
not fold. ``unit_test`` skips that gate and returns null diagnostic
numbers for every id.

``gallery_mask`` is the hook T6 uses for topology and transit-time
limits. This harness refuses a non-None mask: ``rank1_map`` has no mask
argument, so applying one here would not be that protocol. T5 passes
``None``. The payload says the limits were not applied. This module
does not read a fixture file or a label file.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from typing import Literal, TypedDict, cast

from trace_search.fixtures.guards import FixtureSeparationError
from trace_search.reid.eval import EvalItem, rank1_map

# ASCII digits. ``\d`` matches Nd digits that NFKC leaves unchanged.
_HOLD_ID = re.compile(r"^FX-HOLD-[0-9]{3}$")
_SMOKE_PREFIX = "FX-SMOKE"
_CODE_PREFIX = "FX-CODE"
_EVAL_MODE = "eval"
_UNIT_TEST_MODE = "unit_test"
_RANK1_THRESHOLD = 0.90
_MAP_THRESHOLD = 0.70
_DIAGNOSTICS_NOTE = (
    "Not an AC-06 result. AC-06 is DEFERRED in Cut 01; these numbers are not evidence."
)
_TOPOLOGY_NOTE = "not applied (T6 supplies gallery_mask)"
_MASK_REFUSAL = (
    "gallery_mask is not applied in Cut 01; T6 supplies gallery_mask. "
    "rank1_map has no mask argument, so a non-None mask is refused."
)

GalleryMask = Callable[[EvalItem, EvalItem], bool]

__all__ = [
    "Ac06RunResult",
    "Diagnostics",
    "GalleryMask",
    "Thresholds",
    "run",
]


class Thresholds(TypedDict):
    """Rank-1 and mAP bars. They are not compared to the diagnostics."""

    rank1: float
    mAP: float


class Diagnostics(TypedDict):
    """Measured numbers. The note says they are not an AC-06 result."""

    note: str
    rank1: float | None
    mAP: float | None
    num_query: int | None
    num_valid_query: int | None
    num_gallery: int | None
    topology_time_limits: str


class Ac06RunResult(TypedDict):
    """Cut 01 AC-06 payload. ``ac06_score`` is always null."""

    criterion: Literal["AC-06"]
    fixture_id: str
    status: Literal["DEFERRED"]
    evidence: Literal[False]
    ac06_score: None
    thresholds: Thresholds
    placeholder_embedder: bool
    diagnostics: Diagnostics


def run(
    fixture_id: str,
    query: Sequence[EvalItem] | None = None,
    gallery: Sequence[EvalItem] | None = None,
    gallery_mask: GalleryMask | None = None,
    mode: Literal["eval", "unit_test"] = "eval",
) -> Ac06RunResult:
    """Return a Cut 01 AC-06 payload. The payload is never a pass.

    ``mode`` must be ``eval`` or ``unit_test``. Any other value raises
    ``ValueError``. In ``unit_test``, diagnostic numbers are null even
    when arrays were supplied. In ``eval``, a blocked prefix or an id
    that is not ``FX-HOLD`` plus three ASCII digits raises
    ``FixtureSeparationError`` and keeps the original id. Omitted arrays
    are not measured. A non-None ``gallery_mask`` raises ``ValueError``.
    """
    original = _require_fixture_id(fixture_id)
    checked = _require_mode(mode)
    _refuse_mask(gallery_mask)
    if checked == _UNIT_TEST_MODE:
        return _null_result(original)
    _refuse_blocked(original)
    return _eval_hold(original, query, gallery)


def _require_fixture_id(fixture_id: object) -> str:
    if isinstance(fixture_id, str) and fixture_id != "":
        return fixture_id
    raise ValueError(f"fixture_id must be a non-empty string, got {fixture_id!r}")


def _require_mode(mode: str) -> Literal["eval", "unit_test"]:
    if mode == _EVAL_MODE:
        return "eval"
    if mode == _UNIT_TEST_MODE:
        return "unit_test"
    raise ValueError(f"mode must be 'eval' or 'unit_test', got {mode!r}")


def _refuse_mask(gallery_mask: object) -> None:
    """Refuse a mask. T6 owns topology and transit-time limits."""
    if gallery_mask is None:
        return
    raise ValueError(_MASK_REFUSAL)


def _refuse_blocked(original: str) -> None:
    normalised = _normalise_fixture_id(original)
    if _is_smoke(normalised):
        _reject_smoke(original)
    if _is_code(normalised):
        _reject_code(original)
    if not _is_hold(normalised):
        _reject_not_hold(original)


def _eval_hold(
    original: str,
    query: Sequence[EvalItem] | None,
    gallery: Sequence[EvalItem] | None,
) -> Ac06RunResult:
    if _missing_arrays(query, gallery):
        return _null_result(original)
    return _measured(
        original,
        cast(Sequence[EvalItem], query),
        cast(Sequence[EvalItem], gallery),
    )


def _missing_arrays(
    query: Sequence[EvalItem] | None,
    gallery: Sequence[EvalItem] | None,
) -> bool:
    """True when either array was omitted. Empty sequences are present."""
    if query is None:
        return True
    return gallery is None


def _normalise_fixture_id(fixture_id: str) -> str:
    """NFKC, then strip, upper-case, and read ``_`` as ``-``."""
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


def _is_smoke(normalised: str) -> bool:
    return normalised.startswith(_SMOKE_PREFIX)


def _is_code(normalised: str) -> bool:
    return normalised.startswith(_CODE_PREFIX)


def _is_hold(normalised: str) -> bool:
    return _HOLD_ID.fullmatch(normalised) is not None


def _reject_smoke(original: str) -> None:
    raise FixtureSeparationError(f"{original} is refused: normalised id starts with FX-SMOKE")


def _reject_code(original: str) -> None:
    raise FixtureSeparationError(f"{original} is refused: normalised id starts with FX-CODE")


def _reject_not_hold(original: str) -> None:
    raise FixtureSeparationError(
        f"{original} is refused: normalised id does not match ^FX-HOLD-[0-9]{{3}}$"
    )


def _measured(
    fixture_id: str,
    query: Sequence[EvalItem],
    gallery: Sequence[EvalItem],
) -> Ac06RunResult:
    scored = rank1_map(query, gallery)
    return _payload(
        fixture_id,
        scored.rank1,
        scored.mean_ap,
        scored.num_query,
        scored.num_valid_query,
        scored.num_gallery,
    )


def _null_result(fixture_id: str) -> Ac06RunResult:
    return _payload(fixture_id, None, None, None, None, None)


def _payload(
    fixture_id: str,
    rank1: float | None,
    mean_ap: float | None,
    num_query: int | None,
    num_valid_query: int | None,
    num_gallery: int | None,
) -> Ac06RunResult:
    return {
        "criterion": "AC-06",
        "fixture_id": fixture_id,
        "status": _deferred_status(),
        "evidence": _evidence_flag(),
        "ac06_score": cast(None, _null_score()),
        "thresholds": _thresholds(),
        "placeholder_embedder": _placeholder_embedder(),
        "diagnostics": _diagnostics(
            rank1,
            mean_ap,
            num_query,
            num_valid_query,
            num_gallery,
        ),
    }


def _deferred_status() -> Literal["DEFERRED"]:
    """Cut 01 has no pass branch. Thresholds are not applied."""
    return "DEFERRED"


def _evidence_flag() -> Literal[False]:
    """Diagnostics are not evidence for AC-06."""
    return False


def _null_score() -> object:
    """No AC-06 score is produced. The payload casts this to null."""
    return None


def _placeholder_embedder() -> bool:
    """True: this harness does not run a trained embedder."""
    return True


def _thresholds() -> Thresholds:
    return {"rank1": _RANK1_THRESHOLD, "mAP": _MAP_THRESHOLD}


def _diagnostics(
    rank1: float | None,
    mean_ap: float | None,
    num_query: int | None,
    num_valid_query: int | None,
    num_gallery: int | None,
) -> Diagnostics:
    return {
        "note": _DIAGNOSTICS_NOTE,
        "rank1": rank1,
        "mAP": mean_ap,
        "num_query": num_query,
        "num_valid_query": num_valid_query,
        "num_gallery": num_gallery,
        "topology_time_limits": _TOPOLOGY_NOTE,
    }
