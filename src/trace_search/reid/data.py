"""Load ReID samples for training and evaluation, or refuse the batch.

This is the door into ReID training and evaluation. It refuses ids that
normalise to an ``FX-SMOKE`` or ``FX-CODE`` prefix, ``SMOKE_ONLY`` and
``CODE_TEST`` tags, and any unknown tag. A sample path is checked with
``check_fixture_allowed`` and must have a manifest row. The allowed-id
set is empty, so every other id is refused. One refusal refuses the batch.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from trace_search.fixtures.guards import (
    FixtureSeparationError,
    LoaderContext,
    check_fixture_allowed,
    identify_fixture_by_content,
    load_manifest_fixtures,
)

# Empty until a later task adds an explicit id pattern. Nothing matches today.
_ALLOWED_ID_PATTERNS: tuple[re.Pattern[str], ...] = ()

_REID_CONTEXTS = frozenset({LoaderContext.TRAINING, LoaderContext.EVALUATION})
_DENIED_USE_TAGS = frozenset({"SMOKE_ONLY", "CODE_TEST"})
_RECOGNISED_USE_TAGS = frozenset({"SMOKE_ONLY", "CODE_TEST", "TRAIN", "DEV", "HOLD"})


@dataclass(frozen=True)
class ReidSample:
    """One sample the ReID loader may accept.

    ``label`` is an anonymous same-appearance group index used only by the
    metric. It is not a person id.
    """

    fixture_id: str
    use: str
    camera_id: str
    track_id: int
    label: int
    crop: npt.NDArray[np.uint8] | None
    path: Path | None


def load_reid_samples(
    samples: Sequence[ReidSample],
    context: LoaderContext,
    manifest_path: Path,
) -> list[ReidSample]:
    """Return every sample, or raise ``FixtureSeparationError`` and return nothing.

    Only ``LoaderContext.TRAINING`` and ``LoaderContext.EVALUATION`` are
    accepted. Any other context is refused before a sample is read.
    """
    _require_reid_context(context)
    return _accept_batch(samples, context, manifest_path)


def _require_reid_context(context: LoaderContext) -> None:
    if isinstance(context, LoaderContext) and context in _REID_CONTEXTS:
        return
    raise FixtureSeparationError(
        f"ReID loader refuses context {context!r}. Only TRAINING and EVALUATION are accepted."
    )


def _accept_batch(
    samples: Sequence[ReidSample],
    context: LoaderContext,
    manifest_path: Path,
) -> list[ReidSample]:
    """Accept every sample. The first refusal aborts the batch."""
    accepted: list[ReidSample] = []
    for sample in samples:
        _require_sample(sample, context, manifest_path)
        accepted.append(sample)
    return accepted


def _require_sample(sample: ReidSample, context: LoaderContext, manifest_path: Path) -> None:
    normalised_id = _normalise_fixture_id(sample.fixture_id)
    _refuse_smoke_prefix(sample.fixture_id, normalised_id)
    _refuse_code_prefix(sample.fixture_id, normalised_id)
    normalised_use = _normalise_use_tag(sample.use)
    _refuse_forbidden_use_tag(sample.use, normalised_use)
    _refuse_unknown_use_tag(sample.use, normalised_use)
    if sample.path is not None:
        _refuse_by_content_guard(sample.path, context, manifest_path)
        _refuse_without_manifest_row(sample.path, manifest_path)
    _refuse_unless_allowlisted(sample.fixture_id, normalised_id)


def _normalise_fixture_id(fixture_id: str) -> str:
    """NFKC, strip, upper-case, and read ``_`` as ``-``."""
    folded = unicodedata.normalize("NFKC", fixture_id)
    return folded.strip().upper().replace("_", "-")


def _refuse_smoke_prefix(original: str, normalised: str) -> None:
    if not normalised.startswith("FX-SMOKE"):
        return
    raise FixtureSeparationError(f"{original} is refused: normalised id starts with FX-SMOKE")


def _refuse_code_prefix(original: str, normalised: str) -> None:
    if not normalised.startswith("FX-CODE"):
        return
    raise FixtureSeparationError(f"{original} is refused: normalised id starts with FX-CODE")


def _normalise_use_tag(use: str) -> str:
    return use.strip().upper()


def _refuse_forbidden_use_tag(original: str, normalised: str) -> None:
    if normalised not in _DENIED_USE_TAGS:
        return
    raise FixtureSeparationError(f"{original} is refused: forbidden use tag {normalised}")


def _refuse_unknown_use_tag(original: str, normalised: str) -> None:
    if normalised in _RECOGNISED_USE_TAGS:
        return
    raise FixtureSeparationError(f"{original} is refused: unknown use tag {normalised!r}")


def _refuse_by_content_guard(path: Path, context: LoaderContext, manifest_path: Path) -> None:
    check_fixture_allowed(path, context, manifest_path)


def _refuse_without_manifest_row(path: Path, manifest_path: Path) -> None:
    rows = load_manifest_fixtures(manifest_path)
    if identify_fixture_by_content(path, rows) is None:
        raise FixtureSeparationError(f"no manifest row for {path}")


def _id_on_allowlist(normalised: str) -> bool:
    for pattern in _ALLOWED_ID_PATTERNS:
        if pattern.fullmatch(normalised):
            return True
    return False


def _refuse_unless_allowlisted(original: str, normalised: str) -> None:
    if _id_on_allowlist(normalised):
        return
    raise FixtureSeparationError(f"{original} is refused: id is not on the ReID allowlist")
