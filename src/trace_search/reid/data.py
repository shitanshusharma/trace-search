"""Load ReID samples for training and evaluation, or refuse the batch.

This is the door into ReID training and evaluation. It refuses ids that
normalise to an ``FX-SMOKE`` or ``FX-CODE`` prefix, ``SMOKE_ONLY`` and
``CODE_TEST`` tags, and any unknown tag. A sample path is checked against
the caller manifest and the repo smoke manifest. A crop with no path is
hashed the same way. The training id allowlist is empty. Evaluation
accepts only ``FX-SYNREID-nnn``. One refusal refuses the batch.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import imagehash
import numpy as np
import numpy.typing as npt
from PIL import Image

from trace_search.fixtures.content import (
    HAMMING_THRESHOLD,
    ContentCheckError,
    nearest_hamming,
    perceptual_hashes,
    sha256_file,
)
from trace_search.fixtures.guards import (
    FORBIDDEN_TAGS_BY_CONTEXT,
    FixtureSeparationError,
    FixtureUseTag,
    LoaderContext,
    load_manifest_fixtures,
)

# Training accepts no id in Cut 01. Evaluation accepts the synthetic prefix only.
_SYNREID_ID = re.compile(r"FX-SYNREID-[0-9]{3}")
_TRAINING_ID_PATTERNS: tuple[re.Pattern[str], ...] = ()
_EVALUATION_ID_PATTERNS: tuple[re.Pattern[str], ...] = (_SYNREID_ID,)

_REID_CONTEXTS = frozenset({LoaderContext.TRAINING, LoaderContext.EVALUATION})
_DENIED_USE_TAGS = frozenset({FixtureUseTag.SMOKE_ONLY.value, FixtureUseTag.CODE_TEST.value})
_RECOGNISED_USE_TAGS = frozenset(tag.value for tag in FixtureUseTag)
_TRAINING_BLOCKED_TAGS = frozenset(
    {FixtureUseTag.TRAIN.value, FixtureUseTag.HOLD.value, FixtureUseTag.DEV.value}
)
_DEV_TAG = FixtureUseTag.DEV.value

_Rows = dict[str, tuple[FixtureUseTag, str, list[str]]]
_Match = tuple[FixtureUseTag, str]


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


def _repo_manifest_path() -> Path:
    """The repo ``licenses/manifest.json``. Callers cannot pass another path."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "licenses" / "manifest.json"


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
    """Accept every sample. The first refusal aborts the batch.

    Smoke and code prefixes are checked before any other gate, so one
    planted id still names itself when an earlier sample would fail a
    later gate. The caller manifest and the repo manifest are loaded
    once each.
    """
    _refuse_forbidden_ids(samples)
    caller_rows = load_manifest_fixtures(manifest_path)
    repo_rows = load_manifest_fixtures(_repo_manifest_path())
    accepted: list[ReidSample] = []
    for sample in samples:
        _require_sample(sample, context, caller_rows, repo_rows)
        _seal_crop(sample)
        accepted.append(sample)
    return accepted


def _refuse_forbidden_ids(samples: Sequence[ReidSample]) -> None:
    for sample in samples:
        normalised = _normalise_fixture_id(sample.fixture_id)
        _refuse_smoke_prefix(sample.fixture_id, normalised)
        _refuse_code_prefix(sample.fixture_id, normalised)


def _require_sample(
    sample: ReidSample,
    context: LoaderContext,
    caller_rows: _Rows,
    repo_rows: _Rows,
) -> None:
    normalised_id = _normalise_fixture_id(sample.fixture_id)
    _refuse_smoke_prefix(sample.fixture_id, normalised_id)
    _refuse_code_prefix(sample.fixture_id, normalised_id)
    normalised_use = _normalise_use_tag(sample.use)
    _refuse_forbidden_use_tag(sample.use, normalised_use)
    _refuse_unknown_use_tag(sample.use, normalised_use)
    if sample.path is not None:
        _refuse_path(sample, context, caller_rows, repo_rows)
    if sample.crop is not None:
        _refuse_crop(sample, repo_rows)
    _refuse_unless_allowlisted(sample.fixture_id, normalised_id, context)
    _refuse_without_crop_or_path(sample)
    _refuse_context_tag(sample.use, normalised_use, normalised_id, context)


def _normalise_fixture_id(fixture_id: str) -> str:
    """NFKC, strip, upper-case, and read ``_`` as ``-``.

    A non-string id raises ``FixtureSeparationError``.
    """
    if not isinstance(fixture_id, str):
        raise FixtureSeparationError(
            f"fixture id must be a string, got {type(fixture_id).__name__}"
        )
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
    if not isinstance(use, str):
        raise FixtureSeparationError(f"use tag must be a string, got {type(use).__name__}")
    return use.strip().upper()


def _refuse_forbidden_use_tag(original: str, normalised: str) -> None:
    if normalised not in _DENIED_USE_TAGS:
        return
    raise FixtureSeparationError(f"{original} is refused: forbidden use tag {normalised}")


def _refuse_unknown_use_tag(original: str, normalised: str) -> None:
    if normalised in _RECOGNISED_USE_TAGS:
        return
    raise FixtureSeparationError(f"{original} is refused: unknown use tag {normalised!r}")


def _refuse_path(
    sample: ReidSample,
    context: LoaderContext,
    caller_rows: _Rows,
    repo_rows: _Rows,
) -> None:
    path = cast(Path, sample.path)
    digest, hashes = _fingerprint(path)
    caller_match = _match_rows(digest, hashes, caller_rows)
    if caller_match is None:
        raise FixtureSeparationError(f"no manifest row for {path}")
    _refuse_forbidden_match(path, caller_match, context)
    repo_match = _match_rows(digest, hashes, repo_rows)
    _refuse_forbidden_match(path, repo_match, context)
    _refuse_row_mismatch(sample, caller_match)
    if repo_match is not None:
        _refuse_row_mismatch(sample, repo_match)


def _fingerprint(path: Path) -> tuple[str, list[str]]:
    """SHA-256 and pHash for one path. The pHash is computed once."""
    try:
        digest = sha256_file(path)
        hashes = perceptual_hashes(path)
    except ContentCheckError as exc:
        raise FixtureSeparationError(f"FAIL-CLOSED: {exc}") from exc
    return digest, hashes


def _match_rows(digest: str, hashes: list[str], rows: _Rows) -> _Match | None:
    known = rows.get(digest)
    if known is not None:
        use_tag, fixture_id, _phashes = known
        return use_tag, fixture_id
    for _sha, (use_tag, fixture_id, manifest_hashes) in rows.items():
        distance = nearest_hamming(hashes, manifest_hashes)
        if distance is not None and distance <= HAMMING_THRESHOLD:
            return use_tag, fixture_id
    return None


def _refuse_forbidden_match(path: Path, match: _Match | None, context: LoaderContext) -> None:
    if match is None:
        return
    use_tag, fixture_id = match
    forbidden = FORBIDDEN_TAGS_BY_CONTEXT.get(context, set())
    if use_tag not in forbidden:
        return
    raise FixtureSeparationError(
        f"Fixture {path} identified as {fixture_id} with use tag {use_tag.value} "
        f"is FORBIDDEN in {context.value} context (identified by SHA-256 or pHash)."
    )


def _refuse_row_mismatch(sample: ReidSample, match: _Match) -> None:
    use_tag, fixture_id = match
    same_id = _normalise_fixture_id(fixture_id) == _normalise_fixture_id(sample.fixture_id)
    same_use = _normalise_use_tag(use_tag.value) == _normalise_use_tag(sample.use)
    if same_id and same_use:
        return
    raise FixtureSeparationError(
        f"{sample.fixture_id} is refused: manifest row {fixture_id} "
        f"use {use_tag.value} does not match the sample"
    )


def _refuse_crop(sample: ReidSample, repo_rows: _Rows) -> None:
    """Hash an in-memory crop against the repo smoke manifest."""
    crop = cast(npt.NDArray[np.uint8], sample.crop)
    digest, phash = _crop_fingerprint(crop)
    match = _match_rows(digest, [phash], repo_rows)
    if match is None:
        return
    use_tag, fixture_id = match
    raise FixtureSeparationError(
        f"crop on {sample.fixture_id} identified as {fixture_id} "
        f"with use tag {use_tag.value} is FORBIDDEN "
        "(identified by SHA-256 or pHash)."
    )


def _crop_fingerprint(crop: npt.NDArray[np.uint8]) -> tuple[str, str]:
    array = np.ascontiguousarray(crop)
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    image = Image.fromarray(array, mode="RGB")
    return digest, str(imagehash.phash(image))


def _patterns_for(context: LoaderContext) -> tuple[re.Pattern[str], ...]:
    if context is LoaderContext.TRAINING:
        return _TRAINING_ID_PATTERNS
    return _EVALUATION_ID_PATTERNS


def _id_on_allowlist(normalised: str, context: LoaderContext) -> bool:
    for pattern in _patterns_for(context):
        if pattern.fullmatch(normalised):
            return True
    return False


def _refuse_unless_allowlisted(original: str, normalised: str, context: LoaderContext) -> None:
    if _id_on_allowlist(normalised, context):
        return
    raise FixtureSeparationError(f"{original} is refused: id is not on the ReID allowlist")


def _refuse_without_crop_or_path(sample: ReidSample) -> None:
    """Refuse a sample that has no crop and no path."""
    if sample.crop is not None or sample.path is not None:
        return
    raise FixtureSeparationError(
        f"{sample.fixture_id} is refused: sample has neither a crop nor a path"
    )


def _refuse_context_tag(
    original: str,
    normalised: str,
    normalised_id: str,
    context: LoaderContext,
) -> None:
    if context is LoaderContext.TRAINING and normalised in _TRAINING_BLOCKED_TAGS:
        raise FixtureSeparationError(f"{original} is refused in TRAINING")
    if context is not LoaderContext.EVALUATION:
        return
    if _SYNREID_ID.fullmatch(normalised_id) is None:
        return
    if normalised == _DEV_TAG:
        return
    raise FixtureSeparationError(f"{original} is refused in EVALUATION")


def _seal_crop(sample: ReidSample) -> None:
    """Replace the crop with a read-only copy so the caller cannot edit it."""
    crop = sample.crop
    if crop is None:
        return
    copied = np.array(crop, copy=True)
    copied.setflags(write=False)
    object.__setattr__(sample, "crop", copied)
