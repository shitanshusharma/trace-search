"""Load ReID samples for training and evaluation, or refuse the batch.

This is the door into ReID training and evaluation. It refuses ids that
normalise to an ``FX-SMOKE`` or ``FX-CODE`` prefix, ``SMOKE_ONLY`` and
``CODE_TEST`` tags, and any unknown tag. A sample path is checked against
the caller manifest and the repo smoke manifest by SHA-256 and pHash.

A crop with no path is checked by pHash only. Manifest sha256 values are
hashes of media files, so crop bytes are not compared to them. That pHash
can match a whole smoke frame, including one that was only resized. It
does not match a box cut out of a frame. That limit is pinned by
``test_known_gap_centre_crop_of_a_smoke_frame_is_accepted``.

TRAINING refuses every sample before the id allowlist and before any tag
deny-list. The training id allowlist stays empty. Evaluation accepts only
``FX-SYNREID-nnn`` with the ``DEV`` tag. One refusal refuses the batch,
and the caller's samples and arrays are left unchanged.
"""

from __future__ import annotations

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
    sealed = [_require_sample(sample, context, caller_rows, repo_rows) for sample in samples]
    _install_sealed_crops(samples, sealed)
    return list(samples)


def _install_sealed_crops(
    samples: Sequence[ReidSample],
    sealed: Sequence[npt.NDArray[np.uint8] | None],
) -> None:
    """Attach copies only after the whole batch has been accepted.

    A refusal leaves every caller sample and array as it was passed in.
    """
    for sample, crop in zip(samples, sealed, strict=True):
        if crop is None:
            continue
        object.__setattr__(sample, "crop", crop)


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
) -> npt.NDArray[np.uint8] | None:
    normalised_id = _normalise_fixture_id(sample.fixture_id)
    _refuse_smoke_prefix(sample.fixture_id, normalised_id)
    _refuse_code_prefix(sample.fixture_id, normalised_id)
    normalised_use = _normalise_use_tag(sample.use)
    _refuse_forbidden_use_tag(sample.use, normalised_use)
    _refuse_unknown_use_tag(sample.use, normalised_use)
    if sample.path is not None:
        _refuse_path(sample, context, caller_rows, repo_rows)
    sealed = _sealed_crop(sample, repo_rows)
    _refuse_training_context(sample.use, context)
    _refuse_unless_allowlisted(sample.fixture_id, normalised_id, context)
    _refuse_without_crop_or_path(sample)
    if context is LoaderContext.EVALUATION:
        _refuse_evaluation_tag(sample.use, normalised_use, normalised_id)
    return sealed


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


def _sealed_crop(
    sample: ReidSample,
    repo_rows: _Rows,
) -> npt.NDArray[np.uint8] | None:
    """Copy a crop, pHash that copy, and return it still read-only.

    The copy is installed on the sample only after the whole batch
    passes. pHash can match a whole smoke frame. A box cut from a frame,
    such as a centre crop, is a known gap and is not refused. Manifest
    sha256 values are file hashes, so this check does not use them.
    """
    if sample.crop is None:
        return None
    sealed = _readonly_copy(_require_rgb_uint8(sample.crop))
    _refuse_smoke_phash(sample, sealed, repo_rows)
    return sealed


def _require_rgb_uint8(crop: object) -> npt.NDArray[np.uint8]:
    if _is_rgb_uint8(crop):
        return cast(npt.NDArray[np.uint8], crop)
    shape = getattr(crop, "shape", None)
    dtype = getattr(crop, "dtype", None)
    raise FixtureSeparationError(
        f"crop must be a non-empty HxWx3 uint8 array, got shape {shape} dtype {dtype}"
    )


def _is_rgb_uint8(crop: object) -> bool:
    if not isinstance(crop, np.ndarray) or crop.dtype != np.uint8 or crop.ndim != 3:
        return False
    if crop.shape[2] != 3:
        return False
    return crop.shape[0] > 0 and crop.shape[1] > 0


def _readonly_copy(crop: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """A copy whose writeable flag cannot be turned back on."""
    owned = np.array(crop, dtype=np.uint8, copy=True)
    owned.setflags(write=False)
    view = owned.view()
    view.setflags(write=False)
    return view


def _refuse_smoke_phash(
    sample: ReidSample,
    sealed: npt.NDArray[np.uint8],
    repo_rows: _Rows,
) -> None:
    match = _match_phash(_crop_phash(sealed), repo_rows)
    if match is None:
        return
    use_tag, fixture_id = match
    raise FixtureSeparationError(
        f"crop on {sample.fixture_id} identified as {fixture_id} "
        f"with use tag {use_tag.value} is FORBIDDEN (identified by pHash)."
    )


def _crop_phash(crop: npt.NDArray[np.uint8]) -> str:
    image = Image.fromarray(crop, mode="RGB")
    return str(imagehash.phash(image))


def _match_phash(phash: str, rows: _Rows) -> _Match | None:
    for _sha, (use_tag, fixture_id, manifest_hashes) in rows.items():
        distance = nearest_hamming([phash], manifest_hashes)
        if distance is not None and distance <= HAMMING_THRESHOLD:
            return use_tag, fixture_id
    return None


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


def _refuse_training_context(original: str, context: LoaderContext) -> None:
    """TRAINING accepts no sample. The tag and the id are not consulted."""
    if context is not LoaderContext.TRAINING:
        return
    raise FixtureSeparationError(f"{original} is refused in TRAINING")


def _refuse_evaluation_tag(original: str, normalised: str, normalised_id: str) -> None:
    if _SYNREID_ID.fullmatch(normalised_id) is None:
        return
    if normalised == _DEV_TAG:
        return
    raise FixtureSeparationError(f"{original} is refused in EVALUATION")
