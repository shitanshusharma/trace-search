"""Loader guard. Forbidden fixtures are identified by bytes, not by name.

SHA-256 matches the manifest. pHash (Hamming <= 8) catches a re-encode.
Unknown use tags, unknown contexts, missing files, and decode or pHash
errors are denied.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from trace_search.fixtures.content import (
    HAMMING_THRESHOLD,
    nearest_hamming,
    perceptual_hashes,
    sha256_file,
)


class FixtureUseTag(Enum):
    """Fixture usage tags the loader understands. Anything else is denied."""

    SMOKE_ONLY = "SMOKE_ONLY"
    CODE_TEST = "CODE_TEST"
    TRAIN = "TRAIN"
    DEV = "DEV"
    HOLD = "HOLD"


class LoaderContext(Enum):
    """Contexts in which a file may be loaded."""

    SMOKE_TEST = "smoke_test"
    UNIT_TEST = "unit_test"
    TRAINING = "training"
    EVALUATION = "evaluation"
    STUDY = "study"


class FixtureSeparationError(Exception):
    """A file must not be loaded in this context."""


FORBIDDEN_TAGS_BY_CONTEXT = {
    LoaderContext.TRAINING: {FixtureUseTag.SMOKE_ONLY, FixtureUseTag.CODE_TEST},
    LoaderContext.EVALUATION: {FixtureUseTag.SMOKE_ONLY, FixtureUseTag.CODE_TEST},
    LoaderContext.STUDY: {FixtureUseTag.SMOKE_ONLY, FixtureUseTag.CODE_TEST},
}


def _require_context(context: LoaderContext) -> None:
    if not isinstance(context, LoaderContext):
        raise FixtureSeparationError(
            f"Unknown context {context!r}. FAIL-CLOSED: rejecting unknown context."
        )


def load_manifest_fixtures(manifest_path: Path) -> dict[str, tuple[FixtureUseTag, str, list[str]]]:
    """Map SHA-256 to (use tag, fixture id, pHashes). Unknown tags are denied."""
    if not manifest_path.exists():
        raise FixtureSeparationError(f"FAIL-CLOSED: manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureSeparationError(f"FAIL-CLOSED: cannot read manifest: {exc}") from exc

    fixtures: dict[str, tuple[FixtureUseTag, str, list[str]]] = {}
    for entry in manifest.get("entries", []):
        if "use" not in entry:
            continue
        raw_use = entry.get("use")
        try:
            use_tag = FixtureUseTag(raw_use)
        except ValueError as exc:
            raise FixtureSeparationError(
                f"Unknown use tag {raw_use!r}. FAIL-CLOSED: rejecting unknown use tag."
            ) from exc
        sha = entry.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise FixtureSeparationError(
                f"FAIL-CLOSED: manifest row {entry.get('fixture_id', entry.get('name'))} "
                "has use tag but no SHA-256"
            )
        fixture_id = str(entry.get("fixture_id") or "unknown")
        phashes = entry.get("phash") or []
        if not isinstance(phashes, list):
            raise FixtureSeparationError(f"FAIL-CLOSED: pHash list for {fixture_id} is not a list")
        fixtures[sha] = (use_tag, fixture_id, [str(item) for item in phashes])
    return fixtures


def identify_fixture_by_content(
    file_path: Path,
    manifest_fixtures: dict[str, tuple[FixtureUseTag, str, list[str]]],
    hamming_threshold: int = HAMMING_THRESHOLD,
) -> tuple[FixtureUseTag, str] | None:
    """Identify a file by SHA-256 or pHash. Errors are denied, not ignored."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FixtureSeparationError(f"FAIL-CLOSED: missing file {file_path}")
    try:
        file_sha = sha256_file(file_path)
    except Exception as exc:
        raise FixtureSeparationError(f"FAIL-CLOSED: {exc}") from exc

    known = manifest_fixtures.get(file_sha)
    if known is not None:
        use_tag, fixture_id, _phashes = known
        return use_tag, fixture_id

    try:
        file_hashes = perceptual_hashes(file_path)
    except Exception as exc:
        raise FixtureSeparationError(f"FAIL-CLOSED: {exc}") from exc

    for _sha, (use_tag, fixture_id, manifest_hashes) in manifest_fixtures.items():
        distance = nearest_hamming(file_hashes, manifest_hashes)
        if distance is not None and distance <= hamming_threshold:
            return use_tag, fixture_id
    return None


def _enforce(
    file_path: Path,
    context: LoaderContext,
    manifest_fixtures: dict[str, tuple[FixtureUseTag, str, list[str]]],
    hamming_threshold: int,
) -> None:
    identified = identify_fixture_by_content(file_path, manifest_fixtures, hamming_threshold)
    if identified is None:
        return
    use_tag, fixture_id = identified
    forbidden = FORBIDDEN_TAGS_BY_CONTEXT.get(context, set())
    if use_tag in forbidden:
        raise FixtureSeparationError(
            f"Fixture {file_path} identified as {fixture_id} with use tag {use_tag.value} "
            f"is FORBIDDEN in {context.value} context (identified by SHA-256 or pHash)."
        )


def check_fixture_allowed(
    file_path: Path,
    context: LoaderContext,
    manifest_path: Path,
    hamming_threshold: int = HAMMING_THRESHOLD,
) -> None:
    """Deny forbidden fixtures and any failure to identify the file."""
    _require_context(context)
    fixtures = load_manifest_fixtures(manifest_path)
    _enforce(Path(file_path), context, fixtures, hamming_threshold)


def validate_dataset_for_context(
    file_paths: list[Path],
    manifest_path: Path,
    context: LoaderContext,
    hamming_threshold: int = HAMMING_THRESHOLD,
) -> None:
    """Validate every file. One forbidden file or one error denies the dataset."""
    _require_context(context)
    fixtures = load_manifest_fixtures(manifest_path)
    for file_path in file_paths:
        _enforce(Path(file_path), context, fixtures, hamming_threshold)
