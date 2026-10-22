"""The ReID loader refuses smoke and code-test data and fails closed.

Clips are generated into pytest's temp directory. Nothing is read from
``smoke/`` and nothing is written into the repo tree.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

import pytest

import trace_search.reid as reid
import trace_search.reid.data as data_module
from tests.media_helpers import write_mp4v_variant
from trace_search.fixtures.content import sha256_file
from trace_search.fixtures.generator import generate_smoke_fixtures
from trace_search.fixtures.guards import FixtureSeparationError, LoaderContext

ReidSample = data_module.ReidSample
load_reid_samples = data_module.load_reid_samples

_REID_CONTEXTS = (LoaderContext.TRAINING, LoaderContext.EVALUATION)
_OTHER_CONTEXTS = (
    LoaderContext.SMOKE_TEST,
    LoaderContext.UNIT_TEST,
    LoaderContext.STUDY,
)

# U+FF26 U+FF38 then ASCII "-SMOKE-SYN-001". NFKC folds the two letters.
_FULLWIDTH_SMOKE_ID = "\uff26\uff38-SMOKE-SYN-001"

_SMOKE_IDS = (
    "FX-SMOKE-SYN-001",
    "fx-smoke-syn-001",
    "Fx-Smoke-Syn-001",
    "  FX-SMOKE-SYN-001 ",
    "FX_SMOKE_SYN_001",
    _FULLWIDTH_SMOKE_ID,
)
_CODE_IDS = (
    "FX-CODE-001",
    "fx_code_001",
)
_FORBIDDEN_TAGS = (
    ("SMOKE_ONLY", "SMOKE_ONLY"),
    ("smoke_only", "SMOKE_ONLY"),
    (" Smoke_Only ", "SMOKE_ONLY"),
    ("CODE_TEST", "CODE_TEST"),
    ("code_test", "CODE_TEST"),
)
_UNKNOWN_TAGS = ("", "TRAINING")
_UNLISTED_IDS = ("FX-HOLD-001", "anything")

# Test-only. Production ``_ALLOWED_ID_PATTERNS`` stays empty.
_BATCH_PATTERN = re.compile(r"FX-BATCH-\d{3}")


def _sample(
    fixture_id: str,
    use: str = "TRAIN",
    *,
    path: Path | None = None,
    track_id: int = 1,
    label: int = 0,
) -> ReidSample:
    return ReidSample(
        fixture_id=fixture_id,
        use=use,
        camera_id="cam-1",
        track_id=track_id,
        label=label,
        crop=None,
        path=path,
    )


def _write_manifest(directory: Path, entries: Sequence[object] | None = None) -> Path:
    path = directory / "manifest.json"
    payload = {
        "manifest_version": "1.0",
        "entries": [] if entries is None else list(entries),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _refused(
    samples: list[ReidSample],
    context: LoaderContext,
    manifest: Path,
    phrase: str,
) -> str:
    with pytest.raises(FixtureSeparationError) as caught:
        load_reid_samples(samples, context, manifest)
    message = str(caught.value)
    assert phrase in message
    return message


def _entry_clip(directory: Path, entry: dict[str, object]) -> Path:
    fixture_id = str(entry["fixture_id"])
    matches = sorted(directory.glob(f"{fixture_id}_*.mp4"))
    assert len(matches) == 1
    clip = matches[0]
    assert sha256_file(clip) == entry["sha256"]
    return clip


@pytest.fixture(scope="module")
def smoke_clips(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict[str, object]]]:
    """Generate the smoke set once, outside the repo tree."""
    directory = tmp_path_factory.mktemp("reid-smoke-clips")
    repo_root = Path(__file__).resolve().parents[2]
    assert not directory.resolve().is_relative_to(repo_root)
    entries = generate_smoke_fixtures(directory, base_seed=42, num_clips=5)
    assert len(entries) >= 5
    assert all(entry["use"] == "SMOKE_ONLY" for entry in entries)
    return directory, entries


def test_reid_sample_fields_are_the_spec_set() -> None:
    assert [field.name for field in fields(ReidSample)] == [
        "fixture_id",
        "use",
        "camera_id",
        "track_id",
        "label",
        "crop",
        "path",
    ]


def test_label_is_an_anonymous_group_index() -> None:
    doc = ReidSample.__doc__ or ""
    assert "anonymous" in doc
    assert "group index" in doc
    assert "not a person id" in doc


def test_package_exports_the_loader() -> None:
    assert reid.ReidSample is ReidSample
    assert reid.load_reid_samples is load_reid_samples


@pytest.mark.parametrize("fixture_id", _SMOKE_IDS)
@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_smoke_id_is_refused(
    fixture_id: str,
    context: LoaderContext,
    tmp_path: Path,
) -> None:
    message = _refused(
        [_sample(fixture_id)],
        context,
        _write_manifest(tmp_path),
        "normalised id starts with FX-SMOKE",
    )
    assert fixture_id in message


@pytest.mark.parametrize("fixture_id", _CODE_IDS)
@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_code_id_is_refused(
    fixture_id: str,
    context: LoaderContext,
    tmp_path: Path,
) -> None:
    message = _refused(
        [_sample(fixture_id)],
        context,
        _write_manifest(tmp_path),
        "normalised id starts with FX-CODE",
    )
    assert fixture_id in message


@pytest.mark.parametrize(("raw_tag", "normalised"), _FORBIDDEN_TAGS)
@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_forbidden_use_tag_is_refused(
    raw_tag: str,
    normalised: str,
    context: LoaderContext,
    tmp_path: Path,
) -> None:
    message = _refused(
        [_sample("FX-HOLD-001", use=raw_tag)],
        context,
        _write_manifest(tmp_path),
        f"forbidden use tag {normalised}",
    )
    assert raw_tag in message


@pytest.mark.parametrize("raw_tag", _UNKNOWN_TAGS)
@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_unknown_use_tag_is_refused(
    raw_tag: str,
    context: LoaderContext,
    tmp_path: Path,
) -> None:
    normalised = raw_tag.strip().upper()
    message = _refused(
        [_sample("FX-HOLD-001", use=raw_tag)],
        context,
        _write_manifest(tmp_path),
        f"unknown use tag {normalised!r}",
    )
    assert raw_tag in message


@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_smoke_bytes_are_refused_by_sha256(
    context: LoaderContext,
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    directory, entries = smoke_clips
    clip = _entry_clip(directory, entries[0])
    manifest = _write_manifest(tmp_path, entries)
    _refused(
        [_sample("FX-SYNREID-001", path=clip)],
        context,
        manifest,
        "identified by SHA-256 or pHash",
    )


@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_renamed_dat_smoke_bytes_are_refused(
    context: LoaderContext,
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    directory, entries = smoke_clips
    clip = _entry_clip(directory, entries[0])
    renamed = tmp_path / "renamed.dat"
    renamed.write_bytes(clip.read_bytes())
    assert renamed.suffix == ".dat"
    assert sha256_file(renamed) == entries[0]["sha256"]
    manifest = _write_manifest(tmp_path, entries)
    _refused(
        [_sample("FX-SYNREID-001", path=renamed)],
        context,
        manifest,
        "identified by SHA-256 or pHash",
    )


@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_reencoded_smoke_clip_is_refused_by_phash(
    context: LoaderContext,
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    directory, entries = smoke_clips
    clip = _entry_clip(directory, entries[0])
    variant = tmp_path / "reencoded.mp4"
    write_mp4v_variant(clip, variant, scale=0.5)
    assert sha256_file(variant) != entries[0]["sha256"]
    manifest = _write_manifest(tmp_path, entries)
    _refused(
        [_sample("FX-SYNREID-001", path=variant)],
        context,
        manifest,
        "identified by SHA-256 or pHash",
    )


@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_path_with_no_manifest_row_is_refused(
    context: LoaderContext,
    tmp_path: Path,
    smoke_clips: tuple[Path, list[dict[str, object]]],
) -> None:
    directory, entries = smoke_clips
    clip = _entry_clip(directory, entries[0])
    manifest = _write_manifest(tmp_path, [])
    _refused(
        [_sample("FX-SYNREID-001", path=clip)],
        context,
        manifest,
        "no manifest row",
    )


@pytest.mark.parametrize("fixture_id", _UNLISTED_IDS)
@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_empty_allowlist_refuses_other_ids(
    fixture_id: str,
    context: LoaderContext,
    tmp_path: Path,
) -> None:
    message = _refused(
        [_sample(fixture_id)],
        context,
        _write_manifest(tmp_path),
        "not on the ReID allowlist",
    )
    assert fixture_id in message


@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_one_planted_sample_refuses_the_whole_batch(
    context: LoaderContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nine pattern-matching samples and one planted id: the batch is refused.

    The production allowlist is empty, so nothing matches it. This test
    injects a pattern only here, which makes the nine samples good-looking.
    The planted id still refuses the batch, and the caller gets nothing back.
    """
    monkeypatch.setattr(data_module, "_ALLOWED_ID_PATTERNS", (_BATCH_PATTERN,))
    samples = [
        _sample(f"FX-BATCH-{index:03d}", track_id=index, label=index) for index in range(1, 10)
    ]
    planted_id = "fx-smoke-syn-001"
    samples.append(_sample(planted_id, track_id=10, label=10))
    manifest = _write_manifest(tmp_path)
    with pytest.raises(FixtureSeparationError) as caught:
        returned = load_reid_samples(samples, context, manifest)
        pytest.fail(f"loader returned {returned!r}")
    message = str(caught.value)
    assert planted_id in message
    assert "normalised id starts with FX-SMOKE" in message


@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_injected_pattern_returns_a_matching_sample(
    context: LoaderContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A test-only pattern can accept one id. The production tuple stays empty."""
    monkeypatch.setattr(data_module, "_ALLOWED_ID_PATTERNS", (_BATCH_PATTERN,))
    sample = _sample("FX-BATCH-004", track_id=4, label=2)
    loaded = load_reid_samples([sample], context, _write_manifest(tmp_path))
    assert loaded == [sample]


@pytest.mark.parametrize("context", _OTHER_CONTEXTS)
def test_other_contexts_are_refused(context: LoaderContext, tmp_path: Path) -> None:
    _refused([], context, _write_manifest(tmp_path), "Only TRAINING and EVALUATION")


def test_string_context_is_refused(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    with pytest.raises(FixtureSeparationError, match="Only TRAINING and EVALUATION"):
        load_reid_samples([], "training", manifest)  # type: ignore[arg-type]


@pytest.mark.parametrize("context", _REID_CONTEXTS)
def test_empty_batch_is_accepted(context: LoaderContext, tmp_path: Path) -> None:
    assert load_reid_samples([], context, _write_manifest(tmp_path)) == []
