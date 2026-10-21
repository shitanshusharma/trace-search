"""Loader guard: content identity, fail closed, all three scored contexts."""

import json
import os
from pathlib import Path

import pytest

from tests.media_helpers import write_mp4v_variant
from trace_search.fixtures.generator import SyntheticVideoGenerator
from trace_search.fixtures.guards import (
    FixtureSeparationError,
    FixtureUseTag,
    LoaderContext,
    check_fixture_allowed,
    validate_dataset_for_context,
)

SCORED = [LoaderContext.TRAINING, LoaderContext.EVALUATION, LoaderContext.STUDY]


def _clip(path: Path, seed: int) -> SyntheticVideoGenerator:
    path.parent.mkdir(parents=True, exist_ok=True)
    gen = SyntheticVideoGenerator(seed=seed)
    gen.generate_clip(
        output_path=path,
        duration_sec=2.0,
        fps=10,
        width=640,
        height=480,
        num_figures=2,
    )
    return gen


def _manifest(directory: Path, sha: str, phashes: list[str], fixture_id: str, use: str) -> Path:
    path = directory / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "manifest_version": "1.0",
                "last_updated_ist": "2026-10-20",
                "entries": [
                    {
                        "name": "fixture",
                        "version": "1.0.0",
                        "type": "synthetic_clip",
                        "license": "MIT",
                        "source_url": "https://example.com",
                        "sha256": sha,
                        "phash": phashes,
                        "use": use,
                        "fixture_id": fixture_id,
                        "generator": "Test",
                        "generator_version": "1.0",
                        "generator_license": "MIT",
                        "seed": "1",
                    }
                ],
            }
        )
    )
    return path


def test_loader_rejects_renamed_copy(tmp_path: Path):
    original = tmp_path / "original.mp4"
    gen = _clip(original, 123)
    manifest = _manifest(
        tmp_path,
        gen.compute_sha256(original),
        gen.compute_phashes(original),
        "FX-SMOKE-SYN-999",
        "SMOKE_ONLY",
    )
    renamed = tmp_path / "totally_different_name.mp4"
    renamed.write_bytes(original.read_bytes())
    with pytest.raises(FixtureSeparationError, match="SHA-256 or pHash"):
        check_fixture_allowed(renamed, LoaderContext.TRAINING, manifest)


def test_loader_rejects_symlink(tmp_path: Path):
    original = tmp_path / "original.mp4"
    gen = _clip(original, 456)
    manifest = _manifest(
        tmp_path,
        gen.compute_sha256(original),
        gen.compute_phashes(original),
        "FX-SMOKE-SYN-998",
        "SMOKE_ONLY",
    )
    link = tmp_path / "linked.mp4"
    os.symlink(original, link)
    with pytest.raises(FixtureSeparationError, match="FORBIDDEN"):
        validate_dataset_for_context([link], manifest, LoaderContext.EVALUATION)


def test_loader_rejects_relative_path(tmp_path: Path):
    smoke = tmp_path / "smoke"
    train = tmp_path / "train"
    clip = smoke / "FX-SMOKE-SYN-997.mp4"
    gen = _clip(clip, 789)
    manifest = _manifest(
        tmp_path,
        gen.compute_sha256(clip),
        gen.compute_phashes(clip),
        "FX-SMOKE-SYN-997",
        "SMOKE_ONLY",
    )
    train.mkdir()
    relative = train / ".." / "smoke" / "FX-SMOKE-SYN-997.mp4"
    with pytest.raises(FixtureSeparationError, match="FORBIDDEN"):
        check_fixture_allowed(relative, LoaderContext.TRAINING, manifest)


def test_loader_rejects_uppercase_extension(tmp_path: Path):
    original = tmp_path / "test.MP4"
    gen = _clip(original, 111)
    manifest = _manifest(
        tmp_path,
        gen.compute_sha256(original),
        gen.compute_phashes(original),
        "FX-SMOKE-SYN-996",
        "SMOKE_ONLY",
    )
    with pytest.raises(FixtureSeparationError, match="FORBIDDEN"):
        check_fixture_allowed(original, LoaderContext.STUDY, manifest)


@pytest.mark.parametrize("context", SCORED)
def test_code_test_rejected_in_every_scored_context(tmp_path: Path, context: LoaderContext):
    clip = tmp_path / "FX-CODE-001.mp4"
    gen = _clip(clip, 222)
    manifest = _manifest(
        tmp_path,
        gen.compute_sha256(clip),
        gen.compute_phashes(clip),
        "FX-CODE-001",
        "CODE_TEST",
    )
    with pytest.raises(FixtureSeparationError, match="CODE_TEST"):
        validate_dataset_for_context([clip], manifest, context)


@pytest.mark.parametrize("context", SCORED)
def test_reencoded_smoke_rejected_in_every_scored_context(tmp_path: Path, context: LoaderContext):
    original = tmp_path / "original.mp4"
    gen = _clip(original, 42)
    manifest = _manifest(
        tmp_path,
        gen.compute_sha256(original),
        gen.compute_phashes(original),
        "FX-SMOKE-SYN-042",
        "SMOKE_ONLY",
    )
    variant = tmp_path / "train" / "variant.mp4"
    write_mp4v_variant(original, variant, scale=0.5)
    assert gen.compute_sha256(variant) != gen.compute_sha256(original)
    with pytest.raises(FixtureSeparationError, match="SMOKE_ONLY"):
        validate_dataset_for_context([variant], manifest, context)


def test_loader_rejects_unknown_use_tag(tmp_path: Path):
    clip = tmp_path / "test.mp4"
    gen = _clip(clip, 333)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "bad",
                        "sha256": gen.compute_sha256(clip),
                        "phash": gen.compute_phashes(clip),
                        "use": "UNKNOWN_TAG",
                        "fixture_id": "FX-BAD-001",
                    }
                ]
            }
        )
    )
    with pytest.raises(FixtureSeparationError, match="Unknown use tag"):
        check_fixture_allowed(clip, LoaderContext.TRAINING, manifest)


def test_loader_rejects_unknown_context(tmp_path: Path):
    clip = tmp_path / "test.mp4"
    gen = _clip(clip, 444)
    manifest = _manifest(
        tmp_path,
        gen.compute_sha256(clip),
        gen.compute_phashes(clip),
        "FX-TRAIN-001",
        "TRAIN",
    )
    with pytest.raises(FixtureSeparationError, match="Unknown context"):
        check_fixture_allowed(clip, "invalid_context", manifest)


def test_missing_manifest_entry_sha_is_denied(tmp_path: Path):
    clip = tmp_path / "test.mp4"
    _clip(clip, 445)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "incomplete",
                        "use": "SMOKE_ONLY",
                        "fixture_id": "FX-SMOKE-SYN-100",
                    }
                ]
            }
        )
    )
    with pytest.raises(FixtureSeparationError, match="no SHA-256"):
        validate_dataset_for_context([clip], manifest, LoaderContext.TRAINING)


def test_missing_file_is_denied(tmp_path: Path):
    clip = tmp_path / "present.mp4"
    gen = _clip(clip, 446)
    manifest = _manifest(
        tmp_path,
        gen.compute_sha256(clip),
        gen.compute_phashes(clip),
        "FX-TRAIN-002",
        "TRAIN",
    )
    missing = tmp_path / "missing.mp4"
    with pytest.raises(FixtureSeparationError, match="missing file"):
        validate_dataset_for_context([missing], manifest, LoaderContext.STUDY)


def test_hash_exception_is_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    clip = tmp_path / "present.mp4"
    gen = _clip(clip, 447)
    manifest = _manifest(
        tmp_path,
        gen.compute_sha256(clip),
        gen.compute_phashes(clip),
        "FX-TRAIN-003",
        "TRAIN",
    )

    def _boom(_path: Path) -> str:
        raise OSError("disk gone")

    monkeypatch.setattr("trace_search.fixtures.guards.sha256_file", _boom)
    with pytest.raises(FixtureSeparationError, match="FAIL-CLOSED"):
        validate_dataset_for_context([clip], manifest, LoaderContext.TRAINING)


def test_smoke_test_refuses_every_tag_except_smoke_only(tmp_path: Path):
    clip = tmp_path / "tagged.mp4"
    gen = _clip(clip, 601)
    sha = gen.compute_sha256(clip)
    phashes = gen.compute_phashes(clip)
    refused = [tag for tag in FixtureUseTag if tag is not FixtureUseTag.SMOKE_ONLY]
    assert {tag.value for tag in refused} >= {"CODE_TEST", "TRAIN", "DEV", "HOLD"}
    for tag in refused:
        manifest = _manifest(tmp_path, sha, phashes, "FX-TAGGED-001", tag.value)
        with pytest.raises(FixtureSeparationError, match=tag.value):
            check_fixture_allowed(clip, LoaderContext.SMOKE_TEST, manifest)


def test_smoke_test_allows_smoke_only(tmp_path: Path):
    clip = tmp_path / "smoke.mp4"
    gen = _clip(clip, 602)
    manifest = _manifest(
        tmp_path,
        gen.compute_sha256(clip),
        gen.compute_phashes(clip),
        "FX-SMOKE-SYN-602",
        "SMOKE_ONLY",
    )
    check_fixture_allowed(clip, LoaderContext.SMOKE_TEST, manifest)


def test_smoke_test_refuses_a_file_with_no_row(tmp_path: Path):
    clip = tmp_path / "unknown.mp4"
    _clip(clip, 603)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": []}))
    with pytest.raises(FixtureSeparationError, match="no manifest row"):
        check_fixture_allowed(clip, LoaderContext.SMOKE_TEST, manifest)


@pytest.mark.parametrize(
    "context",
    [
        LoaderContext.UNIT_TEST,
        LoaderContext.TRAINING,
        LoaderContext.EVALUATION,
        LoaderContext.STUDY,
    ],
)
def test_other_contexts_still_allow_an_unidentified_file(tmp_path: Path, context: LoaderContext):
    clip = tmp_path / "unknown.mp4"
    _clip(clip, 604)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": []}))
    check_fixture_allowed(clip, context, manifest)


def test_loader_allows_train_in_training(tmp_path: Path):
    clip = tmp_path / "train.mp4"
    gen = _clip(clip, 555)
    manifest = _manifest(
        tmp_path,
        gen.compute_sha256(clip),
        gen.compute_phashes(clip),
        "FX-TRAIN-001",
        "TRAIN",
    )
    validate_dataset_for_context([clip], manifest, LoaderContext.TRAINING)
