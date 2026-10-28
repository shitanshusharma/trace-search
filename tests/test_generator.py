"""Tests for synthetic video generation."""

import tempfile
from pathlib import Path

import pytest

from trace_search.fixtures.generator import (
    SyntheticVideoGenerator,
    generate_smoke_fixtures,
)


def test_generator_determinism():
    """Same seed produces identical videos (same SHA-256)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        gen1 = SyntheticVideoGenerator(seed=42)
        path1 = output_dir / "test1.mp4"
        gen1.generate_clip(
            output_path=path1,
            duration_sec=2.0,
            fps=10,
            width=640,
            height=480,
            num_figures=2,
        )

        gen2 = SyntheticVideoGenerator(seed=42)
        path2 = output_dir / "test2.mp4"
        gen2.generate_clip(
            output_path=path2,
            duration_sec=2.0,
            fps=10,
            width=640,
            height=480,
            num_figures=2,
        )

        sha1 = SyntheticVideoGenerator.compute_sha256(path1)
        sha2 = SyntheticVideoGenerator.compute_sha256(path2)

        assert sha1 == sha2, "Same seed should produce identical videos"


def test_generator_different_seeds():
    """Different seeds produce different videos."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        gen1 = SyntheticVideoGenerator(seed=42)
        path1 = output_dir / "test1.mp4"
        gen1.generate_clip(
            output_path=path1,
            duration_sec=2.0,
            fps=10,
            width=640,
            height=480,
        )

        gen2 = SyntheticVideoGenerator(seed=123)
        path2 = output_dir / "test2.mp4"
        gen2.generate_clip(
            output_path=path2,
            duration_sec=2.0,
            fps=10,
            width=640,
            height=480,
        )

        sha1 = SyntheticVideoGenerator.compute_sha256(path1)
        sha2 = SyntheticVideoGenerator.compute_sha256(path2)

        assert sha1 != sha2, "Different seeds should produce different videos"


def test_phash_computation():
    """pHash computation returns 10 hashes in correct format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        gen = SyntheticVideoGenerator(seed=42)
        path = output_dir / "test.mp4"
        gen.generate_clip(
            output_path=path,
            duration_sec=3.0,
            fps=10,
            width=640,
            height=480,
        )

        phashes = SyntheticVideoGenerator.compute_phashes(path, num_samples=10)

        assert len(phashes) == 10, "Should return 10 pHashes"
        assert len(set(phashes)) >= 8, "Most pHashes should be unique (allow some similarity)"
        assert all(len(h) == 16 for h in phashes), "Each pHash should be 16 hex chars"


def test_generate_smoke_fixtures_minimum(
    canonical_smoke: tuple[Path, list[dict[str, object]]],
) -> None:
    """generate_smoke_fixtures creates at least 5 clips."""
    _clips, manifest_entries = canonical_smoke

    assert len(manifest_entries) == 5, "Should generate 5 clips"

    for entry in manifest_entries:
        assert entry["type"] == "synthetic_clip"
        assert entry["use"] == "SMOKE_ONLY"
        assert entry["generator"] == "SyntheticVideoGenerator"
        assert "fixture_id" in entry
        assert entry["fixture_id"].startswith("FX-SMOKE-")
        assert len(entry["sha256"]) == 64
        assert len(entry["phash"]) == 10


def test_generate_smoke_fixtures_has_lowlight(
    canonical_smoke: tuple[Path, list[dict[str, object]]],
) -> None:
    """At least one fixture has low brightness."""
    _clips, manifest_entries = canonical_smoke

    has_lowlight = False
    for entry in manifest_entries:
        seed_data = entry["seed"]
        assert isinstance(seed_data, str)
        if "lowlight" in seed_data.lower():
            has_lowlight = True
            break

    assert has_lowlight, "Should have at least one low-light clip"


def test_generate_smoke_fixtures_determinism(
    canonical_smoke: tuple[Path, list[dict[str, object]]],
    tmp_path: Path,
) -> None:
    """Same seed produces same manifest entries."""
    _clips, entries1 = canonical_smoke
    entries2 = generate_smoke_fixtures(
        output_dir=tmp_path / "smoke2",
        base_seed=42,
        num_clips=5,
    )

    for e1, e2 in zip(entries1, entries2, strict=True):
        assert e1["sha256"] == e2["sha256"], "Same seed should produce same SHA-256"
        assert e1["phash"] == e2["phash"], "Same seed should produce same pHashes"


def test_generate_smoke_fixtures_rejects_too_few():
    """Generating fewer than 5 clips raises an error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "smoke"

        with pytest.raises(ValueError, match="at least 5 clips"):
            generate_smoke_fixtures(
                output_dir=output_dir,
                base_seed=42,
                num_clips=3,
            )
